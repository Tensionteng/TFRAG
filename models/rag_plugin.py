"""
RAG (Retrieval-Augmented Generation) Plugin Module

This module provides a plug-and-play RAG+RL enhancement for time series forecasting models.
Usage:
    1. Wrap your model with RAGPlugin to enable RAG+RL
    2. Or use RAGPlugin as a standalone component in your experiment

Example:
    >>> from models.rag_plugin import RAGPlugin
    >>> model = YourModel(args)
    >>> rag_model = RAGPlugin(model, args)
    >>> outputs = rag_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np

from models.Memory import MemoryBankWithRetrieval
from utils.tools import mae_reward_func, mse_reward_func


class PolicyHead(nn.Module):
    """
    Generic policy head for RL-based adjustment.
    Can be attached to any forecasting model.
    """
    
    def __init__(
        self,
        d_model: int,
        pred_len: int,
        c_out: int,
        hidden_dim: int = 128,
        mode: str = "concat",  # "concat" or "diff"
    ):
        super().__init__()
        self.mode = mode
        self.pred_len = pred_len
        self.c_out = c_out
        
        # Input dimension depends on mode
        # concat: pred_len * 2 (output + retrieved)
        # diff: pred_len (output - retrieved)
        input_dim = d_model * 2 if mode == "concat" else d_model
        
        # Policy network layers
        self.action_mean = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, c_out, kernel_size=3, padding=1),
        )
        self.action_logstd = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, c_out, kernel_size=3, padding=1),
        )
        
    def forward(self, outputs: torch.Tensor, retrieved: torch.Tensor) -> torch.distributions.Normal:
        """
        Compute action distribution given outputs and retrieved sequences.
        
        Args:
            outputs: [B, pred_len, c_out] model predictions
            retrieved: [B, pred_len, c_out] retrieved ground truth
            
        Returns:
            Normal distribution for RL sampling
        """
        if self.mode == "concat":
            # Concatenate along feature dimension
            x = torch.cat([outputs, retrieved], dim=-1)  # [B, pred_len, 2*c_out]
        else:  # diff
            x = retrieved - outputs  # [B, pred_len, c_out]
            
        # Transpose for Conv1d: [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1)
        
        action_mean = self.action_mean(x).permute(0, 2, 1)  # [B, pred_len, c_out]
        action_logstd = self.action_logstd(x).permute(0, 2, 1)
        action_logstd = torch.clamp(action_logstd, min=-5, max=-1)
        action_std = torch.exp(action_logstd)
        
        return torch.distributions.Normal(action_mean, action_std)


class RAGPlugin(nn.Module):
    """
    Plug-and-play RAG+RL wrapper for time series forecasting models.
    
    This wrapper adds retrieval-augmented generation and reinforcement learning
    capabilities to any base forecasting model without modifying its architecture.
    
    Args:
        base_model: The base forecasting model (e.g., iTransformer, PatchTST)
        args: Configuration arguments containing:
            - use_rag: bool, whether to enable RAG
            - num_retrieve: int, number of samples to retrieve
            - seq_len: int, input sequence length
            - pred_len: int, prediction length
            - enc_in: int, encoder input dimension
            - d_model: int, model dimension
            - gemma_1: float, weight for base loss
            - gemma_2: float, weight for RL loss
            - use_gpu: bool, whether to use GPU
            - gpu: int, GPU device index
    """
    
    def __init__(self, base_model: nn.Module, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.use_rag = getattr(args, 'use_rag', False)
        
        if self.use_rag:
            # Initialize memory bank
            self.memory_bank = MemoryBankWithRetrieval(
                seq_len=args.seq_len,
                dim=args.enc_in,
                pred_len=args.pred_len,
                use_gpu=getattr(args, 'use_gpu', True) and torch.cuda.is_available(),
                gpu_index=getattr(args, 'gpu', 0),
            )
            
            # Initialize policy head
            self.policy_head = PolicyHead(
                d_model=args.d_model,
                pred_len=args.pred_len,
                c_out=args.c_out,
                mode="concat",
            )
            
            self.num_retrieve = getattr(args, 'num_retrieve', 5)
            self.num_samples = getattr(args, 'num_rl_samples', 8)
            self.gamma_1 = getattr(args, 'gamma_1', 0.5)
            self.gamma_2 = getattr(args, 'gamma_2', 0.5)
            
        self._training_mode = True
        
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with optional RAG+RL enhancement.
        
        Returns:
            dict containing:
                - 'outputs': model predictions [B, pred_len, c_out]
                - 'loss': total loss (if in training and use_rag=True)
                - 'base_loss': base MSE loss
                - 'rl_loss': reinforcement learning loss (if use_rag)
                - 'dist': action distribution (if use_rag)
                - 'adjusted_outputs': adjusted predictions (if use_rag)
        """
        # Get base model predictions
        if hasattr(self.base_model, 'forecast'):
            outputs = self.base_model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            outputs = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
        result = {'outputs': outputs}
        
        # If not using RAG, return base outputs only
        if not self.use_rag or not self.training:
            return result
            
        # RAG+RL enhancement
        batch_y = kwargs.get('batch_y')
        if batch_y is None:
            raise ValueError("batch_y is required for RAG training")
            
        # Retrieve similar sequences
        similar_seqs, similar_seqs_mark, similar_seqs_gt, distances = \
            self.memory_bank.retrieve_similar(x_enc, k=self.num_retrieve)
        
        # Compute similarity weights
        d_min = distances.min(dim=1, keepdim=True)[0]
        d_max = distances.max(dim=1, keepdim=True)[0]
        weights = (distances - d_min) / (d_max - d_min + 1e-4)
        weights = torch.exp(-weights)
        weights = F.softmax(weights, dim=1)
        
        # Weighted average of retrieved sequences
        retrieved_gt = torch.sum(
            weights.unsqueeze(-1).unsqueeze(-1) * similar_seqs_gt,
            dim=1,
        )
        
        # Get action distribution from policy head
        dist = self.policy_head(outputs, retrieved_gt)
        
        # RL sampling
        if self.training:
            log_probs = []
            rewards = []
            
            for _ in range(self.num_samples):
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Smooth action with pooling
                action = F.avg_pool1d(
                    action.permute(0, 2, 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).permute(0, 2, 1)
                
                adjusted = outputs + action
                
                # Compute reward
                reward = mae_reward_func(outputs, adjusted, batch_y) + \
                        mse_reward_func(outputs, adjusted, batch_y)
                
                log_probs.append(log_prob)
                rewards.append(reward)
            
            log_probs = torch.stack(log_probs, dim=0)
            rewards = torch.stack(rewards, dim=0)
            
            # Normalize rewards
            mean_reward = rewards.mean(dim=0, keepdim=True)
            std_reward = rewards.std(dim=0, keepdim=True) + 1e-4
            advantages = (rewards - mean_reward) / std_reward
            
            # Compute RL loss
            per_step_loss = log_probs * advantages
            rl_loss = -per_step_loss.mean()
            
            # Base MSE loss
            base_loss = F.mse_loss(outputs, batch_y)
            
            # Total loss
            total_loss = self.gamma_1 * base_loss + self.gamma_2 * rl_loss
            
            result.update({
                'loss': total_loss,
                'base_loss': base_loss,
                'rl_loss': rl_loss,
                'dist': dist,
                'adjusted_outputs': outputs + dist.mean,
            })
        else:
            # Inference: use mean action for adjustment
            with torch.no_grad():
                adjusted = outputs + dist.mean
                result['adjusted_outputs'] = adjusted
                
        return result
        
    def load_memory_bank(self, train_loader):
        """Load training data into memory bank."""
        if self.use_rag:
            self.memory_bank.load_dataset(train_loader)
            
    def train(self, mode: bool = True):
        """Set training mode."""
        self._training_mode = mode
        super().train(mode)
        return self
        
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
        
    @property
    def training(self):
        return self._training_mode
        
    def get_base_model(self):
        """Get the underlying base model."""
        return self.base_model
