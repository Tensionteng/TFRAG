"""
Model Factory for creating forecasting models with optional RAG+RL enhancement.

This factory provides a unified interface for creating models, automatically
wrapping them with RAGPlugin when use_rag is enabled.

Example:
    >>> from models.model_factory import create_model
    >>> model = create_model(args)
    >>> # If args.use_rag=True, model is automatically wrapped with RAGPlugin
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import importlib

from models.rag_plugin import RAGPlugin


def create_model(args, model_class: Optional[type] = None) -> nn.Module:
    """
    Create a forecasting model with optional RAG+RL enhancement.
    
    Args:
        args: Configuration arguments
        model_class: Optional specific model class to instantiate.
                    If None, uses args.model to determine the class.
    
    Returns:
        nn.Module: The created model (possibly wrapped with RAGPlugin)
    
    Example:
        >>> args.model = 'iTransformer'
        >>> args.use_rag = True
        >>> model = create_model(args)
        >>> # model is now a RAGPlugin wrapping iTransformer
    """
    # Import the model module dynamically
    if model_class is None:
        model_name = args.model
        try:
            model_module = importlib.import_module(f'models.{model_name}')
            model_class = model_module.Model
        except ImportError as e:
            raise ImportError(
                f"Failed to import model '{model_name}'. "
                f"Make sure models/{model_name}.py exists."
            ) from e
    
    # Create base model
    base_model = model_class(args).float()
    
    # Wrap with RAGPlugin if enabled
    if getattr(args, 'use_rag', False):
        base_model = RAGPlugin(base_model, args)
        print(f"[RAG] Model wrapped with RAGPlugin (k={args.num_retrieve})")
    
    # Handle multi-GPU
    if getattr(args, 'use_multi_gpu', False) and getattr(args, 'use_gpu', True):
        if hasattr(args, 'device_ids'):
            base_model = nn.DataParallel(base_model, device_ids=args.device_ids)
    
    return base_model


def load_pretrained_model(
    checkpoint_path: str,
    args,
    model_class: Optional[type] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        args: Configuration arguments
        model_class: Optional model class
        strict: Whether to strictly enforce state_dict matching
    
    Returns:
        nn.Module: Loaded model
    """
    model = create_model(args, model_class)
    
    if hasattr(model, 'module'):
        # Unwrap DataParallel if needed
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
    
    print(f"[Model] Loaded checkpoint from {checkpoint_path}")
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: The model to inspect
    
    Returns:
        dict with model information
    """
    info = {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'is_rag_enabled': isinstance(model, RAGPlugin) or 
                         (hasattr(model, 'module') and isinstance(model.module, RAGPlugin)),
    }
    
    if info['is_rag_enabled']:
        info['rag_config'] = {
            'num_retrieve': getattr(model, 'num_retrieve', 'N/A'),
            'gamma_1': getattr(model, 'gamma_1', 'N/A'),
            'gamma_2': getattr(model, 'gamma_2', 'N/A'),
        }
    
    return info


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model to get the base model.
    
    This handles:
    - DataParallel wrapper
    - RAGPlugin wrapper
    
    Args:
        model: Possibly wrapped model
    
    Returns:
        nn.Module: The unwrapped base model
    """
    # Unwrap DataParallel
    if hasattr(model, 'module'):
        model = model.module
    
    # Unwrap RAGPlugin
    if isinstance(model, RAGPlugin):
        model = model.get_base_model()
    
    return model
