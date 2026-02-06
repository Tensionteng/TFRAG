"""
Refactored Long-term Forecasting Experiment with RAG Plugin Support.

This version uses the RAGPlugin architecture for cleaner code and easier maintenance.
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    visual,
    visual_adjustment,
)
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime

from models.model_factory import create_model, unwrap_model
from models.rag_plugin import RAGPlugin

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    Long-term forecasting experiment class with optional RAG+RL enhancement.
    
    The RAG functionality is now handled by RAGPlugin, making this class
    much cleaner and model-agnostic.
    """
    
    def __init__(self, args):
        super().__init__(args)
        
    def _build_model(self):
        """Build model using the factory."""
        model = create_model(self.args)
        return model

    def _get_data(self, flag):
        """Get data loader for the specified split."""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Select optimizer for training."""
        model = unwrap_model(self.model)
        return optim.Adam(model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """Select loss function."""
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """Validation loop."""
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            batch_y=batch_y[:, -self.args.pred_len :, :].to(self.device) if self.args.use_rag else None
                        )
                else:
                    result = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        batch_y=batch_y[:, -self.args.pred_len :, :].to(self.device) if self.args.use_rag else None
                    )
                
                # Handle RAGPlugin output
                if isinstance(result, dict):
                    outputs = result['outputs']
                else:
                    outputs = result
                    
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.detach().cpu())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """Training loop with optional RAG+RL."""
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        # Load memory bank if using RAG
        if self.args.use_rag and isinstance(self.model, (RAGPlugin, nn.DataParallel)):
            model_to_load = unwrap_model(self.model)
            if isinstance(model_to_load, RAGPlugin):
                print("[RAG] Loading training data into memory bank...")
                model_to_load.load_memory_bank(train_loader)

        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # Forward pass
                f_dim = -1 if self.args.features == "MS" else 0
                target = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            batch_y=target
                        )
                else:
                    result = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        batch_y=target
                    )

                # Handle output based on whether RAG is used
                if isinstance(result, dict):
                    # RAGPlugin output
                    outputs = result['outputs']
                    loss = result.get('loss')
                    if loss is None:
                        loss = criterion(outputs, target)
                    
                    # Optional: visualization
                    if 'dist' in result and i % 100 == 0:
                        self._visualize_training(
                            batch_x, target, result, epoch, i
                        )
                else:
                    # Standard model output
                    outputs = result
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(outputs, target)

                train_loss.append(loss.item())

                # Progress logging
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                # Backward pass
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Load best model
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _visualize_training(self, batch_x, batch_y, result, epoch, iter_idx):
        """Visualize training progress (optional)."""
        if not hasattr(self, '_viz_counter'):
            self._viz_counter = 0
        self._viz_counter += 1
        
        if self._viz_counter % 5 != 0:
            return
            
        outputs = result['outputs']
        adjusted = result.get('adjusted_outputs', outputs)
        dist = result.get('dist')
        
        # Convert to numpy for visualization
        input_np = batch_x.detach().cpu().numpy()
        y_np = batch_y.detach().cpu().numpy()
        pred_np = outputs.detach().cpu().numpy()
        adj_np = adjusted.detach().cpu().numpy()
        
        # Concatenate for visualization
        gt = np.concatenate((input_np[0, :, -1], y_np[0, :, -1]), axis=0)
        pd = np.concatenate((input_np[0, :, -1], pred_np[0, :, -1]), axis=0)
        ad = np.concatenate((input_np[0, :, -1], adj_np[0, :, -1]), axis=0)
        
        folder = f"./adjust_result/{self.args.model_id}_{self.args.model}/"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # Simplified visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(gt, label='Ground Truth', alpha=0.7)
        plt.plot(pd, label='Prediction', alpha=0.7)
        plt.plot(ad, label='Adjusted', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(folder, f"epoch{epoch}_{iter_idx}.png"))
        plt.close()

    def test(self, setting, test=0):
        """Test loop."""
        folder_path = os.path.join("./test_results", setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        test_data, test_loader = self._get_data(flag="test")
        
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Handle output
                if isinstance(result, dict):
                    outputs = result['outputs']
                else:
                    outputs = result

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # Inverse transform if needed
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])]
                        )
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)
                
                # Visualization
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(
                            input_np.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".png"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # Save results
        folder_path = os.path.join("./results", setting + "/")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
