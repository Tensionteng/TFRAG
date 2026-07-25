"""Long-term forecasting experiment, with optional CRAFT (retrieval + RL corrector).

The RAG machinery lives in models/rag_plugin.py; this class only has to
(a) hand the corrector the extra inputs it needs during training (targets and,
when temporal exclusion is on, dataset indices), and (b) evaluate the *bare
backbone*, because that is the deployed artifact.
"""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from data_provider.indexed import indexed_loader, unpack_batch
from exp.exp_basic import Exp_Basic
from models.model_factory import create_model, extract_base_state_dict, unwrap_dataparallel
from models.rag_plugin import RAGPlugin
from utils.losses_freq import build_criterion
from utils.metrics import metric
from utils.run_logger import log_run, variant_name
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        return create_model(self.args)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        # All trainable parameters, including the policy head. Unwrapping here is
        # what previously left the corrector frozen at initialization.
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_all = sum(1 for _ in self.model.parameters())
        if len(params) != n_all:
            print(f"[opt] {len(params)}/{n_all} parameter tensors trainable (rest frozen)")
        return optim.Adam(params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return build_criterion(getattr(self.args, "loss", "MSE"))

    @property
    def _rag(self):
        inner = unwrap_dataparallel(self.model)
        return inner if isinstance(inner, RAGPlugin) else None

    def _dec_inp(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        return (
            torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .to(self.device)
        )

    @staticmethod
    def _outputs_of(result):
        return result["outputs"] if isinstance(result, dict) else result

    # -------------------------------------------------------------- eval loops

    def vali(self, vali_data, vali_loader, criterion):
        """Validation on the backbone alone -- no retrieval, no corrector.

        Model selection therefore optimises the quantity we actually deploy.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = self._dec_inp(batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = self._outputs_of(result)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                target = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                total_loss.append(criterion(outputs, target).detach().cpu())

        self.model.train()
        return float(np.average(total_loss))

    # ------------------------------------------------------------------ train

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        rag = self._rag
        if rag is not None:
            # Bank order must equal dataset index order, so it is built from the
            # dataset (sequentially), never from the shuffled training loader.
            print("[RAG] building memory bank ...")
            rag.load_memory_bank(
                train_data,
                batch_size=max(self.args.batch_size, 64),
                num_workers=self.args.num_workers,
            )
            if rag.exclusion_radius > 0:
                # Queries must carry their dataset index for exclusion to work.
                train_loader = indexed_loader(
                    train_data,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                )
                print(f"[RAG] temporal exclusion radius = {rag.exclusion_radius} steps")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        history = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, rl_loss_log, reward_log = [], [], []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, idx = unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = self._dec_inp(batch_y)

                f_dim = -1 if self.args.features == "MS" else 0
                target = batch_y[:, -self.args.pred_len :, f_dim:]
                kwargs = {}
                if rag is not None:
                    kwargs["batch_y"] = target
                    if idx is not None:
                        kwargs["query_idx"] = idx.to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, **kwargs
                        )
                else:
                    result = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, **kwargs
                    )

                if isinstance(result, dict) and result.get("loss") is not None:
                    loss = result["loss"]
                    rl_loss_log.append(float(result["rl_loss"].item()))
                    reward_log.append(float(result["reward_mean"].item()))
                else:
                    outputs = self._outputs_of(result)[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(outputs, target)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                        f" | {speed:.4f}s/iter; left {left:.1f}s"
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = float(np.average(train_loss))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            msg = (
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} "
                f"({time.time() - epoch_time:.1f}s)"
            )
            if rl_loss_log:
                msg += f" | RL: {np.mean(rl_loss_log):.5f} reward: {np.mean(reward_log):.3f}"
            print(msg)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "test_loss": test_loss,
                    "rl_loss": float(np.mean(rl_loss_log)) if rl_loss_log else None,
                    "reward_mean": float(np.mean(reward_log)) if reward_log else None,
                }
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self._history = history
        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # The deployment artifact: backbone weights with the wrapper stripped.
        torch.save(extract_base_state_dict(self.model), os.path.join(path, "base_model.pth"))
        print(f"[ckpt] backbone-only weights saved to {path}/base_model.pth")
        if getattr(self.args, "slim_ckpt", False):
            # The wrapper checkpoint is redundant once the backbone is extracted and
            # the model is already loaded in memory for testing.
            os.remove(best_model_path)
            print("[ckpt] removed the full wrapper checkpoint (--slim_ckpt)")
        return self.model

    # ------------------------------------------------------------------- test

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            ckpt = os.path.join("./checkpoints", setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        folder_path = os.path.join("./test_results", setting)
        os.makedirs(folder_path, exist_ok=True)

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = self._dec_inp(batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = self._outputs_of(result)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

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

                preds.append(outputs[:, :, f_dim:])
                trues.append(batch_y[:, :, f_dim:])

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inp.shape
                        inp = test_data.inverse_transform(
                            inp.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((inp[0, :, -1], trues[-1][0, :, -1]), axis=0)
                    pd_ = np.concatenate((inp[0, :, -1], preds[-1][0, :, -1]), axis=0)
                    visual(gt, pd_, os.path.join(folder_path, f"{i}.png"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)

        out_dir = os.path.join("./results", setting)
        os.makedirs(out_dir, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}")

        with open("result_long_term_forecast.txt", "a") as f:
            f.write(setting + "  \n")
            f.write(f"mse:{mse}, mae:{mae}\n\n")

        np.save(os.path.join(out_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        if getattr(self.args, "save_arrays", True):
            np.save(os.path.join(out_dir, "pred.npy"), preds.astype(np.float32))
            np.save(os.path.join(out_dir, "true.npy"), trues.astype(np.float32))
        else:
            print("[io] skipped pred/true arrays (--no_save_arrays)")
        # Per-sample errors: needed for the difficulty-quartile analysis and for
        # paired tests that go below the dataset level.
        np.save(
            os.path.join(out_dir, "per_sample_mse.npy"),
            ((preds - trues) ** 2).mean(axis=(1, 2)).astype(np.float32),
        )
        np.save(
            os.path.join(out_dir, "per_sample_mae.npy"),
            np.abs(preds - trues).mean(axis=(1, 2)).astype(np.float32),
        )

        log_run(
            self.args,
            setting,
            {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe},
            extra={
                "variant": variant_name(self.args),
                "n_test_windows": int(preds.shape[0]),
                "results_dir": out_dir,
                "history": getattr(self, "_history", None),
            },
        )
        return mse, mae
