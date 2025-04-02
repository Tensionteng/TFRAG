from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.Memory import MemoryBankWithRetrieval
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    visual,
    mae_reward_func,
    mse_reward_func,
    visual_similar,
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
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from datetime import datetime
from layers.Autoformer_EncDec import series_decomp

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        if args.use_rag:
            self.memory_bank = MemoryBankWithRetrieval(
                window_size=args.seq_len,
                dim=args.enc_in,
                pred_len=args.pred_len,
                use_gpu=True,
                gpu_index=1,
            )
            self.num_retrieve = args.num_retrieve
            self.old_model = self.model

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _rag_loss(self, pred, y_true, trend_weight=0.1, freq_weight=0.1):
        # 原始 MSE 损失
        mse = nn.MSELoss()(pred, y_true)
        # 从 memory_bank 中检索相似序列
        similar_seqs = self.memory_bank.get_similar_seqs()

        # 计算趋势损失
        trend_loss = self.memory_bank.compute_trend_loss(pred, similar_seqs)

        # 计算频率损失
        freq_loss = self.memory_bank.compute_frequency_loss(pred, similar_seqs)

        # 总损失
        total_loss = mse + trend_weight * trend_loss + freq_weight * freq_loss
        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(outputs, batch_y)

                total_loss.append(loss.detach().cpu())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        if self.args.use_rag:
            print("Loading memory bank...")
            self.memory_bank.load_dataset(train_loader)
            print("Memory bank loading finished.")

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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )

                        if self.args.use_rag:
                            loss = criterion(
                                outputs, batch_y, trend_weight=0.1, freq_weight=0.1
                            )
                        else:
                            loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)

                    if self.args.use_rag:
                        (
                            similar_seqs,
                            similar_seqs_mark_x,
                            similar_seqs_gt,
                            distances,
                        ) = self.memory_bank.retrieve_similar(
                            batch_x,
                            k=5,
                        )

                        batch_size, k, seq_len, dim = similar_seqs.size()
                        _, _, pred_len, _ = similar_seqs_gt.size()
                        _, _, _, x_mark_dim = similar_seqs_mark_x.size()

                        d_min = distances.min(dim=1, keepdim=True)[0]
                        d_max = distances.max(dim=1, keepdim=True)[0]
                        weights = (distances - d_min) / (d_max - d_min + 1e-8)
                        weights = torch.exp(-weights)
                        weights = F.softmax(weights, dim=1)

                        # 计算加权平均
                        similar_seqs_gt = torch.sum(
                            weights.unsqueeze(-1).unsqueeze(-1) * similar_seqs_gt,
                            dim=1,
                        )

                        # diff版本
                        # diff_y = similar_seqs_gt - outputs.detach()
                        # dist = self.model.get_dist(diff_y, None, None, None)

                        # concat版本
                        dist = self.model.get_dist_cat(
                            torch.cat([outputs, similar_seqs_gt], dim=1)
                        )
                        num_samples = 10
                        log_probs = []
                        rewards = []
                        adjustment_limitation = []
                        # n次采样
                        for j in range(num_samples):
                            action = dist.rsample()
                            # 平均池化，每3步做平均
                            action = F.avg_pool1d(
                                action.permute(0, 2, 1),
                                kernel_size=3,
                                stride=1,
                                padding=1,
                            ).permute(0, 2, 1)
                            adjusted_ouput = outputs + action
                            log_prob = dist.log_prob(action)
                            reward = mae_reward_func(
                                outputs, adjusted_ouput, batch_y
                            ) + mse_reward_func(outputs, adjusted_ouput, batch_y)
                            # 可视化调整后的结果
                            if reward > 0 and i % 100 == 0 and j % 5 == 0:
                                input = batch_x.detach().cpu().numpy()
                                y = batch_y.detach().cpu().numpy()
                                pred = outputs.detach().cpu().numpy()
                                ad_pred = adjusted_ouput.detach().cpu().numpy()
                                gt = np.concatenate(
                                    (input[0, :, -1], y[0, :, -1]), axis=0
                                )
                                pd = np.concatenate(
                                    (input[0, :, -1], pred[0, :, -1]), axis=0
                                )
                                ad = np.concatenate(
                                    (input[0, :, -1], ad_pred[0, :, -1]), axis=0
                                )
                                floder = "./adjust_result/ETTh1_96_96_iTransformer/"
                                if not os.path.exists(floder):
                                    os.makedirs(floder)

                                visual_adjustment(
                                    gt,
                                    pd,
                                    ad,
                                    name=os.path.join(
                                        floder,
                                        f"epoch{epoch}_{i}_iter{j}_reward_{reward}.png",
                                    ),
                                )
                            log_probs.append(log_prob)
                            rewards.append(reward)
                            # 调整值的L2范数，值在340左右
                            adjustment_limitation.append(action.norm(p=2))

                        log_probs = torch.stack(log_probs, dim=0)
                        rewards = torch.tensor(
                            rewards, dtype=torch.float32, device=self.device
                        )

                        mean = rewards.mean(dim=0, keepdim=True)
                        std = rewards.std(dim=0, keepdim=True) + 1e-8
                        advantages = (rewards - mean) / std  # 标准化奖励
                        # 均值大小也在340左右
                        rl_loss = -(
                            torch.mean(
                                torch.exp(log_probs)
                                * advantages.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                            )
                            - 0.01 * torch.mean(torch.stack(adjustment_limitation))
                        )

                        loss = loss + rl_loss

                    train_loss.append(loss.item())

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

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # 可视化检索结果，前三相似的序列和gt高度重叠
                # if similar_seqs is not None and  i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     batch_y = batch_y.detach().cpu().numpy()
                #     similar_seqs = similar_seqs.detach().cpu().numpy()
                #     similar_seqs_gt = similar_seqs_gt.detach().cpu().numpy()
                #     if train_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = train_data.inverse_transform(
                #             input.reshape(shape[0] * shape[1], -1)
                #         ).reshape(shape)
                #         similar_seqs = train_data.inverse_transform(
                #             similar_seqs.reshape(shape[0] * shape[1], -1)
                #         ).reshape(shape)
                #     x = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                #     similar_seqs = np.concatenate(
                #         (similar_seqs[0, :, :, -1], similar_seqs_gt[0, :, :, -1]),
                #         axis=1,
                #     )
                #     visual_similar(
                #         x, similar_seqs, os.path.join("./similar", str(i) + ".png")
                #     )

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

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
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

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".png"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = os.path.join("./results", setting + "/")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = "Not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
