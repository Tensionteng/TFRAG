import os
from tkinter import NO

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 3
                else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
            )
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    elif args.lradj == "type4":
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def visual_similar(x, similar_x, name="./pic/similar.pdf"):
    """
    Retrieval results visualization
    """
    plt.figure()
    for i in range(similar_x.shape[0]):
        plt.plot(similar_x[i, :], label=f"Similar_{i+1}", linewidth=2)
    plt.plot(x, label="X", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def visual_adjustment(
    true,
    s_gt,
    preds=None,
    adjusted_preds=None,
    mean=None,
    diff=None,
    action=None,
    name="./pic/adjustment.pdf",
):
    """
    Adjustment results visualization
    """
    plt.figure()
    plt.plot(true, label="GT", linewidth=2)
    plt.plot(s_gt, label="Similar_GT", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    if adjusted_preds is not None:
        plt.plot(adjusted_preds, label="Adjusted Prediction", linewidth=2)
    if mean is not None:
        plt.plot(mean, label="Mean", linewidth=2)
    if diff is not None:
        plt.plot(diff, label="Diff", linewidth=2)
    if action is not None:
        plt.plot(action, label="Action", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_rfft(signal, freq_ratio=0.1):
    num_dims = signal.dim()

    if num_dims not in {3, 4}:
        raise ValueError("Input signal must be 3D or 4D tensor.")

    # 选择 FFT 的维度
    fft_dim = 1 if num_dims == 3 else 2

    # 计算 FFT
    fft_coeffs = torch.fft.rfft(signal, dim=fft_dim)

    # 计算截止频率
    seq_len = signal.shape[fft_dim - 1]
    cutoff = int(seq_len * freq_ratio)

    # 创建高频和低频掩码
    mask = torch.zeros_like(fft_coeffs)
    mask[..., cutoff:, :] = 1.0

    # 分离高频和低频分量
    high_coeffs = fft_coeffs * mask
    low_coeffs = fft_coeffs * (1 - mask)

    # 逆 FFT
    high_signal = torch.fft.irfft(high_coeffs, dim=fft_dim)
    low_signal = torch.fft.irfft(low_coeffs, dim=fft_dim)

    return high_signal, low_signal


def mse_reward_func(output, adjusted_output, gt):
    mse_output = ((output - gt) ** 2).mean(dim=(1, 2))
    mse_adjusted = ((adjusted_output - gt) ** 2).mean(dim=(1, 2))
    # mse_output = (output - gt) ** 2
    # mse_adjusted = (adjusted_output - gt) ** 2
    # return torch.where(
    #     mse_adjusted < mse_output,
    #     torch.tensor(1.0, device=output.device),
    #     torch.tensor(0.0, device=output.device),
    # )
    return mse_output - mse_adjusted


def mae_reward_func(output, adjusted_output, gt):
    mae_output = torch.abs(output - gt).mean(dim=(1, 2))
    mae_adjusted = torch.abs(adjusted_output - gt).mean(dim=(1, 2))
    # mae_output = torch.abs(output - gt)
    # mae_adjusted = torch.abs(adjusted_output - gt)
    # return torch.where(
    #     mae_adjusted < mae_output,
    #     torch.tensor(1.0, device=output.device),
    #     torch.tensor(0.0, device=output.device),
    # )
    return mae_output - mae_adjusted

def frequcncy_reward_func(output, adjusted_output, gt, high_freq_cutoff_ratio=0.25):
    return torch.abs(output - gt).mean(dim=(1, 2)) - torch.abs(adjusted_output - gt).mean(dim=(1, 2))
    


def sample_bernoulli(weights, k):
    """根据概率张量进行k次伯努利采样

    Args:
        p (torch.Tensor): 形状为 (bs, len, dim) 的概率张量，每个元素在0到1之间
        k (int): 采样次数

    Returns:
        torch.Tensor: 形状为 (k, bs, len, dim) 的布尔张量，每个元素表示采样结果
    """
    # 生成形状为(k, bs, len, dim)的均匀分布随机数
    randoms = torch.rand(
        (k, *weights.shape), device=weights.device, dtype=weights.dtype
    )
    # 通过比较随机数与概率得到采样结果
    samples = randoms < weights
    return samples


def cal_accuracy(y_pred, y_true):
    """计算每个batch的准确率

    Args:
        y_pred (torch.Tensor): 预测值，形状为(bs, len, dim)
        y_true (torch.Tensor): 真实值，形状为(bs, len, dim)

    Returns:
        torch.Tensor: 每个batch的准确率，形状为(bs,)
    """
    # 计算每个样本的正确预测数量
    correct = (y_pred == y_true).sum(dim=(1, 2))
    # 计算每个样本的总预测数量
    total = y_true.shape[1] * y_true.shape[2]
    # 返回每个batch的准确率
    return correct.float() / total
