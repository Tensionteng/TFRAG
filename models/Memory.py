import torch
import torch.nn as nn
import faiss
import torch.nn.functional as F
import numpy as np


class MemoryBankWithRetrieval:
    def __init__(self, window_size, dim, pred_len, use_gpu=False, gpu_index=0):
        """
        Initialize a memory bank to store and retrieve feature vectors using FAISS.

        Args:
            seq_len (int): Length of the input sequence (x).
            dim (int): Dimensionality of the feature vectors.
            pred_len (int, optional): Length of the output sequence (y). If None, assumed equal to seq_len.
            use_gpu (bool): Whether to use GPU for FAISS.
            gpu_index (int): GPU device index to use.
        """
        self.window_size = window_size
        self.dim = dim
        self.pred_len = pred_len
        self.use_gpu = use_gpu
        self.device = torch.device(f"cuda:{gpu_index}" if use_gpu else "cpu")

        # 初始化 FAISS 索引（只存储 x）
        self.faiss_dim = window_size * dim
        index = faiss.IndexFlatL2(self.faiss_dim)  # L2 距离索引
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, gpu_index, index)
        else:
            self.index = index

        self.y_store = None  # 用于存储所有的ground truth y
        self.x_mark_store = None  # 用于存储所有的x_mark

    def load_dataset(self, dataset_loader):
        """
        Load a dataset into the memory bank.

        Args:
            dataset (Dataset): Dataset to load.
        """
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataset_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_x_mark = batch_x_mark.to(self.device)
            batch_y_mark = batch_y_mark.to(self.device)

            # 更新内存库
            self.update(batch_x, batch_x_mark, batch_y, batch_y_mark)

    def update(self, x, x_mark, y, y_mark):
        """
        Update the memory bank with new input x and corresponding y.

        Args:
            x (Tensor): Input sequence, shape (batch_size, seq_len, dim).
            y (Tensor): Output sequence, shape (batch_size, pred_len, dim).
        """
        y = y[:, -self.pred_len :, :]
        x, x_mark, y, y_mark = (
            x.float().detach(),
            x_mark.float().detach(),
            y.float().detach(),
            y_mark.float().detach(),
        )
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1).cpu().numpy()  # (batch_size, seq_len * dim)
        y_flat = y.view(batch_size, -1)  # (batch_size, pred_len * dim)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1).cpu().numpy()  # (batch_size, seq_len * dim)
        y_flat = y.view(batch_size, -1)  # (batch_size, pred_len * dim)
        x_mark = x_mark.view(batch_size, -1)  # (batch_size, seq_len * dim)

        # 添加 x 到 FAISS 索引
        self.index.add(x_flat)

        # 更新 y_store
        if self.y_store is None:
            self.y_store = y_flat
        else:
            self.y_store = torch.cat([self.y_store, y_flat], dim=0)

        if self.x_mark_store is None:
            self.x_mark_store = x_mark
        else:
            self.x_mark_store = torch.cat([self.x_mark_store, x_mark], dim=0)

    def retrieve_similar(self, query, k):
        """
        Retrieve the top-k most similar y vectors from the memory bank using FAISS.

        Args:
            query (Tensor): Query sequence, shape (batch_size, seq_len, dim).
            k (int): Number of similar vectors to retrieve.

        Returns:
            Tensor: Retrieved top-k vectors, shape (batch_size, k, seq_len, dim).
            Tensor: Ground true for retrieved top-k vectors, shape (batch_size, k, pred_len, dim).
        """

        batch_size, seq_len, dim = query.size()

        query_flat = (
            query.view(batch_size, -1).detach().cpu().numpy()
        )  # (batch_size, seq_len * dim)

        # 检索 top-k 相似向量
        distances, indices, embeddings = self.index.search_and_reconstruct(
            query_flat, k
        )

        # 将 indices 转换为张量，并从 y_store 中获取对应的 y
        embeddings = embeddings.reshape(batch_size, k, seq_len, self.dim)

        # 将 indices 转换为张量，并从 y_store 中获取对应的 y
        indices_tensor = torch.tensor(
            indices, dtype=torch.long, device=self.y_store.device
        )

        topk_y_flat = self.y_store[
            indices_tensor
        ]  # 形状为 (batch_size, k, pred_len * dim)

        mark_x = self.x_mark_store[
            indices_tensor
        ]  # 形状为 (batch_size, k, seq_len * dim)

        topk_y_gt = topk_y_flat.reshape(
            batch_size, k, self.pred_len, self.dim
        )  # 重塑为 (batch_size, k, pred_len, dim)

        mark_x = mark_x.reshape(batch_size, k, seq_len, -1)

        return (
            torch.from_numpy(embeddings).to(query.device),
            mark_x.to(query.device),
            topk_y_gt.to(query.device),
            torch.from_numpy(distances).to(query.device),
        )

    def compute_trend_loss(self, pred_seq, similar_seqs, kernel_size=8):
        """
        Compute trend loss using average pooling and MSE.

        Args:
            pred_seq (Tensor): Predicted sequence, shape (batch_size, pred_len, dim).
            similar_seqs (Tensor): Retrieved sequences, shape (batch_size, k, pred_len, dim).
            kernel_size (int): Size of the pooling kernel.

        Returns:
            Tensor: Trend loss.
        """
        pred_trend = F.avg_pool1d(
            pred_seq.permute(0, 2, 1), kernel_size=kernel_size
        ).permute(0, 2, 1)
        similar_trend = F.avg_pool1d(
            similar_seqs.mean(dim=1).permute(0, 2, 1), kernel_size=kernel_size
        ).permute(0, 2, 1)
        return F.mse_loss(pred_trend, similar_trend)

    def compute_frequency_loss(self, pred_seq, similar_seqs):
        """
        Compute frequency loss using FFT and MSE.

        Args:
            pred_seq (Tensor): Predicted sequence, shape (batch_size, pred_len, dim).
            similar_seqs (Tensor): Retrieved sequences, shape (batch_size, k, pred_len, dim).

        Returns:
            Tensor: Frequency loss.
        """
        pred_fft = torch.fft.rfft(pred_seq, dim=1)
        pred_fft_mag = torch.abs(pred_fft[:, 1:, :])  # 去除直流分量
        similar_fft = torch.fft.rfft(similar_seqs.mean(dim=1), dim=1)
        similar_fft_mag = torch.abs(similar_fft[:, 1:, :])
        epsilon = 1e-8
        return F.mse_loss(
            torch.log(pred_fft_mag + epsilon), torch.log(similar_fft_mag + epsilon)
        )
