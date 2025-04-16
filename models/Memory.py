import torch
import torch.nn as nn
import faiss
import torch.nn.functional as F
import numpy as np

try:
    import faiss.contrib.torch_utils
except ImportError:
    print(
        "faiss.contrib.torch_utils not found. GPU FAISS operations might require manual numpy conversion."
    )


class MemoryBankWithRetrieval:
    def __init__(self, seq_len, dim, pred_len, use_gpu=False, gpu_index=0):
        """
        Initialize a memory bank to store and retrieve feature vectors using FAISS.

        Args:
            seq_len (int): Length of the input sequence (x).
            dim (int): Dimensionality of the feature vectors.
            pred_len (int, optional): Length of the output sequence (y). If None, assumed equal to seq_len.
            use_gpu (bool): Whether to use GPU for FAISS.
            gpu_index (int): GPU device index to use.
        """
        self.seq_len = seq_len
        self.dim = dim
        self.pred_len = pred_len
        self.use_gpu = use_gpu
        self.device = torch.device(f"cuda:{gpu_index}" if use_gpu else "cpu")

        # 初始化 FAISS 索引（只存储 x）
        self.faiss_dim = seq_len * dim
        if use_gpu:
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = gpu_index
            # 使用 GpuIndexFlatL2，支持直接操作 GPU 上的数据
            self.index = faiss.GpuIndexFlatL2(res, self.faiss_dim, config)
        else:
            self.index = faiss.IndexFlatL2(self.faiss_dim)

        self.y_store = []  # 用于存储所有的ground truth y
        self.x_mark_store = []  # 用于存储所有的x_mark
        self.y_mark_store = []  # 用于存储所有的ground truth y

    def load_dataset(self, dataset_loader):
        """
        Load a dataset into the memory bank in one time.
        Note: you may need more GPU memory to load the whole dataset.

        Args:
            dataset (Dataset): Dataset to load.
        """

        num_batches = len(dataset_loader)
        y_store_list = []
        x_mark_store_list = []
        y_mark_store_list = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            dataset_loader
        ):
            batch_x = batch_x.float().to(self.device).detach()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()

            y_store_list.append(batch_y[:, -self.pred_len :, :])
            x_mark_store_list.append(batch_x_mark)
            y_mark_store_list.append(batch_y_mark)

            self.update(batch_x)

            print(
                f"Processing batch {i + 1}/{num_batches} for memory bank...",
                end="\r",
            )

        self.y_store = torch.cat(y_store_list, dim=0).to(self.device)
        self.x_mark_store = torch.cat(x_mark_store_list, dim=0).to(self.device)
        self.y_mark_store = torch.cat(y_mark_store_list, dim=0).to(self.device)
        print(
            f"Memory bank built successfully. Index size: {self.index.ntotal}, Store size: {len(self.y_store)}"
        )

    def update(self, x: torch.Tensor):
        """
        Update the memory bank with new x vectors.

        Args:
            x (Tensor): Input sequence, shape (total_samples, seq_len, dim).
        """
        total_samples = x.size(0)
        x_flat = x.view(total_samples, -1)  # (total_samples, seq_len * dim)

        if self.use_gpu:
            # 直接添加 GPU 上的数据
            self.index.add(x_flat)  # GpuIndexFlatL2 直接接受 PyTorch 张量
        else:
            # 如果没有 GPU，则需要转换为 NumPy
            self.index.add(x_flat.cpu().numpy())

    def retrieve_similar(self, query: torch.Tensor, k: int):
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
            query.view(batch_size, -1).to(self.device).detach()
        )  # (batch_size, seq_len * dim)

        # 检索 top-k 相似向量
        distances, indices, embeddings = self.index.search_and_reconstruct(
            query_flat, k
        )

        embeddings = embeddings.reshape(batch_size, k, seq_len, self.dim)

        # 将 indices 转换为张量，并从 y_store 中获取对应的 y
        indices_tensor = (
            indices.to(self.device)
            if self.use_gpu
            else torch.tensor(indices, dtype=torch.long, device=self.y_store.device)
        )

        topk_y_gt = self.y_store[
            indices_tensor
        ]  # 形状为 (batch_size, k, pred_len, dim)

        mark_x = self.x_mark_store[
            indices_tensor
        ]  # 形状为 (batch_size, k, seq_len, dim)

        return (
            (
                embeddings.to(query.device)
                if self.use_gpu
                else torch.from_numpy(embeddings).to(query.device)
            ),
            mark_x.to(query.device),
            topk_y_gt.to(query.device),
            (
                (distances).to(query.device)
                if self.use_gpu
                else torch.from_numpy(distances).to(query.device)
            ),
        )
