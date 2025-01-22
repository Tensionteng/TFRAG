import torch
import torch.nn as nn
import faiss
import torch.nn.functional as F


class MemoryBankWithRetrieval:
    def __init__(self, seq_len, feature_dim, use_gpu=False):
        """
        Initialize a memory bank to store and retrieve feature vectors using FAISS.
        Args:
            seq_len (int): Length of the sequence.
            feature_dim (int): Dimensionality of the feature vectors.
            use_gpu (bool): Whether to use GPU for FAISS.
        """
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(7 * seq_len)  # L2 distance for similarity
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def update(self, features):
        """
        Update the memory bank with new features.
        Args:
            features (Tensor): New feature vectors to add (batch_size, seq_len, feature_dim).
        """
        batch_size = features.size(0)

        features_np = features.cpu().detach().numpy()
        features_flat = features_np.reshape(
            batch_size, -1
        )  # (batch_size, seq_len * feature_dim)
        self.index.add(features_flat)

    def retrieve_similar(self, query, k=5):
        """
        Retrieve the top-k most similar vectors from the memory bank using FAISS.
        Args:
            query (Tensor): Query feature vector (batch_size, seq_len, feature_dim).
            k (int): Number of most similar vectors to retrieve.
        Returns:
            Tensor: Retrieved top-k vectors (batch_size, k, seq_len, feature_dim).
        """
        batch_size, seq_len, feature_dim = query.shape

        query_flat_np = (
            query.reshape(batch_size, -1).detach().cpu().numpy()
        )  # (batch_size, seq_len * feature_dim)

        distances, indices, reconstructed_vectors = self.index.search_and_reconstruct(
            query_flat_np, k
        )  # (batch_size, k), (batch_size, k, seq_len * feature_dim)

        # Reshape reconstructed vectors to original sequence format
        reconstructed_vectors = reconstructed_vectors.reshape(
            batch_size, k, seq_len, feature_dim
        )
        topk_vectors = torch.tensor(reconstructed_vectors, device=query.device)

        return topk_vectors

    def fuse_sequences(self, pred_seq: torch.Tensor, similar_seqs, fusion_mode="mean"):
        """
        Fuse the predicted sequence with the retrieved similar sequences.
        Args:
            pred_seq (Tensor): Predicted sequence (batch_size, seq_len, feature_dim).
            similar_seqs (Tensor): Retrieved similar sequences (batch_size, k, seq_len, feature_dim).
            fusion_mode (str): Fusion mode ('mean' or 'mlp').
        Returns:
            Tensor: Fused sequence (batch_size, seq_len, feature_dim).
        """
        batch_size, k, seq_len, feature_dim = similar_seqs.size()
        similar_seqs = similar_seqs.mean(dim=1).squeeze(
            1
        )  # (batch_size, seq_len, feature_dim)

        if fusion_mode == "mean":
            fused_seq = (pred_seq + similar_seqs) / 2  # Average fusion

        elif fusion_mode == "mlp":

            combined = torch.cat(
                [pred_seq, similar_seqs], dim=-1
            )  # (batch_size, seq_len, feature_dim * 2)

            mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim * 4),
                nn.ReLU(),
                nn.Linear(feature_dim *4 , feature_dim),
            ).to(pred_seq.device)
            fused_seq = mlp(combined)  # (batch_size, seq_len, feature_dim)
        else:
            raise ValueError("Fusion mode must be 'mean' or 'mlp'")

        return fused_seq

    def compute_trend_loss(self, pred_seq, similar_seqs, kernel_size=8):
        """
        Compute trend loss using average pooling and MSE.
        Args:
            pred_seq (Tensor): Predicted sequence (batch_size, seq_len, feature_dim).
            similar_seqs (Tensor): Retrieved similar sequences (batch_size, k, seq_len, feature_dim).
        Returns:
            Tensor: Trend loss.
        """
        pred_trend = F.avg_pool1d(pred_seq.permute(0, 2, 1), kernel_size).permute(
            0, 2, 1
        )

        similar_seqs = similar_seqs.mean(dim=1).squeeze(
            1
        )  # (batch_size, seq_len, feature_dim)

        similar_trends = F.avg_pool1d(
            similar_seqs.permute(0, 2, 1),
            kernel_size=8,
        ).permute(0, 2, 1)

        trend_loss = F.mse_loss(pred_trend, similar_trends)
        return trend_loss

    def compute_frequency_loss(self, pred_seq, similar_seqs):
        """
        Compute frequency loss using FFT and MSE.
        Args:
            pred_seq (Tensor): Predicted sequence (batch_size, seq_len, feature_dim).
            similar_seqs (Tensor): Retrieved similar sequences (batch_size, k, seq_len, feature_dim).
        Returns:
            Tensor: Frequency loss.
        """
        pred_fft = torch.fft.rfft(pred_seq, dim=1)

        similar_seqs = similar_seqs.mean(dim=1).squeeze(
            1
        )  # (batch_size, seq_len, feature_dim)

        similar_ffts = torch.fft.rfft(similar_seqs, dim=1)

        energy_diff = torch.abs(pred_fft - similar_ffts)
        frequency_loss = torch.mean(energy_diff**2)
        return frequency_loss
