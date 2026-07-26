"""FAISS-backed retrieval database for CRAFT.

The bank stores, for every training window i, the flattened lookback x_i (indexed
by FAISS) and the corresponding future y_i (kept in a side tensor). Bank position
equals the training-dataset index, which is what makes temporal exclusion possible:
window i covers timesteps [i, i + seq_len + pred_len).

Retrieval modes
---------------
``nn``     : true nearest neighbours (default).
``random`` : uniformly sampled bank entries, preserving the RL signal but
             destroying the semantic query-neighbour relation. This is the
             control that isolates "does retrieval structure matter?".

Exclusion
---------
``exclusion_radius`` r drops every neighbour j with |i - j| <= r for query i.
 * r = 0            : self-exclusion only is *not* applied; the query itself can
                      be returned (distance 0). Use only to reproduce the
                      no-safeguard condition.
 * r = 1            : self-exclusion.
 * r >= pred_len    : query and neighbour target windows cannot overlap.
 * r >= seq_len + pred_len : no overlap at all between the query's target window
                      and the neighbour's input or target window.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import faiss
except ImportError as _e:  # pragma: no cover - faiss is a hard requirement for RAG
    raise ImportError(
        "faiss is required for the CRAFT retrieval bank. Install faiss-gpu "
        "(recommended) or faiss-cpu."
    ) from _e

try:
    import faiss.contrib.torch_utils  # noqa: F401  enables torch tensors on GPU indexes

    _TORCH_UTILS = True
except ImportError:  # pragma: no cover
    _TORCH_UTILS = False


class MemoryBankWithRetrieval:
    def __init__(
        self,
        seq_len,
        dim,
        pred_len,
        use_gpu=False,
        gpu_index=0,
        store_on_cpu=False,
    ):
        """
        Args:
            seq_len:   lookback length L.
            dim:       number of variates D.
            pred_len:  horizon P.
            use_gpu:   put the FAISS index on GPU (needs a GPU faiss build).
            gpu_index: CUDA device ordinal for the index.
            store_on_cpu: keep the y side-store in host memory (slower gather,
                       much lower GPU footprint; needed for Traffic-scale data).
        """
        self.seq_len = seq_len
        self.dim = dim
        self.pred_len = pred_len
        self.use_gpu = use_gpu
        self.faiss_dim = seq_len * dim
        self.index_device = torch.device(f"cuda:{gpu_index}" if use_gpu else "cpu")
        self.store_device = torch.device("cpu") if store_on_cpu else self.index_device

        if use_gpu and not hasattr(faiss, "GpuIndexFlatL2"):
            # faiss-cpu is what pyproject.toml pins. Fall back loudly rather than
            # aborting a multi-day campaign on its first run; retrieval is then
            # slower but numerically identical (both are exact flat L2 indexes).
            print(
                "[RAG][warn] this faiss build has no GPU support (faiss-cpu); "
                "falling back to a CPU index. Install faiss-gpu for speed."
            )
            use_gpu = False
            self.use_gpu = False
            self.index_device = torch.device("cpu")
            self.store_device = torch.device("cpu") if store_on_cpu else self.index_device

        if use_gpu:
            res = faiss.StandardGpuResources()
            res.setDefaultNullStreamAllDevices()
            config = faiss.GpuIndexFlatConfig()
            config.device = gpu_index
            self.index = faiss.GpuIndexFlatL2(res, self.faiss_dim, config)
        else:
            self.index = faiss.IndexFlatL2(self.faiss_dim)

        self.y_store = None  # [N, P, D]
        self.n_total = 0

    # ------------------------------------------------------------------ build

    def build_from_dataset(self, dataset, batch_size=64, num_workers=0):
        """Index a dataset in its natural order, so bank position == dataset index.

        Order matters: a shuffled loader would make `exclusion_radius` meaningless.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        xs, ys = [], []
        for batch in loader:
            batch_x, batch_y = batch[0], batch[1]
            xs.append(batch_x.float())
            ys.append(batch_y[:, -self.pred_len :, :].float())

        x_all = torch.cat(xs, dim=0)
        self.y_store = torch.cat(ys, dim=0).to(self.store_device)
        del xs, ys

        self._add_with_cpu_fallback(x_all)
        del x_all

        self.n_total = self.index.ntotal
        assert self.n_total == len(self.y_store), (
            f"index size {self.n_total} != store size {len(self.y_store)}; "
            "the bank was built from a shuffled or truncated loader"
        )
        assert self.n_total == len(dataset), (
            f"bank size {self.n_total} != dataset size {len(dataset)}; "
            "bank position no longer equals dataset index, exclusion would be wrong"
        )
        print(f"[RAG] memory bank built: {self.n_total} windows, dim={self.faiss_dim}")
        return self

    def load_dataset(self, dataset_loader):
        """Legacy entry point (kept so old scripts keep working).

        Builds from an existing loader. If that loader shuffles, bank position does
        NOT equal dataset index, so temporal exclusion is disabled.
        """
        xs, ys = [], []
        for batch_x, batch_y, _, _ in dataset_loader:
            xs.append(batch_x.float())
            ys.append(batch_y[:, -self.pred_len :, :].float())
        x_all = torch.cat(xs, dim=0)
        self.y_store = torch.cat(ys, dim=0).to(self.store_device)
        self._add(x_all)
        self.n_total = self.index.ntotal
        return self

    def _add_with_cpu_fallback(self, x):
        """Populate the index, dropping to a CPU index if the GPU cannot hold it.

        A flat L2 index over ECL/Traffic is ~3 GB (12k windows x 96 x 862 floats).
        With the model also resident that does not fit on an 11 GB card, and FAISS
        raises 'alloc fail type FlatData ... cudaMalloc error out of memory'. Both
        index types are exact, so falling back costs speed and nothing else --
        far better than losing the cell from the results table.
        """
        if not self.use_gpu:
            self._add(x)
            return
        try:
            self._add(x)
        except (RuntimeError, MemoryError) as e:
            msg = str(e)
            if "out of memory" not in msg and "alloc" not in msg:
                raise
            need = x.numel() * 4 / 1e9
            print(
                f"[RAG][warn] GPU FAISS could not allocate the index (~{need:.1f} GB): "
                f"{msg.splitlines()[0][:120]}\n"
                f"[RAG][warn] rebuilding it on CPU; retrieval will be slower but identical."
            )
            self.use_gpu = False
            self.index_device = torch.device("cpu")
            self.index = faiss.IndexFlatL2(self.faiss_dim)
            self.store_device = torch.device("cpu")
            if self.y_store is not None:
                self.y_store = self.y_store.cpu()
            torch.cuda.empty_cache()
            self._add(x)

    def _add(self, x):
        x_flat = x.reshape(x.size(0), -1).contiguous()
        if self.use_gpu and _TORCH_UTILS:
            self.index.add(x_flat.to(self.index_device))
        else:
            self.index.add(np.ascontiguousarray(x_flat.cpu().numpy()))

    # --------------------------------------------------------------- retrieve

    def _search(self, query_flat, k):
        """Returns (distances, indices) as torch tensors on the query device."""
        if self.use_gpu and _TORCH_UTILS:
            d, i = self.index.search(query_flat.to(self.index_device).contiguous(), k)
            return d, i.long()
        d, i = self.index.search(
            np.ascontiguousarray(query_flat.detach().cpu().numpy()), k
        )
        return torch.from_numpy(d), torch.from_numpy(i).long()

    def retrieve(
        self,
        query,
        k,
        query_idx=None,
        exclusion_radius=0,
        mode="nn",
        generator=None,
    ):
        """Retrieve k futures per query.

        Args:
            query:  [B, L, D] lookback batch.
            k:      neighbours to return.
            query_idx: [B] dataset indices of the queries. Required when
                    exclusion_radius > 0.
            exclusion_radius: drop neighbours with |i - j| <= radius.
            mode:   'nn' or 'random'.
        Returns:
            y_neighbours [B, k, P, D], distances [B, k] (both on query.device)
        """
        if self.y_store is None:
            raise RuntimeError("memory bank is empty; call build_from_dataset first")
        B = query.size(0)
        out_device = query.device

        if exclusion_radius > 0 and query_idx is None:
            raise ValueError(
                "exclusion_radius > 0 requires query_idx; run with an indexed "
                "train loader (use_rag enables this automatically)"
            )

        if mode == "random":
            idx, dist = self._random_indices(B, k, query_idx, exclusion_radius, generator)
        elif mode == "nn":
            idx, dist = self._nn_indices(query, k, query_idx, exclusion_radius)
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        y = self.y_store[idx.reshape(-1).to(self.y_store.device)]
        y = y.reshape(B, k, self.pred_len, self.dim)
        return y.to(out_device), dist.to(out_device)

    def _nn_indices(self, query, k, query_idx, radius):
        # Over-retrieve so that, after masking out the excluded neighbourhood, at
        # least k candidates survive. The excluded set spans 2*radius+1 positions.
        pad = 0 if radius == 0 else 2 * radius + 1
        k_search = min(k + pad, self.n_total)
        query_flat = query.reshape(query.size(0), -1).contiguous()
        dist, idx = self._search(query_flat, k_search)
        dist, idx = dist.to(query.device), idx.to(query.device)

        if radius == 0:
            return idx[:, :k], dist[:, :k]

        qi = query_idx.to(query.device).view(-1, 1)
        keep = (idx - qi).abs() > radius
        # FAISS returns -1 when fewer than k_search vectors exist; never keep those.
        keep &= idx >= 0
        return self._take_first_k(idx, dist, keep, k)

    def _random_indices(self, B, k, query_idx, radius, generator):
        # Oversample, then filter, mirroring the NN path so both modes see the
        # same exclusion rule.
        pad = 0 if radius == 0 else 2 * radius + 1
        k_draw = min(k + pad + k, self.n_total)
        device = self.y_store.device
        idx = torch.randint(
            0, self.n_total, (B, k_draw), device=device, generator=generator
        )
        dist = torch.zeros(B, k_draw, device=device)
        if radius == 0:
            return idx[:, :k], dist[:, :k]
        qi = query_idx.to(device).view(-1, 1)
        keep = (idx - qi).abs() > radius
        return self._take_first_k(idx, dist, keep, k)

    @staticmethod
    def _take_first_k(idx, dist, keep, k):
        """Per row, gather the first k entries where keep is True.

        Rows with fewer than k survivors fall back to their own first k entries,
        which can only happen on pathologically small banks; we surface that as an
        error instead of silently leaking an excluded neighbour.
        """
        if int(keep.sum(dim=1).min()) < k:
            raise RuntimeError(
                "not enough non-excluded neighbours; lower exclusion_radius or "
                "raise the retrieval over-fetch"
            )
        # Stable ordering: rank kept entries by their column position.
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :k]
        return torch.gather(idx, 1, order), torch.gather(dist, 1, order)

    # ------------------------------------------------------------ compat shim

    def retrieve_similar(self, query, k):
        """Old signature used by pre-refactor code; NN, no exclusion."""
        y, d = self.retrieve(query, k, mode="nn", exclusion_radius=0)
        return None, None, y, d
