"""Dataset wrapper that also yields the sample's dataset index.

CRAFT's temporal-exclusion safeguard needs to know *which* training window a
query came from, so it can refuse to retrieve its own temporal neighbourhood.
The stock TSLib datasets return only (x, y, x_mark, y_mark); this wrapper appends
the index without touching them.
"""

from torch.utils.data import DataLoader, Dataset


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return (*self.base[index], index)

    def __getattr__(self, name):
        # Forward attribute access (e.g. .scale, .inverse_transform) to the wrapped
        # dataset so callers cannot tell the difference.
        return getattr(self.base, name)


def indexed_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def unpack_batch(batch):
    """Return (x, y, x_mark, y_mark, idx_or_None) for 4- or 5-tuple batches."""
    if len(batch) == 5:
        return batch[0], batch[1], batch[2], batch[3], batch[4]
    return batch[0], batch[1], batch[2], batch[3], None
