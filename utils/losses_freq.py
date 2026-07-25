"""Frequency-domain training objectives used as CRAFT's external baselines.

These exist so the FreDF / FFL comparison can be run on every dataset instead of
two, which is what the reviewers asked for. All of them are drop-in replacements
for nn.MSELoss on [B, P, C] tensors, with the FFT taken along the time axis.

  FreDFLoss        - forecast in the frequency domain (Wang et al., 2024): error
                     on the complex rFFT coefficients. alpha mixes it with time
                     domain MSE (alpha=1.0 is pure frequency).
  FocalFrequencyLoss - focal frequency loss (Jiang et al., 2021): squared spectral
                     distance re-weighted by a detached focal weight, so hard
                     frequency bins dominate.
  BandWeightedMSE   - time-domain MSE plus an explicit high-band spectral penalty;
                     the simplest "just add a frequency term" control.
"""

import torch
import torch.nn as nn


class FreDFLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, norm: str = "l1"):
        super().__init__()
        self.alpha = alpha
        self.norm = norm

    def forward(self, pred, true):
        fp = torch.fft.rfft(pred, dim=1)
        ft = torch.fft.rfft(true, dim=1)
        diff = fp - ft
        freq_loss = diff.abs().mean() if self.norm == "l1" else diff.abs().pow(2).mean()
        if self.alpha >= 1.0:
            return freq_loss
        time_loss = torch.mean((pred - true) ** 2)
        return self.alpha * freq_loss + (1.0 - self.alpha) * time_loss


class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, true):
        fp = torch.fft.rfft(pred, dim=1)
        ft = torch.fft.rfft(true, dim=1)
        # Squared distance per frequency bin, on the (real, imag) plane.
        sq = (fp.real - ft.real) ** 2 + (fp.imag - ft.imag) ** 2
        with torch.no_grad():
            w = sq.detach() ** self.alpha
            w = w / (w.amax(dim=1, keepdim=True) + 1e-8)
        return (w * sq).mean()


class BandWeightedMSE(nn.Module):
    def __init__(self, high_band_start: float = 0.2, band_weight: float = 1.0):
        super().__init__()
        self.high_band_start = high_band_start
        self.band_weight = band_weight

    def forward(self, pred, true):
        mse = torch.mean((pred - true) ** 2)
        n = pred.size(1)
        freqs = torch.fft.rfftfreq(n, d=1.0, device=pred.device)  # 0 .. 0.5
        mask = freqs >= self.high_band_start
        if not bool(mask.any()):
            return mse
        fp = torch.fft.rfft(pred, dim=1)[:, mask, :]
        ft = torch.fft.rfft(true, dim=1)[:, mask, :]
        return mse + self.band_weight * (fp - ft).abs().pow(2).mean()


def build_criterion(name: str):
    """Map a --loss string to a criterion. Unknown names raise, never silently MSE."""
    key = (name or "MSE").lower()
    table = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": lambda: nn.HuberLoss(delta=1.0),
        "fredf": FreDFLoss,
        "ffl": FocalFrequencyLoss,
        "bandmse": BandWeightedMSE,
    }
    if key not in table:
        raise ValueError(f"unknown --loss {name!r}; choose from {sorted(table)}")
    return table[key]()
