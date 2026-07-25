#!/usr/bin/env python
"""Band-wise spectral analysis of a base run vs a CRAFT run -- corrected edition.

WHY THIS SCRIPT EXISTS
----------------------
The submitted tables titled "Prediction Error Energy" / "Error Energy Reduction (%)"
did not report error energy. The old code (freq_analysis_all.py) computed the PSD of
the *prediction itself* and reported (E_craft - E_base)/E_base, so a band where the
prediction gained energy was printed as a large positive "reduction" -- which is why
some cells exceeded 100%. Two different sign conventions also coexisted in the same
row (FRMSE promotion was better-is-positive; the band columns were not).

This script reports three clearly-separated quantities per band, each with an
explicit sign convention:

  1. RESIDUAL ENERGY  E_res = mean |FFT(pred - true)|^2 over the band.
     Lower is better. `res_reduction_pct = 100 * (E_res_base - E_res_craft)/E_res_base`
     is a true error reduction, positive = CRAFT better, and cannot exceed 100%
     unless CRAFT's residual is negative (impossible).

  2. SPECTRAL AMPLITUDE MISMATCH  |E_pred - E_true| where E is band energy of the
     signal. This is what the old table was groping towards: it says whether the
     prediction carries the right *amount* of energy in the band. Reported with
     GT energy alongside, so a reader can see whether CRAFT moved toward the target
     or overshot it. `gap_reduction_pct` is positive when CRAFT is closer to GT.

  3. FRMSE, the paper's Section 5.3 definition: RMS difference of amplitude spectra,
     Hanning-windowed, matching the original implementation so numbers stay
     comparable with the submission.

Inputs are the results/ directories written by run.py (pred.npy, true.npy).

Usage
-----
  python experiments/freq_band_analysis.py \
      --base   results/<base_setting> \
      --craft  results/<craft_setting> \
      --dataset ECL --model iTransformer \
      --out analysis/freq_bands.csv

  # many pairs at once, from a manifest of "dataset,model,base_dir,craft_dir" lines
  python experiments/freq_band_analysis.py --manifest analysis/pairs.csv --out analysis/freq_bands.csv
"""

import argparse
import csv
import os

import numpy as np

BANDS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]


def load_pair(base_dir, craft_dir):
    def _load(d):
        pred = np.load(os.path.join(d, "pred.npy"))
        true = np.load(os.path.join(d, "true.npy"))
        return pred.astype(np.float64), true.astype(np.float64)

    pb, tb = _load(base_dir)
    pc, tc = _load(craft_dir)
    if pb.shape != pc.shape:
        raise ValueError(
            f"shape mismatch base {pb.shape} vs craft {pc.shape}; the two runs must "
            "share dataset, pred_len and test split"
        )
    if not np.allclose(tb, tc, atol=1e-5):
        raise ValueError(
            "ground truth differs between the two directories -- different splits or "
            "different --inverse setting. Refusing to compare."
        )
    return pb, pc, tb


def band_masks(n_time):
    """Normalised rFFT bin frequencies in [0, 0.5]; one boolean mask per band."""
    freqs = np.fft.rfftfreq(n_time, d=1.0)
    masks = []
    for lo, hi in BANDS:
        # Half-open [lo, hi), except the last band which includes Nyquist.
        m = (freqs >= lo) & (freqs < hi) if hi < 0.5 else (freqs >= lo) & (freqs <= hi)
        masks.append(m)
    return freqs, masks


def band_energy(x, masks):
    """Mean per-bin |FFT|^2 within each band, averaged over samples and channels.

    x: [N, P, C]. Returns array of length len(BANDS).
    """
    spec = np.fft.rfft(x, axis=1)
    power = np.abs(spec) ** 2
    return np.array([power[:, m, :].mean() if m.any() else np.nan for m in masks])


def frmse(pred, true):
    """Paper definition: RMS difference of Hanning-windowed amplitude spectra."""
    w = np.hanning(pred.shape[1])[None, :, None]
    ap = np.abs(np.fft.rfft(pred * w, axis=1)) / pred.shape[1]
    at = np.abs(np.fft.rfft(true * w, axis=1)) / true.shape[1]
    return float(np.sqrt(((ap - at) ** 2).mean()))


def pct(before, after):
    """Reduction of a lower-is-better quantity, in percent. Positive = improved."""
    if before is None or after is None or not np.isfinite(before) or before == 0:
        return float("nan")
    return 100.0 * (before - after) / before


def analyse(dataset, model, base_dir, craft_dir):
    pred_b, pred_c, true = load_pair(base_dir, craft_dir)
    _, masks = band_masks(true.shape[1])

    res_b = band_energy(pred_b - true, masks)
    res_c = band_energy(pred_c - true, masks)
    sig_t = band_energy(true, masks)
    sig_b = band_energy(pred_b, masks)
    sig_c = band_energy(pred_c, masks)

    rows = []
    for i, (lo, hi) in enumerate(BANDS):
        gap_b = abs(sig_b[i] - sig_t[i])
        gap_c = abs(sig_c[i] - sig_t[i])
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "band": f"{lo:.1f}-{hi:.1f}",
                # 1. true error energy (lower better)
                "res_energy_base": res_b[i],
                "res_energy_craft": res_c[i],
                "res_reduction_pct": pct(res_b[i], res_c[i]),
                # 2. amplitude mismatch vs ground truth (lower better)
                "signal_energy_gt": sig_t[i],
                "signal_energy_base": sig_b[i],
                "signal_energy_craft": sig_c[i],
                "gap_to_gt_base": gap_b,
                "gap_to_gt_craft": gap_c,
                "gap_reduction_pct": pct(gap_b, gap_c),
                # direction flags, so nobody has to re-derive the sign by hand
                "craft_energy_moved": "up" if sig_c[i] > sig_b[i] else "down",
                "craft_overshoots_gt": bool(sig_c[i] > sig_t[i]),
            }
        )

    overall = {
        "dataset": dataset,
        "model": model,
        "band": "ALL",
        "res_energy_base": res_b.mean(),
        "res_energy_craft": res_c.mean(),
        "res_reduction_pct": pct(res_b.mean(), res_c.mean()),
        "signal_energy_gt": sig_t.mean(),
        "signal_energy_base": sig_b.mean(),
        "signal_energy_craft": sig_c.mean(),
        "gap_to_gt_base": abs(sig_b - sig_t).mean(),
        "gap_to_gt_craft": abs(sig_c - sig_t).mean(),
        "gap_reduction_pct": pct(abs(sig_b - sig_t).mean(), abs(sig_c - sig_t).mean()),
        "craft_energy_moved": "",
        "craft_overshoots_gt": "",
        "frmse_base": frmse(pred_b, true),
        "frmse_craft": frmse(pred_c, true),
        "mse_base": float(((pred_b - true) ** 2).mean()),
        "mse_craft": float(((pred_c - true) ** 2).mean()),
    }
    overall["frmse_reduction_pct"] = pct(overall["frmse_base"], overall["frmse_craft"])
    rows.append(overall)
    return rows


FIELDS = [
    "dataset", "model", "band",
    "res_energy_base", "res_energy_craft", "res_reduction_pct",
    "signal_energy_gt", "signal_energy_base", "signal_energy_craft",
    "gap_to_gt_base", "gap_to_gt_craft", "gap_reduction_pct",
    "craft_energy_moved", "craft_overshoots_gt",
    "frmse_base", "frmse_craft", "frmse_reduction_pct", "mse_base", "mse_craft",
]


def to_latex(rows, path):
    """Camera-ready table: residual reduction (the honest error metric) per band."""
    per = {}
    for r in rows:
        if r["band"] == "ALL":
            continue
        per.setdefault((r["dataset"], r["model"]), {})[r["band"]] = r["res_reduction_pct"]
    bands = [f"{lo:.1f}-{hi:.1f}" for lo, hi in BANDS]
    with open(path, "w") as f:
        f.write("% Positive = CRAFT reduces residual energy in the band. Cannot exceed 100.\n")
        f.write("\\begin{tabular}{ll" + "r" * len(bands) + "}\n\\toprule\n")
        f.write("Dataset & Model & " + " & ".join(bands) + " \\\\\n\\midrule\n")
        for (ds, mdl), vals in sorted(per.items()):
            cells = [f"{vals.get(b, float('nan')):.2f}" for b in bands]
            f.write(f"{ds} & {mdl} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[latex] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--craft")
    ap.add_argument("--dataset", default="unknown")
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--manifest", help="CSV: dataset,model,base_dir,craft_dir")
    ap.add_argument("--out", default="analysis/freq_bands.csv")
    ap.add_argument("--latex", default=None)
    args = ap.parse_args()

    pairs = []
    if args.manifest:
        with open(args.manifest) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ds, mdl, b, c = [p.strip() for p in line.split(",")]
                pairs.append((ds, mdl, b, c))
    else:
        if not (args.base and args.craft):
            ap.error("provide --base and --craft, or --manifest")
        pairs.append((args.dataset, args.model, args.base, args.craft))

    rows, failures = [], []
    for ds, mdl, b, c in pairs:
        try:
            rows.extend(analyse(ds, mdl, b, c))
            print(f"[ok] {ds}/{mdl}")
        except Exception as e:  # keep going; report at the end, never silently skip
            failures.append(f"{ds}/{mdl}: {e}")
            print(f"[FAIL] {ds}/{mdl}: {e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {args.out}  ({len(rows)} rows)")

    if args.latex:
        to_latex(rows, args.latex)

    if failures:
        print("\n=== pairs that could not be analysed ===")
        for x in failures:
            print(" -", x)

    # Headline summary: is the 0.2-0.5 concentration claim supported?
    mid_high, low = [], []
    for r in rows:
        if r["band"] in ("0.2-0.3", "0.3-0.4", "0.4-0.5"):
            mid_high.append(r["res_reduction_pct"])
        elif r["band"] in ("0.0-0.1", "0.1-0.2"):
            low.append(r["res_reduction_pct"])
    if mid_high and low:
        print(
            f"\nmean residual-energy reduction: low bands (0.0-0.2) "
            f"{np.nanmean(low):+.2f}%  |  mid-high bands (0.2-0.5) {np.nanmean(mid_high):+.2f}%"
        )
        print(
            "The frequency-selectivity claim holds only if mid-high > low. "
            "Report whichever way this comes out."
        )


if __name__ == "__main__":
    main()
