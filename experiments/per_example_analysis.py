#!/usr/bin/env python
"""Difficulty-quartile and distribution-shift breakdown, from saved predictions.

Two questions, both answered post-hoc from results/*/pred.npy -- no retraining:

  1. QUARTILES. Split test windows into quartiles by base-model difficulty and
     report the per-quartile change. This is the "capacity reallocation" table; it
     is also where the submission's deployment problem showed up (degradation on
     easy windows), so it needs to be recomputed under the real protocol.

  2. SHIFT. The test split is chronological, so its tail is the most temporally
     distant from training. Reporting the last 20% separately is a cheap proxy for
     "how does a corrector-free deployment behave under drift?".

Both comparisons are paired at the window level, so a Wilcoxon signed-rank test on
per-window errors is available even from a single seed -- far more informative than
comparing two scalars.

Usage
  python experiments/per_example_analysis.py --base results/<base> --craft results/<craft> \
      --dataset ETTh1 --model iTransformer --out analysis/per_example.csv
  python experiments/per_example_analysis.py --manifest analysis/pairs.csv --out analysis/per_example.csv
"""

import argparse
import csv
import os

import numpy as np

try:
    from scipy import stats

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def per_window_mse(d):
    p = np.load(os.path.join(d, "pred.npy")).astype(np.float64)
    t = np.load(os.path.join(d, "true.npy")).astype(np.float64)
    return ((p - t) ** 2).mean(axis=(1, 2)), t


def analyse(dataset, model, base_dir, craft_dir, shift_frac=0.2):
    eb, tb = per_window_mse(base_dir)
    ec, tc = per_window_mse(craft_dir)
    if eb.shape != ec.shape:
        raise ValueError(f"window count differs: {eb.shape} vs {ec.shape}")
    if not np.allclose(tb, tc, atol=1e-5):
        raise ValueError("ground truth differs between runs; refusing to compare")

    rows = []

    def block(name, mask):
        b, c = eb[mask], ec[mask]
        row = {
            "dataset": dataset,
            "model": model,
            "split": name,
            "n_windows": int(mask.sum()),
            "mse_base": b.mean(),
            "mse_craft": c.mean(),
            "delta_pct": 100.0 * (b.mean() - c.mean()) / b.mean(),
            "craft_better_frac": float((c < b).mean()),
        }
        if HAVE_SCIPY and mask.sum() > 10 and not np.allclose(b, c):
            row["p_wilcoxon"] = float(stats.wilcoxon(b, c, zero_method="zsplit").pvalue)
        else:
            row["p_wilcoxon"] = float("nan")
        return row

    n = eb.size
    rows.append(block("all", np.ones(n, dtype=bool)))

    # Quartiles by base difficulty.
    order = np.argsort(eb)
    qmask = np.zeros(n, dtype=int)
    for q, chunk in enumerate(np.array_split(order, 4)):
        qmask[chunk] = q
    for q, label in enumerate(["Q1_easy", "Q2", "Q3", "Q4_hard"]):
        rows.append(block(label, qmask == q))

    # Chronological tail: drift proxy.
    cut = int(n * (1.0 - shift_frac))
    idx = np.arange(n)
    rows.append(block(f"first_{int((1-shift_frac)*100)}pct", idx < cut))
    rows.append(block(f"last_{int(shift_frac*100)}pct_shift", idx >= cut))
    return rows


FIELDS = [
    "dataset", "model", "split", "n_windows", "mse_base", "mse_craft",
    "delta_pct", "craft_better_frac", "p_wilcoxon",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--craft")
    ap.add_argument("--dataset", default="unknown")
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--manifest", help="CSV: dataset,model,base_dir,craft_dir")
    ap.add_argument("--shift_frac", type=float, default=0.2)
    ap.add_argument("--out", default="analysis/per_example.csv")
    a = ap.parse_args()

    pairs = []
    if a.manifest:
        with open(a.manifest) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ds, mdl, b, c = [x.strip() for x in line.split(",")]
                    pairs.append((ds, mdl, b, c))
    else:
        if not (a.base and a.craft):
            ap.error("provide --base and --craft, or --manifest")
        pairs.append((a.dataset, a.model, a.base, a.craft))

    rows, failures = [], []
    for ds, mdl, b, c in pairs:
        try:
            rows.extend(analyse(ds, mdl, b, c, a.shift_frac))
            print(f"[ok] {ds}/{mdl}")
        except Exception as e:
            failures.append(f"{ds}/{mdl}: {e}")
            print(f"[FAIL] {ds}/{mdl}: {e}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {a.out} ({len(rows)} rows)")

    overall = [r for r in rows if r["split"] == "all"]
    if overall:
        d = np.array([r["delta_pct"] for r in overall])
        print(
            f"\ndeployed-backbone change on {len(d)} pairs: mean {d.mean():+.2f}%, "
            f"improved on {int((d>0).sum())}/{len(d)}"
        )
    if not HAVE_SCIPY:
        print("[warn] scipy missing: no Wilcoxon p-values")
    for x in failures:
        print(" - FAILED:", x)


if __name__ == "__main__":
    main()
