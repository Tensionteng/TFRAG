#!/usr/bin/env python
"""One table: every variant against the baseline, paired by seed.

Answers "which configuration actually wins, and is the margin real?" in a single
view, instead of one aggregate run per treatment. Pairs strictly by (dataset, model,
pred_len, seed) and refuses to pair across protocols.

Usage
  python experiments/compare_variants.py --runs runs
  python experiments/compare_variants.py --runs runs --dataset ETTh1 --markdown out.md
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.aggregate_results import (  # noqa: E402
    HAVE_SCIPY,
    PROTOCOL_KEYS,
    load_runs,
    protocol_of,
)

if HAVE_SCIPY:
    from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--metric", default="mse", choices=["mse", "mae"])
    ap.add_argument("--markdown", default=None)
    a = ap.parse_args()

    runs = load_runs(a.runs)
    if a.dataset:
        runs = [r for r in runs if r["data"] == a.dataset]

    # (data, model, pred_len) -> variant -> seed -> run
    idx = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        idx[(r["data"], r["model"], r["pred_len"])][r["variant"]][r["seed"]] = r

    rows = []
    for cell, variants in sorted(idx.items(), key=lambda kv: [str(x) for x in kv[0]]):
        base = variants.get(a.baseline)
        if not base:
            continue
        for v, treat in sorted(variants.items()):
            if v == a.baseline:
                continue
            seeds = sorted(set(base) & set(treat))
            if not seeds:
                continue
            if {protocol_of(base[s]) for s in seeds} != {protocol_of(treat[s]) for s in seeds}:
                rows.append({"cell": cell, "variant": v, "note": "PROTOCOL MISMATCH"})
                continue
            b = np.array([base[s][a.metric] for s in seeds])
            t = np.array([treat[s][a.metric] for s in seeds])
            d = 100.0 * (b - t) / b  # positive = treatment better
            row = {
                "cell": cell,
                "variant": v,
                "n": len(seeds),
                "base": b.mean(),
                "treat": t.mean(),
                "treat_std": t.std(ddof=1) if len(t) > 1 else 0.0,
                "delta_pct": d.mean(),
                "wins": int((t < b).sum()),
                "p": float("nan"),
                "note": "",
            }
            if HAVE_SCIPY and len(seeds) >= 2 and not np.allclose(b, t):
                row["p"] = float(stats.ttest_rel(b, t).pvalue)
            if len(seeds) < 3:
                row["note"] = f"only {len(seeds)} seed(s)"
            rows.append(row)

    hdr = (
        f"| dataset | pl | variant | n | base {a.metric} | variant {a.metric} | "
        f"delta% | wins | p | note |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        if r.get("note") == "PROTOCOL MISMATCH":
            lines.append(
                f"| {r['cell'][0]} | {r['cell'][2]} | {r['variant']} | | | | | | | "
                f"**protocol mismatch, not compared** |"
            )
            continue
        p = "n/a" if not np.isfinite(r["p"]) else f"{r['p']:.3f}"
        star = " **" if np.isfinite(r["p"]) and r["p"] < 0.05 and r["delta_pct"] > 0 else ""
        lines.append(
            f"| {r['cell'][0]} | {r['cell'][2]} | `{r['variant']}` | {r['n']} | "
            f"{r['base']:.5f} | {r['treat']:.5f} ±{r['treat_std']:.5f} | "
            f"{r['delta_pct']:+.2f}{star} | {r['wins']}/{r['n']} | {p} | {r['note']} |"
        )

    out = "\n".join(lines)
    print(out)
    print("\npositive delta% = variant beats the baseline; ** marks p<0.05 in its favour")
    if a.markdown:
        os.makedirs(os.path.dirname(a.markdown) or ".", exist_ok=True)
        with open(a.markdown, "w") as f:
            f.write(f"# Variant comparison ({a.metric}, paired by seed)\n\n{out}\n")
        print(f"[md] {a.markdown}")


if __name__ == "__main__":
    main()
