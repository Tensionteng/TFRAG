#!/usr/bin/env python
"""Aggregate runs/*.json into seeded means, paired significance tests and CIs.

This is what answers "are the reported gains above the noise floor?". It refuses
to compare conditions that differ in anything other than the variant and the seed,
so a protocol mismatch surfaces as a skipped comparison instead of a wrong p-value.

Outputs
-------
  <out>/per_run.csv        one row per run, full config
  <out>/per_cell.csv       mean/std over seeds for each (dataset, model, pred_len, variant)
  <out>/paired_tests.csv   craft-vs-base paired t-test + Wilcoxon per cell and per dataset
  <out>/summary.md         human-readable digest, incl. bootstrap CI on the aggregate gain

Usage
  python experiments/aggregate_results.py --runs runs --out analysis
  python experiments/aggregate_results.py --runs runs --out analysis --treatment craft_random
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy import stats

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

# A cell is one experimental condition; runs inside a cell may differ only by seed.
CELL_KEYS = ["data", "model", "pred_len", "variant"]
# Everything here must match between base and treatment for a paired test to be fair.
PROTOCOL_KEYS = [
    "seq_len", "pred_len", "train_epochs", "learning_rate", "batch_size",
    "d_model", "d_ff", "e_layers", "n_heads", "lradj", "features", "patience",
]


def _variant_of(record):
    """Re-derive the variant label from the stored config.

    Deliberately not trusting the string saved at run time: when the naming rules
    are refined (e.g. to separate a gamma_2 sweep that previously collapsed into one
    label), old records must pick up the new rules instead of silently mixing
    several conditions under one name.
    """
    cfg = record.get("config") or {}
    stored = record.get("variant", "base")
    if "use_rag" not in cfg:
        # Too old / too sparse to re-derive; the stored label is all we have.
        return stored
    try:
        from utils.run_logger import variant_name

        return variant_name(SimpleNamespace(**cfg))
    except Exception:
        return stored


def load_runs(runs_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        cfg, m = r.get("config", {}), r.get("metrics", {})
        if m.get("mse") is None:
            print(f"[skip] {os.path.basename(p)}: no mse")
            continue
        runs.append(
            {
                "file": os.path.basename(p),
                "setting": r.get("setting"),
                "variant": _variant_of(r),
                "mse": float(m["mse"]),
                "mae": float(m["mae"]),
                "seed": cfg.get("seed"),
                **{k: cfg.get(k) for k in set(CELL_KEYS + PROTOCOL_KEYS) - {"variant"}},
            }
        )
    if not runs:
        raise SystemExit(f"no usable run records in {runs_dir}/")
    print(f"[load] {len(runs)} runs from {runs_dir}/")
    return runs


def cell_of(r):
    return tuple(r[k] for k in CELL_KEYS)


def protocol_of(r):
    return tuple(r[k] for k in PROTOCOL_KEYS)


def write_csv(path, rows, fields):
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path} ({len(rows)} rows)")


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. Deterministic given seed."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(base_by_seed, treat_by_seed, metric):
    """Pair strictly by seed; unmatched seeds are dropped and counted."""
    seeds = sorted(set(base_by_seed) & set(treat_by_seed))
    if len(seeds) < 2:
        return None
    b = np.array([base_by_seed[s][metric] for s in seeds], dtype=float)
    t = np.array([treat_by_seed[s][metric] for s in seeds], dtype=float)
    out = {
        "n_seeds": len(seeds),
        "seeds": ";".join(str(s) for s in seeds),
        f"{metric}_base_mean": b.mean(),
        f"{metric}_base_std": b.std(ddof=1) if len(b) > 1 else 0.0,
        f"{metric}_treat_mean": t.mean(),
        f"{metric}_treat_std": t.std(ddof=1) if len(t) > 1 else 0.0,
        f"{metric}_delta_pct": 100.0 * (b.mean() - t.mean()) / b.mean(),
        f"{metric}_wins": int((t < b).sum()),
    }
    if HAVE_SCIPY:
        if np.allclose(b, t):
            out[f"{metric}_p_ttest"] = 1.0
            out[f"{metric}_p_wilcoxon"] = 1.0
        else:
            out[f"{metric}_p_ttest"] = float(stats.ttest_rel(b, t).pvalue)
            try:
                out[f"{metric}_p_wilcoxon"] = float(
                    stats.wilcoxon(b, t, zero_method="zsplit").pvalue
                )
            except ValueError:
                out[f"{metric}_p_wilcoxon"] = float("nan")
        d = b - t
        sd = d.std(ddof=1)
        out[f"{metric}_cohens_dz"] = float(d.mean() / sd) if sd > 0 else float("nan")
    lo, hi = bootstrap_ci(100.0 * (b - t) / b)
    out[f"{metric}_delta_pct_ci_lo"] = lo
    out[f"{metric}_delta_pct_ci_hi"] = hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--treatment", default="craft")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    os.makedirs(args.out, exist_ok=True)

    write_csv(
        os.path.join(args.out, "per_run.csv"),
        runs,
        ["file", "setting", "variant", "data", "model", "pred_len", "seed", "mse", "mae"]
        + PROTOCOL_KEYS,
    )

    # ---- per cell -------------------------------------------------------
    by_cell = defaultdict(list)
    for r in runs:
        by_cell[cell_of(r)].append(r)

    cell_rows = []
    for cell, rs in sorted(by_cell.items(), key=lambda kv: [str(x) for x in kv[0]]):
        mse = np.array([r["mse"] for r in rs])
        mae = np.array([r["mae"] for r in rs])
        seeds = [r["seed"] for r in rs]
        if len(set(seeds)) != len(seeds):
            print(f"[warn] duplicate seeds in cell {cell}: {seeds}")
        cell_rows.append(
            {
                **dict(zip(CELL_KEYS, cell)),
                "n_seeds": len(rs),
                "seeds": ";".join(str(s) for s in sorted(seeds)),
                "mse_mean": mse.mean(),
                "mse_std": mse.std(ddof=1) if len(mse) > 1 else 0.0,
                "mae_mean": mae.mean(),
                "mae_std": mae.std(ddof=1) if len(mae) > 1 else 0.0,
            }
        )
    write_csv(
        os.path.join(args.out, "per_cell.csv"),
        cell_rows,
        CELL_KEYS + ["n_seeds", "seeds", "mse_mean", "mse_std", "mae_mean", "mae_std"],
    )

    # ---- paired tests ---------------------------------------------------
    index = defaultdict(dict)  # (data, model, pred_len) -> variant -> {seed: run}
    dup_warnings = []
    for r in runs:
        bucket = index[(r["data"], r["model"], r["pred_len"])].setdefault(r["variant"], {})
        if r["seed"] in bucket:
            # Never overwrite: a duplicated (cell, variant, seed) means two runs are
            # claiming the same condition, and silently keeping one would fabricate
            # a clean pairing out of ambiguous data.
            dup_warnings.append(
                f"{(r['data'], r['model'], r['pred_len'])} variant={r['variant']} "
                f"seed={r['seed']}: duplicate records {bucket[r['seed']]['file']} and {r['file']}"
            )
            continue
        bucket[r["seed"]] = r

    test_rows, skipped, all_mse_deltas, all_mae_deltas = [], [], list(), list()
    seed_deltas = []
    skipped.extend(dup_warnings)
    for key, variants in sorted(index.items(), key=lambda kv: [str(x) for x in kv[0]]):
        base, treat = variants.get(args.baseline), variants.get(args.treatment)
        if not base or not treat:
            skipped.append(f"{key}: missing {args.baseline if not base else args.treatment}")
            continue
        # Refuse to pair across protocols -- this is the check whose absence caused
        # the submission's cross-table comparisons to be unverifiable.
        pb = {protocol_of(r) for r in base.values()}
        pt = {protocol_of(r) for r in treat.values()}
        if pb != pt:
            skipped.append(f"{key}: protocol mismatch between {args.baseline} and {args.treatment}")
            continue

        row = {"data": key[0], "model": key[1], "pred_len": key[2]}
        ok = True
        for metric in ("mse", "mae"):
            res = paired_test(base, treat, metric)
            if res is None:
                ok = False
                skipped.append(f"{key}: fewer than 2 shared seeds")
                break
            row.update(res)
        if not ok:
            continue
        test_rows.append(row)
        all_mse_deltas.append(row["mse_delta_pct"])
        all_mae_deltas.append(row["mae_delta_pct"])
        # Per-seed deltas, pooled across cells. The cell-mean bootstrap below is
        # degenerate when only one cell exists, so this gives a CI that still has
        # sampling variation to work with.
        for s in sorted(set(base) & set(treat)):
            seed_deltas.append(
                100.0 * (base[s]["mse"] - treat[s]["mse"]) / base[s]["mse"]
            )

    fields = ["data", "model", "pred_len", "n_seeds", "seeds"]
    for m in ("mse", "mae"):
        fields += [
            f"{m}_base_mean", f"{m}_base_std", f"{m}_treat_mean", f"{m}_treat_std",
            f"{m}_delta_pct", f"{m}_delta_pct_ci_lo", f"{m}_delta_pct_ci_hi",
            f"{m}_wins", f"{m}_p_ttest", f"{m}_p_wilcoxon", f"{m}_cohens_dz",
        ]
    write_csv(os.path.join(args.out, "paired_tests.csv"), test_rows, fields)

    # ---- summary --------------------------------------------------------
    lines = [
        f"# CRAFT results summary ({args.treatment} vs {args.baseline})", "",
        f"- run records: {len(runs)}",
        f"- comparable cells: {len(test_rows)}",
        f"- scipy available: {HAVE_SCIPY}" + ("" if HAVE_SCIPY else "  **install scipy for p-values**"),
        "",
    ]
    if test_rows:
        mse_d = np.array(all_mse_deltas)
        mae_d = np.array(all_mae_deltas)
        lo, hi = bootstrap_ci(mse_d)
        lo2, hi2 = bootstrap_ci(mae_d)
        n_sig = sum(
            1 for r in test_rows if np.isfinite(r.get("mse_p_ttest", np.nan)) and r["mse_p_ttest"] < 0.05
        )
        n_sig_better = sum(
            1
            for r in test_rows
            if np.isfinite(r.get("mse_p_ttest", np.nan))
            and r["mse_p_ttest"] < 0.05
            and r["mse_delta_pct"] > 0
        )
        slo, shi = bootstrap_ci(np.array(seed_deltas), seed=1)
        lines += [
            "## Aggregate",
            f"- mean MSE change: **{mse_d.mean():+.2f}%** (95% bootstrap CI over "
            f"{len(mse_d)} cell{'s' if len(mse_d) != 1 else ''} {lo:+.2f}% .. {hi:+.2f}%"
            + ("; degenerate with a single cell -- use the per-seed CI below" if len(mse_d) < 2 else "")
            + ")",
            f"- mean MSE change, pooled over {len(seed_deltas)} per-seed paired deltas: "
            f"**{np.mean(seed_deltas):+.2f}%** (95% bootstrap CI {slo:+.2f}% .. {shi:+.2f}%)",
            f"- mean MAE change: **{mae_d.mean():+.2f}%** (95% bootstrap CI {lo2:+.2f}% .. {hi2:+.2f}%)",
            f"- cells improved: {int((mse_d > 0).sum())}/{len(mse_d)}",
            f"- cells with p<0.05 (paired t, MSE): {n_sig}, of which favour {args.treatment}: {n_sig_better}",
            "",
            "> Positive = treatment better. If the CI includes 0, the headline number is "
            "not distinguishable from noise and must be reported as such.",
            "",
            "## Per dataset",
            "",
            "| dataset | cells | mean dMSE% | CI low | CI high | improved |",
            "|---|---|---|---|---|---|",
        ]
        by_ds = defaultdict(list)
        for r in test_rows:
            by_ds[r["data"]].append(r["mse_delta_pct"])
        for ds, vals in sorted(by_ds.items()):
            v = np.array(vals)
            l, h = bootstrap_ci(v)
            lines.append(
                f"| {ds} | {len(v)} | {v.mean():+.2f} | {l:+.2f} | {h:+.2f} | {int((v>0).sum())}/{len(v)} |"
            )
    if skipped:
        lines += ["", "## Skipped comparisons", ""] + [f"- {s}" for s in skipped]

    path = os.path.join(args.out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[md] {path}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
