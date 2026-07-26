#!/usr/bin/env python
"""Backfill model-specific architecture knobs into run records written before they
were logged, and quarantine the ones whose value cannot be recovered.

Why this is safe for most models: the per-model branches of
scripts/craft_rebuttal/common.sh have never changed except for MixLinear, so for
FACT/SegRNN/TimesNet/PatchTST/DLinear the value a past run used is fully determined
by (model, dataset) -- this reconstructs it rather than guessing.

Why MixLinear at lookback 96 is different: that branch DID change (period_len 24 ->
4), and one host ran the old version for part of the campaign, so a record written
then is genuinely ambiguous. Those are moved aside instead of being labelled with a
value that may be wrong. At lookback 720 both versions agree on 24, so those are
backfilled.

    python experiments/backfill_arch_keys.py --runs runs [--apply]
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.run_logger import arch_keys_for  # noqa: E402

# (model, dataset-or-None) -> {key: value}, transcribed from common.sh's model_args.
FACT_DEFAULT = {"core": 0.5, "num_kernels": 4, "use_norm": 1}
FACT_BY_DS = {
    "traffic": {"core": 0.5, "num_kernels": 4, "use_norm": 1},
    "electricity": {"core": 0.1, "num_kernels": 4, "use_norm": 1},
}
# (period_len, lpf, alpha). period_len is the value at lookback >= 480; below that it
# is 24 in the stale common.sh and 4 in the fixed one -- hence AMBIGUOUS_BELOW_480.
MIXLINEAR_BY_DS = {
    "ETTh2": (24, 19, 0.5),
    "ETTm1": (2, 15, 0.01),
    "weather": (4, 15, 0.01),
    "electricity": (24, 19, 0.5),
    "traffic": (24, 19, 0.5),
}
MIXLINEAR_OTHER = (24, 1, 0.95)
# ETTm1 and Weather hardcode period_len (2 and 4) in BOTH versions of common.sh, so
# their short-lookback records are unaffected by the change and stay recoverable.
# Every other dataset reads $mp and is therefore ambiguous below lookback 480.
MIXLINEAR_UNAMBIGUOUS = {"ETTm1", "weather"}


def _dataset_of(cfg):
    stem = os.path.splitext(os.path.basename(cfg.get("data_path") or ""))[0]
    return stem or cfg.get("data")


def resolve(cfg):
    """Values for this run's arch keys, or None if they cannot be recovered."""
    model, ds = cfg.get("model"), _dataset_of(cfg)
    if model == "FACT":
        return dict(FACT_BY_DS.get(ds, FACT_DEFAULT))
    if model == "MixLinear":
        if int(cfg.get("seq_len") or 0) < 480 and ds not in MIXLINEAR_UNAMBIGUOUS:
            return None  # 24 (stale common.sh) or 4 (fixed): unknowable after the fact
        p, l, a = MIXLINEAR_BY_DS.get(ds, MIXLINEAR_OTHER)
        return {"period_len": p, "lpf": l, "alpha": a}
    if model == "SegRNN":
        return {"seg_len": 48}
    if model == "TimesNet":
        return {"top_k": 5}
    if model == "TimeXer":
        return {"patch_len": 16}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--quarantine", default="runs_ambiguous")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    a = ap.parse_args()

    filled = skipped = moved = 0
    for name in sorted(os.listdir(a.runs)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(a.runs, name)
        with open(path) as f:
            rec = json.load(f)
        cfg = rec.get("config", {})
        keys = arch_keys_for(cfg.get("model"))
        if not keys or all(cfg.get(k) is not None for k in keys):
            skipped += 1
            continue
        vals = resolve(cfg)
        if vals is None:
            moved += 1
            print(f"[ambiguous] {name}  {cfg.get('model')} seq_len={cfg.get('seq_len')}")
            if a.apply:
                os.makedirs(a.quarantine, exist_ok=True)
                shutil.move(path, os.path.join(a.quarantine, name))
            continue
        missing = {k: vals[k] for k in keys if cfg.get(k) is None and k in vals}
        if not missing:
            skipped += 1
            continue
        filled += 1
        print(f"[fill] {name}  {missing}")
        if a.apply:
            cfg.update(missing)
            rec["config"] = cfg
            with open(path, "w") as f:
                json.dump(rec, f, indent=2, sort_keys=True)

    verb = "wrote" if a.apply else "would write"
    print(f"\n{verb}: {filled} backfilled, {moved} quarantined, {skipped} already complete")
    if not a.apply:
        print("dry run -- pass --apply to make the changes")


if __name__ == "__main__":
    main()
