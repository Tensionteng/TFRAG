#!/usr/bin/env python
"""Evaluate the DEPLOYED artifact: the CRAFT-trained backbone, alone.

This is the experiment that decides whether "improves the base model at zero
inference cost" survives. It rebuilds the backbone with use_rag=False, loads the
backbone weights out of a CRAFT checkpoint, and runs the standard test loop, so the
numbers are directly comparable to a plain base run.

It reads the run record written by run.py, so the evaluated config is guaranteed to
match the trained one rather than being retyped on the command line.

Usage
  python experiments/eval_extracted_base.py --run_json runs/<craft_setting>.json
  python experiments/eval_extracted_base.py --run_json runs/<s>.json --tag deploy
"""

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: E402
from models.model_factory import strip_base_prefix  # noqa: E402
from run import build_setting  # noqa: E402


class Args:
    """Minimal argparse.Namespace stand-in built from a saved run record."""

    def __init__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return f"Args({sorted(self.__dict__)})"


# Defaults for fields the run record does not carry (they never affect results here).
FALLBACK = dict(
    num_workers=4, itr=1, use_amp=False, inverse=False, use_dtw=False,
    expand=2, d_conv=4, moving_avg=25, activation="gelu", dropout=0.1,
    channel_independence=1, decomp_method="moving_avg", use_norm=1,
    down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
    seg_len=96, top_k=5, num_kernels=6, distil=True, patch_len=16,
    p_hidden_dims=[128, 128], p_hidden_layers=2, seasonal_patterns="Monthly",
    mask_rate=0.25, anomaly_ratio=0.25, augmentation_ratio=0, target="OT", freq="h",
    checkpoints="./checkpoints/", use_multi_gpu=False, devices="0,1",
    gpu_type="cuda", lradj="type1", patience=3, tag="", des="Exp",
    fusion_mode="mean", w_trend=0.25, w_frequency=0.25,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_json", required=True, help="runs/<craft_setting>.json")
    ap.add_argument("--checkpoint", default=None, help="override the checkpoint path")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="deploy", help="suffix distinguishing this eval")
    args_cli = ap.parse_args()

    with open(args_cli.run_json) as f:
        record = json.load(f)
    cfg = {k: v for k, v in record["config"].items() if v is not None}
    if not cfg.get("use_rag"):
        print("[warn] this run was not a CRAFT run; extraction is a no-op")

    train_setting = record["setting"]
    ckpt_dir = os.path.join("./checkpoints", train_setting)
    ckpt = args_cli.checkpoint or os.path.join(ckpt_dir, "base_model.pth")
    if not os.path.exists(ckpt):
        # Fall back to the full wrapper checkpoint and strip it here.
        ckpt = os.path.join(ckpt_dir, "checkpoint.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint under {ckpt_dir}")

    merged = copy.deepcopy(FALLBACK)
    merged.update(cfg)
    merged["use_rag"] = False  # the whole point: deploy the backbone alone
    merged["is_training"] = 0
    merged["use_gpu"] = torch.cuda.is_available()
    merged["gpu"] = args_cli.gpu
    merged["tag"] = (cfg.get("tag") or "") + "_" + args_cli.tag
    # Keep this record out of the 'base' cell it would otherwise collide with.
    merged["variant_override"] = (
        "craft_deployed" if cfg.get("use_rag") else "base_reeval"
    )
    args = Args(merged)
    args.device = torch.device(
        f"cuda:{args.gpu}" if args.use_gpu else "cpu"
    )

    exp = Exp_Long_Term_Forecast(args)

    state = torch.load(ckpt, map_location="cpu")
    state = strip_base_prefix(state)
    missing, unexpected = exp.model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(
            f"{len(missing)} backbone weights missing from {ckpt} (first: {missing[:3]}). "
            "Extraction failed -- do NOT report these numbers."
        )
    if unexpected:
        print(f"[info] ignored {len(unexpected)} non-backbone keys (policy head etc.)")
    print(f"[load] backbone weights from {ckpt}")

    setting = build_setting(args, 0)
    mse, mae = exp.test(setting, test=0)
    print(f"\nDEPLOYED BACKBONE  mse={mse:.6f}  mae={mae:.6f}")
    print(f"trained as: {train_setting}")
    print(f"eval id   : {setting}")


if __name__ == "__main__":
    main()
