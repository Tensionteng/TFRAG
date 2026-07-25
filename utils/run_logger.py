"""One JSON record per finished run, so aggregation never has to parse log text.

Every record carries the full resolved config. That is the fix for the protocol
ambiguity the reviewers flagged: whatever ends up in a table can be traced back to
an exact epochs/seed/lr/variant tuple.
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone

RUNS_DIR = "runs"

# Fields that define an experimental condition; everything else is incidental.
_CONFIG_KEYS = [
    "variant_override",
    "task_name", "model", "model_id", "data", "data_path", "root_path", "features",
    "seq_len", "label_len", "pred_len", "d_model", "n_heads", "e_layers", "d_layers",
    "d_ff", "factor", "embed", "distil", "dropout", "enc_in", "dec_in", "c_out",
    "batch_size", "learning_rate", "train_epochs", "patience", "lradj", "loss",
    "seed", "itr", "des", "tag", "use_amp", "inverse",
    "use_rag", "num_retrieve", "num_rl_samples", "gamma_1", "gamma_2", "gamma_3",
    "distill_target", "distill_tau", "distill_only_positive", "lambda_reg",
    "kappa", "reward_level", "reward_type", "rl_sampling", "detach_yhat",
    "retrieval_mode", "exclusion_radius", "policy_hidden", "policy_mode",
    "freeze_policy", "rl_sampling", "detach_yhat", "reward_level", "reward_type",
]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def variant_name(args):
    """Short label for the condition, used for grouping in the aggregator.

    The canonical method is 'craft': nearest-neighbour retrieval WITH temporal
    exclusion, discrete rewards, no detach. Only deviations get a suffix, and the
    suffix never contains a numeric value that varies with pred_len -- otherwise
    the same condition would get a different name at each horizon and the paired
    tests would find nothing to compare.
    """
    # Explicit override, used by eval_extracted_base.py: a re-evaluated CRAFT
    # backbone must NOT be logged as 'base', or it collides with the real base run
    # of the same cell and seed and silently breaks the paired comparison.
    override = getattr(args, "variant_override", None)
    if override:
        return str(override)
    if not getattr(args, "use_rag", False):
        loss = str(getattr(args, "loss", "MSE")).lower()
        return "base" if loss == "mse" else f"base_{loss}"
    bits = ["craft"]
    g3 = float(getattr(args, "gamma_3", 0.0))
    if g3 > 0:
        # The value belongs in the name: a gamma_3 sweep is several conditions, not
        # one, and merging them would look like duplicate seeds. Unlike the exclusion
        # radius, gamma_3 does not vary with pred_len, so the name stays stable.
        bits.append(f"distill{g3:g}")
        if getattr(args, "distill_target", "best") != "best":
            bits.append(getattr(args, "distill_target"))
    if getattr(args, "retrieval_mode", "nn") != "nn":
        bits.append(getattr(args, "retrieval_mode"))
    if int(getattr(args, "exclusion_radius", 0)) <= 0:
        bits.append("noexcl")  # safeguard disabled: an ablation, not the method
    # Core hyperparameters that a sweep varies must appear in the name, or several
    # distinct conditions collapse into one label and look like duplicated seeds.
    # Only non-default values are appended, so the canonical condition stays "craft".
    g2 = float(getattr(args, "gamma_2", 0.5))
    if abs(g2 - 0.5) > 1e-12:
        bits.append(f"g2{g2:g}")
    k = int(getattr(args, "num_retrieve", 5))
    if k != 5:
        bits.append(f"k{k}")
    ns = int(getattr(args, "num_rl_samples", 8))
    if ns != 8:
        bits.append(f"ns{ns}")
    lam = float(getattr(args, "lambda_reg", 0.0))
    if lam:
        bits.append(f"lam{lam:g}")
    if getattr(args, "freeze_policy", False):
        bits.append("frozen")
    if getattr(args, "detach_yhat", False):
        bits.append("detach")
    if getattr(args, "reward_type", "discrete") != "discrete":
        bits.append(getattr(args, "reward_type"))
    return "_".join(bits)


def log_run(args, setting, metrics, extra=None, runs_dir=RUNS_DIR):
    """Write runs/<setting>.json. Returns the path."""
    os.makedirs(runs_dir, exist_ok=True)
    record = {
        "setting": setting,
        "variant": variant_name(args),
        "metrics": {k: (None if v is None else float(v)) for k, v in metrics.items()},
        "config": {k: _jsonable(getattr(args, k, None)) for k in _CONFIG_KEYS},
        "env": {
            "git_commit": _git_commit(),
            "host": platform.node(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    if extra:
        record["extra"] = {k: _jsonable(v) for k, v in extra.items()}
    path = os.path.join(runs_dir, f"{setting}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"[run-log] {path}")
    return path


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)
