#!/bin/bash
# B9 -- FACT and MixLinear (both ICLR 2026) as baselines, and as CRAFT backbones.
#
# WHAT REVIEWER ErQJ ASKED (weakness 2):
#   "Evaluated baselines are outdated. No 2026, 2025 models and only 2 of 8 are from
#    2024. Given that time-series forecasting is a rapidly evolving field, whether the
#    method can be applied to the latest models is a crucial issue."
#
# Two things get produced, and they answer different halves of that sentence:
#   PART A  the baselines themselves at the paper's protocol (lookback 96), so the
#           main table stops being a 2022-2023 comparison.
#   PART B  MixLinear at its OWN lookback 720. Its scripts use 720 everywhere; running
#           a long-lookback model only at 96 would under-report it, which is the exact
#           criticism we are trying to answer.
#   PART C  CRAFT applied on top of both, which is the "can it be applied to the latest
#           models" half. MixLinear is also the weakest backbone available anywhere in
#           this repo (0.1K parameters), so it is the sharpest test of the submission's
#           claim that gains concentrate on weak backbones.
#
# Hyperparameters come from the authors' own scripts (see model_args/lr_for/bs_for/
# epochs_for in common.sh), including their training budgets: FACT 15 epochs,
# MixLinear 30. Budgets are matched WITHIN each model's base-vs-CRAFT pair, so every
# paired test stays fair; they differ ACROSS models exactly as the source papers do.
#
# Usage
#   GPUS="0 1 2 3" bash scripts/craft_rebuttal/parallel.sh \
#       "PART=A bash scripts/craft_rebuttal/b9_new_baselines.sh"
source "$(dirname "$0")/common.sh"

PART=${PART:-A}
PL=${PL:-96}
B9_SEEDS=${B9_SEEDS:-"2021 1 2"}

SMALL=(
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
  "ETTh2|./dataset/ETT-small/|ETTh2.csv|ETTh2|7"
  "ETTm1|./dataset/ETT-small/|ETTm1.csv|ETTm1|7"
  "ETTm2|./dataset/ETT-small/|ETTm2.csv|ETTm2|7"
  "Weather|./dataset/weather/|weather.csv|custom|21"
  "Exchange|./dataset/exchange_rate/|exchange_rate.csv|custom|8"
)
BIG=("ECL|./dataset/electricity/|electricity.csv|custom|321")

sweep() { # sweep <array> <models> <extra...>
  local -n DS=$1; shift
  local models=$1; shift
  for MODEL in $models; do
    for entry in "${DS[@]}"; do
      IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
      for SEED in $B9_SEEDS; do
        run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$@"
      done
    done
  done
}

craft_arm() { echo "--use_rag --gamma_1 1.0 --gamma_2 ${GAMMA2:-0.5} --num_retrieve 5 \
  --num_rl_samples 8 --exclusion_radius $PL --retrieval_mode nn"; }

if [ "$PART" = "A" ] || [ "$PART" = "all" ]; then
  echo "### PART A: FACT + MixLinear at the paper's lookback 96"
  sweep SMALL "FACT MixLinear"
  sweep BIG   "FACT MixLinear"
fi

if [ "$PART" = "B" ] || [ "$PART" = "all" ]; then
  echo "### PART B: MixLinear at its native lookback 720"
  # Only the low-variate datasets: at L=720 the CRAFT index would be 720*enc_in wide,
  # which is not the point of this part anyway -- these are baseline rows.
  SEQ_LEN=720 sweep SMALL "MixLinear"
fi

if [ "$PART" = "C" ] || [ "$PART" = "all" ]; then
  echo "### PART C: CRAFT on top of both 2026 backbones (lookback 96)"
  sweep SMALL "FACT MixLinear" "$(craft_arm)"
  sweep BIG   "FACT MixLinear" "$(craft_arm)"
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo
echo "### baseline table (each model vs its own CRAFT arm)"
$PY experiments/compare_variants.py --runs runs --markdown analysis/b9_new_baselines.md \
  | grep -E "FACT|MixLinear|dataset|---" | sed 's/^/  /'
echo
echo "### absolute scores, for the main-table rows"
$PY - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from collections import defaultdict
from experiments.aggregate_results import load_runs

runs = [r for r in load_runs("runs") if r["model"] in ("FACT", "MixLinear")]
g = defaultdict(list)
for r in runs:
    g[(r["model"], r["dataset"], r["pred_len"], r["seq_len"], r["variant"])].append(r)
print(f"{'model':<11}{'dataset':<14}{'L':>5}{'pl':>5}{'variant':<10}{'n':>3}{'MSE':>10}{'MAE':>10}")
for k in sorted(g, key=lambda x: (x[0], str(x[1]), x[3])):
    v = g[k]
    mse = np.mean([x["mse"] for x in v]); mae = np.mean([x["mae"] for x in v])
    print(f"{k[0]:<11}{str(k[1]):<14}{k[3]:>5}{k[2]:>5}{k[4]:<10}{len(v):>3}{mse:>10.4f}{mae:>10.4f}")
EOF
