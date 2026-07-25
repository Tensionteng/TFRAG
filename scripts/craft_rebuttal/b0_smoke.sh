#!/bin/bash
# B0 -- 10-minute correctness gate. Run this FIRST on any new machine.
# Verifies: unit tests pass, a base run works, a CRAFT run works (bank + exclusion
# + RL + backbone extraction), the deployed-backbone eval works, and all three
# analysis scripts produce output. Nothing here is reportable; it only proves the
# plumbing is intact before spending GPU days.
source "$(dirname "$0")/common.sh"

EPOCHS=1
PATIENCE=1
DES=Smoke
SEED=${SEED:-2021}
PL=96
NAME=ETTh1; ROOT=./dataset/ETT-small/; DPATH=ETTh1.csv; DATA=ETTh1; ENC=7
MODEL=${MODEL:-DLinear}

echo "### 1/5 unit tests"
$PY -m pytest tests/ -q || { echo "UNIT TESTS FAILED -- stop here"; exit 1; }

echo "### 2/5 base run"
run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED"

echo "### 3/5 CRAFT run"
run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$(craft_flags $PL)"

echo "### 4/5 deployed-backbone eval"
CRAFT_JSON=$(ls -t runs/*Smoke*rag*.json 2>/dev/null | head -1)
if [ -n "$CRAFT_JSON" ]; then
  $PY experiments/eval_extracted_base.py --run_json "$CRAFT_JSON" --gpu "$GPU" \
    > "$LOGDIR/smoke_deploy.log" 2>&1 \
    && tail -3 "$LOGDIR/smoke_deploy.log" \
    || { echo "deploy eval FAILED"; tail -20 "$LOGDIR/smoke_deploy.log"; }
else
  echo "no CRAFT run record found"
fi

echo "### 5/5 analysis scripts"
BASE_DIR=$(ls -dt results/*Smoke*_0 2>/dev/null | grep -v rag | head -1)
CRAFT_DIR=$(ls -dt results/*Smoke*rag* 2>/dev/null | grep -v deploy | head -1)
if [ -n "$BASE_DIR" ] && [ -n "$CRAFT_DIR" ]; then
  if [ -f "$BASE_DIR/pred.npy" ]; then
    $PY experiments/freq_band_analysis.py --base "$BASE_DIR" --craft "$CRAFT_DIR" \
        --dataset "$NAME" --model "$MODEL" --out analysis/smoke_freq.csv
  else
    echo "skipping freq analysis: SLIM=1 did not write pred.npy (expected)"
  fi
  $PY experiments/per_example_analysis.py --base "$BASE_DIR" --craft "$CRAFT_DIR" \
      --dataset "$NAME" --model "$MODEL" --out analysis/smoke_per_example.csv
fi
$PY experiments/aggregate_results.py --runs runs --out analysis/smoke_agg

echo
echo "### smoke test done. Delete the smoke artifacts before the real campaign:"
echo "  rm -rf runs/*Smoke* results/*Smoke* checkpoints/*Smoke* test_results/*Smoke* analysis/smoke_*"
