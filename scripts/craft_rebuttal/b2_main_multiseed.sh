#!/bin/bash
# B2 -- the decision-critical experiment: multi-seed main results with paired tests.
#
# WHAT REVIEWERS ASKED: "Can you provide multi-seed std and a paired significance
# test for the main table, not just the ablations? How is the headline 7.23%/5.89%
# shown to be above the noise floor?" (ErQJ Q2; 8ivr; AC)
#
# WHAT THIS PRODUCES: for every (dataset x pred_len) on the chosen backbone, N seeds
# of base and N seeds of CRAFT under ONE identical protocol, then
# analysis/paired_tests.csv + analysis/summary.md with per-cell std, paired t-test,
# Wilcoxon, Cohen's dz and a bootstrap CI on the aggregate gain.
#
# COST: 2 conditions x |SEEDS| x |PRED_LENS| x |datasets| runs.
#   Default = 2 x 5 x 4 x 8 = 320 runs. On one A100 that is roughly 2-4 days for
#   iTransformer; ECL and Traffic dominate. Start with SCOPE=pilot.
#
# Usage
#   SCOPE=pilot bash scripts/craft_rebuttal/b2_main_multiseed.sh      # 2 datasets, pl=96
#   SCOPE=full  bash scripts/craft_rebuttal/b2_main_multiseed.sh
#   DRY_RUN=1 SCOPE=full bash scripts/craft_rebuttal/b2_main_multiseed.sh | wc -l
source "$(dirname "$0")/common.sh"

MODEL=${MODEL:-iTransformer}
SCOPE=${SCOPE:-pilot}

if [ "$SCOPE" = "pilot" ]; then
  DS=("${DATASETS_SMALL[@]}"); PRED_LENS=${PRED_LENS_OVERRIDE:-"96"}
else
  DS=("${DATASETS_ALL[@]}")
fi

echo "B2: model=$MODEL scope=$SCOPE seeds=[$SEEDS] pred_lens=[$PRED_LENS] epochs=$EPOCHS"
echo "    conditions: base, craft (nn retrieval + temporal exclusion)"

for entry in "${DS[@]}"; do
  IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
  for PL in $PRED_LENS; do
    for SEED in $SEEDS; do
      # Identical protocol; the ONLY difference is the corrector.
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED"
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$(craft_flags $PL)"
    done
  done
done

echo
echo "### aggregating"
$PY experiments/aggregate_results.py --runs runs --out analysis
echo
echo "Read analysis/summary.md. If the bootstrap CI on the aggregate MSE change"
echo "includes 0, the headline claim must be restated, not re-run with more seeds."
