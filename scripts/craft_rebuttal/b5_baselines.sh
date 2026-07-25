#!/bin/bash
# B5 -- modern backbones and frequency-aware baselines, on ALL datasets.
#
# WHAT REVIEWERS ASKED:
#  * ErQJ weakness 2: "Evaluated baselines are outdated. No 2026, 2025 models and
#    only 2 of 8 are from 2024."
#  * 8ivr: "comparison with existing methods for the spectral bias problem is
#    limited to arbitrary datasets and deferred to the appendix. A more
#    comprehensive evaluation with current methods across the five datasets would
#    establish their significance better."
#  * zxKS: asks for comparison with methods that explicitly target spectral bias.
#
# Part 1 adds backbones already vendored in models/ but never evaluated with CRAFT
# (TimeXer, WPMixer, MultiPatchFormer), so this is config work, not new modelling.
# Part 2 runs FreDF / FFL / BandWeightedMSE as *losses* on every dataset -- the
# submission only had Weather and ECL, which is what made the comparison look
# cherry-picked.
#
# Usage
#   PART=backbones bash scripts/craft_rebuttal/b5_baselines.sh
#   PART=losses    bash scripts/craft_rebuttal/b5_baselines.sh
#   PART=all       bash scripts/craft_rebuttal/b5_baselines.sh
source "$(dirname "$0")/common.sh"

PART=${PART:-all}
PL=${PL:-96}
B5_SEEDS=${B5_SEEDS:-"2021 1 2"}
MODELS=${MODELS:-"TimeXer WPMixer MultiPatchFormer PatchTST DLinear TimeMixer"}
DS=("${DATASETS_ALL[@]}")
if [ "${B5_SMALL:-0}" = "1" ]; then DS=("${DATASETS_SMALL[@]}"); fi

if [ "$PART" = "backbones" ] || [ "$PART" = "all" ]; then
  echo "### plug-and-play across backbones: base vs CRAFT, $B5_SEEDS seeds"
  for MODEL in $MODELS; do
    for entry in "${DS[@]}"; do
      IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
      for SEED in $B5_SEEDS; do
        run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED"
        run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$(craft_flags $PL)"
      done
    done
  done
fi

if [ "$PART" = "losses" ] || [ "$PART" = "all" ]; then
  echo "### frequency-aware training objectives on every dataset"
  MODEL=${LOSS_MODEL:-iTransformer}
  for LOSS in MSE mae huber fredf ffl bandmse; do
    echo "--- loss=$LOSS"
    for entry in "${DS[@]}"; do
      IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
      for SEED in $B5_SEEDS; do
        run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" --loss "$LOSS"
      done
    done
  done
  # CRAFT for the same cells, so the table has a matched CRAFT row -- the submitted
  # frequency-baseline table promised one in its caption and did not have it.
  for entry in "${DS[@]}"; do
    IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
    for SEED in $B5_SEEDS; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$(craft_flags $PL)"
    done
  done
fi

echo
echo "### aggregating"
$PY experiments/aggregate_results.py --runs runs --out analysis/b5
for L in base_mae base_huber base_fredf base_ffl base_bandmse; do
  $PY experiments/aggregate_results.py --runs runs --out "analysis/b5_$L" \
      --baseline base --treatment "$L" 2>&1 | tail -3
done
