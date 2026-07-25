#!/bin/bash
# B7 -- the reviewer questions that still have no data. Nothing here is optional.
#
# Each part names the exact question it answers, so this script can be checked against
# the review rather than against anyone's plan.
#
#  PART 1  ErQJ Q2 + Q3, 8ivr "missing stds"
#          "Can you provide multi-seed std and a paired significance test for the main
#           Table, not just the ablations?" / "Table 12 shows the extracted base model
#           is worse on 5/6 datasets."
#          -> base vs craft on ALL 8 benchmarks, 3 seeds, one protocol. Every metric is
#             the deployed backbone alone, which is exactly the artifact Q3 is about.
#
#  PART 2  ErQJ Q4
#          "If random retrieval matches NN retrieval, what supports the
#           delta-informativeness assumption in Proposition 2?"
#          -> the random-retrieval control, seeded, with the temporal exclusion
#             safeguard actually enforced (the submitted ablation had neither).
#
#  PART 3  ErQJ Q5 + 8ivr Q3
#          "Reconcile Tables 21/26 and clarify Table 27's sign convention."
#          -> keeps pred/true arrays so experiments/freq_band_analysis.py can emit the
#             corrected band table on the three datasets those tables cover.
#             Also feeds the chronological-drift split for 8ivr Q3.
#
# Usage
#   GPUS="4 5 6 7" bash scripts/craft_rebuttal/parallel.sh \
#       "PART=1 bash scripts/craft_rebuttal/b7_reviewer_questions.sh"
#   PART=all GPU=0 bash scripts/craft_rebuttal/b7_reviewer_questions.sh   # serial
source "$(dirname "$0")/common.sh"

MODEL=${MODEL:-iTransformer}
PART=${PART:-all}
PL=${PL:-96}
Q_SEEDS=${Q_SEEDS:-"2021 1 2"}

# All eight benchmarks the paper's main table reports.
ALL=("${DATASETS_ALL[@]}")
# The six the submitted random-retrieval ablation used.
SIX=(
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
  "ETTh2|./dataset/ETT-small/|ETTh2.csv|ETTh2|7"
  "ETTm1|./dataset/ETT-small/|ETTm1.csv|ETTm1|7"
  "ETTm2|./dataset/ETT-small/|ETTm2.csv|ETTm2|7"
  "Weather|./dataset/weather/|weather.csv|custom|21"
  "ECL|./dataset/electricity/|electricity.csv|custom|321"
)
# The three the submitted band-energy tables cover.
THREE=(
  "Weather|./dataset/weather/|weather.csv|custom|21"
  "ECL|./dataset/electricity/|electricity.csv|custom|321"
  "Traffic|./dataset/traffic/|traffic.csv|custom|862"
)

sweep() { # sweep <array name> <seeds> <extra flags...>
  local -n DS=$1; shift
  local seeds=$1; shift
  for entry in "${DS[@]}"; do
    IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
    for SEED in $seeds; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$@"
    done
  done
}

CRAFT="--use_rag --gamma_1 1.0 --gamma_2 0.5 --num_retrieve 5 --num_rl_samples 8 \
       --exclusion_radius $PL --retrieval_mode nn"

if [ "$PART" = "1" ] || [ "$PART" = "all" ]; then
  echo "### PART 1 (ErQJ Q2/Q3): base vs craft, 8 benchmarks, $Q_SEEDS"
  sweep ALL "$Q_SEEDS"
  sweep ALL "$Q_SEEDS" $CRAFT
fi

if [ "$PART" = "2" ] || [ "$PART" = "all" ]; then
  echo "### PART 2 (ErQJ Q4): random-retrieval control, 6 benchmarks"
  sweep SIX "$Q_SEEDS" --use_rag --gamma_1 1.0 --gamma_2 0.5 --num_retrieve 5 \
        --num_rl_samples 8 --exclusion_radius "$PL" --retrieval_mode random
fi

if [ "$PART" = "3" ] || [ "$PART" = "all" ]; then
  echo "### PART 3 (ErQJ Q5 / 8ivr Q3): keep arrays for the corrected band table"
  # SLIM must NOT apply here: the frequency analysis needs pred/true. One seed is
  # enough for a spectral table, which is a within-run comparison.
  ARTIFACT_FLAGS=""
  sweep THREE "2021" --tag freqarrays
  sweep THREE "2021" $CRAFT --tag freqarrays
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo
echo "### analysis"
$PY experiments/aggregate_results.py --runs runs --out analysis --treatment craft
$PY experiments/compare_variants.py --runs runs --markdown analysis/variants.md | tail -25
echo
echo "### ErQJ Q4: does random retrieval match NN?"
$PY experiments/compare_variants.py --runs runs 2>/dev/null \
  | grep -E "craft_random|craft\`" | sed 's/^/    /'
echo
echo "Next: bash scripts/craft_rebuttal/b1_freq_analysis.sh   (ErQJ Q5)"
echo "      bash scripts/craft_rebuttal/b3_deployment.sh      (ErQJ Q3 quartiles)"
