#!/bin/bash
# B4 -- gamma_2 sensitivity + the component ablations, all with multiple seeds.
#
# WHAT REVIEWERS ASKED:
#  * zxKS: "In Table 1(b) the baseline appears to outperform CRAFT for several
#    gamma_2. Is the benefit only in a narrow range, and how sensitive is it?"
#    -- the submitted table claims [0.5, 1.0] is robust, but its own Weather row
#    at gamma_2=1.0 (0.1740) is worse than the baseline (0.1736). This produces a
#    real curve with error bars so the claim can be stated correctly.
#  * ErQJ Q4: "If random retrieval matches NN retrieval, what supports the
#    delta-informativeness assumption?" -- the random control, now seeded.
#  * The retrieval safeguard: the submitted paper reports self-exclusion and a P/2
#    temporal radius, but the released code implemented neither. This runs the
#    exclusion sweep for real (0 = none, P/2, P, 2P).
#  * The detach ablation and the discrete-vs-continuous reward comparison.
#
# Usage
#   WHICH=gamma     bash scripts/craft_rebuttal/b4_gamma_and_ablations.sh
#   WHICH=retrieval bash scripts/craft_rebuttal/b4_gamma_and_ablations.sh
#   WHICH=all       bash scripts/craft_rebuttal/b4_gamma_and_ablations.sh
source "$(dirname "$0")/common.sh"

MODEL=${MODEL:-iTransformer}
WHICH=${WHICH:-all}
PL=${PL:-96}
ABL_SEEDS=${ABL_SEEDS:-"2021 1 2"}
DS=("${DATASETS_SMALL[@]}")
if [ "${ABL_WITH_ECL:-0}" = "1" ]; then
  DS+=("ECL|./dataset/electricity/|electricity.csv|custom|321")
fi

run_ds() { # run_ds <extra flags...>
  for entry in "${DS[@]}"; do
    IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
    for SEED in $ABL_SEEDS; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$@"
    done
  done
}

if [ "$WHICH" = "gamma" ] || [ "$WHICH" = "all" ]; then
  echo "### gamma_2 sweep (gamma_2=0 is the plain MSE baseline)"
  run_ds   # gamma_2 = 0 <=> base run, no corrector
  for G2 in 0.1 0.25 0.5 0.75 1.0 2.0 5.0; do
    echo "--- gamma_2=$G2"
    run_ds --use_rag --gamma_1 1.0 --gamma_2 "$G2" --num_retrieve 5 --num_rl_samples 8 \
           --exclusion_radius "$PL" --retrieval_mode nn --tag g2sweep
  done
fi

if [ "$WHICH" = "retrieval" ] || [ "$WHICH" = "all" ]; then
  echo "### retrieval structure: NN vs random, matched exclusion"
  run_ds "$(craft_flags $PL)"                                          # canonical
  run_ds --use_rag --gamma_1 1.0 --gamma_2 "${GAMMA2:-0.5}" --num_retrieve 5 \
         --num_rl_samples 8 --exclusion_radius "$PL" --retrieval_mode random

  echo "### exclusion radius sweep: none / P-half / P / 2P"
  for R in 0 $((PL / 2)) "$PL" $((PL * 2)); do
    echo "--- exclusion_radius=$R"
    run_ds --use_rag --gamma_1 1.0 --gamma_2 "${GAMMA2:-0.5}" --num_retrieve 5 \
           --num_rl_samples 8 --exclusion_radius "$R" --retrieval_mode nn --tag exclsweep
  done

  echo "### k sweep"
  for K in 1 3 5 10 20; do
    run_ds --use_rag --gamma_1 1.0 --gamma_2 "${GAMMA2:-0.5}" --num_retrieve "$K" \
           --num_rl_samples 8 --exclusion_radius "$PL" --tag ksweep
  done
fi

if [ "$WHICH" = "mechanism" ] || [ "$WHICH" = "all" ]; then
  echo "### detach ablation (blocks the RL gradient path into theta)"
  run_ds "$(craft_flags $PL)" --detach_yhat

  echo "### discrete vs continuous reward"
  run_ds --use_rag --gamma_1 1.0 --gamma_2 "${GAMMA2:-0.5}" --num_retrieve 5 \
         --num_rl_samples 8 --exclusion_radius "$PL" --reward_type continuous

  echo "### L2 action penalty (paper Eq. 12, absent from the released code)"
  for LAM in 0.001 0.01 0.1; do
    run_ds "$(craft_flags $PL)" --lambda_reg "$LAM" --tag lamsweep
  done

  echo "### N_s sweep"
  for NS in 2 4 8 16; do
    run_ds --use_rag --gamma_1 1.0 --gamma_2 "${GAMMA2:-0.5}" --num_retrieve 5 \
           --num_rl_samples "$NS" --exclusion_radius "$PL" --tag nssweep
  done
fi

echo
echo "### aggregating each treatment against the shared base runs"
for V in craft craft_random craft_noexcl craft_detach craft_continuous; do
  echo "--- $V"
  $PY experiments/aggregate_results.py --runs runs --out "analysis/abl_$V" \
      --baseline base --treatment "$V" 2>&1 | tail -4
done
