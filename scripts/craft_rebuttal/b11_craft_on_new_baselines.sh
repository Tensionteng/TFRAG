#!/bin/bash
# Does CRAFT improve the 2026 baselines? Small datasets, unified lookback.
#
# The question reviewer ErQJ's W2 really poses is not "are there newer models" but
# "does the plug-in still pay off on them". Answering it needs base and CRAFT at the
# SAME lookback, paired by seed -- which is what this script enforces: SEQ_LEN is set
# once per phase and both arms inherit it.
#
# It also fixes an unfairness in the earlier b9 runs. Those used gamma_2 = 0.5, the
# default tuned (loosely) for iTransformer. On ETTh1 the only CRAFT arm that has ever
# shown a positive paired delta is gamma_2 = 0.1 with a frozen policy, so testing a
# new backbone at 0.5 alone would report "CRAFT fails" when the honest statement is
# "CRAFT at an untuned strength fails". Both strengths are swept here.
#
#   PHASE=A bash scripts/craft_rebuttal/b11_craft_on_new_baselines.sh   # L=96
#   PHASE=B ...                                                        # L=720

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/common.sh"

PHASE=${PHASE:-A}
SEEDS=${SEEDS:-"2021 1 2"}
PL=${PL:-96}
MODELS=${MODELS:-"MixLinear FACT"}

# Small datasets only: an ETT run is minutes, so a full grid lands the same day.
SMALL=(
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
  "ETTh2|./dataset/ETT-small/|ETTh2.csv|ETTh2|7"
  "ETTm1|./dataset/ETT-small/|ETTm1.csv|ETTm1|7"
  "ETTm2|./dataset/ETT-small/|ETTm2.csv|ETTm2|7"
)

CRAFT="--use_rag --num_retrieve 5 --exclusion_radius 96 --policy_hidden 128 --gamma_1 1.0"

# Arms, in increasing corrector strength. 'base' is the paired reference; every CRAFT
# arm differs from it in the corrector alone.
arms_for_phase() {
  case "$PHASE" in
    # L=96: full sweep, including the default 0.5 so the earlier b9 rows stay
    # comparable and the gamma_2 dependence is visible rather than assumed.
    A) echo "base|:g2:0.05:frozen|:g2:0.1:frozen|:g2:0.1:live|:g2:0.5:live" ;;
    # L=720 is MixLinear's own operating point and quadruples the cost, so only the
    # reference and the arm that wins at L=96 are carried over.
    B) echo "base|:g2:0.1:frozen" ;;
  esac
}

run_arm() {  # run_arm <arm> <name> <root> <dpath> <data> <enc> <model> <seed>
  local arm=$1; shift
  local name=$1 root=$2 dpath=$3 data=$4 enc=$5 model=$6 seed=$7
  if [ "$arm" = "base" ]; then
    run_one "$name" "$root" "$dpath" "$data" "$enc" "$model" "$PL" "$seed" --skip_if_done
    return
  fi
  IFS=':' read -r _ _ g2 mode <<< "$arm"
  local frozen=""
  [ "$mode" = "frozen" ] && frozen="--freeze_policy"
  run_one "$name" "$root" "$dpath" "$data" "$enc" "$model" "$PL" "$seed" \
    $CRAFT --gamma_2 "$g2" $frozen --skip_if_done
}

case "$PHASE" in
  A) export SEQ_LEN=96  ;;
  B) export SEQ_LEN=720 ;;
  *) echo "PHASE must be A or B"; exit 1 ;;
esac

IFS='|' read -ra ARMS <<< "$(arms_for_phase)"
for entry in "${SMALL[@]}"; do
  IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
  for MODEL in $MODELS; do
    for arm in "${ARMS[@]}"; do
      for s in $SEEDS; do
        run_arm "$arm" "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$s"
      done
    done
  done
done

echo "b11 phase $PHASE done (L=$SEQ_LEN)"
