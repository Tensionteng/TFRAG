#!/bin/bash
# ETTh1-only search for a CRAFT configuration that actually beats the baseline.
#
# Everything else in this campaign asked "is the submitted claim true?" (it is not).
# This asks the different question the authors need answered: does ANY configuration
# of retrieval + RL improve the deployed backbone, and how good can the absolute
# number get. ETTh1 only, so a wave completes in hours rather than days.
#
#   WAVE=1 bash scripts/craft_rebuttal/b10_etth1_sota.sh    # no code changes needed
#   WAVE=2 ...                                              # needs the loss/mixup/freq flags
#
# Every arm is paired against a base run at the SAME seq_len, lr and batch size, so
# a positive delta means the corrector helped, not that the protocol changed.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/common.sh"

WAVE=${WAVE:-1}
SEEDS=${SEEDS:-"2021 1 2"}
PL=${PL:-96}
D="ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$D"

# CRAFT flags shared by every arm: the safeguards the paper describes, actually on.
CRAFT_BASE="--use_rag --num_retrieve 5 --exclusion_radius 96 --policy_hidden 128"

# ---------------------------------------------------------------------- wave 1
# a) How good can the backbone alone get? Lookback is a free parameter for a
#    plug-and-play claim, and ETTh1 rewards it heavily. This sets the target.
# b) gamma_2 micro-sweep with a FROZEN policy. This is the only CRAFT arm that has
#    ever shown a positive paired delta (+0.75%, p=0.097 at g2=0.1), and the
#    mechanism -- a fixed random critic injecting structured noise into the
#    backbone's gradient -- predicts the optimum sits well below the default 0.5.
# c) gamma_3 distillation: hand the backbone the corrector's best accepted action
#    as a target. More samples means a better max, so Ns is raised with it.
if [ "$WAVE" = "1" ]; then
  for SL in 96 336 512 720; do
    for s in $SEEDS; do
      SEQ_LEN=$SL run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        --skip_if_done
    done
  done

  for g2 in 0.01 0.03 0.05 0.1 0.2; do
    for s in $SEEDS; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        $CRAFT_BASE --gamma_1 1.0 --gamma_2 "$g2" --freeze_policy --skip_if_done
    done
  done

  for g3 in 0.5 1 2 5; do
    for s in $SEEDS; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        $CRAFT_BASE --gamma_1 1.0 --gamma_2 0.1 --gamma_3 "$g3" \
        --num_rl_samples 16 --detach_yhat --skip_if_done
    done
  done

  # CRAFT at the long lookback, in case the corrector only pays off once the
  # backbone has enough context for the retrieved reference to be consistent.
  for SL in 336 720; do
    for s in $SEEDS; do
      SEQ_LEN=$SL run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        $CRAFT_BASE --gamma_1 1.0 --gamma_2 0.1 --freeze_policy --skip_if_done
    done
  done
fi

# ---------------------------------------------------------------------- wave 2
# The plugin hardcoded MSE as its base loss, so CRAFT had never been stacked on the
# objectives that DO beat MSE here (FreDF +4.46%, MAE +1.89%). With --loss honoured,
# these arms test the composition. SL is set to whatever wave 1 found best.
if [ "$WAVE" = "2" ]; then
  SL=${SL:-96}
  for L in fredf mae huber; do
    for s in $SEEDS; do
      SEQ_LEN=$SL run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        --loss "$L" --skip_if_done
      SEQ_LEN=$SL run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        --loss "$L" $CRAFT_BASE --gamma_1 1.0 --gamma_2 0.1 --freeze_policy --skip_if_done
    done
  done

  # Retrieval-manifold mixup: interpolate a window with its own nearest neighbour.
  # Unlike random mixup this stays on the data manifold, and unlike the corrector it
  # is a genuine extra information channel -- the neighbour's future is a second
  # observation of a similar state, which the MSE loss on a single target never sees.
  for a in 0.2 0.4 0.8; do
    for s in $SEEDS; do
      SEQ_LEN=$SL run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        $CRAFT_BASE --gamma_2 0 --rag_mixup_p 0.5 --rag_mixup_alpha "$a" --skip_if_done
    done
  done

  # Frequency-domain reward: credit a correction only when it reduces high-band
  # residual energy. This is the reward Proposition 1 actually implies, and it is
  # the one reward under which distillation transfers something MSE does not.
  for g3 in 1 2; do
    for s in $SEEDS; do
      SEQ_LEN=$SL run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" iTransformer "$PL" "$s" \
        $CRAFT_BASE --gamma_1 1.0 --gamma_2 0.1 --gamma_3 "$g3" --reward_type freq \
        --num_rl_samples 16 --detach_yhat --skip_if_done
    done
  done
fi

echo "b10 wave $WAVE done"
