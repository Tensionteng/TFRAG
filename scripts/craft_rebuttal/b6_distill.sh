#!/bin/bash
# B6 -- can CRAFT actually beat the baseline? Testing the distillation channel.
#
# WHY: the pilot showed the submitted method LOSES to plain MSE (ETTh1 -0.65% over
# 5 seeds, p=0.26; Weather -22% on seed 2021). The cause is mechanistic. The RL loss
# reaches theta only through the state s = [Y_hat; Y_ref], so its gradient pushes
# Y_hat towards states where good actions are more *probable* -- which is unrelated
# to Y_hat being accurate. The paper measured that gradient at ~1e-7 and concluded
# it was harmless; it was small only because the optimizer bug left the policy head
# frozen at initialization. With the corrector actually training, the perturbation
# is large enough to do real damage.
#
# THE FIX (--gamma_3): give theta an explicit, useful target instead of hoping for
# zero-order transfer. Per timestep, take the highest-reward sampled correction,
# keep it only if it improved something, and distil y_hat towards y_hat + a. The
# target is pooled (kappa=3) and improvement-screened, so it is a denoised step
# towards the truth rather than the raw label.
#
# This also turns the paper's weakest section into its strongest: "internalisation"
# stops being three hypothesised zero-order mechanisms and becomes a first-order
# channel you can point at, ablate, and measure.
#
# FAIRNESS: every arm below shares one protocol, architecture, lr, batch and seed
# set. gamma_3 is selected on VALIDATION loss (early stopping already does this);
# no arm gets extra tuning, and no test-set selection happens anywhere. The
# baseline is the same iTransformer config used by the CRAFT arms.
#
# Usage
#   GPUS="1 2 3" bash scripts/craft_rebuttal/parallel.sh \
#       "bash scripts/craft_rebuttal/b6_distill.sh"
#   # or serially:
#   GPU=1 bash scripts/craft_rebuttal/b6_distill.sh
source "$(dirname "$0")/common.sh"

MODEL=${MODEL:-iTransformer}
PL=${PL:-96}
D_SEEDS=${D_SEEDS:-"2021 1 2"}
G3_GRID=${G3_GRID:-"0.5 1.0 2.0"}
DS=("${DATASETS_SMALL[@]}")
if [ "${WITH_ECL:-0}" = "1" ]; then
  DS+=("ECL|./dataset/electricity/|electricity.csv|custom|321")
fi
if [ "${WITH_ETTM:-0}" = "1" ]; then
  DS+=("ETTm1|./dataset/ETT-small/|ETTm1.csv|ETTm1|7")
  DS+=("ETTh2|./dataset/ETT-small/|ETTh2.csv|ETTh2|7")
fi

run_ds() {
  for entry in "${DS[@]}"; do
    IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
    for SEED in $D_SEEDS; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" "$@"
    done
  done
}

# Arm 1: baseline (shared by every comparison).
echo "### arm 1/4: base (plain MSE)"
run_ds

# Arm 2: submitted method, for the record. Expected to lose; that is the point.
echo "### arm 2/4: craft as submitted (RL term only, gamma_3=0)"
run_ds --use_rag --gamma_1 1.0 --gamma_2 0.5 --num_retrieve 5 --num_rl_samples 8 \
       --exclusion_radius "$PL" --retrieval_mode nn

# Arm 3: RL trains the corrector only, no distillation. Isolates "does removing the
# harmful RL->theta perturbation alone recover the baseline?"
echo "### arm 3/4: craft + detach (RL trains the corrector only)"
run_ds --use_rag --gamma_1 1.0 --gamma_2 0.5 --num_retrieve 5 --num_rl_samples 8 \
       --exclusion_radius "$PL" --retrieval_mode nn --detach_yhat

# Arm 4: the proposed transfer channel, over a small gamma_3 grid.
echo "### arm 4/4: craft + distillation"
for G3 in $G3_GRID; do
  echo "--- gamma_3=$G3 (best-of-N target)"
  run_ds --use_rag --gamma_1 1.0 --gamma_2 0.5 --num_retrieve 5 --num_rl_samples 8 \
         --exclusion_radius "$PL" --retrieval_mode nn --detach_yhat \
         --gamma_3 "$G3" --distill_target best
done
if [ "${WITH_ADV:-0}" = "1" ]; then
  echo "--- advantage-weighted target"
  run_ds --use_rag --gamma_1 1.0 --gamma_2 0.5 --num_retrieve 5 --num_rl_samples 8 \
         --exclusion_radius "$PL" --retrieval_mode nn --detach_yhat \
         --gamma_3 1.0 --distill_target advantage
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo
echo "### comparing every arm against the shared baseline"
# Discover the variants actually present rather than reconstructing their names in
# bash: python formats gamma_3=1.0 as "1", and guessing that mapping is a bug farm.
VARIANTS=$($PY - <<'EOF'
import glob, json
vs = set()
for p in glob.glob("runs/*.json"):
    v = json.load(open(p)).get("variant", "")
    if v.startswith("craft"):
        vs.add(v)
print(" ".join(sorted(vs)))
EOF
)
echo "found arms: $VARIANTS"
for V in $VARIANTS; do
  echo "--- $V"
  $PY experiments/aggregate_results.py --runs runs --out "analysis/b6_$V" \
      --baseline base --treatment "$V" 2>&1 \
      | grep -E "mean MSE|cells improved|p<0.05" | sed 's/^/    /'
done
echo
echo "gamma_3 is chosen on validation loss (early stopping). Report the selected"
echo "value and the whole grid, so the selection is visible rather than implied."
