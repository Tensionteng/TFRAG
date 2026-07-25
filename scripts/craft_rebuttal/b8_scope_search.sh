#!/bin/bash
# B8 -- where, if anywhere, does CRAFT genuinely help?
#
# Everything measured so far is iTransformer @ horizon 96 on ETTh1 and Weather, where
# CRAFT loses. That is a narrow slice, and three regimes remain untested -- two of them
# the ones the submission itself points at:
#
#   1. WEAK BACKBONES. The submitted heatmap reports 38/40 improved combinations, and
#      Reviewer ErQJ observed the gains concentrate in weak backbones. A regulariser
#      plausibly helps a linear model with headroom more than a tuned Transformer.
#   2. ECL. The submitted per-example table is positive on ECL alone (+4.2%) and
#      negative on the other five datasets. That asymmetry may be real.
#   3. LONGER HORIZONS. The main table averages {96,192,336,720}; only 96 is measured.
#
# METHOD -- this is a search, so it is run as one:
#   Phase 1 (this script) is EXPLORATORY. 2 seeds per cell, wide grid. Every cell is
#   reported, including losses. A cell looking good here proves nothing: with ~35 cells
#   at 2 seeds, several will look positive by chance.
#   Phase 2 (b9_confirm.sh, generated below) takes only the cells that passed Phase 1
#   and re-tests them on 5 FRESH seeds. A result that survives that is reportable.
#
# Reporting the whole grid is not optional. Selecting the winners from an unreported
# search is precisely the practice that makes a reviewer distrust everything else.
source "$(dirname "$0")/common.sh"

PHASE=${PHASE:-1}
EXP_SEEDS=${EXP_SEEDS:-"2021 1"}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"7 8 9 10 11"}   # disjoint from EXP_SEEDS by design

craft_arm() { echo "--use_rag --gamma_1 1.0 --gamma_2 ${GAMMA2:-0.5} --num_retrieve 5 \
  --num_rl_samples 8 --exclusion_radius $1 --retrieval_mode nn"; }

# ---------------------------------------------------------------- phase 1 grid
# Backbones present in this repo, weakest first. RLinear/FITS/ModernTCN/GPT4TS from the
# paper's baseline table are NOT vendored here, so they cannot be run.
BACKBONES=${BACKBONES:-"DLinear TSMixer SegRNN FreTS PatchTST TimesNet iTransformer"}
B8_DATASETS=(
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
  "ETTm2|./dataset/ETT-small/|ETTm2.csv|ETTm2|7"
  "Weather|./dataset/weather/|weather.csv|custom|21"
  "ECL|./dataset/electricity/|electricity.csv|custom|321"
)
HORIZONS=${HORIZONS:-"96 336"}

if [ "$PHASE" = "1" ]; then
  echo "### B8 phase 1 (exploratory): backbones x datasets x horizons, $EXP_SEEDS"
  for MODEL in $BACKBONES; do
    for entry in "${B8_DATASETS[@]}"; do
      IFS='|' read -r NAME ROOT DPATH DATA ENC <<< "$entry"
      for PL in $HORIZONS; do
        for SEED in $EXP_SEEDS; do
          run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED"
          run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" \
                  "$(craft_arm "$PL")"
        done
      done
    done
  done
fi

# ---------------------------------------------------------------- phase 2 list
if [ "$PHASE" = "2" ]; then
  echo "### B8 phase 2 (confirmation): only Phase-1 candidates, fresh seeds $CONFIRM_SEEDS"
  if [ ! -s analysis/b8_candidates.txt ]; then
    echo "no analysis/b8_candidates.txt -- run the selection step below first"; exit 1
  fi
  while IFS=, read -r NAME ROOT DPATH DATA ENC MODEL PL; do
    [ -z "$NAME" ] && continue
    echo "--- confirming $NAME/$MODEL/pl$PL"
    for SEED in $CONFIRM_SEEDS; do
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED"
      run_one "$NAME" "$ROOT" "$DPATH" "$DATA" "$ENC" "$MODEL" "$PL" "$SEED" \
              "$(craft_arm "$PL")"
    done
  done < analysis/b8_candidates.txt
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

# --------------------------------------------------------------- selection step
echo
echo "### full grid (every cell, wins and losses)"
$PY experiments/compare_variants.py --runs runs --markdown analysis/b8_grid.md | tail -60

echo
echo "### selecting Phase-2 candidates: craft ahead of base on ALL exploratory seeds"
$PY - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd())
from collections import defaultdict
from experiments.aggregate_results import load_runs

DS = {  # name -> root, data_path, data, enc_in  (must match common.sh)
    "ETTh1": ("./dataset/ETT-small/", "ETTh1.csv", "ETTh1", 7),
    "ETTm2": ("./dataset/ETT-small/", "ETTm2.csv", "ETTm2", 7),
    "Weather": ("./dataset/weather/", "weather.csv", "custom", 21),
    "ECL": ("./dataset/electricity/", "electricity.csv", "custom", 321),
}
STEM2NAME = {"ETTh1": "ETTh1", "ETTm2": "ETTm2", "weather": "Weather", "electricity": "ECL"}

runs = load_runs("runs")
idx = defaultdict(lambda: defaultdict(dict))
for r in runs:
    idx[(r["dataset"], r["model"], r["pred_len"])][r["variant"]][r["seed"]] = r

cands, examined = [], 0
for (ds, model, pl), v in sorted(idx.items(), key=lambda kv: [str(x) for x in kv[0]]):
    base, craft = v.get("base"), v.get("craft")
    if not base or not craft:
        continue
    seeds = sorted(set(base) & set(craft))
    if len(seeds) < 2:
        continue
    examined += 1
    wins = sum(craft[s]["mse"] < base[s]["mse"] for s in seeds)
    if wins == len(seeds):  # unanimous on the exploratory seeds
        name = STEM2NAME.get(ds)
        if name:
            root, dpath, data, enc = DS[name]
            cands.append(f"{name},{root},{dpath},{data},{enc},{model},{pl}")

os.makedirs("analysis", exist_ok=True)
with open("analysis/b8_candidates.txt", "w") as f:
    f.write("\n".join(cands) + ("\n" if cands else ""))
print(f"examined {examined} cells; {len(cands)} unanimous candidate(s)")
for c in cands:
    print("  ", c)
print()
print(f"With {examined} cells at 2 seeds, roughly {examined/4:.1f} unanimous cells are")
print("expected by chance alone. Phase 2 on fresh seeds is what separates signal from that.")
EOF

echo
echo "### next: PHASE=2 bash scripts/craft_rebuttal/b8_scope_search.sh"
