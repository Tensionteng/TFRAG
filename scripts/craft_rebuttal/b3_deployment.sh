#!/bin/bash
# B3 -- does the DEPLOYED artifact actually improve? (ErQJ Q3 / weakness 4)
#
# WHAT REVIEWERS ASKED: "Table 12 shows the extracted base model is worse on 5/6
# datasets (up to -20.8%). If only the base model is deployed, how does this
# reconcile with the main improvement claim?"
#
# The submitted evidence was single-run, 10-epoch. This re-runs the extraction and
# evaluation under the SAME protocol as B2, over all seeds, and adds a per-window
# paired test plus the difficulty-quartile breakdown.
#
# PREREQUISITE: run B2 first -- this consumes its checkpoints and run records.
#
# NOTE ON WHAT IS BEING MEASURED: exp.test() already evaluates the backbone alone
# (the plugin is a pass-through in eval mode), so a correct B2 CRAFT run and this
# script should agree. They are computed by different code paths on purpose: if the
# numbers diverge, the extraction or the eval path is wrong and nothing downstream
# is trustworthy. The script reports the discrepancy explicitly.
source "$(dirname "$0")/common.sh"

echo "B3: re-evaluating extracted backbones from every CRAFT run record"
mapfile -t JSONS < <(ls runs/*rag*.json 2>/dev/null | grep -v Smoke)
if [ ${#JSONS[@]} -eq 0 ]; then
  echo "no CRAFT run records in runs/ -- run b2_main_multiseed.sh first"; exit 1
fi
echo "found ${#JSONS[@]} CRAFT runs"

for J in "${JSONS[@]}"; do
  echo "=== $(basename "$J")"
  if [ "$DRY_RUN" = "1" ]; then
    echo "$PY experiments/eval_extracted_base.py --run_json $J --gpu $GPU"
    continue
  fi
  LOG="$LOGDIR/deploy_$(basename "$J" .json).log"
  if $PY experiments/eval_extracted_base.py --run_json "$J" --gpu "$GPU" > "$LOG" 2>&1; then
    grep -E "DEPLOYED BACKBONE" "$LOG" | sed 's/^/    /'
    # Cross-check against the in-training test metric from the same run.
    $PY - "$J" "$LOG" <<'EOF'
import json, re, sys
rec = json.load(open(sys.argv[1]))
m = re.search(r"mse=([0-9.]+)", open(sys.argv[2]).read())
if m:
    a, b = rec["metrics"]["mse"], float(m.group(1))
    rel = abs(a - b) / max(a, 1e-12)
    flag = "OK" if rel < 1e-3 else "MISMATCH"
    print(f"    cross-check {flag}: in-training mse={a:.6f} extracted mse={b:.6f} (rel {rel:.2e})")
    if flag == "MISMATCH":
        print("    ^^ extraction or eval path disagrees -- investigate before reporting")
EOF
  else
    echo "    !! FAILED -- $LOG"; tail -5 "$LOG" | sed 's/^/    /'
  fi
done

echo
echo "### per-window quartile + drift analysis"
# Build a manifest pairing each base results dir with its CRAFT counterpart.
$PY - <<'EOF'
import glob, json, os
rows, missing = [], []
recs = [json.load(open(p)) for p in glob.glob("runs/*.json") if "Smoke" not in p]
base = {}
for r in recs:
    c = r["config"]
    key = (c["data"], c["model"], c["pred_len"], c["seed"], c["train_epochs"])
    if r.get("variant") == "base":
        base[key] = r["setting"]
for r in recs:
    c = r["config"]
    if r.get("variant", "").startswith("craft") and "deploy" not in (c.get("tag") or ""):
        key = (c["data"], c["model"], c["pred_len"], c["seed"], c["train_epochs"])
        if key in base:
            rows.append(f'{c["data"]},{c["model"]},results/{base[key]},results/{r["setting"]}')
        else:
            missing.append(str(key))
os.makedirs("analysis", exist_ok=True)
with open("analysis/pairs.csv", "w") as f:
    f.write("\n".join(rows) + ("\n" if rows else ""))
print(f"[manifest] analysis/pairs.csv: {len(rows)} pairs")
if missing:
    print(f"[warn] {len(missing)} CRAFT runs had no matching base run at the same seed/protocol")
EOF

if [ -s analysis/pairs.csv ]; then
  $PY experiments/per_example_analysis.py --manifest analysis/pairs.csv --out analysis/per_example.csv
else
  echo "no pairs -- nothing to analyse"
fi

echo
echo "Deliverable: analysis/per_example.csv (splits all / Q1..Q4 / first80 / last20)"
echo "The 'all' rows replace submitted Table 12. Report the sign honestly."
