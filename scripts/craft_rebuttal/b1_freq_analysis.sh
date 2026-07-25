#!/bin/bash
# B1 -- corrected band-wise spectral analysis. Post-hoc: no training, minutes to run.
#
# WHAT REVIEWERS ASKED (ErQJ Q5, weakness 6): "Please reconcile the conflicting
# values between Tables 21 and 26 for iTransformer-ECL, and clarify Table 27's sign
# convention -- do positive percentages mean reduction or increase, and how can a
# 'reduction' exceed 100%?"
#
# The answer is in the old code: freq_analysis_all.py computed the PSD of the
# PREDICTION, not of the residual, and printed (E_craft - E_base)/E_base under a
# header that said "Error Energy Reduction". A band where the prediction gained
# energy therefore appeared as a large positive "reduction", which is why cells
# exceeded 100%.
#
# This regenerates the analysis with three separated, explicitly signed quantities:
#   res_reduction_pct  -- true residual-energy reduction, positive = better, <= 100
#   gap_reduction_pct  -- movement of band energy TOWARDS the ground-truth energy
#   frmse_reduction_pct -- the paper's Section 5.3 FRMSE, unchanged definition
#
# PREREQUISITE: B2 (and optionally B5) so that results/*/pred.npy exist.
source "$(dirname "$0")/common.sh"

echo "### building base/CRAFT pair manifest from run records"
$PY - <<'EOF'
import glob, json, os
recs = [json.load(open(p)) for p in glob.glob("runs/*.json") if "Smoke" not in p]
base = {}
for r in recs:
    c = r["config"]
    if r.get("variant") == "base":
        base[(c["data"], c["model"], c["pred_len"], c["seed"], c["train_epochs"])] = r["setting"]
rows, unmatched = [], 0
for r in recs:
    c = r["config"]
    if r.get("variant") == "craft":
        k = (c["data"], c["model"], c["pred_len"], c["seed"], c["train_epochs"])
        if k in base:
            rows.append(f'{c["data"]},{c["model"]},results/{base[k]},results/{r["setting"]}')
        else:
            unmatched += 1
os.makedirs("analysis", exist_ok=True)
with open("analysis/freq_pairs.csv", "w") as f:
    f.write("\n".join(rows) + ("\n" if rows else ""))
print(f"[manifest] analysis/freq_pairs.csv: {len(rows)} pairs ({unmatched} CRAFT runs unmatched)")
EOF

if [ ! -s analysis/freq_pairs.csv ]; then
  echo "no pairs found -- run b2_main_multiseed.sh first"; exit 1
fi

$PY experiments/freq_band_analysis.py \
    --manifest analysis/freq_pairs.csv \
    --out analysis/freq_bands.csv \
    --latex analysis/freq_bands.tex

echo
echo "### band summary averaged over all pairs"
$PY - <<'EOF'
import csv
from collections import defaultdict
rows = list(csv.DictReader(open("analysis/freq_bands.csv")))
agg = defaultdict(list)
for r in rows:
    if r["band"] == "ALL":
        continue
    try:
        agg[r["band"]].append(float(r["res_reduction_pct"]))
    except ValueError:
        pass
print(f"{'band':<10} {'n':>4} {'mean residual reduction %':>26}")
for b in sorted(agg):
    v = agg[b]
    print(f"{b:<10} {len(v):>4} {sum(v)/len(v):>25.2f}")
print()
print("The paper claims gains concentrate in 0.2-0.5. That holds only if those")
print("three bands beat 0.0-0.2 here. Whatever the table says is what goes in.")
EOF

echo
echo "Deliverables: analysis/freq_bands.csv, analysis/freq_bands.tex"
