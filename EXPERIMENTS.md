# CRAFT rebuttal experiments — build, run, deliver

This file is the complete brief for the agent (or person) who runs these experiments.
Read all of §0–§3 before launching anything.

- **Paper**: CRAFT: Corrector-Retrieval Augmented Fine-Tuning for Time Series Forecasting
  (NeurIPS 2026 submission 18915). Source in `craft/`, reviews in `craft/openreview_reviews.md`.
- **Scores**: 4 (borderline accept), 5 (accept), 2 (reject). The AC named *empirical
  robustness and significance* as the decision-critical issue.
- **Goal of this campaign**: replace every disputed number with one produced under a
  single documented protocol, with seeds and significance tests, and correct two
  analyses that were wrong as printed.
- **Non-goal**: making CRAFT look good. Several of these experiments can come out
  against the paper. §7 pre-registers what we conclude in each case, so the outcome
  is decided by the data and not by whoever reads it first.

---

## 0. Read this first: the released code could not run CRAFT at all

At commit `9972e5f` (the state before this work), `--use_rag` crashed in four
independent ways. All are fixed, but it means **no number in the submission can be
reproduced from the released code**, and the re-run is not optional.

| Failure | Where | Effect |
|---|---|---|
| `from tkinter import NO` | `utils/tools.py:2` | `ImportError` on any headless machine — the whole repo failed to import |
| Policy head sized from `d_model*2` but fed `2*c_out` channels | `models/rag_plugin.py` | `RuntimeError` unless `d_model == c_out`; crashed for every real config |
| `training` declared as a read-only property | `models/rag_plugin.py` | `nn.Module.train()` raised `AttributeError: property has no setter` |
| Reward `[Ns,B,P,C]` multiplied by log-prob `[Ns,B,P]` | `models/rag_plugin.py` | `RuntimeError` on broadcast |
| `.forecast()` called with 4 args | `models/rag_plugin.py` | `TypeError` on DLinear/RLinear, whose `forecast(x_enc)` takes one |
| Optimizer built from `unwrap_model(...)` | `exp/exp_long_term_forecasting.py` | Policy head got gradients but was **never stepped** — the corrector stayed at initialization |
| `args.gemma_1` | `run.py` (inference branch) | `AttributeError`; root `*.sh` also passed `--gemma_*` vs the declared `--gamma_*` |
| `fix_seed = 2021` hardcoded | `run.py` | `--seed` only fed augmentation ⇒ multi-seed results were impossible from the CLI |

Three things the paper describes were also **absent** from the code and are now
implemented. Treat them as new, not as reproductions:

1. **Retrieval safeguards** (§3.1: self-exclusion + temporal exclusion radius). There
   was no exclusion of any kind, and the bank was built from a *shuffled* loader so
   the information needed to exclude did not exist. During training the query's own
   window was therefore retrievable at distance 0 and received the largest softmax
   weight.
2. **L2 action penalty** `λ‖a'‖₂` (Eq. 12), described as load-bearing for
   Requirement 2. Not present in the loss. Now `--lambda_reg`, default `0.0`.
3. **REINFORCE sampling**. The code used `rsample()` (reparameterised), which is not
   the score-function estimator the paper specifies. Default is now `sample()`;
   `--rl_sampling rsample` reproduces the old behaviour.

Also note `PolicyHead` is a kernel-3 **temporal CNN**, not the "lightweight MLP" of
§3.2. The code is kept as-is (locality is a real design choice that pairs with the
pooling step) and the paper text should be corrected instead.

---

## 1. What was added

### Core (changed behaviour — read before trusting output)
| File | Change |
|---|---|
| `models/rag_plugin.py` | Rewritten. Correct channel arithmetic, timestep-level discrete reward matching log-prob shape, `--detach_yhat`, `--lambda_reg`, `--rl_sampling`, retrieval mode/exclusion plumbing, `extract_base_state_dict()`. Eval mode is a strict pass-through (asserted in tests). |
| `models/Memory.py` | Rewritten. Bank built from the dataset **in order** so bank position == dataset index; `retrieve(..., query_idx, exclusion_radius, mode)` with `nn`/`random`; drops the unused `search_and_reconstruct` (large memory saving); refuses to return an excluded neighbour rather than silently falling back. |
| `exp/exp_long_term_forecasting.py` | Optimizer covers **all** parameters; indexed train loader when exclusion is on; saves `base_model.pth`, `per_sample_mse.npy`, `per_sample_mae.npy`; writes a run record; `--loss` selector. Validation and test evaluate the backbone alone. |
| `run.py` | `--seed` now seeds training; ~15 new CRAFT flags; `build_setting()` includes lr/bs/epochs/loss/seed/variant so two conditions can no longer overwrite each other's `results/` directory (they previously could, and did); `detect_anomaly` off by default (it was globally on, a large slowdown). |
| `utils/tools.py` | tkinter import removed; `discrete_reward` / `continuous_reward` with `level ∈ {step,item}`. |

### New
| File | Purpose |
|---|---|
| `data_provider/indexed.py` | Dataset wrapper yielding the sample index, for temporal exclusion. |
| `utils/losses_freq.py` | FreDF, Focal Frequency Loss, BandWeightedMSE, Huber — so the frequency-baseline comparison can run on every dataset. |
| `utils/run_logger.py` | One JSON per run with the full resolved config. This is the fix for the protocol ambiguity: every reported number is traceable to an exact (epochs, lr, batch, seed, variant) tuple. |
| `experiments/freq_band_analysis.py` | **Corrected** band-wise spectral analysis (§2, B1). |
| `experiments/aggregate_results.py` | Seeded means, paired *t*-test, Wilcoxon, Cohen's *dz*, bootstrap CI. Refuses to pair across protocols. |
| `experiments/eval_extracted_base.py` | Evaluates the deployed backbone alone from a saved run record. |
| `experiments/per_example_analysis.py` | Difficulty quartiles + chronological-tail drift split, paired per window. |
| `tests/test_craft.py` | 43 CPU tests, no data or GPU needed. |
| `scripts/craft_rebuttal/*.sh` | Launchers (§3). |

---

## 2. The one analysis error you must understand

The submitted tables titled **"Prediction Error Energy"** and **"Error Energy
Reduction (%)"** did not report error energy. `freq_analysis_all.py:421-449` computes
the PSD **of the prediction itself**, and the percentage column is
`(E_craft − E_base)/E_base`. So:

- A band where the prediction *gained* energy printed as a large positive
  "reduction" — e.g. ECL/iTransformer 0.1–0.2 Hz went `12.06 → 39.04` and was
  reported as `+223.8%` improvement. That is why cells exceed an impossible 100%.
- The sign convention was also **inconsistent within a single row**: the FRMSE
  promotion column is better-is-positive, the band columns are not.
- The main-text band table and the appendix band table disagree on the same
  ECL/iTransformer cell (0.1–0.2: `12.3` vs `223.8`).

`experiments/freq_band_analysis.py` replaces this with three separated, explicitly
signed quantities per band:

| Column | Meaning | Sign |
|---|---|---|
| `res_reduction_pct` | reduction in **residual** energy, `FFT(pred − true)` | positive = CRAFT better, cannot exceed 100 |
| `gap_reduction_pct` | movement of band energy **towards** the ground-truth band energy (reported with `signal_energy_gt` and an overshoot flag) | positive = closer to GT |
| `frmse_reduction_pct` | the paper's §5.3 FRMSE, definition unchanged | positive = CRAFT better |

The paper's third core claim — gains concentrated in 0.2–0.5 — must be re-derived
from `res_reduction_pct`. The script prints the low-band vs mid-high-band means so
the answer is unambiguous. **It may flip.** See §7.

---

## 3. How to run

### Setup
```bash
git pull
uv sync --extra dev            # or use your own env and export PY="python"
# datasets under ./dataset/ : ETT-small/ weather/ electricity/ traffic/ exchange_rate/
```
FAISS: `pyproject.toml` pins `faiss-cpu`. The bank runs on GPU only with a GPU FAISS
build; without one, pass `--no_faiss_gpu` (the launchers do not, so install
`faiss-gpu` on gpu6 or add the flag in `common.sh`). For ECL/Traffic the launchers
already pass `--memory_store_cpu`.

### B0 — correctness gate (~10 min). Always first.
```bash
bash scripts/craft_rebuttal/b0_smoke.sh
```
Runs the unit tests, one base run, one CRAFT run, the deployment eval and all three
analysis scripts. **If the unit tests fail, stop and report — do not run the
campaign.** Then delete the smoke artifacts as the script instructs.

### B2 — main multi-seed results (the decision-critical one)
```bash
SCOPE=pilot bash scripts/craft_rebuttal/b2_main_multiseed.sh   # 2 datasets, pl=96, ~20 runs
SCOPE=full  bash scripts/craft_rebuttal/b2_main_multiseed.sh   # 320 runs
DRY_RUN=1 SCOPE=full bash scripts/craft_rebuttal/b2_main_multiseed.sh   # print commands only
```
Knobs: `SEEDS` (default `2021 1 2 3 4`), `PRED_LENS`, `MODEL` (default
`iTransformer`), `EPOCHS` (default 10 — **one budget for the whole campaign**), `GPU`.
Run the pilot first and check `analysis/summary.md` before committing to `full`.

### B1 — corrected frequency analysis (post-hoc, minutes)
```bash
bash scripts/craft_rebuttal/b1_freq_analysis.sh     # needs B2's results/
```

### B3 — deployment claim (post-hoc + one eval pass per CRAFT run)
```bash
bash scripts/craft_rebuttal/b3_deployment.sh        # needs B2's checkpoints/
```
Cross-checks the extracted-backbone metric against the in-training test metric; they
must agree to ~1e-3 relative. A `MISMATCH` line means extraction or eval is broken
and nothing downstream is reportable.

### B4 — γ₂ sweep and component ablations
```bash
WHICH=gamma     bash scripts/craft_rebuttal/b4_gamma_and_ablations.sh
WHICH=retrieval bash scripts/craft_rebuttal/b4_gamma_and_ablations.sh
WHICH=mechanism bash scripts/craft_rebuttal/b4_gamma_and_ablations.sh
```
Covers γ₂ ∈ {0,0.1,0.25,0.5,0.75,1,2,5}, NN vs random retrieval, exclusion radius
∈ {0, P/2, P, 2P}, k ∈ {1,3,5,10,20}, N_s ∈ {2,4,8,16}, detach, discrete vs
continuous reward, λ ∈ {0.001,0.01,0.1}. 3 seeds each.

### B5 — modern backbones and frequency-aware baselines
```bash
PART=backbones bash scripts/craft_rebuttal/b5_baselines.sh
PART=losses    bash scripts/craft_rebuttal/b5_baselines.sh
```

### Cost guide (single A100, 10 epochs)
ETT ≈ minutes/run · Weather ≈ 10–20 min · Exchange ≈ minutes · ECL ≈ 1–2 h ·
Traffic ≈ 2–4 h. CRAFT adds roughly 1.3–2.3× at `N_s=8`.
**B2 full ≈ 2–4 GPU-days.** Order of value if compute is short: B0 → B2 pilot →
B1 → B3 → B2 full → B4 → B5.

---

## 4. Deliverables (交付物)

Commit `analysis/` and `runs/` (they are small; `results/*.npy` are not — keep those
on disk). For each item state the protocol inline: seeds, epochs, backbone.

| # | File | Becomes | Must contain |
|---|---|---|---|
| D1 | `analysis/summary.md`, `analysis/paired_tests.csv` | replacement for the main results table + a new significance table | per-cell mean±std over ≥5 seeds, paired *t* and Wilcoxon *p*, Cohen's *dz*, bootstrap CI on the aggregate MSE/MAE change |
| D2 | `analysis/freq_bands.csv`, `analysis/freq_bands.tex` | replacement for **both** band-energy tables and the main-text band table | per band: residual energy base/CRAFT + reduction %, GT band energy, gap-to-GT, overshoot flag, FRMSE. One stated sign convention. |
| D3 | `analysis/per_example.csv` | replacement for the per-example/deployment table | `all` + Q1–Q4 + `first_80pct`/`last_20pct_shift` rows, per-window Wilcoxon, for every dataset |
| D4 | `analysis/abl_*/summary.md` | γ₂ curve, retrieval control, exclusion sweep, k/N_s, detach, reward type, λ | 3-seed mean±std and Δ% vs the shared base runs |
| D5 | `analysis/b5*/summary.md` | modern-backbone heatmap + frequency-loss table on **all** datasets | a matched CRAFT row in the frequency-loss table (the submission's caption promised one and the table had none) |
| D6 | **Protocol table** (write by hand from `runs/*.json`) | new appendix table | one row per reported table: epochs, lr grid, batch, seeds, early stopping, backbone. This is the direct answer to "what distinguishes the full protocol from the shortened one?" |
| D7 | **Changelog** (extend §0 of this file) | appendix subsection + code release note | the implementation corrections, stated plainly, including that safeguards/L2 were absent before |

Every number in the paper must be traceable to a `runs/*.json`. If a number has no
record, it does not go in.

---

## 5. Rules that keep this defensible

1. **One protocol.** `EPOCHS` is fixed campaign-wide. If you change it, re-run both
   arms of every affected comparison. Never compare across budgets — the aggregator
   refuses to, and that refusal is a feature.
2. **Base and CRAFT differ only by the corrector.** Same lr, batch, architecture,
   epochs, seed. `common.sh` enforces this; don't tune one arm.
3. **Never delete a run because it looks wrong.** Failed and unflattering runs belong
   in the register. `logs/` keeps stdout for every run.
4. **Report the deployed artifact.** All headline numbers come from the backbone
   alone. `adjusted_outputs` is never a reportable metric.
5. **Exclusion on by default.** The canonical variant is `craft` = NN retrieval +
   `exclusion_radius = pred_len`. `craft_noexcl` is an ablation, not the method. If
   the gain exists only without exclusion, that is a finding to report, not to hide.

---

## 6. Known gaps in this harness

- `--itr` still loops inside one process with one seed; use `SEEDS` instead.
- The exclusion radius is measured in dataset-index units, which equals timesteps for
  the ETT/custom loaders (window *i* starts at timestep *i*). It is not meaningful for
  M4 or the classification/anomaly loaders.
- Only `long_term_forecast` is CRAFT-aware; the other `Exp_*` classes ignore `--use_rag`.
- `random` retrieval draws with replacement and can return the same entry twice.
- Multi-GPU (`--use_multi_gpu`) is untested with the plugin; the bank lives on one device.
- Hessian / loss-surface / gradient-covariance probes from the appendix are **not**
  ported. If those tables stay in the paper they need the same protocol treatment.

---

## 7. Pre-registered decision rules

Fix these before seeing results.

**D1 — aggregate gain.** Use the per-seed pooled CI when the campaign covers only a
handful of cells; the cell-mean CI is degenerate at one cell. If the 95% CI on the mean MSE change excludes 0
and the majority of cells have *p* < 0.05 in CRAFT's favour, the headline claim
stands with the CI reported alongside. If the CI includes 0, the abstract must drop
"7.23%/5.89%" and state that CRAFT is competitive with, and complementary to, tuned
MSE without a significant aggregate advantage. Do not add seeds until the CI excludes
0 — report the CI at the pre-declared seed count.

**D2 — frequency selectivity.** The claim survives only if mean `res_reduction_pct`
over 0.2–0.5 exceeds that over 0.0–0.2. If it does not, replace the claim with what
the corrected numbers show, and say explicitly that the submitted table's sign
convention was wrong.

**D3 — deployment.** If the extracted backbone is significantly worse on a majority
of datasets, the "improves the base model at zero inference cost" claim must be
narrowed to the datasets where it holds, and the abstract reworded. This is the claim
most likely to fail; it failing is survivable, overselling it is not.

**D4 — retrieval.** If `craft_random` matches `craft` within CI, retrieval is a
stabilising reference rather than semantic guidance: reframe the mechanism, weaken the
δ-informativeness premise to "δ→0 predicts the observed insensitivity", and consider
whether "Retrieval-Augmented" belongs in the title.

**D4b — exclusion.** If the gain disappears once `exclusion_radius ≥ pred_len`, the
previously reported gains were partly leakage. Report that outcome directly.

When finished, write `analysis/REPORT.md`: every trial (including failures), the
observed numbers against these rules, and which paper claims change.
