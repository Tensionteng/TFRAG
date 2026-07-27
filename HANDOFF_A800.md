# Handoff — CRAFT (NeurIPS 2026 submission 18915) — continuing on 8×A800

You are taking over an experimental campaign. Read this file completely before running
anything. It records what was tried, what is ruled out, what is still open, and — most
importantly — the infrastructure invariants that, when violated, silently produce
**wrong numbers that look right**. Three separate false results have already been
produced and caught in this campaign. Each one is described below so you do not
reproduce it.

---

## 1. The mission, stated honestly

**Goal:** find a configuration of the paper's method (retrieval + RL corrector, applied
at training time only) that beats strong baselines with statistical significance.

**Current status:** across **731 run records / ~34 distinct configurations**, no
configuration of the method beats its own paired baseline at p<0.05. The best is
`craft_distill5_g20.1_ns16_detach` on ETTh1: **+0.85%, p=0.303**. Meanwhile a one-line
loss-function change (MAE, FreDF) achieves +1.9% to +8.1% at p<0.01 on the same cells.
At its best-tuned setting on the two 2026 backbones the corrector's mean effect over 8
paired cells is **+0.01%** (§6.6) — zero, not a weak positive.

You are not being asked to confirm this. You are being asked to make a genuine attempt
at the remaining untested directions (§7) with far more compute than was available.
But you must know what has already been excluded, or you will spend 8×A800 rediscovering
it.

**The constraint that makes this hard, and that any successful trick must break:**
the RL reward is computed from `batch_y` — the same ground-truth tensor the primary MSE
loss already consumes. A corrector conditioned only on that signal carries no
information the loss does not already carry, so there is nothing for the backbone to
internalise. This is why the null control (`--detach_yhat`, which provably reduces the
objective to the plain baseline) performs *identically* to full CRAFT. **Any direction
that does not introduce information or an inductive bias absent from the MSE objective
is predicted to fail, and every such direction tested so far has failed.** Judge new
ideas against this criterion first; it is cheaper than a GPU-week.

---

## 2. Repo, sync, environment

### Use git, not rsync

Code moves by git. The agent on the A800 box should `git pull` / `git commit` / `git push`
so that work is reviewable and reversible.

> **This is not a style preference.** In the previous campaign, code was pushed to one
> host by `rsync` and that host's `scripts/craft_rebuttal/common.sh` silently stayed one
> revision behind. It set MixLinear's `period_len` to 24 instead of 4 at lookback 96,
> which clamps the model's `lpf` from 19 down to 4 and guts its frequency branch. 31 runs
> of a crippled baseline were produced and had to be quarantined. `git status` would have
> shown it instantly; `rsync` cannot.

```bash
git clone <repo-url> TFRAG && cd TFRAG
git log --oneline -8          # confirm you have the arch-key fix and later
```

Datasets are **not** in git. Transfer `dataset/` once (scp/rsync), or re-download from
the links in `README.md`. Expected layout:

```
dataset/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
dataset/weather/weather.csv
dataset/electricity/electricity.csv
dataset/traffic/traffic.csv
dataset/exchange_rate/exchange_rate.csv
```

Run records (`runs/*.json`) from the previous campaign are the accumulated evidence base.
Copy them over — they let you skip work via `--skip_if_done` and they are the input to
every analysis script. Transfer as a tarball; ~700 small files over scp is slow.

### Environment

`uv` is the package manager. `requirements.txt` is stale upstream pinning — trust
`pyproject.toml` / `uv.lock`.

```bash
uv sync
uv run python -c "import faiss, torch; print(faiss.__version__, torch.__version__, torch.cuda.is_available())"
```

- **Python is pinned to 3.13** in `.python-version`. Do not raise it. On 3.14 the
  DataLoader default start method changes from `fork` to `forkserver`, which pickles the
  dataset; that exposed an `IndexedDataset.__getattr__` recursion (fixed, but 3.13 is
  the tested configuration) and there was no `faiss-gpu` wheel for 3.14.
- **faiss**: `Memory.py` uses `faiss.StandardGpuResources` / `GpuIndexFlatL2` when a GPU
  is visible. `faiss-gpu-cu12` is preferred; `faiss-cpu` works via an automatic fallback
  (`_add_with_cpu_fallback`) but is slower to build the bank. On 80 GB A800s the GPU
  index will fit comfortably for every dataset including Traffic (~3.8 GB).
- `run.py` leaves `torch.autograd.set_detect_anomaly(True)` on globally. It is a real
  slowdown. Turn it off for throughput runs; turn it back on when debugging the RL loss.

---

## 3. What the paper claims and why this matters

**CRAFT** (a.k.a. SRRF in the code) is a *training-time-only* plug-in: FAISS retrieval of
similar training windows → a Gaussian policy head proposes an additive correction to the
backbone's prediction → REINFORCE with a discrete reward ("did the correction reduce
MSE/MAE at this timestep?"). All auxiliary modules are discarded after training, so
inference is the bare backbone.

The submitted abstract claims **7.23% MSE / 5.89% MAE** average reduction across 8
backbones and 5 datasets. Reviewer scores were 4 / 5 / **2**, with the 2 (reviewer ErQJ)
having verified individual table cells and found six substantive problems.

Anything you produce must survive that reviewer. That is the bar.

---

## 4. Code map

Upstream Time-Series-Library except:

| File | Role |
|---|---|
| `models/rag_plugin.py` | `RAGPlugin` wrapper + `PolicyHead`. The method. |
| `models/Memory.py` | `MemoryBankWithRetrieval`: FAISS bank, `nn`/`random` retrieval, exclusion radius |
| `models/model_factory.py` | `create_model` (dynamic import + optional wrap), `unwrap_model` |
| `exp/exp_long_term_forecasting.py` | the only SRRF-aware experiment class |
| `run.py` | ~100 args; `build_setting()` builds the experiment identity string |
| `utils/run_logger.py` | per-run JSON record, `variant_name()`, `ARCH_KEYS_BY_MODEL` |
| `utils/losses_freq.py` | `build_criterion`: mse / mae / huber / fredf / ffl / bandmse |
| `models/FACT.py`, `models/MixLinear.py` | ICLR-2026 baselines added for reviewer ErQJ's W2 |
| `experiments/aggregate_results.py` | paired t / Wilcoxon / Cohen's dz / bootstrap CI |
| `experiments/compare_variants.py` | one table, every variant vs base |
| `experiments/backfill_arch_keys.py` | repairs records written before arch knobs were logged |
| `scripts/craft_rebuttal/*.sh` | the campaign; `common.sh` holds all reference configs |
| `tests/test_craft.py` | 62 CPU tests. **Run these after any change.** |

`uv run --extra dev pytest tests/test_craft.py -q` — must stay green.

### Key flags added for this campaign

`--gamma_1/2/3` (base / RL / distillation weights), `--freeze_policy`, `--detach_yhat`,
`--retrieval_mode {nn,random}`, `--exclusion_radius`, `--lambda_reg`, `--reward_type`,
`--reward_level`, `--num_rl_samples`, `--distill_target {best,advantage}`, `--seed`,
`--skip_if_done`, `--slim_ckpt`, `--no_save_arrays`, `--loss`.

---

## 5. Bugs already found and fixed — do not reintroduce

Six crashes and four silent-corruption bugs. The crashes are fixed and covered by tests.
The silent ones are the dangerous class; each produced a plausible wrong number.

### 5.1 Crashes (fixed)

1. `utils/tools.py` had a stray `from tkinter import NO` — broke every import headless.
2. `PolicyHead` sized its `Conv1d` from `d_model*2` but `forward` concatenates to
   `2*c_out` channels. `--use_rag` only ran when `d_model == c_out`.
3. A read-only `training` property shadowed `nn.Module.training`.
4. Reward/log-prob shapes disagreed; now both `[Ns, B, P]` with an explicit check.
5. `_run_base` called `.forecast(...)`, which DLinear/RLinear define with a different arity.
6. `Exp_Basic._acquire_device` hardcoded `CUDA_VISIBLE_DEVICES="0,1"`, killing 8 of the
   first 9 runs on a 4-GPU host.

### 5.2 Silent corruption (the important ones)

**(a) The policy head was never optimized.** `_select_optimizer` called
`unwrap_model(self.model)`, a helper that strips *both* DataParallel *and* the RAGPlugin,
before collecting parameters. The optimizer received only backbone weights. Gradients
still reached the policy head (so inspecting `.grad` looked fine) and the logged
`rl_loss` still moved between epochs (because the backbone changed, so the policy's
*input* changed) — but `optimizer.step()` never touched it. **Every number in the
submitted paper was produced by a corrector frozen at random initialization.**
Introduced in commit `9972e5f`; the original `89b5032` was correct.
Verify with: 8 policy-head tensors, 0 in the old optimizer, 8 in the fixed one.

**(b) Learning rates were guessed, not read from the reference scripts.** Weather ran at
5e-4 (reference: 1e-4) and Traffic at 5e-4 (reference: 1e-3 with `e_layers 4`). A
too-high LR destabilises the MSE baseline, which flatters any bounded-gradient
objective: MAE appeared to beat MSE by **20.5% (p=0.002)** on Weather. The correct
figure is **+3.8%**. `lr_for()` / `bs_for()` / `model_args()` in `common.sh` now carry a
comment naming the source script for each value. **Do not change them without checking
the reference script.**

**(c) Cell identity omitted the protocol.** The analysis keyed cells on
`(dataset, model, pred_len)`, so the mistuned Weather runs and their corrected reruns
landed in the same cell and one silently overwrote the other. Fixed in `cbf8520`:
`protocol_of(r)` now contributes to the key.

**(d) Cell identity omitted model-specific architecture knobs.** `period_len`, `lpf`,
`alpha` (MixLinear), `core`, `num_kernels`, `use_norm` (FACT), `seg_len`, `top_k`,
`patch_len` appeared in *none* of `build_setting`, the run record, or `PROTOCOL_KEYS`.
Two MixLinear configs whose frequency branch saw 4 vs 24 FFT bins produced the **same
setting string**, so `--skip_if_done` skipped the rerun and the records overwrote each
other. Fixed: `ARCH_KEYS_BY_MODEL` in `utils/run_logger.py`, keyed by model so a
MixLinear-only flag cannot split an iTransformer cell.

> **The invariant behind (b)(c)(d):** *every quantity that changes the result must appear
> in the run identity and in the cell key.* If you add a flag that affects training, add
> it to `build_setting`, `_CONFIG_KEYS`, and (if model-specific) `ARCH_KEYS_BY_MODEL` —
> and add a test to `tests/test_craft.py` asserting that two runs differing in it are
> never paired. There are already three such tests; copy one.

### 5.3 Data-loss traps

- `rsync --delete-excluded` once deleted the staged datasets. Never use it against a host
  that holds `dataset/`.
- Repeated `pkill -f <pattern>` from an interactive shell matches the shell itself and
  self-kills (exit 255). Write the kill into a script file and `setsid nohup` it.
- Checkpoints fill disks fast. Use `SLIM=1` (`--no_save_arrays --slim_ckpt`) for anything
  that does not need `pred.npy`. The frequency analysis is the only consumer of those arrays.

---

## 6. Results so far — what is ruled out

716 records. Excluding the mistuned-LR runs (Weather @5e-4, Traffic @5e-4), 653 usable.
All numbers below are **paired by seed** at Input-96 / Predict-96 unless stated.

### 6.1 CRAFT vs its own baseline

| | value |
|---|---|
| aggregate ΔMSE over 40 cells | **−13.32%**, 95% bootstrap CI [−26.78, −4.33] |
| pooled over 99 per-seed deltas | −12.07%, CI [−20.21, −5.69] |
| cells improved | 9 / 40 |
| cells with p<0.05 | 7, of which **1** favours CRAFT |

Per backbone (mean Δ%, default γ₂=0.5): TimesNet +0.1, DLinear −0.3, MixLinear −2.7,
SegRNN −4.9, iTransformer −5.9, FACT −7.4, TSMixer −8.2, PatchTST −16.2, FreTS −75.7.
**The "CRAFT helps weak backbones" hypothesis fails in the wrong direction** — the
weakest backbones are hurt most.

### 6.2 The null control — the most important single result

| variant | dataset | n | base | treat | Δ% | p |
|---|---|---|---|---|---|---|
| `craft_detach` | ETTh1 | 10 | 0.3927 | 0.3917 | +0.24 | 0.338 |
| `craft_frozen` | ETTh1 | 10 | 0.3927 | 0.3916 | +0.26 | 0.256 |

`--detach_yhat` blocks the RL gradient path to θ, reducing the objective to the plain
baseline. `--freeze_policy` reproduces the submitted code path (§5.2a). Both are
statistically indistinguishable from the baseline **and from full CRAFT**. The corrector
contributes nothing.

### 6.3 γ₂ is flat — there is no operating point

ETTh1, iTransformer, frozen policy, 5–10 seeds:

| γ₂ | 0.01 | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|---|---|
| Δ% | +0.18 | +0.18 | +0.19 | +0.36 | +0.34 | +0.26 |
| p | 0.43 | 0.42 | 0.40 | 0.26 | 0.24 | 0.26 |

An earlier 3-seed run showed γ₂=0.1 at +0.75% (p=0.097). It did not survive more seeds.
**Treat any 3-seed positive as unconfirmed until it replicates at n≥5.**

### 6.4 γ₃ distillation — best CRAFT arm, still not significant

`craft_distill5_g20.1_ns16_detach`, ETTh1, n=5: 0.3937 → **0.3904**, +0.85%, **p=0.303**,
4/5 seeds. Effect (0.0033) is smaller than the std (0.0063). γ₃ ∈ {0.5, 1, 2} are all
≈0. This is the only arm worth more seeds (§7).

### 6.5 Lookback does not rescue it (iTransformer, ETTh1, 10 epochs)

| L | 96 | 336 | 512 | 720 |
|---|---|---|---|---|
| base MSE | **0.3925** | 0.4081 | 0.4174 | 0.4081 |

Longer lookback is *worse* at this budget. At L=720, CRAFT beats base (+0.84%, p=0.461)
but both are worse than L=96 base. If you want long-lookback numbers to be competitive
you must also raise the epoch budget — untested.

### 6.6 Modern backbones (b11, complete): γ₂ was genuinely mistuned, and tuning reaches exactly neutral

200 runs, FACT and MixLinear on the four ETT datasets, **unified lookback 96**, 5 paired
seeds per cell, base and CRAFT differing only in the corrector.

Aggregated over cells:

| arm | cells | mean Δ% | positive | significantly positive |
|---|---|---|---|---|
| `craft` (default γ₂=0.5) | 12 | **−7.45** | 1/12 | 0 |
| `craft_g20.1` (live policy) | 8 | −1.33 | 1/8 | 0 |
| `craft_g20.05_frozen` | 8 | −0.39 | 1/8 | 0 |
| **`craft_g20.1_frozen`** | 8 | **+0.01** | 3/8 | 0 |

Per cell, the two extremes:

| backbone | dataset | γ₂=0.5 (default) | γ₂=0.1 frozen |
|---|---|---|---|
| FACT | ETTh1 | **−28.14%** (p=.062) | +0.22% (p=.838) |
| FACT | ETTh2 | **−8.27%** (p=.027) | −0.16% |
| FACT | ETTm1 | −18.70% | −0.41% |
| FACT | ETTm2 | −23.30% (p=.069) | +0.15% |
| MixLinear | ETTh1 | −0.45% | **+0.74%** (p=.263) |
| MixLinear | ETTh2 | −0.80% | −0.14% |
| MixLinear | ETTm1 | **−8.29%** (p=.018) | −0.18% |
| MixLinear | ETTm2 | −0.18% | −0.14% |

Two conclusions, both worth stating in the paper:

1. The catastrophic numbers are an **untuned-strength artifact**, not a property of the
   backbones. The defensible claim is "harmful at the default strength, neutral when
   tuned" — which is both more accurate and less damaging than "destroys modern backbones".
2. At its best tuning the corrector's mean effect is **+0.01% over 8 cells**. This is not
   a weak positive; it is zero to three significant figures, and it is the cleanest
   statement of the §1 argument in the whole dataset.

**Trap noted for MixLinear.** Forcing `period_len 4` at lookback 96 (to keep the authors'
`lpf=19` from being clamped) makes their model *worse* on ETTh1: 0.4004 → 0.4589. ETTh1
takes the `lpf=1` branch, so `period_len` only alters the time-domain reshaping there, and
24 suits it better. Report MixLinear per-dataset at its best config rather than imposing
one; its own operating point is lookback 720 (ETTh1 0.3640, ETTh2 0.2834, Weather 0.1729).

### 6.7 What DOES beat the baseline (these are existing published methods, not ours)

iTransformer, 3 paired seeds, per-dataset reference configs:

| loss | ETTh1 | ETTh2 | ETTm1 | ETTm2 | ECL | Exch | Traffic | Weather | mean |
|---|---|---|---|---|---|---|---|---|---|
| **MAE** | +1.89✓ | +2.83 | +8.11✓ | +4.55 | −0.48 | +0.69 | −1.67 | +3.77 | +2.46 |
| **FreDF** | +4.46✓ | +2.81✓ | +5.69✓ | +4.77 | +0.40 | −3.28 | −0.73 | +3.11 | +2.15 |
| **Huber** | +1.04 | +1.87✓ | +2.44✓ | +3.12 | +0.59✓ | +1.54✓ | +0.06✓ | +1.78 | +1.55 |
| FFL | −8.30✓ | −10.31 | −4.58✓ | −3.22 | −5.23✓ | −2.96✓ | −1.63✓ | — | −5.13 |
| BandMSE | −17.19✓ | −0.40 | −8.64✓ | −3.10 | −18.48✓ | −3.62 | −22.81✓ | — | −10.56 |

(✓ = p<0.05; bootstrap CIs: MAE [+0.55,+4.55], FreDF [+0.08,+4.04], Huber [+0.92,+2.20].)

Two things follow. (i) Huber wins on **8/8** datasets with a CI strictly above zero — it
is the strongest simple intervention. (ii) FFL and BandMSE, which are *also* explicitly
frequency-aware, lose on every dataset — so the result is not "any change to MSE helps",
it is specifically the objectives that fix **gradient attenuation**. That is exactly what
the paper's Proposition 1 predicts, and it is the paper's theory surviving.

### 6.8 Best absolute MSE seen, Input-96 / Predict-96

| dataset | best | config |
|---|---|---|
| ETTh1 | **0.3775** | iTransformer + FreDF |
| ETTh2 | **0.2940** | iTransformer + MAE / FreDF |
| ETTm1 | **0.3213** | iTransformer + MAE |
| ETTm2 | **0.1757** | FACT base |
| ECL | **0.1352** | FACT base |
| Exchange | **0.0886** | iTransformer + Huber |
| Traffic | **0.3922** | iTransformer + CRAFT (n=4, +0.20%, p=0.529) |
| Weather | **0.1577** | FACT base |

At lookback 720, MixLinear base reaches ETTh1 0.3640 / ETTh2 0.2834 / Weather 0.1729.

---

## 7. Open directions, ranked

Judge each against the criterion in §1: *does it introduce information or an inductive
bias that the MSE objective does not already have?* The first three do; the rest are
listed because they are cheap, not because they are likely.

### 7.1 CRAFT stacked on FreDF / Huber — **untested, highest priority**

`rag_plugin.py` hardcoded `base_loss = F.mse_loss(y_hat, batch_y)`, ignoring `--loss`
entirely. The plug-in has therefore **never** been combined with the objectives that
actually work. Fix that line to use `build_criterion(args.loss)` (already prepared), then
test `--use_rag --loss fredf` against `--loss fredf` alone.

Why it might work: the corrector's failure mode is that its reward duplicates MSE's
information. Against a *frequency-domain* base loss the corrector is optimising a
different criterion than the loss, so the composition is not obviously redundant.

Why it might not: the reward is still computed from `batch_y` by
`discrete_reward`/`continuous_reward`, which are MSE/MAE indicators. Consider pairing
this with 7.2.

### 7.2 Frequency-domain reward — **untested**

Reward the correction only when it reduces **high-band residual energy**, i.e.
`|FFT(pred − true)|` above a cutoff, rather than pointwise MSE/MAE improvement. Add as
`--reward_type freq` in `utils/tools.py` alongside `discrete_reward`.

This is the reward Proposition 1 actually implies, and it is the only reward under which
γ₃ distillation transfers something the MSE loss is not already transferring. It is the
most theoretically coherent untested idea and it keeps the paper's main line intact.

### 7.3 Retrieval-manifold mixup — **untested, best odds of a real gain**

Interpolate a training window with its own nearest neighbour:
`x' = λx + (1−λ)x_n`, `y' = λy + (1−λ)y_n`, `λ ~ Beta(α,α)`.

Unlike random mixup this stays on the data manifold, and unlike the corrector it is a
**genuine extra information channel** — the neighbour's future is a second observation of
a similar state, which a single-target MSE loss never sees. Mixup is a reliable
generalisation improver, and this is the one direction that clearly satisfies the §1
criterion.

Implementation note: `MemoryBankWithRetrieval` currently stores only `y_store` (futures).
You must also store `x` to mix inputs. For ETT this is trivial (8545 × 96 × 7 ≈ 23 MB);
for Traffic/ECL keep it on CPU (`--memory_store_cpu`).

This is arguably "retrieval-augmented training" in a more defensible sense than the
current corrector, and it preserves the paper's title and Proposition 2's role.

### 7.4 More seeds on the one arm that leads

`craft_distill5_g20.1_ns16_detach` at n=5 gives +0.85%, p=0.303. With 8×A800, n=20 is
cheap. **But see §8 before you interpret the result** — this arm was selected *because*
it led, so a fresh confirmation on disjoint seeds is mandatory.

### 7.5 Cheap, low-prior

- γ₃ > 5, and `--distill_target advantage` with a temperature sweep.
- `--num_rl_samples` 32/64 (better max for `distill_target=best`).
- Long lookback **with a raised epoch budget** (§6.5 only tested 10 epochs).
- CRAFT on top of the FACT/MixLinear configs at their own lookback 720.

### 7.6 Already excluded — do not re-run

Default γ₂=0.5 on any backbone; the γ₂ sweep 0.01–1.0 at L=96; random vs NN retrieval
(indistinguishable, 3 seeds, exclusion enforced); the weak-backbone hypothesis;
lookback 336/512/720 at 10 epochs; γ₃ ∈ {0.5,1,2} with Ns=16; FFL and BandMSE as losses.

---

## 8. Methodology — the part that decides whether a win counts

Your objective is phrased as "beat the baselines with statistical significance". Read this
section as the definition of what that means, because the obvious way to achieve it is
also the way to produce a result that reviewer ErQJ will destroy.

**~30 configurations have already been tested.** At α=0.05 you expect roughly 1.5 false
positives from that many comparisons on pure noise. If you run 200 configurations and
report the one with p<0.05, you have found nothing, and the paper will be checked by a
reviewer who has already caught six real errors in it.

**Required protocol for any claimed win:**

1. **Pre-register.** Before running, write the hypothesis, the exact config, the primary
   metric, and the decision rule into `FINDINGS.md`. Commit it. Then run.
2. **Split the seeds.** Explore on seeds {1,2,3}. Any configuration that looks promising
   must then be **confirmed on a disjoint seed set** (e.g. {11..25}) with the config
   frozen. Only the confirmation run may be reported as evidence.
3. **Correct for multiplicity.** If a sweep of *k* configurations produced the candidate,
   report Holm–Bonferroni or report the sweep size alongside the p-value. Never report a
   swept winner as a single test.
4. **Pair strictly.** Base and treatment must share seed, seq_len, pred_len, lr, batch
   size, epochs, and every architecture knob. `experiments/compare_variants.py` refuses to
   pair across protocols — do not work around it.
5. **A win must generalise.** One cell at p=0.04 is not a result. Require consistency
   across ≥5 of the 8 datasets, or an effect large enough to survive Holm correction.
6. **Report negatives.** They are already an asset: the appendix's honest negative results
   were singled out as a strength by reviewer ErQJ ("I highly appreciate this honesty").

**The comparison that matters is against the strongest baseline, not the weakest.**
Beating vanilla-MSE iTransformer is not enough now — FreDF reaches 0.3775 on ETTh1 and
FACT reaches 0.1577 on Weather. A method that beats MSE but loses to a one-line loss
change will not be accepted.

---

## 9. Running experiments

```bash
# one run
uv run python run.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
  --model_id ETTh1_96_96 --model iTransformer --features M \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 3 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 --n_heads 8 \
  --learning_rate 0.0001 --batch_size 32 --train_epochs 10 --patience 3 \
  --seed 1 --des A800 --itr 1 --skip_if_done \
  --use_rag --gamma_1 1.0 --gamma_2 0.1 --exclusion_radius 96 --freeze_policy

# a campaign, sharded across GPUs, resumable
GPUS="0 1 2 3 4 5 6 7" SLIM=1 bash scripts/craft_rebuttal/parallel.sh \
  "WAVE=1 bash scripts/craft_rebuttal/b10_etth1_sota.sh"

# inspect the queue without running
GPUS="0 1" PLAN_ONLY=1 bash scripts/craft_rebuttal/parallel.sh "<generator cmd>"

# analysis
uv run python experiments/compare_variants.py --runs runs --markdown analysis/variants.md
uv run python experiments/aggregate_results.py --runs runs --out analysis --treatment craft
```

`parallel.sh` generates its queue by running the campaign script with `DRY_RUN=1`, so
there is one source of truth for the configs. Every command carries `--skip_if_done`:
safe to interrupt, safe to relaunch, safe to add GPUs.

Existing campaigns: `b1` frequency analysis, `b2` main multi-seed, `b3` deployment,
`b4` γ ablations, `b5` baselines, `b6` distillation, `b7` reviewer questions, `b8` scope
search, `b9` FACT/MixLinear, `b10` ETTh1 SOTA search, `b11` CRAFT on new baselines.

**On 8×A800 raise `GPUS` to all 8 and consider 2 workers per GPU** for the ETT datasets —
iTransformer over 7 variates does not saturate an A800. Traffic (862 variates) and ECL
(321) should get one worker per GPU.

`SEEDS` is set in `common.sh` (`2021 1 2 3 4`) and a campaign script's own
`SEEDS=${SEEDS:-...}` default will **not** override it, because `common.sh` is sourced
first. Set `SEEDS` in the environment.

---

## 10. Deliverables

1. `FINDINGS.md` — pre-registrations, then outcomes, in chronological order. Negative
   results included.
2. `runs/*.json` — every run. Never delete; `--skip_if_done` depends on them.
3. `analysis/` — regenerated tables. `paired_tests.csv` is the one that goes in the paper.
4. Any claimed win: the pre-registration, the exploration result, and the **disjoint-seed
   confirmation**, with the sweep size stated.
5. Green `pytest tests/test_craft.py` on every commit.

The two rebuttal drafts (`craft/rebuttal_final_en.md`, `craft/rebuttal_final_zh.md`) are
written against the current evidence and will need rewriting if any of §7 succeeds.
`craft/` is a separate LaTeX repo with its own remote — do not push it to this one.

---

## 11. If nothing in §7 works

That is a real possibility and it is not a failure of the campaign. The fallback, already
drafted, keeps the paper's theory (Propositions 1 and 6, the spectral-bias diagnosis, the
frequency analysis) and replaces the corrector with the interventions that do work
(FreDF / Huber / MAE), reporting the corrector as a documented negative result with a
mechanism. §6.7 is the empirical core of that version, and it is stronger evidence for the
paper's own theory than the corrector ever was.

Do not let the goal in §1 turn into a search for a p-value. A smaller true claim survives
review; a larger false one does not.
