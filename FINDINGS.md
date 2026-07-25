# CRAFT: what the re-run actually shows

Empirical record from the corrected implementation. Every number here comes from a
`runs/*.json` record under one protocol: iTransformer, lookback 96, horizon 96,
10 epochs, patience 3, lr 1e-4 (ETTh1) / 5e-4 (Weather), batch 32, temporal
exclusion radius = pred_len. Deployed-backbone metrics only (the corrector is a
strict pass-through at evaluation, asserted by test).

Reproduce with `bash scripts/craft_rebuttal/b6_distill.sh` and read the table with
`python experiments/compare_variants.py --runs runs`.

## 1. Result

ETTh1, horizon 96, paired by seed:

| arm | n seeds | base MSE | arm MSE | Δ% | wins | p (paired t) |
|---|---|---|---|---|---|---|
| `craft` — as submitted (trained policy, RL gradient reaches θ) | 5 | 0.39374 | 0.39631 ±0.0035 | **−0.65** | 1/5 | 0.259 |
| `craft_detach` — RL gradient blocked from θ | 10 | 0.39268 | 0.39174 ±0.0024 | +0.24 | 8/10 | 0.338 |
| `craft_frozen` — policy head frozen at init | 10 | 0.39268 | 0.39164 ±0.0018 | +0.26 | 7/10 | 0.256 |
| `craft_distill0.5_detach` — corrector distilled into θ | 3 | 0.39515 | 0.39303 ±0.0030 | +0.53 | 3/3 | 0.359 |
| `craft_distill1_detach` | 3 | 0.39515 | 0.39392 ±0.0034 | +0.31 | 1/3 | 0.656 |
| `craft_distill2_detach` | 3 | 0.39515 | 0.39381 ±0.0035 | +0.34 | 1/3 | 0.650 |
| `craft_g20.1_frozen` | 3 | 0.39515 | 0.39219 ±0.0024 | +0.75 | 3/3 | 0.097 |
| `craft_g21_frozen` | 3 | 0.39515 | 0.39338 ±0.0025 | +0.45 | 3/3 | 0.194 |

Weather, horizon 96, paired by seed (3 seeds):

| arm | base MSE | arm MSE | Δ% | wins | p |
|---|---|---|---|---|---|
| `craft` — as submitted | 0.20552 | 0.26806 ±0.0207 | **−30.48** | 0/3 | **0.039** |
| `craft_detach` — null control | 0.20552 | 0.20022 ±0.0035 | +2.55 | 3/3 | 0.271 |
| `craft_frozen` | 0.20552 | 0.20102 ±0.0030 | +2.18 | 3/3 | 0.122 |
| `craft_g21_frozen` (γ₂=1) | 0.20552 | 0.20849 ±0.0112 | −1.49 | 2/3 | 0.734 |

**No arm beats the baseline significantly on either dataset. The only significant
result in the whole set is that the submitted configuration is *worse* on Weather
(p = 0.039).**

On both datasets the null control (`craft_detach`, which by construction trains θ with
the baseline's exact objective) scores at least as well as the arm where the RL term
actually acts on θ: +2.55% vs +2.18% on Weather, +0.24% vs +0.26% on ETTh1. A
mechanism that cannot outperform its own placebo is not doing the work.

Note the seed-count effect, which is itself a finding: at 3 seeds `craft_frozen`
looked like +0.74% (3/3 wins, p=0.095); at 10 seeds it is +0.26% (p=0.256). The
submitted paper's mechanism ablations are single-run, and this is how much they move.

## 2. Why it cannot work as formulated

Three independent reasons, in increasing order of how fundamental they are.

### 2.1 The `detach` arm is a null control, and it matches the RL arm

With `--detach_yhat`, `L_RL` has no gradient path to θ (`tests/test_craft.py::
test_detach_blocks_rl_gradient_to_backbone`). With γ₁ = 1.0 the backbone's objective
is then **exactly** the baseline's MSE. The only difference between a
`craft_detach` run and a `base` run at the same seed is the RNG stream consumed by
action sampling, which perturbs dropout masks and shuffling order.

So `craft_detach` measures pure trajectory noise — and `craft_frozen`, in which the
RL term *is* active on θ, produces the same +0.25% as the null control. Whatever the
small positive delta is, it is not the RL mechanism.

### 2.2 The corrector holds no information the base loss lacks

The reward (Eq. 11) is computed from `y_gt`. The base MSE loss already uses `y_gt`
directly and exactly. Retrieved neighbours contribute *other* windows' futures, which
are strictly less informative about this window's target than the target itself.

A training-time module built from `y_gt` therefore cannot add information to θ's
training signal. It can only reshape the optimisation path — which is precisely what
the paper's "zero-order transfer" hypothesis amounts to, and why the appendix found
∥∇_θ L_RL∥ ~ 1e-7, why the detach ablation changed nothing, and why random retrieval
matched nearest-neighbour retrieval. Those three results are not puzzles to be
explained; they are what this argument predicts.

### 2.3 Even the distilled target is a noisier version of the label

`--gamma_3` gives θ the corrector's accepted output `ŷ + a` as an explicit target,
which is the strongest form of transfer available. But `a` is a Gaussian sample with
σ ∈ [0.0067, 0.368] (log σ clamped to [−5, −1]), average-pooled over κ=3 and screened
by best-of-N reward. Best-of-8 random perturbations screened for improvement is a
high-variance finite-difference estimate of the descent direction — strictly noisier
than the exact gradient θ already receives from MSE against the same `y_gt`.

This is borne out: distillation does not beat the `detach` null control (+0.53% at
γ₃=0.5 versus +0.24%, both inside noise), and larger γ₃ makes it worse.

## 3. The noise floor, and what it implies for the submitted numbers

ETTh1 base MSE across 10 seeds: **0.39268, σ ≈ 0.0024 (0.6% of the mean)**.
Weather: the `craft_detach` null control — an objective *identical* to the baseline's,
differing only in RNG stream — scores **+2.55%** with 3/3 wins. That is a direct
measurement of Weather's trajectory noise: roughly **±2.5%**, from nothing but a
different random stream.

The submitted headline gains — ECL 3.4%, Exchange 2.8%, Weather 2.4%, Traffic 1.6% —
sit at or inside this per-dataset seed spread. That is the same conclusion Reviewer
ErQJ reached from Table 13, and it is now measured rather than inferred.

## 4. What was verified along the way

- The submitted implementation could not run `--use_rag` at all (six crashes; see
  `EXPERIMENTS.md` §0). No submitted number is reproducible from the released code.
- The policy head was never optimised, because `_select_optimizer` unwrapped the
  plugin. The reported numbers therefore came from a *frozen random* corrector — which
  is why `--freeze_policy` exists as an explicit arm here.
- Retrieval self-exclusion and the temporal exclusion radius described in §3.1 were
  not implemented; the query's own window was retrievable at distance 0 with the
  largest softmax weight. Now implemented and enabled by default.
- The L2 action penalty of Eq. 12 was absent from the loss. Now `--lambda_reg`.
- The band-energy tables report the PSD of the *prediction*, not of the residual,
  under an "Error Energy Reduction" header, which is why entries exceed 100%.

## 5. Honest options

Ordered by how much of the current paper each preserves.

1. **Reframe as an analysis/negative-result contribution.** §2 is a clean argument for
   why training-time corrector methods cannot mitigate spectral bias, supported by the
   null-control design of §2.1 and the measured noise floor of §3. This is honest and
   genuinely useful to the field, but it is a different paper and a weaker venue.
2. **Pivot to frequency-domain objectives.** The submission's own Table 8 shows FreDF
   improving Weather by 4.5% MSE — a real, reproducible gain, unlike CRAFT's.
   `utils/losses_freq.py` implements FreDF, FFL and BandWeightedMSE, and
   `scripts/craft_rebuttal/b5_baselines.sh` runs them on every dataset. The existing
   spectral-bias theory (Propositions 1, 6) transfers directly to this framing.
3. **Move retrieval to inference.** Conditioning predictions on retrieved exemplars at
   test time adds information and can genuinely improve accuracy. This abandons the
   zero-overhead claim, which is the paper's main selling point, but it is the version
   of the idea that can work.
4. **Do not** keep the current claims with better-looking numbers. The gains are inside
   the seed noise, and Reviewer ErQJ has already verified individual table cells; a
   result obtained by seed selection or an under-tuned baseline is the one outcome
   worse than a negative result.
