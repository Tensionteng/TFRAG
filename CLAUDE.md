# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) that adds **SRRF** — a plug-and-play RAG + reinforcement-learning *training* enhancement for time series forecasting. The claim being tested is that retrieval + RL during training lets the backbone internalize high-frequency dynamics, so **inference is unchanged**: no retrieval, no extra params, no architecture change at test time.

Everything under `models/` (except `rag_plugin.py`, `Memory.py`, `model_factory.py`), `layers/`, `data_provider/`, `utils/`, and `scripts/` is upstream TSLib. The SRRF work lives in the four files listed under Architecture below plus the root-level `*.sh` / analysis scripts.

`craft/` is an untracked LaTeX paper (NeurIPS/ICLR style, `main.tex` + `sections/`) with its own git repo — it is the write-up of these experiments, not code.

## Environment & commands

`uv` is the package manager (migrated from conda in `0fe6da6`); `requirements.txt` is stale upstream pinning — trust `pyproject.toml` / `uv.lock`.

```bash
uv sync                                    # create .venv from uv.lock
uv run python run.py --task_name ... --is_training 1 --model_id ... --model ... --data ...
bash any_model_with_rag_ETTh1.sh           # SRRF experiment scripts (root level)
bash ablation_study.sh                     # gamma_2 sweep on weather
bash scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh   # upstream baselines
```

There is no test suite and no CI. Dev extras (`uv sync --extra dev`) provide `pytest`, `black`, `isort`, `flake8`; black/isort are configured for `line-length = 100`.

Datasets are **not** in the repo — download from the links in `README.md` and place under `./dataset/` (`dataset/ETT-small/`, `dataset/weather/`, `dataset/electricity/`, `dataset/traffic/`, `dataset/exchange_rate/`). Scripts pass these via `--root_path`.

Outputs: `./checkpoints/<setting>/checkpoint.pth`, `./results/<setting>/{pred,true,metrics}.npy`, `./test_results/<setting>/*.png`, appended lines in `result_long_term_forecast.txt`, and RAG training plots in `./adjust_result/`. `<setting>` is the long hyperparameter string built in `run.py:394`, suffixed with `_rag_<gamma_1>_<gamma_2>` when RAG is on.

## Architecture

Entry point `run.py` parses ~80 args, picks an `Exp_*` class by `--task_name`, then loops `--itr` times calling `exp.train(setting)` / `exp.test(setting)`.

**Only `long_term_forecast` is SRRF-aware.** `exp/exp_long_term_forecasting.py` was rewritten (`9972e5f`) to build models through `models/model_factory.py:create_model`, which imports `models/<name>.py` dynamically via `importlib` and wraps it in `RAGPlugin` when `--use_rag` is set. The other four `Exp_*` classes still instantiate from `Exp_Basic.model_dict` (`exp/exp_basic.py`) and ignore `--use_rag` entirely. Adding a model therefore needs both: a `models/<Name>.py` exposing `Model`, **and** an entry in `model_dict` for the non-forecasting tasks.

The RAG path (`models/rag_plugin.py`):

1. `MemoryBankWithRetrieval` (`models/Memory.py`) FAISS-indexes the **whole training set** — flattened `seq_len * enc_in` vectors — once at the start of `train()`, keeping `y_store` / `x_mark_store` / `y_mark_store` resident on device. Memory scales with dataset size; this is why RAG runs need large GPU memory.
2. Per batch, `retrieve_similar` returns top-k neighbours' ground-truth futures; they are softmax-weighted by normalized L2 distance into a single `retrieved_gt`.
3. `PolicyHead` maps `(outputs, retrieved_gt)` to a diagonal `Normal` over an additive correction (`adjusted = outputs + action`).
4. `num_rl_samples` actions are `rsample()`d; reward is a *binary, per-timestep* indicator ("did the adjustment reduce MAE/MSE at this step?" — `utils/tools.py:205,218`, changed from item-level in `50fdd52`); rewards are standardized across samples into advantages; loss is `gamma_1 * MSE + gamma_2 * (-log_prob * advantage).mean()`.
5. `RAGPlugin.forward` returns a **dict** (`outputs`, `loss`, `base_loss`, `rl_loss`, `dist`, `adjusted_outputs`) in training and a dict with just `outputs` in eval. Every call site in the exp class branches on `isinstance(result, dict)`, which is how one experiment class serves both wrapped and bare models.

`test()` reports metrics on `result['outputs']` — the **base** prediction, never `adjusted_outputs`. That is deliberate: the RL head is a training-time critic, so evaluation must be inference-cost-free.

Analysis of finished runs: `freq_analysis_all.py` and `freq_metric_plot.py` (matplotlib/scipy FFT comparisons of base vs `+SRRF` `pred.npy`), plus `demo.ipynb`, `best_results.csv`, `result_summary.parquet`. These read hardcoded paths/filenames and assume older `<setting>` / `<setting>_rag` directory names — expect to fix paths before rerunning them. Comments and print output in the SRRF code are mixed Chinese/English.

## Known rough edges (verified in this checkout)

These are live, not hypothetical — check them before concluding a run "just fails":

- `utils/tools.py:2` has a stray upstream `from tkinter import NO`, which makes **any** import of `utils.tools` (and therefore all of `models/Memory.py`, `rag_plugin.py`, every exp) fail with `ModuleNotFoundError: No module named 'tkinter'` on a headless install. Deleting the line is safe — `NO` is unused.
- `PolicyHead.__init__` sizes its `Conv1d` with `input_dim = d_model * 2`, but `forward` concatenates along the feature axis producing `2 * c_out` channels. Confirmed crash: `d_model=128, c_out=7` → `expected input[2, 14, 96] to have 256 channels`. `--use_rag` only runs when `d_model == c_out`.
- `_select_optimizer` (`exp/exp_long_term_forecasting.py:53`) calls `unwrap_model(self.model)`, which strips the `RAGPlugin`. The optimizer therefore only sees backbone params — `policy_head` receives gradients but is never stepped, so it stays at initialization.
- The root `*.sh` scripts still pass `--gemma_1` / `--gemma_2`; `run.py` renamed these to `--gamma_1` / `--gamma_2`, so those scripts die on argparse. `run.py:458` also still reads `args.gemma_1` in the `--is_training 0` branch (`AttributeError`).
- `uv.lock` pins **faiss-cpu**, but `Memory.py` uses `faiss.StandardGpuResources` / `GpuIndexFlatL2` whenever `use_gpu` is true (default when CUDA is visible). RAG on GPU needs a faiss build with GPU support.
- `Exp_Basic._acquire_device` hardcodes `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"` after the device is chosen, so `--gpu` beyond that range behaves unexpectedly.
- `run.py:14` leaves `torch.autograd.set_detect_anomaly(True)` on globally — correct for debugging the RL loss, but a real slowdown for benchmark timing.
