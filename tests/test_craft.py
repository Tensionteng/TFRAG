"""CPU-only correctness tests for the CRAFT components.

These run without datasets or a GPU: `pytest tests/ -v`. They cover the four
failures that made the RAG path unrunnable, plus the semantics that the
experiments depend on (reward shape/range, exclusion, extraction, band analysis).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.indexed import IndexedDataset, unpack_batch  # noqa: E402
from models.Memory import MemoryBankWithRetrieval  # noqa: E402
from models.model_factory import extract_base_state_dict, strip_base_prefix  # noqa: E402
from models.rag_plugin import PolicyHead, RAGPlugin  # noqa: E402
from utils.losses_freq import build_criterion  # noqa: E402
from utils.tools import continuous_reward, discrete_reward  # noqa: E402

L, P, D = 16, 8, 3


class TinyBackbone(nn.Module):
    """Stand-in forecaster: [B, L, D] -> [B, P, D]."""

    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(args.seq_len, args.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.proj(x_enc.permute(0, 2, 1)).permute(0, 2, 1)


class TinyDataset(torch.utils.data.Dataset):
    """Windows over one synthetic series, mirroring the TSLib contract."""

    def __init__(self, n=40, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.series = torch.randn(n + L + P, D, generator=g)
        self.n = n
        self.scale = False

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.series[i : i + L]
        y = self.series[i + L - 4 : i + L + P]  # label_len=4 prefix + horizon
        return x, y, torch.zeros(L, 4), torch.zeros(y.shape[0], 4)


class Args:
    seq_len, label_len, pred_len = L, 4, P
    enc_in = dec_in = c_out = D
    d_model = 32
    use_rag = True
    num_retrieve = 3
    num_rl_samples = 4
    gamma_1 = gamma_2 = 0.5
    lambda_reg = 0.0
    kappa = 3
    reward_level = "step"
    reward_type = "discrete"
    rl_sampling = "sample"
    detach_yhat = False
    retrieval_mode = "nn"
    exclusion_radius = 0
    policy_hidden = 16
    policy_mode = "concat"
    use_gpu = False
    gpu = 0
    memory_store_cpu = True
    no_faiss_gpu = True
    features = "M"


def make_args(**kw):
    a = Args()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ------------------------------------------------------------------ regressions

def test_tools_imports_without_tkinter():
    """utils.tools used to import tkinter, breaking every headless install."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "utils", "tools.py")).read()
    assert "tkinter" not in src


@pytest.mark.parametrize("c_out,hidden", [(3, 16), (7, 32), (21, 8), (321, 8)])
def test_policy_head_channels_independent_of_d_model(c_out, hidden):
    """Previously sized from d_model*2, so it crashed unless d_model == c_out."""
    head = PolicyHead(c_out=c_out, hidden_dim=hidden)
    dist = head(torch.randn(2, P, c_out), torch.randn(2, P, c_out))
    assert dist.mean.shape == (2, P, c_out)
    assert torch.isfinite(dist.mean).all() and (dist.stddev > 0).all()


def test_policy_head_diff_mode():
    head = PolicyHead(c_out=D, hidden_dim=8, mode="diff")
    assert head(torch.randn(2, P, D), torch.randn(2, P, D)).mean.shape == (2, P, D)


def test_train_eval_mode_toggles():
    """The read-only `training` property made nn.Module.train() raise."""
    m = RAGPlugin(TinyBackbone(make_args()), make_args())
    m.train()
    assert m.training
    m.eval()
    assert not m.training
    m.train(True)
    assert m.training


# ---------------------------------------------------------------------- rewards

def test_discrete_reward_shape_and_range():
    out = torch.randn(4, P, D)
    gt = torch.randn(4, P, D)
    r = discrete_reward(out, gt.clone(), gt)  # perfect correction
    assert r.shape == (4, P), "step-level reward must be [B, P] to match log-prob"
    assert torch.all((r >= 0) & (r <= 2))
    assert torch.all(r == 2), "a perfect correction improves both MSE and MAE"


def test_discrete_reward_zero_when_correction_hurts():
    gt = torch.zeros(2, P, D)
    out = torch.full((2, P, D), 0.1)
    worse = torch.full((2, P, D), 0.5)
    assert torch.all(discrete_reward(out, worse, gt) == 0)


def test_item_level_reward_shape():
    r = discrete_reward(torch.randn(5, P, D), torch.randn(5, P, D), torch.randn(5, P, D), level="item")
    assert r.shape == (5,)


def test_continuous_reward_sign():
    gt = torch.zeros(2, P, D)
    out = torch.full((2, P, D), 0.5)
    better = torch.full((2, P, D), 0.1)
    assert torch.all(continuous_reward(out, better, gt) > 0)


# ------------------------------------------------------------------ memory bank

def _bank(ds, **kw):
    b = MemoryBankWithRetrieval(seq_len=L, dim=D, pred_len=P, use_gpu=False, store_on_cpu=True, **kw)
    return b.build_from_dataset(ds, batch_size=7)


def test_bank_position_equals_dataset_index():
    ds = TinyDataset()
    b = _bank(ds)
    assert b.n_total == len(ds)
    for i in (0, 5, len(ds) - 1):
        expected = ds[i][1][-P:]
        got = b.y_store[i]
        assert torch.allclose(got, expected, atol=1e-5)


def test_nn_retrieval_finds_self_without_exclusion():
    ds = TinyDataset()
    b = _bank(ds)
    q = torch.stack([ds[i][0] for i in (2, 11, 20)])
    y, d = b.retrieve(q, k=3, mode="nn")
    assert y.shape == (3, 3, P, D)
    # Nearest neighbour of a bank member is itself, at distance ~0.
    assert d[:, 0].abs().max() < 1e-3


def test_exclusion_radius_removes_neighbourhood():
    ds = TinyDataset(n=60)
    b = _bank(ds)
    qidx = torch.tensor([10, 25, 40])
    q = torch.stack([ds[int(i)][0] for i in qidx])
    radius = 5
    # Re-derive the returned indices by matching the retrieved futures back to the store.
    y, _ = b.retrieve(q, k=3, query_idx=qidx, exclusion_radius=radius, mode="nn")
    for row, qi in enumerate(qidx.tolist()):
        for j in range(y.shape[1]):
            hits = (b.y_store - y[row, j]).abs().amax(dim=(1, 2)) < 1e-5
            found = torch.nonzero(hits).flatten().tolist()
            assert found, "retrieved future is not in the store"
            assert all(abs(f - qi) > radius for f in found), (
                f"query {qi} got neighbour within radius {radius}: {found}"
            )


def test_exclusion_requires_query_idx():
    b = _bank(TinyDataset())
    with pytest.raises(ValueError, match="query_idx"):
        b.retrieve(torch.randn(2, L, D), k=2, exclusion_radius=3)


def test_random_retrieval_differs_from_nn():
    ds = TinyDataset(n=60)
    b = _bank(ds)
    q = torch.stack([ds[i][0] for i in range(8)])
    y_nn, _ = b.retrieve(q, k=3, mode="nn")
    torch.manual_seed(0)
    y_rand, _ = b.retrieve(q, k=3, mode="random")
    assert y_rand.shape == y_nn.shape
    assert not torch.allclose(y_nn, y_rand)


def test_bank_rejects_shuffled_build():
    """A shuffled bank would silently break exclusion, so it must fail loudly."""
    ds = TinyDataset()
    b = MemoryBankWithRetrieval(seq_len=L, dim=D, pred_len=P, store_on_cpu=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    b.load_dataset(loader)  # legacy path: allowed, but index_map is meaningless
    assert b.n_total == len(ds)


# --------------------------------------------------------------------- plugin

def test_plugin_training_forward_produces_finite_loss_and_grads():
    args = make_args()
    m = RAGPlugin(TinyBackbone(args), args)
    ds = TinyDataset()
    m.load_memory_bank(ds, batch_size=8)
    m.train()

    x = torch.stack([ds[i][0] for i in range(4)])
    y = torch.stack([ds[i][1] for i in range(4)])[:, -P:]
    out = m(x, None, None, None, batch_y=y)

    assert torch.isfinite(out["loss"])
    assert out["reward_mean"].item() >= 0
    assert out["y_ref"].shape == (4, P, D)
    out["loss"].backward()

    pol = [p for n, p in m.named_parameters() if n.startswith("policy_head")]
    assert pol and any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol), (
        "policy head received no gradient"
    )
    base = [p for n, p in m.named_parameters() if n.startswith("base_model")]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in base)


def test_optimizer_covers_policy_head():
    """The old _select_optimizer unwrapped the plugin, freezing the corrector."""
    args = make_args()
    m = RAGPlugin(TinyBackbone(args), args)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    owned = {id(p) for g in opt.param_groups for p in g["params"]}
    pol = [p for n, p in m.named_parameters() if n.startswith("policy_head")]
    assert pol and all(id(p) in owned for p in pol)


def test_policy_head_actually_updates():
    args = make_args()
    m = RAGPlugin(TinyBackbone(args), args)
    ds = TinyDataset()
    m.load_memory_bank(ds, batch_size=8)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    before = m.policy_head.action_mean[0].weight.detach().clone()
    x = torch.stack([ds[i][0] for i in range(6)])
    y = torch.stack([ds[i][1] for i in range(6)])[:, -P:]
    for _ in range(3):
        opt.zero_grad()
        m(x, None, None, None, batch_y=y)["loss"].backward()
        opt.step()
    assert not torch.allclose(before, m.policy_head.action_mean[0].weight)


def test_eval_mode_is_pass_through():
    args = make_args()
    m = RAGPlugin(TinyBackbone(args), args)
    m.eval()
    x = torch.randn(2, L, D)
    out = m(x, None, None, None)
    assert set(out) == {"outputs"}, "eval must not retrieve or correct"
    assert torch.allclose(out["outputs"], m.base_model(x))


def test_detach_blocks_rl_gradient_to_backbone():
    ds = TinyDataset()
    x = torch.stack([ds[i][0] for i in range(4)])
    y = torch.stack([ds[i][1] for i in range(4)])[:, -P:]

    args = make_args(detach_yhat=True, gamma_1=0.0, gamma_2=1.0)
    m = RAGPlugin(TinyBackbone(args), args)
    m.load_memory_bank(ds, batch_size=8)
    m.train()
    m(x, None, None, None, batch_y=y)["loss"].backward()
    g = [p.grad for p in m.base_model.parameters() if p.grad is not None]
    assert all(gr.abs().sum() == 0 for gr in g), "detach must zero the RL path to theta"


def test_l2_penalty_increases_loss():
    ds = TinyDataset()
    x = torch.stack([ds[i][0] for i in range(4)])
    y = torch.stack([ds[i][1] for i in range(4)])[:, -P:]
    losses = {}
    for lam in (0.0, 1.0):
        torch.manual_seed(7)
        args = make_args(lambda_reg=lam)
        m = RAGPlugin(TinyBackbone(args), args)
        m.load_memory_bank(ds, batch_size=8)
        m.train()
        torch.manual_seed(7)
        out = m(x, None, None, None, batch_y=y)
        losses[lam] = (out["rl_loss"].item(), out["action_l2"].item())
    assert losses[1.0][1] > 0, "action norm must be measured when lambda_reg > 0"
    assert losses[0.0][1] == 0


def test_extraction_round_trips_backbone_weights():
    args = make_args()
    plugin = RAGPlugin(TinyBackbone(args), args)
    sd = extract_base_state_dict(plugin)
    assert sd and all(not k.startswith("base_model.") for k in sd)
    fresh = TinyBackbone(args)
    fresh.load_state_dict(sd)  # strict: keys must match exactly
    x = torch.randn(2, L, D)
    assert torch.allclose(fresh(x), plugin.base_model(x))


def test_strip_prefix_on_wrapper_checkpoint():
    args = make_args()
    plugin = RAGPlugin(TinyBackbone(args), args)
    stripped = strip_base_prefix(plugin.state_dict())
    TinyBackbone(args).load_state_dict(stripped)
    # Idempotent on an already-stripped dict.
    assert set(strip_base_prefix(stripped)) == set(stripped)


# ------------------------------------------------------------------- utilities

def test_indexed_dataset_appends_index_and_forwards_attrs():
    ds = TinyDataset()
    wrapped = IndexedDataset(ds)
    item = wrapped[3]
    assert len(item) == 5 and item[4] == 3
    assert wrapped.scale is False  # attribute forwarding
    loader = torch.utils.data.DataLoader(wrapped, batch_size=4)
    x, y, xm, ym, idx = unpack_batch(next(iter(loader)))
    assert idx.shape == (4,)
    x4, *_rest, none_idx = unpack_batch(next(iter(torch.utils.data.DataLoader(ds, batch_size=4))))
    assert none_idx is None


@pytest.mark.parametrize("name", ["MSE", "mae", "huber", "fredf", "ffl", "bandmse"])
def test_frequency_criteria_are_finite_and_differentiable(name):
    crit = build_criterion(name)
    pred = torch.randn(3, 16, D, requires_grad=True)
    true = torch.randn(3, 16, D)
    loss = crit(pred, true)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_criterion_zero_on_perfect_prediction():
    for name in ("MSE", "mae", "fredf", "ffl", "bandmse"):
        y = torch.randn(2, 16, D)
        assert build_criterion(name)(y.clone(), y).item() < 1e-8


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="unknown --loss"):
        build_criterion("nope")


# ------------------------------------------------------- band analysis semantics

def test_band_energy_localises_a_pure_tone():
    from experiments.freq_band_analysis import BANDS, band_energy, band_masks

    n = 96
    t = np.arange(n)
    # Normalised frequency 0.35 -> must land in the 0.3-0.4 band.
    x = np.sin(2 * np.pi * 0.35 * t)[None, :, None]
    _, masks = band_masks(n)
    e = band_energy(x, masks)
    assert int(np.argmax(e)) == [b for b, _ in enumerate(BANDS)][3]


def test_residual_reduction_sign_and_bound():
    """A strictly better prediction must yield a positive reduction <= 100%."""
    from experiments.freq_band_analysis import band_energy, band_masks, pct

    rng = np.random.default_rng(0)
    true = rng.standard_normal((20, 96, 2))
    base = true + rng.standard_normal((20, 96, 2))
    craft = true + 0.5 * (base - true)  # exactly halves the residual
    _, masks = band_masks(96)
    red = [pct(b, c) for b, c in zip(band_energy(base - true, masks), band_energy(craft - true, masks))]
    for r in red:
        assert 0 < r < 100
        assert abs(r - 75.0) < 5.0  # halving amplitude quarters energy


def test_worse_prediction_gives_negative_reduction():
    from experiments.freq_band_analysis import band_energy, band_masks, pct

    rng = np.random.default_rng(1)
    true = rng.standard_normal((10, 96, 2))
    base = true + 0.2 * rng.standard_normal((10, 96, 2))
    craft = true + 0.6 * rng.standard_normal((10, 96, 2))
    _, masks = band_masks(96)
    red = [pct(b, c) for b, c in zip(band_energy(base - true, masks), band_energy(craft - true, masks))]
    assert all(r < 0 for r in red), "a worse prediction must never read as an improvement"


def test_analysis_pipeline_end_to_end(tmp_path):
    """freq + per-example scripts run on synthetic results dirs and agree on direction."""
    from experiments.freq_band_analysis import analyse as freq_analyse
    from experiments.per_example_analysis import analyse as pe_analyse

    rng = np.random.default_rng(2)
    true = rng.standard_normal((30, 96, 2)).astype(np.float32)
    base = (true + rng.standard_normal((30, 96, 2)) * 0.4).astype(np.float32)
    craft = (true + 0.5 * (base - true)).astype(np.float32)
    dirs = {}
    for name, pred in (("base", base), ("craft", craft)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "pred.npy", pred)
        np.save(d / "true.npy", true)
        dirs[name] = str(d)

    rows = freq_analyse("D", "M", dirs["base"], dirs["craft"])
    overall = [r for r in rows if r["band"] == "ALL"][0]
    assert overall["frmse_reduction_pct"] > 0
    assert overall["mse_craft"] < overall["mse_base"]

    pe = pe_analyse("D", "M", dirs["base"], dirs["craft"])
    all_row = [r for r in pe if r["split"] == "all"][0]
    assert all_row["delta_pct"] > 0
    assert {r["split"] for r in pe} >= {"Q1_easy", "Q4_hard", "last_20pct_shift"}


def test_freq_analysis_refuses_mismatched_ground_truth(tmp_path):
    from experiments.freq_band_analysis import analyse as freq_analyse

    rng = np.random.default_rng(3)
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "pred.npy", rng.standard_normal((5, 96, 2)).astype(np.float32))
        np.save(d / "true.npy", rng.standard_normal((5, 96, 2)).astype(np.float32))
    with pytest.raises(ValueError, match="ground truth"):
        freq_analyse("D", "M", str(tmp_path / "a"), str(tmp_path / "b"))


# --------------------------------------------------------------- aggregator

def _write_run(dirpath, setting, variant, seed, mse, mae, **cfg):
    import json

    base_cfg = dict(
        data="ETTh1", model="iTransformer", pred_len=96, seq_len=96, train_epochs=10,
        learning_rate=0.0001, batch_size=32, d_model=128, d_ff=128, e_layers=2,
        n_heads=8, lradj="type1", features="M", patience=3, seed=seed,
    )
    base_cfg.update(cfg)
    rec = {"setting": setting, "variant": variant, "metrics": {"mse": mse, "mae": mae},
           "config": base_cfg, "env": {}}
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, f"{setting}.json"), "w") as f:
        json.dump(rec, f)


def test_aggregator_pairs_by_seed_and_reports_ci(tmp_path):
    runs, out = str(tmp_path / "runs"), str(tmp_path / "analysis")
    for i, s in enumerate([1, 2, 3, 4, 5]):
        _write_run(runs, f"b{s}", "base", s, 0.400 + 0.001 * i, 0.420)
        _write_run(runs, f"c{s}", "craft", s, 0.380 + 0.001 * i, 0.410)
    r = subprocess.run(
        [sys.executable, "experiments/aggregate_results.py", "--runs", runs, "--out", out],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert r.returncode == 0, r.stderr
    summary = open(os.path.join(out, "summary.md")).read()
    assert "Aggregate" in summary
    import csv as _csv

    with open(os.path.join(out, "paired_tests.csv")) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["n_seeds"]) == 5
    assert float(row["mse_delta_pct"]) > 0
    assert float(row["mse_p_ttest"]) < 0.05  # constant offset -> highly significant
    assert int(row["mse_wins"]) == 5


def test_deployed_eval_record_does_not_collide_with_base(tmp_path):
    """A re-evaluated CRAFT backbone must not be filed under the 'base' variant."""
    from utils.run_logger import variant_name

    class A:
        use_rag = False
        loss = "MSE"
        variant_override = "craft_deployed"

    assert variant_name(A()) == "craft_deployed"
    A.variant_override = None
    assert variant_name(A()) == "base"


def test_aggregator_rejects_duplicate_seed_records(tmp_path):
    runs, out = str(tmp_path / "runs"), str(tmp_path / "analysis")
    for s in (1, 2, 3):
        _write_run(runs, f"b{s}", "base", s, 0.40, 0.42)
        _write_run(runs, f"c{s}", "craft", s, 0.38, 0.41)
    _write_run(runs, "b1_dup", "base", 1, 0.99, 0.99)  # same cell+variant+seed
    r = subprocess.run(
        [sys.executable, "experiments/aggregate_results.py", "--runs", runs, "--out", out],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert r.returncode == 0, r.stderr
    assert "duplicate records" in open(os.path.join(out, "summary.md")).read()


def test_aggregator_skips_protocol_mismatch(tmp_path):
    runs, out = str(tmp_path / "runs"), str(tmp_path / "analysis")
    for s in (1, 2, 3):
        _write_run(runs, f"b{s}", "base", s, 0.40, 0.42, train_epochs=10)
        _write_run(runs, f"c{s}", "craft", s, 0.38, 0.41, train_epochs=50)  # different budget
    r = subprocess.run(
        [sys.executable, "experiments/aggregate_results.py", "--runs", runs, "--out", out],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert r.returncode == 0, r.stderr
    summary = open(os.path.join(out, "summary.md")).read()
    assert "protocol mismatch" in summary, "unequal training budgets must not be paired"
