"""CRAFT: training-time corrector wrapper for any forecasting backbone.

Wrap a backbone in :class:`RAGPlugin` and it gains a retrieval-conditioned RL
corrector during training. At eval time the wrapper is a pass-through: no
retrieval, no policy, no extra parameters on the forward path. After training,
`extract_base_state_dict` recovers the backbone alone for deployment.

Correspondence to the paper (Section 3):
    Y_ref      = distance-weighted mean of k retrieved futures   (Eq. 8)
    s          = [Y_hat ; Y_ref]                                  (Section 3.2)
    a ~ N(mu, sigma), a' = Pool(a; kappa)                         (Eq. 9-10)
    r in {0,1,2} from MSE/MAE improvement indicators              (Eq. 11)
    A_j        = (r_j - mean r) / (std r + eps)                   (Eq. 12)
    L_RL       = -mean(log pi(a_j|s) * A_j) + lambda * mean||a'_j||_2
    L_total    = gamma_1 * MSE + gamma_2 * L_RL                   (Eq. 14)

Example:
    >>> model = RAGPlugin(iTransformer(args), args)
    >>> out = model(x, x_mark, dec_in, y_mark, batch_y=target, query_idx=idx)
    >>> out['loss'].backward()
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Memory import MemoryBankWithRetrieval
from utils.tools import continuous_reward, discrete_reward


class PolicyHead(nn.Module):
    """Gaussian policy over an additive correction, conditioned on [Y_hat; Y_ref].

    A depthwise-in-time CNN rather than a per-timestep MLP: the kernel-3
    convolutions let the correction at step t depend on its neighbours, which is
    the same locality the pooling step relies on.
    """

    def __init__(
        self,
        c_out: int,
        hidden_dim: int = 128,
        mode: str = "concat",
        logstd_min: float = -5.0,
        logstd_max: float = -1.0,
    ):
        super().__init__()
        if mode not in ("concat", "diff"):
            raise ValueError(f"unknown policy mode: {mode}")
        self.mode = mode
        self.c_out = c_out
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        # State channels: 2*c_out when concatenating, c_out when using the residual.
        in_ch = c_out * 2 if mode == "concat" else c_out

        def trunk():
            return nn.Sequential(
                nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, c_out, kernel_size=3, padding=1),
            )

        self.action_mean = trunk()
        self.action_logstd = trunk()

    def forward(self, outputs: torch.Tensor, retrieved: torch.Tensor):
        """outputs/retrieved: [B, P, c_out] -> Normal over [B, P, c_out]."""
        if self.mode == "concat":
            x = torch.cat([outputs, retrieved], dim=-1)
        else:
            x = retrieved - outputs

        x = x.permute(0, 2, 1)  # [B, C, P] for Conv1d
        mean = self.action_mean(x).permute(0, 2, 1)
        logstd = self.action_logstd(x).permute(0, 2, 1)
        logstd = torch.clamp(logstd, min=self.logstd_min, max=self.logstd_max)
        return torch.distributions.Normal(mean, torch.exp(logstd))


class RAGPlugin(nn.Module):
    """Plug-and-play CRAFT wrapper.

    Reads from ``args``: use_rag, num_retrieve, num_rl_samples, gamma_1, gamma_2,
    lambda_reg, kappa, reward_level, reward_type, rl_sampling, detach_yhat,
    retrieval_mode, exclusion_radius, policy_hidden, policy_mode, seq_len,
    pred_len, enc_in, c_out, use_gpu, gpu, memory_store_cpu.
    """

    def __init__(self, base_model: nn.Module, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.use_rag = bool(getattr(args, "use_rag", False))

        self.num_retrieve = int(getattr(args, "num_retrieve", 5))
        self.num_samples = int(getattr(args, "num_rl_samples", 8))
        self.gamma_1 = float(getattr(args, "gamma_1", 0.5))
        self.gamma_2 = float(getattr(args, "gamma_2", 0.5))
        self.gamma_3 = float(getattr(args, "gamma_3", 0.0))
        self.distill_target = getattr(args, "distill_target", "best")
        self.distill_tau = float(getattr(args, "distill_tau", 1.0))
        self.distill_only_positive = bool(getattr(args, "distill_only_positive", True))
        self.lambda_reg = float(getattr(args, "lambda_reg", 0.0))
        self.kappa = int(getattr(args, "kappa", 3))
        self.reward_level = getattr(args, "reward_level", "step")
        self.reward_type = getattr(args, "reward_type", "discrete")
        self.rl_sampling = getattr(args, "rl_sampling", "sample")
        self.detach_yhat = bool(getattr(args, "detach_yhat", False))
        self.retrieval_mode = getattr(args, "retrieval_mode", "nn")
        self.exclusion_radius = int(getattr(args, "exclusion_radius", 0))

        if self.use_rag:
            self.memory_bank = MemoryBankWithRetrieval(
                seq_len=args.seq_len,
                dim=args.enc_in,
                pred_len=args.pred_len,
                use_gpu=bool(getattr(args, "use_gpu", True))
                and torch.cuda.is_available()
                and not bool(getattr(args, "no_faiss_gpu", False)),
                gpu_index=int(getattr(args, "gpu", 0)),
                store_on_cpu=bool(getattr(args, "memory_store_cpu", False)),
            )
            self.policy_head = PolicyHead(
                c_out=args.c_out,
                hidden_dim=int(getattr(args, "policy_hidden", 128)),
                mode=getattr(args, "policy_mode", "concat"),
            )

    # ------------------------------------------------------------------ setup

    def load_memory_bank(self, dataset, batch_size=64, num_workers=0):
        if self.use_rag:
            self.memory_bank.build_from_dataset(
                dataset, batch_size=batch_size, num_workers=num_workers
            )

    def get_base_model(self) -> nn.Module:
        return self.base_model

    def extract_base_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict of the deployable backbone, with the wrapper prefix removed."""
        prefix = "base_model."
        return {
            k[len(prefix) :]: v
            for k, v in self.state_dict().items()
            if k.startswith(prefix)
        }

    # ---------------------------------------------------------------- forward

    def _run_base(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Always go through forward(): it is the one entry point every TSLib model
        # implements with this signature. Calling .forecast() directly breaks on
        # backbones that define it as forecast(x_enc) only (DLinear, RLinear, ...).
        return self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def _build_reference(self, x_enc, query_idx):
        """Y_ref: distance-weighted average of retrieved futures (Eq. 8)."""
        neighbours, distances = self.memory_bank.retrieve(
            x_enc,
            k=self.num_retrieve,
            query_idx=query_idx,
            exclusion_radius=self.exclusion_radius,
            mode=self.retrieval_mode,
        )
        # Min-max normalise distances per query before the softmax so the weights
        # do not depend on the absolute scale of the dataset.
        d_min = distances.min(dim=1, keepdim=True)[0]
        d_max = distances.max(dim=1, keepdim=True)[0]
        d_norm = (distances - d_min) / (d_max - d_min + 1e-4)
        weights = F.softmax(torch.exp(-d_norm), dim=1)
        return torch.sum(weights[:, :, None, None] * neighbours, dim=1), distances

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        batch_y: Optional[torch.Tensor] = None,
        query_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self._run_base(x_enc, x_mark_enc, x_dec, x_mark_dec)
        result: Dict[str, Any] = {"outputs": outputs}

        # Eval / non-RAG: identical to the bare backbone.
        if not self.use_rag or not self.training:
            return result
        if batch_y is None:
            raise ValueError("batch_y is required while training with use_rag")

        pred_len = batch_y.size(1)
        y_hat = outputs[:, -pred_len:, -batch_y.size(-1) :]

        y_ref, distances = self._build_reference(x_enc, query_idx)
        state_pred = y_hat.detach() if self.detach_yhat else y_hat
        dist = self.policy_head(state_pred, y_ref)

        log_probs, rewards, action_norms, actions = [], [], [], []
        reward_fn = discrete_reward if self.reward_type == "discrete" else continuous_reward

        for _ in range(self.num_samples):
            # REINFORCE: the score-function estimator needs a non-reparameterised
            # sample. 'rsample' is kept only as a sensitivity knob.
            action = dist.rsample() if self.rl_sampling == "rsample" else dist.sample()
            # log pi(a|s) summed over channels -> one term per timestep, matching
            # the timestep-level reward granularity.
            log_prob = dist.log_prob(action).sum(dim=-1)

            smoothed = F.avg_pool1d(
                action.permute(0, 2, 1),
                kernel_size=self.kappa,
                stride=1,
                padding=self.kappa // 2,
            ).permute(0, 2, 1)
            adjusted = y_hat.detach() + smoothed

            with torch.no_grad():
                rewards.append(
                    reward_fn(y_hat.detach(), adjusted, batch_y, level=self.reward_level)
                )
            log_probs.append(log_prob)
            action_norms.append(smoothed.pow(2).sum(dim=-1).sqrt())
            if self.gamma_3:
                actions.append(smoothed.detach())

        log_probs = torch.stack(log_probs, dim=0)  # [Ns, B, P]
        rewards = torch.stack(rewards, dim=0).to(log_probs.dtype)  # [Ns, B, P]
        if rewards.shape != log_probs.shape:
            raise RuntimeError(
                f"reward shape {tuple(rewards.shape)} != log-prob shape "
                f"{tuple(log_probs.shape)}; check reward_level"
            )

        # Advantage normalised across the Ns samples of the same state (Eq. 13).
        advantages = (rewards - rewards.mean(dim=0, keepdim=True)) / (
            rewards.std(dim=0, keepdim=True) + 1e-4
        )
        pg_loss = -(log_probs * advantages).mean()
        reg = torch.stack(action_norms, dim=0).mean() if self.lambda_reg else y_hat.new_zeros(())
        rl_loss = pg_loss + self.lambda_reg * reg

        base_loss = F.mse_loss(y_hat, batch_y)

        # ---- corrector -> backbone distillation (the internalisation channel) ----
        # The RL term alone gives theta no useful signal: its gradient pushes y_hat
        # towards states where good actions are more *probable*, which is unrelated
        # to y_hat being accurate. This term instead hands theta the corrector's own
        # accepted output as a target, so what the corrector learns is transferred
        # by first-order descent instead of hoped for via zero-order side effects.
        distill_loss = y_hat.new_zeros(())
        if self.gamma_3 and actions:
            acts = torch.stack(actions, dim=0)  # [Ns, B, P, C], detached
            R = rewards  # [Ns, B, P]
            if self.distill_target == "best":
                # Per timestep, the single highest-reward correction.
                idx = R.argmax(dim=0)  # [B, P]
                sel = idx[None, :, :, None].expand(1, -1, -1, acts.size(-1))
                a_sel = acts.gather(0, sel).squeeze(0)
            else:  # 'advantage': reward-softmax weighted mean of the corrections
                w = torch.softmax(R / max(self.distill_tau, 1e-6), dim=0)
                a_sel = (w.unsqueeze(-1) * acts).sum(dim=0)
            if self.distill_only_positive:
                # Only distil corrections that actually improved something; a
                # zero-reward correction carries no information worth copying.
                a_sel = a_sel * (R.amax(dim=0) > 0).to(a_sel.dtype).unsqueeze(-1)
            # Target is a temporally-pooled, improvement-screened step from y_hat
            # towards the truth -- not the raw label, so it acts as a denoised target.
            target = (y_hat.detach() + a_sel).detach()
            distill_loss = F.mse_loss(y_hat, target)

        result.update(
            {
                "loss": self.gamma_1 * base_loss
                + self.gamma_2 * rl_loss
                + self.gamma_3 * distill_loss,
                "distill_loss": distill_loss,
                "base_loss": base_loss,
                "rl_loss": rl_loss,
                "pg_loss": pg_loss,
                "action_l2": reg,
                "reward_mean": rewards.mean(),
                "dist": dist,
                "y_ref": y_ref,
                "retrieval_distance": distances.mean(),
                "adjusted_outputs": y_hat + dist.mean,
            }
        )
        return result
