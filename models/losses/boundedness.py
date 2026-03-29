from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def get_boundedness_weight(
    epoch: int,
    *,
    enabled: bool,
    base_weight: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    if (not enabled) or base_weight <= 0.0:
        return 0.0
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)

    ramp_progress = min(max(epoch - warmup_epochs, 0) / float(ramp_epochs), 1.0)
    return float(base_weight) * ramp_progress


class BoundednessLoss(nn.Module):
    """
    Soft dissipativity regularizer for diagonal latent SDEs:

        2 <z, f(z, u)> + ||G(z)||_F^2 <= -2 alpha ||z||^2 + beta
    """

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        beta: float = 1.0,
        tau: float = 1.0,
        late_only: bool = True,
        late_ratio: float = 0.5,
        beta_mode: str = "fixed",
        beta_quantile: float = 0.95,
        detach_beta_stat: bool = True,
        norm_mode: str = "sum",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.late_only = bool(late_only)
        self.late_ratio = float(late_ratio)
        self.beta_mode = str(beta_mode)
        self.beta_quantile = float(beta_quantile)
        self.detach_beta_stat = bool(detach_beta_stat)
        self.norm_mode = str(norm_mode)

    def _reduce_sq(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "sum":
            return tensor.pow(2).sum(dim=-1)
        if self.norm_mode == "mean":
            return tensor.pow(2).mean(dim=-1)
        raise ValueError(f"Unknown norm_mode={self.norm_mode}")

    def _time_weights(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        weights = torch.ones(steps, device=device, dtype=dtype)
        if self.late_only:
            start = int((1.0 - self.late_ratio) * steps)
            weights[:start] = 0.25
        return weights

    def _beta_from_raw(self, raw_term: torch.Tensor) -> torch.Tensor:
        if self.beta_mode == "fixed":
            return raw_term.new_tensor(self.beta)
        if self.beta_mode == "quantile":
            beta = torch.quantile(raw_term.reshape(-1), self.beta_quantile)
            return beta.detach() if self.detach_beta_stat else beta
        raise ValueError(f"Unknown beta_mode={self.beta_mode}")

    def forward(
        self,
        *,
        state_seq: torch.Tensor,
        drift_seq: torch.Tensor,
        sigma_seq: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state_seq.shape != drift_seq.shape or state_seq.shape != sigma_seq.shape:
            raise ValueError(
                "state_seq, drift_seq, sigma_seq must share shape [B, T, Dz], "
                f"got {tuple(state_seq.shape)}, {tuple(drift_seq.shape)}, {tuple(sigma_seq.shape)}"
            )

        drift_term = 2.0 * (state_seq * drift_seq).sum(dim=-1)
        diffusion_energy = self._reduce_sq(sigma_seq)
        z_norm_sq = self._reduce_sq(state_seq)
        raw_term = drift_term + diffusion_energy + 2.0 * self.alpha * z_norm_sq
        beta = self._beta_from_raw(raw_term)
        violation = raw_term - beta

        time_weights = self._time_weights(state_seq.size(1), state_seq.device, state_seq.dtype).unsqueeze(0)
        if valid_mask is None:
            valid_mask = torch.ones_like(violation)
        total_mask = valid_mask.to(violation.dtype) * time_weights

        penalty = F.softplus(violation / self.tau) * self.tau
        denom = total_mask.sum().clamp_min(1.0)
        loss = (penalty * total_mask).sum() / denom

        with torch.no_grad():
            valid_denom = valid_mask.to(violation.dtype).sum().clamp_min(1.0)
            stats = {
                "loss_bnd": loss.detach(),
                "bnd_beta": beta.detach(),
                "bnd_violation_mean": (violation * valid_mask).sum().detach() / valid_denom,
                "bnd_violation_pos_rate": (((violation > 0).to(violation.dtype)) * valid_mask).sum().detach() / valid_denom,
                "bnd_drift_term_mean": (drift_term * valid_mask).sum().detach() / valid_denom,
                "bnd_diffusion_energy_mean": (diffusion_energy * valid_mask).sum().detach() / valid_denom,
                "bnd_z_norm_sq_mean": (z_norm_sq * valid_mask).sum().detach() / valid_denom,
            }

        return loss, stats
