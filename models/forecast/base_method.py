from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import nn


@dataclass
class LossOutput:
    total: torch.Tensor
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class PredictionOutput:
    samples: torch.Tensor
    trace_samples: Optional[torch.Tensor] = None
    trace_times: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseForecastMethod(nn.Module):
    def training_step(self, batch, log_dict=None) -> LossOutput:
        raise NotImplementedError

    def predict(self, batch, num_samples: int, return_trace: bool = False) -> PredictionOutput:
        raise NotImplementedError

    def forward(self, batch, log_dict=None):
        loss_out = self.training_step(batch, log_dict=log_dict)
        metrics = loss_out.metrics
        zero = torch.zeros(1, device=loss_out.total.device, dtype=loss_out.total.dtype)
        reg = metrics.get("reg", metrics.get("recon", zero))
        cls = metrics.get("cls", metrics.get("kl", zero))
        vel = metrics.get("vel", metrics.get("dyn", zero))
        ctrl = metrics.get("ctrl", metrics.get("constraint", zero))
        stab = metrics.get("stab", metrics.get("dissipation", zero))
        return loss_out.total, reg, cls, vel, ctrl, stab

    def sample(
        self,
        batch,
        num_trajs: int,
        return_all_states: bool = False,
        collect_trace: bool = False,
    ):
        pred = self.predict(
            batch,
            num_samples=num_trajs,
            return_trace=(return_all_states or collect_trace),
        )

        samples = pred.samples
        if samples.dim() == 5:
            samples = samples.flatten(start_dim=-2)

        trace_samples = pred.trace_samples
        if trace_samples is None:
            trace_samples = pred.samples.unsqueeze(1)
        if trace_samples.dim() == 6:
            trace_samples = trace_samples.flatten(start_dim=-2)

        trace_times = pred.trace_times
        if trace_times is None:
            trace_times = torch.arange(
                trace_samples.shape[1], device=samples.device, dtype=samples.dtype
            )

        scores = pred.scores
        if scores is None:
            B, K, A = samples.shape[:3]
            scores = torch.zeros(B, K, A, device=samples.device, dtype=samples.dtype)

        return samples, trace_samples, trace_times, trace_samples, scores
