from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .base_constraint import BaseConstraint


class DissipativityConstraint(BaseConstraint):
    def __init__(self, state_dim: int, ctrl_dim: int, hidden_dim: int, state_key: str = "state_seq", margin: float = 0.0):
        super().__init__()
        self.state_key = state_key
        self.margin = margin
        self.storage_fn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.supply_fn = nn.Sequential(
            nn.Linear(state_dim + ctrl_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trace: Dict[str, torch.Tensor], batch, model) -> Dict[str, torch.Tensor]:
        state_seq = trace.get(self.state_key)
        ctrl_seq = trace.get("ctrl_seq")
        if state_seq is None or ctrl_seq is None or state_seq.shape[1] < 2:
            zero = next(self.parameters()).new_zeros(1)
            return {"loss": zero, "residual": zero}

        V_t = self.storage_fn(state_seq[:, :-1]).squeeze(-1)
        V_tp1 = self.storage_fn(state_seq[:, 1:]).squeeze(-1)
        supply = self.supply_fn(torch.cat([state_seq[:, :-1], ctrl_seq[:, :-1]], dim=-1)).squeeze(-1)
        residual = V_tp1 - V_t - supply
        loss = F.relu(residual + self.margin).mean()
        return {
            "loss": loss,
            "residual": residual.mean(),
            "storage": V_t.mean(),
            "supply": supply.mean(),
        }
