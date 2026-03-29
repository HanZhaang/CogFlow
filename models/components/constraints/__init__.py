from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from .dissipativity import DissipativityConstraint


class ConstraintCollection(nn.Module):
    def __init__(self, items: List[Tuple[str, float, nn.Module]]):
        super().__init__()
        self.names = [name for name, _, _ in items]
        self.weights = {name: weight for name, weight, _ in items}
        self.modules_dict = nn.ModuleDict({name: module for name, _, module in items})

    def forward(self, trace: Dict[str, torch.Tensor], batch, model):
        if len(self.modules_dict) == 0:
            state_seq = trace.get("state_seq")
            if state_seq is None:
                any_tensor = next(model.parameters())
                zero = any_tensor.new_zeros(1)
            else:
                zero = state_seq.new_zeros(1)
            return zero, {}

        total = None
        metrics = {}
        for name in self.names:
            out = self.modules_dict[name](trace, batch, model)
            weighted = self.weights[name] * out["loss"]
            total = weighted if total is None else total + weighted
            metrics[f"{name}_loss"] = out["loss"]
            for key, value in out.items():
                if key != "loss":
                    metrics[f"{name}_{key}"] = value

        metrics["constraint"] = total
        return total, metrics


def build_constraints(cfg, state_dim: int, ctrl_dim: int):
    constraints_cfg = cfg.get("CONSTRAINTS", None)
    if constraints_cfg is None or not constraints_cfg.get("ENABLED", False):
        return ConstraintCollection([])

    items = []
    for item_cfg in constraints_cfg.get("ITEMS", []):
        name = item_cfg["NAME"]
        weight = float(item_cfg.get("WEIGHT", 1.0))
        if name == "dissipativity":
            module = DissipativityConstraint(
                state_dim=state_dim,
                ctrl_dim=ctrl_dim,
                hidden_dim=int(item_cfg.get("HIDDEN_DIM", state_dim)),
                state_key=str(item_cfg.get("STATE_KEY", "state_seq")),
                margin=float(item_cfg.get("MARGIN", 0.0)),
            )
        else:
            raise ValueError(f"Unknown constraint '{name}'")
        items.append((name, weight, module))
    return ConstraintCollection(items)
