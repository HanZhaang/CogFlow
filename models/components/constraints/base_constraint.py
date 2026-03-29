from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class BaseConstraint(nn.Module):
    def forward(self, trace: Dict[str, torch.Tensor], batch, model) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
