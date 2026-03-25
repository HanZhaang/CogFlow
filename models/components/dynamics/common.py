from __future__ import annotations

import math

import torch
from torch import nn


class GaussianHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim * 2),
        )

    def forward(self, x):
        mean, logvar = self.net(x).chunk(2, dim=-1)
        logvar = logvar.clamp(min=-8.0, max=6.0)
        return mean, logvar


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
    if not sample:
        return mean
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (math.log(2.0 * math.pi) + logvar + (target - mean).pow(2) / logvar.exp())


def gaussian_kl(
    post_mean: torch.Tensor,
    post_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        prior_logvar
        - post_logvar
        + (post_logvar.exp() + (post_mean - prior_mean).pow(2)) / prior_logvar.exp()
        - 1.0
    )
