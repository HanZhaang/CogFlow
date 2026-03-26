from __future__ import annotations

import torch
from torch import nn

from .common import GaussianHead


class LatentARGRU(nn.Module):
    def __init__(self, latent_dim: int, ctrl_dim: int, ctx_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.init_hidden = nn.Linear(ctx_dim, hidden_dim)
        self.init_latent = nn.Linear(ctx_dim, latent_dim)
        self.gru = nn.GRUCell(latent_dim + ctrl_dim + ctx_dim, hidden_dim)
        self.prior_head = GaussianHead(hidden_dim + ctx_dim, latent_dim)

    def init_state(self, scene_ctx: torch.Tensor):
        return self.init_hidden(scene_ctx), self.init_latent(scene_ctx)

    def step(self, prev_latent, ctrl_t, scene_ctx, state):
        inp = torch.cat([prev_latent, ctrl_t, scene_ctx], dim=-1)
        state = self.gru(inp, state)
        mean, logvar = self.prior_head(torch.cat([state, scene_ctx], dim=-1))
        return state, mean, logvar
