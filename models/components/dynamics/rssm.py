from __future__ import annotations

import torch
from torch import nn

from .common import GaussianHead


class RSSMDynamics(nn.Module):
    def __init__(self, stoch_dim: int, ctrl_dim: int, ctx_dim: int, det_dim: int, obs_dim: int):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.det_dim = det_dim
        self.init_h = nn.Linear(ctx_dim, det_dim)
        self.init_s = nn.Linear(ctx_dim, stoch_dim)
        self.det_cell = nn.GRUCell(stoch_dim + ctrl_dim + ctx_dim, det_dim)
        self.prior_head = GaussianHead(det_dim + ctx_dim, stoch_dim)
        self.post_head = GaussianHead(det_dim + ctx_dim + obs_dim, stoch_dim)

    def init_state(self, scene_ctx):
        return self.init_h(scene_ctx), self.init_s(scene_ctx)

    def det_step(self, prev_s, ctrl_t, scene_ctx, prev_h):
        return self.det_cell(torch.cat([prev_s, ctrl_t, scene_ctx], dim=-1), prev_h)

    def prior(self, h_t, scene_ctx):
        return self.prior_head(torch.cat([h_t, scene_ctx], dim=-1))

    def posterior(self, h_t, obs_feat, scene_ctx):
        return self.post_head(torch.cat([h_t, scene_ctx, obs_feat], dim=-1))
