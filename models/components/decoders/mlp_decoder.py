from __future__ import annotations

import torch
from torch import nn

from .utils import expand_latent_for_agents


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, ctx_dim: int, out_dim: int, hidden_dim: int, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.agent_proj = nn.Linear(ctx_dim, hidden_dim)
        self.scene_proj = nn.Linear(ctx_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, latent_seq: torch.Tensor, agent_ctx: torch.Tensor, scene_ctx: torch.Tensor) -> torch.Tensor:
        latent_tokens = expand_latent_for_agents(latent_seq, self.num_agents)
        B, K, A, T, _ = latent_tokens.shape

        lat = self.latent_proj(latent_tokens)
        agent = self.agent_proj(agent_ctx)[:, None, :, None, :].expand(B, K, A, T, -1)
        scene = self.scene_proj(scene_ctx)[:, None, None, None, :].expand(B, K, A, T, -1)
        return self.head(torch.cat([lat, agent, scene], dim=-1))
