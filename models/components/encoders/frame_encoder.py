from __future__ import annotations

import torch
from torch import nn


class SkeletonFrameEncoder(nn.Module):
    def __init__(self, num_agents: int, frame_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = num_agents * frame_dim
        self.num_agents = num_agents
        self.frame_dim = frame_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 3:
            return self.net(frames.reshape(frames.shape[0], -1))
        if frames.dim() != 4:
            raise ValueError(f"Expected [B,T,A,D] or [B,A,D], got {tuple(frames.shape)}")

        B, T, A, D = frames.shape
        assert A == self.num_agents and D == self.frame_dim
        x = frames.reshape(B * T, A * D)
        z = self.net(x)
        return z.reshape(B, T, -1)
