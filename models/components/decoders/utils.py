from __future__ import annotations

import torch


def expand_latent_for_agents(latent_seq: torch.Tensor, num_agents: int) -> torch.Tensor:
    if latent_seq.dim() == 5:
        return latent_seq
    if latent_seq.dim() != 4:
        raise ValueError(f"Expected [B,K,T,Z] or [B,K,A,T,Z], got {tuple(latent_seq.shape)}")
    return latent_seq.unsqueeze(2).expand(-1, -1, num_agents, -1, -1)
