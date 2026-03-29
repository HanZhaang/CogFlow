from __future__ import annotations

import torch
from torch import nn

from .common import GaussianHead


class LatentARTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        ctrl_dim: int,
        ctx_dim: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_dim: int | None = None,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.init_latent = nn.Linear(ctx_dim, latent_dim)
        self.init_token = nn.Linear(ctx_dim, d_model)
        self.token_proj = nn.Linear(latent_dim + ctrl_dim + ctx_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim or d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.prior_head = GaussianHead(d_model + ctx_dim, latent_dim)

    def init_state(self, scene_ctx: torch.Tensor):
        init_memory = self.init_token(scene_ctx).unsqueeze(1)
        state = {"memory": init_memory}
        return state, self.init_latent(scene_ctx)

    def step(self, prev_latent, ctrl_t, scene_ctx, state):
        token_t = self.token_proj(torch.cat([prev_latent, ctrl_t, scene_ctx], dim=-1)).unsqueeze(1)
        memory = torch.cat([state["memory"], token_t], dim=1)
        if memory.shape[1] > self.max_seq_len:
            memory = memory[:, -self.max_seq_len :, :]

        seq_len = memory.shape[1]
        pos_ids = torch.arange(seq_len, device=memory.device)
        x = memory + self.pos_embedding(pos_ids).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=memory.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(x, mask=causal_mask)
        last_hidden = self.out_norm(hidden[:, -1, :])
        mean, logvar = self.prior_head(torch.cat([last_hidden, scene_ctx], dim=-1))
        return {"memory": memory}, mean, logvar
