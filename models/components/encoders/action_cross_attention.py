from __future__ import annotations

import torch
from torch import nn


class ActionCrossAttention(nn.Module):
    def __init__(
        self,
        action_dim: int,
        query_dim: int,
        model_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        include_history: bool = True,
        use_raw_ctrl_residual: bool = True,
    ):
        super().__init__()
        self.include_history = include_history
        self.use_raw_ctrl_residual = use_raw_ctrl_residual
        self.max_seq_len = max_seq_len

        self.action_proj = nn.Linear(action_dim, model_dim)
        self.query_proj = nn.Linear(query_dim, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if use_raw_ctrl_residual:
            self.raw_ctrl_proj = nn.Linear(action_dim, model_dim)
            self.out_proj = nn.Linear(model_dim * 2, out_dim)
        else:
            self.raw_ctrl_proj = None
            self.out_proj = nn.Linear(model_dim, out_dim)

    def build_memory(self, batch, fut_ctrl: torch.Tensor):
        ctrl_seq = fut_ctrl
        hist_ctrl = batch.get("hist_cond_cue")
        if self.include_history and hist_ctrl is not None:
            ctrl_seq = torch.cat([hist_ctrl, fut_ctrl], dim=1)

        if ctrl_seq.shape[1] > self.max_seq_len:
            ctrl_seq = ctrl_seq[:, -self.max_seq_len :, :]

        pos_ids = torch.arange(ctrl_seq.shape[1], device=ctrl_seq.device)
        tokens = self.action_proj(ctrl_seq) + self.pos_embedding(pos_ids).unsqueeze(0)
        return self.action_encoder(tokens)

    def forward(self, memory: torch.Tensor, query: torch.Tensor, current_ctrl: torch.Tensor | None = None):
        query_token = self.query_proj(query).unsqueeze(1)
        attn_out, _ = self.cross_attn(query_token, memory, memory, need_weights=False)
        attn_out = attn_out.squeeze(1)
        if self.use_raw_ctrl_residual and current_ctrl is not None:
            raw = self.raw_ctrl_proj(current_ctrl)
            return self.out_proj(torch.cat([attn_out, raw], dim=-1))
        return self.out_proj(attn_out)
