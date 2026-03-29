from __future__ import annotations

import torch
from torch import nn

from models.motion_decoder import build_decoder
from models.utils.common_layers import build_mlps

from .utils import expand_latent_for_agents


class StructuredMoFlowDecoder(nn.Module):
    def __init__(self, model_cfg, latent_dim: int, ctx_dim: int, out_dim: int, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        self.d_model = model_cfg.MOTION_DECODER.D_MODEL
        self.time_chunk_size = int(model_cfg.get("DECODER_TIME_CHUNK", 0))
        self.latent_proj = nn.Linear(latent_dim, self.d_model)
        self.agent_proj = nn.Linear(ctx_dim, self.d_model)
        self.scene_proj = nn.Linear(ctx_dim, self.d_model)
        self.motion_query_embedding = nn.Embedding(model_cfg.NUM_PROPOSED_QUERY, self.d_model)
        self.agent_order_embedding = nn.Embedding(num_agents, self.d_model)
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.post_pe_cat_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.motion_decoder = build_decoder(model_cfg.MOTION_DECODER, use_pre_norm=False, use_adaln=False)
        self.reg_head = build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model, out_dim],
            ret_before_act=True,
            without_norm=True,
        )

    def _decode_chunk(self, fused: torch.Tensor, k_pe: torch.Tensor, a_pe: torch.Tensor) -> torch.Tensor:
        query_token = self.post_pe_cat_mlp(fused + k_pe + a_pe)
        readout_token = self.motion_decoder(query_token)
        return self.reg_head(readout_token)

    def forward(self, latent_seq: torch.Tensor, agent_ctx: torch.Tensor, scene_ctx: torch.Tensor) -> torch.Tensor:
        latent_tokens = expand_latent_for_agents(latent_seq, self.num_agents)
        B, K, A, T, _ = latent_tokens.shape

        lat = self.latent_proj(latent_tokens)
        agent = self.agent_proj(agent_ctx)[:, None, :, None, :].expand(B, K, A, T, -1)
        scene = self.scene_proj(scene_ctx)[:, None, None, None, :].expand(B, K, A, T, -1)
        fused = self.init_emb_fusion_mlp(torch.cat([lat, agent, scene], dim=-1))

        k_pe = self.motion_query_embedding(torch.arange(K, device=latent_seq.device))
        a_pe = self.agent_order_embedding(torch.arange(A, device=latent_seq.device))
        k_pe = k_pe[None, :, None, None, :]
        a_pe = a_pe[None, None, :, None, :]

        if self.time_chunk_size <= 0 or T <= self.time_chunk_size:
            return self._decode_chunk(fused, k_pe, a_pe)

        outputs = []
        for start in range(0, T, self.time_chunk_size):
            end = min(start + self.time_chunk_size, T)
            outputs.append(self._decode_chunk(fused[:, :, :, start:end], k_pe, a_pe))
        return torch.cat(outputs, dim=3)
