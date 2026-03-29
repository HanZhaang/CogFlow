from __future__ import annotations

import torch
from torch import nn

from models.context_encoder import build_context_encoder


class ForecastHistoryEncoder(nn.Module):
    def __init__(self, model_cfg, cfg):
        super().__init__()
        use_pre_norm = model_cfg.get("USE_PRE_NORM", False)
        self.context_encoder = build_context_encoder(
            model_cfg.CONTEXT_ENCODER, use_pre_norm=use_pre_norm, device=cfg.device
        )
        self.d_model = model_cfg.CONTEXT_ENCODER.D_MODEL
        cue_dim = model_cfg.get("COND_D_CUE", 0)

        if cue_dim > 0:
            self.ctrl_gru = nn.GRU(
                input_size=cue_dim,
                hidden_size=self.d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.scene_proj = nn.Linear(self.d_model * 2, self.d_model)
            self.agent_proj = nn.Linear(self.d_model * 2, self.d_model)
        else:
            self.ctrl_gru = None
            self.scene_proj = nn.Identity()
            self.agent_proj = nn.Identity()

    def forward(self, batch):
        past = batch.get("past_traj_original_scale", batch["past_traj"])
        agent_ctx = self.context_encoder(past)
        scene_ctx = agent_ctx.mean(dim=1)

        ctrl_summary = None
        hist_ctrl = batch.get("hist_cond_cue")
        if self.ctrl_gru is not None and hist_ctrl is not None:
            _, h = self.ctrl_gru(hist_ctrl)
            ctrl_summary = torch.cat([h[0], h[1]], dim=-1)
            scene_ctx = self.scene_proj(torch.cat([scene_ctx, ctrl_summary], dim=-1))
            ctrl_expand = ctrl_summary[:, None, :].expand(-1, agent_ctx.shape[1], -1)
            agent_ctx = self.agent_proj(torch.cat([agent_ctx, ctrl_expand], dim=-1))

        return scene_ctx, agent_ctx, {"ctrl_summary": ctrl_summary}
