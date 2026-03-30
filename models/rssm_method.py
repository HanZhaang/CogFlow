from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict

from models.components.constraints import build_constraints
from models.components.decoders import build_sequence_decoder
from models.components.dynamics import RSSMDynamics, gaussian_kl, reparameterize
from models.components.encoders import ActionCrossAttention, ForecastHistoryEncoder, SkeletonFrameEncoder
from models.forecast import BaseForecastMethod, LossOutput, PredictionOutput
from models.model_registry import register_model


class RSSMMethod(BaseForecastMethod):
    def __init__(self, cfg, logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.future_frames = cfg.future_frames
        self.num_agents = cfg.agents
        self.frame_dim = cfg.MODEL.get("AGENT_DIM", 2)
        self.ctrl_dim = cfg.MODEL.get("COND_D_CUE", 0)
        self.stoch_dim = int(cfg.MODEL.get("RSSM_STOCH_DIM", cfg.MODEL.get("COG_D_Z", 64)))
        self.det_dim = int(cfg.MODEL.get("RSSM_DET_DIM", cfg.MODEL.CONTEXT_ENCODER.D_MODEL))
        self.obs_dim = int(cfg.MODEL.get("RSSM_OBS_DIM", self.stoch_dim))
        self.decoder_latent_dim = int(cfg.MODEL.get("RSSM_DECODER_LATENT_DIM", self.stoch_dim))
        self.kl_beta = float(cfg.get("RSSM_KL_BETA", 0.1))

        self.history_encoder = ForecastHistoryEncoder(cfg.MODEL, cfg)
        self.obs_encoder = SkeletonFrameEncoder(
            num_agents=self.num_agents,
            frame_dim=self.frame_dim,
            latent_dim=self.obs_dim,
            hidden_dim=self.det_dim,
        )
        self.dynamics = RSSMDynamics(
            stoch_dim=self.stoch_dim,
            ctrl_dim=self.ctrl_dim,
            ctx_dim=cfg.MODEL.CONTEXT_ENCODER.D_MODEL,
            det_dim=self.det_dim,
            obs_dim=self.obs_dim,
        )
        self.action_fusion = self._build_action_fusion()
        self.state_proj = torch.nn.Linear(self.det_dim + self.stoch_dim, self.decoder_latent_dim)

        decoder_name = cfg.METHOD.get("DECODER", cfg.get("decoder_name", "moflow_structured"))
        self.decoder = build_sequence_decoder(
            name=decoder_name,
            model_cfg=cfg.MODEL,
            latent_dim=self.decoder_latent_dim,
            ctx_dim=cfg.MODEL.CONTEXT_ENCODER.D_MODEL,
            out_dim=self.frame_dim,
            num_agents=self.num_agents,
        )
        self.constraints = build_constraints(
            cfg, state_dim=self.det_dim + self.stoch_dim, ctrl_dim=self.ctrl_dim
        )

    def _build_action_fusion(self):
        fusion_cfg = self.cfg.MODEL.get("ACTION_FUSION", EasyDict())
        if not isinstance(fusion_cfg, EasyDict):
            fusion_cfg = EasyDict(fusion_cfg)
        fusion_name = str(fusion_cfg.get("NAME", "none"))
        if fusion_name in {"none", "", "null"} or self.ctrl_dim <= 0:
            return None
        if fusion_name != "cross_attention":
            raise ValueError(f"Unsupported action fusion '{fusion_name}'")
        return ActionCrossAttention(
            action_dim=self.ctrl_dim,
            query_dim=self.det_dim + self.stoch_dim,
            model_dim=int(fusion_cfg.get("D_MODEL", self.cfg.MODEL.CONTEXT_ENCODER.D_MODEL)),
            out_dim=self.ctrl_dim,
            num_heads=int(fusion_cfg.get("NUM_HEADS", 4)),
            num_layers=int(fusion_cfg.get("NUM_LAYERS", 1)),
            dropout=float(fusion_cfg.get("DROPOUT", 0.1)),
            max_seq_len=int(fusion_cfg.get("MAX_SEQ_LEN", self.future_frames * 2)),
            include_history=bool(fusion_cfg.get("INCLUDE_HISTORY", True)),
            use_raw_ctrl_residual=bool(fusion_cfg.get("USE_RAW_CTRL_RESIDUAL", True)),
        )

    def _future_ctrl(self, batch):
        if "fut_cond_cue" in batch:
            return batch["fut_cond_cue"]
        hist = batch["hist_cond_cue"]
        return hist[:, -1:, :].expand(-1, self.future_frames, -1)

    def _fuse_ctrl(self, batch, ctrl_seq, h_t, s_t, action_memory, t: int):
        ctrl_t = ctrl_seq[:, t]
        if self.action_fusion is None:
            return ctrl_t
        query = torch.cat([h_t, s_t], dim=-1)
        return self.action_fusion(action_memory, query, ctrl_t)

    def training_step(self, batch, log_dict=None) -> LossOutput:
        scene_ctx, agent_ctx, _ = self.history_encoder(batch)
        fut_traj = batch["fut_traj"]
        fut_seq = rearrange(fut_traj, "b a t d -> b t a d")
        ctrl_seq = self._future_ctrl(batch)
        obs_seq = self.obs_encoder(fut_seq)

        h_t, s_t = self.dynamics.init_state(scene_ctx)
        action_memory = self.action_fusion.build_memory(batch, ctrl_seq) if self.action_fusion is not None else None
        joint_states, h_seq, s_seq, kl_terms = [], [], [], []
        for t in range(self.future_frames):
            ctrl_t = self._fuse_ctrl(batch, ctrl_seq, h_t, s_t, action_memory, t)
            h_t = self.dynamics.det_step(s_t, ctrl_t, scene_ctx, h_t)
            prior_mean, prior_logvar = self.dynamics.prior(h_t, scene_ctx)
            post_mean, post_logvar = self.dynamics.posterior(h_t, obs_seq[:, t], scene_ctx)
            s_t = reparameterize(post_mean, post_logvar, sample=True)
            joint_states.append(torch.cat([h_t, s_t], dim=-1))
            h_seq.append(h_t)
            s_seq.append(s_t)
            kl_terms.append(gaussian_kl(post_mean, post_logvar, prior_mean, prior_logvar))

        joint_seq = torch.stack(joint_states, dim=1)
        decoder_latents = self.state_proj(joint_seq)
        decoded = self.decoder(decoder_latents.unsqueeze(1), agent_ctx, scene_ctx).squeeze(1)
        decoded = rearrange(decoded, "b a t d -> b a t d")

        recon = F.mse_loss(decoded, fut_traj)
        kl = torch.stack(kl_terms, dim=1).mean()

        trace = {
            "state_seq": joint_seq,
            "det_state_seq": torch.stack(h_seq, dim=1),
            "stoch_state_seq": torch.stack(s_seq, dim=1),
            "ctrl_seq": ctrl_seq,
            "decoded_seq": rearrange(decoded, "b a t d -> b t a d"),
            "scene_ctx": scene_ctx,
            "agent_ctx": agent_ctx,
        }
        constraint_loss, constraint_metrics = self.constraints(trace, batch, self)
        total = recon + self.kl_beta * kl + constraint_loss
        metrics = {"recon": recon, "kl": kl, "constraint": constraint_loss}
        metrics.update(constraint_metrics)
        return LossOutput(total=total, metrics=metrics)

    def predict(self, batch, num_samples: int, return_trace: bool = False) -> PredictionOutput:
        scene_ctx, agent_ctx, _ = self.history_encoder(batch)
        ctrl_seq = self._future_ctrl(batch)
        B = scene_ctx.shape[0]
        K = num_samples

        scene_ctx_bk = scene_ctx[:, None, :].expand(B, K, -1).reshape(B * K, -1)
        ctrl_bk = ctrl_seq[:, None, :, :].expand(B, K, -1, -1).reshape(B * K, self.future_frames, -1)
        batch_bk = dict(batch)
        if "hist_cond_cue" in batch:
            batch_bk["hist_cond_cue"] = batch["hist_cond_cue"][:, None, :, :].expand(B, K, -1, -1).reshape(B * K, batch["hist_cond_cue"].shape[1], -1)
        if "fut_cond_cue" in batch:
            batch_bk["fut_cond_cue"] = ctrl_bk

        h_t, s_t = self.dynamics.init_state(scene_ctx_bk)
        action_memory = self.action_fusion.build_memory(batch_bk, ctrl_bk) if self.action_fusion is not None else None
        joint_states = []
        for t in range(self.future_frames):
            ctrl_t = self._fuse_ctrl(batch_bk, ctrl_bk, h_t, s_t, action_memory, t)
            h_t = self.dynamics.det_step(s_t, ctrl_t, scene_ctx_bk, h_t)
            prior_mean, prior_logvar = self.dynamics.prior(h_t, scene_ctx_bk)
            s_t = reparameterize(prior_mean, prior_logvar, sample=True)
            joint_states.append(torch.cat([h_t, s_t], dim=-1))

        joint_seq = torch.stack(joint_states, dim=1).reshape(B, K, self.future_frames, -1)
        decoder_latents = self.state_proj(joint_seq)
        decoded = self.decoder(decoder_latents, agent_ctx, scene_ctx)
        trace_samples = decoded.unsqueeze(1) if return_trace else None
        trace_times = torch.linspace(1.0, 1.0, steps=1, device=decoded.device)
        scores = torch.zeros(B, K, self.num_agents, device=decoded.device, dtype=decoded.dtype)
        return PredictionOutput(
            samples=decoded,
            trace_samples=trace_samples,
            trace_times=trace_times,
            scores=scores,
            extras={"joint_state_seq": joint_seq},
        )


@register_model("rssm")
def build_rssm(cfg, args, logger):
    return RSSMMethod(cfg=cfg, logger=logger)
