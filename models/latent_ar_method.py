from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict

from models.components.constraints import build_constraints
from models.components.decoders import build_sequence_decoder
from models.components.dynamics import build_latent_ar_dynamics, gaussian_nll, reparameterize
from models.components.encoders import ActionCrossAttention, ForecastHistoryEncoder, SkeletonFrameEncoder
from models.forecast import BaseForecastMethod, LossOutput, PredictionOutput
from models.model_registry import register_model


class LatentARMethod(BaseForecastMethod):
    def __init__(self, cfg, logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.future_frames = cfg.future_frames
        self.num_agents = cfg.agents
        self.ctrl_dim = cfg.MODEL.get("COND_D_CUE", 0)
        self.frame_dim = cfg.MODEL.get("AGENT_DIM", 2)
        self.latent_dim = int(cfg.MODEL.get("LATENT_DIM", cfg.MODEL.get("COG_D_Z", 64)))
        self.hidden_dim = int(cfg.MODEL.get("LATENT_AR_HIDDEN_DIM", cfg.MODEL.CONTEXT_ENCODER.D_MODEL))
        self.loss_weights = cfg.get("BASELINE_LOSS_WEIGHTS", {"recon": 1.0, "latent_nll": 0.1})
        self.teacher_forcing = bool(cfg.get("LATENT_AR_TEACHER_FORCING", True))
        self.variant = str(cfg.METHOD.get("VARIANT", "gru"))

        self.history_encoder = ForecastHistoryEncoder(cfg.MODEL, cfg)
        self.frame_encoder = SkeletonFrameEncoder(
            num_agents=self.num_agents,
            frame_dim=self.frame_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )
        self.dynamics = self._build_dynamics()
        self.action_fusion = self._build_action_fusion()

        decoder_name = cfg.METHOD.get("DECODER", cfg.get("decoder_name", "moflow_structured"))
        self.decoder = build_sequence_decoder(
            name=decoder_name,
            model_cfg=cfg.MODEL,
            latent_dim=self.latent_dim,
            ctx_dim=cfg.MODEL.CONTEXT_ENCODER.D_MODEL,
            out_dim=self.frame_dim,
            num_agents=self.num_agents,
        )
        self.constraints = build_constraints(cfg, state_dim=self.latent_dim, ctrl_dim=self.ctrl_dim)

    def _action_query_dim(self):
        if self.variant == "gru":
            return self.hidden_dim
        if self.variant == "transformer":
            dynamics_cfg = self.cfg.MODEL.get("LATENT_AR_DYNAMICS", EasyDict())
            if not isinstance(dynamics_cfg, EasyDict):
                dynamics_cfg = EasyDict(dynamics_cfg)
            return int(dynamics_cfg.get("D_MODEL", self.hidden_dim))
        return self.hidden_dim

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
            query_dim=self._action_query_dim(),
            model_dim=int(fusion_cfg.get("D_MODEL", self.cfg.MODEL.CONTEXT_ENCODER.D_MODEL)),
            out_dim=self.ctrl_dim,
            num_heads=int(fusion_cfg.get("NUM_HEADS", 4)),
            num_layers=int(fusion_cfg.get("NUM_LAYERS", 1)),
            dropout=float(fusion_cfg.get("DROPOUT", 0.1)),
            max_seq_len=int(fusion_cfg.get("MAX_SEQ_LEN", self.future_frames * 2)),
            include_history=bool(fusion_cfg.get("INCLUDE_HISTORY", True)),
            use_raw_ctrl_residual=bool(fusion_cfg.get("USE_RAW_CTRL_RESIDUAL", True)),
        )

    def _build_dynamics(self):
        dynamics_cfg = self.cfg.MODEL.get("LATENT_AR_DYNAMICS", EasyDict())
        if not isinstance(dynamics_cfg, EasyDict):
            dynamics_cfg = EasyDict(dynamics_cfg)
        dynamics_name = dynamics_cfg.get("NAME", self.cfg.METHOD.get("VARIANT", "gru"))

        common_kwargs = dict(
            latent_dim=self.latent_dim,
            ctrl_dim=self.ctrl_dim,
            ctx_dim=self.cfg.MODEL.CONTEXT_ENCODER.D_MODEL,
        )
        if dynamics_name == "gru":
            return build_latent_ar_dynamics(
                "gru",
                hidden_dim=int(dynamics_cfg.get("HIDDEN_DIM", self.hidden_dim)),
                **common_kwargs,
            )
        if dynamics_name == "transformer":
            d_model = int(dynamics_cfg.get("D_MODEL", self.hidden_dim))
            return build_latent_ar_dynamics(
                "transformer",
                d_model=d_model,
                num_layers=int(dynamics_cfg.get("NUM_LAYERS", 2)),
                num_heads=int(dynamics_cfg.get("NUM_HEADS", 4)),
                dropout=float(dynamics_cfg.get("DROPOUT", 0.1)),
                ffn_dim=int(dynamics_cfg.get("FFN_DIM", d_model * 4)),
                max_seq_len=int(dynamics_cfg.get("MAX_SEQ_LEN", self.future_frames + 1)),
                **common_kwargs,
            )
        raise ValueError(f"Unsupported latent_ar dynamics variant '{dynamics_name}'")

    def _future_ctrl(self, batch):
        if "fut_cond_cue" in batch:
            return batch["fut_cond_cue"]
        hist = batch["hist_cond_cue"]
        return hist[:, -1:, :].expand(-1, self.future_frames, -1)

    def _state_query(self, dyn_state, prev_latent):
        if self.variant == "gru":
            return dyn_state
        if self.variant == "transformer":
            return dyn_state["memory"][:, -1, :]
        return prev_latent

    def _fuse_ctrl(self, batch, ctrl_seq, dyn_state, prev_latent, action_memory, t: int):
        ctrl_t = ctrl_seq[:, t]
        if self.action_fusion is None:
            return ctrl_t
        query = self._state_query(dyn_state, prev_latent)
        return self.action_fusion(action_memory, query, ctrl_t)

    def _teacher_forced_rollout(self, batch, scene_ctx, target_latents, ctrl_seq):
        dyn_state, prev_latent = self.dynamics.init_state(scene_ctx)
        action_memory = self.action_fusion.build_memory(batch, ctrl_seq) if self.action_fusion is not None else None
        mean_seq, logvar_seq = [], []
        for t in range(self.future_frames):
            ctrl_t = self._fuse_ctrl(batch, ctrl_seq, dyn_state, prev_latent, action_memory, t)
            dyn_state, mean_t, logvar_t = self.dynamics.step(prev_latent, ctrl_t, scene_ctx, dyn_state)
            mean_seq.append(mean_t)
            logvar_seq.append(logvar_t)
            prev_latent = target_latents[:, t] if self.teacher_forcing else mean_t

        return torch.stack(mean_seq, dim=1), torch.stack(logvar_seq, dim=1)

    def training_step(self, batch, log_dict=None) -> LossOutput:
        scene_ctx, agent_ctx, _ = self.history_encoder(batch)
        fut_traj = batch["fut_traj"]
        fut_seq = rearrange(fut_traj, "b a t d -> b t a d")
        ctrl_seq = self._future_ctrl(batch)

        target_latents = self.frame_encoder(fut_seq)
        prior_mean, prior_logvar = self._teacher_forced_rollout(batch, scene_ctx, target_latents, ctrl_seq)
        decoded = self.decoder(prior_mean.unsqueeze(1), agent_ctx, scene_ctx).squeeze(1)
        decoded = rearrange(decoded, "b a t d -> b a t d")

        recon = F.mse_loss(decoded, fut_traj)
        latent_nll = gaussian_nll(target_latents, prior_mean, prior_logvar).mean()

        trace = {
            "state_seq": prior_mean,
            "ctrl_seq": ctrl_seq,
            "decoded_seq": rearrange(decoded, "b a t d -> b t a d"),
            "scene_ctx": scene_ctx,
            "agent_ctx": agent_ctx,
        }
        constraint_loss, constraint_metrics = self.constraints(trace, batch, self)

        total = (
            float(self.loss_weights.get("recon", 1.0)) * recon
            + float(self.loss_weights.get("latent_nll", 0.1)) * latent_nll
            + constraint_loss
        )
        metrics = {"recon": recon, "kl": latent_nll, "constraint": constraint_loss}
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

        dyn_state, prev_latent = self.dynamics.init_state(scene_ctx_bk)
        action_memory = self.action_fusion.build_memory(batch_bk, ctrl_bk) if self.action_fusion is not None else None
        rollout = []
        for t in range(self.future_frames):
            ctrl_t = self._fuse_ctrl(batch_bk, ctrl_bk, dyn_state, prev_latent, action_memory, t)
            dyn_state, mean_t, logvar_t = self.dynamics.step(prev_latent, ctrl_t, scene_ctx_bk, dyn_state)
            prev_latent = reparameterize(mean_t, logvar_t, sample=True)
            rollout.append(prev_latent)

        latent_seq = torch.stack(rollout, dim=1).reshape(B, K, self.future_frames, self.latent_dim)
        decoded = self.decoder(latent_seq, agent_ctx, scene_ctx)
        trace_samples = decoded.unsqueeze(1) if return_trace else None
        trace_times = torch.linspace(1.0, 1.0, steps=1, device=decoded.device)
        scores = torch.zeros(B, K, self.num_agents, device=decoded.device, dtype=decoded.dtype)
        return PredictionOutput(
            samples=decoded,
            trace_samples=trace_samples,
            trace_times=trace_times,
            scores=scores,
            extras={"latent_seq": latent_seq},
        )


@register_model("latent_ar")
def build_latent_ar(cfg, args, logger):
    return LatentARMethod(cfg=cfg, logger=logger)
