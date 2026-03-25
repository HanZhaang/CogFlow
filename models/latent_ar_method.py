from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from models.components.constraints import build_constraints
from models.components.decoders import build_sequence_decoder
from models.components.dynamics import LatentARGRU, gaussian_nll, reparameterize
from models.components.encoders import ForecastHistoryEncoder, SkeletonFrameEncoder
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

        self.history_encoder = ForecastHistoryEncoder(cfg.MODEL, cfg)
        self.frame_encoder = SkeletonFrameEncoder(
            num_agents=self.num_agents,
            frame_dim=self.frame_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )
        self.dynamics = LatentARGRU(
            latent_dim=self.latent_dim,
            ctrl_dim=self.ctrl_dim,
            ctx_dim=cfg.MODEL.CONTEXT_ENCODER.D_MODEL,
            hidden_dim=self.hidden_dim,
        )

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

    def _future_ctrl(self, batch):
        if "fut_cond_cue" in batch:
            return batch["fut_cond_cue"]
        hist = batch["hist_cond_cue"]
        return hist[:, -1:, :].expand(-1, self.future_frames, -1)

    def _teacher_forced_rollout(self, scene_ctx, target_latents, ctrl_seq):
        hidden, prev_latent = self.dynamics.init_state(scene_ctx)
        mean_seq, logvar_seq = [], []
        for t in range(self.future_frames):
            hidden, mean_t, logvar_t = self.dynamics.step(prev_latent, ctrl_seq[:, t], scene_ctx, hidden)
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
        prior_mean, prior_logvar = self._teacher_forced_rollout(scene_ctx, target_latents, ctrl_seq)
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

        hidden, prev_latent = self.dynamics.init_state(scene_ctx_bk)
        rollout = []
        for t in range(self.future_frames):
            hidden, mean_t, logvar_t = self.dynamics.step(prev_latent, ctrl_bk[:, t], scene_ctx_bk, hidden)
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
