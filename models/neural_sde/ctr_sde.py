# SPDX-License-Identifier: MIT
"""
controlled_sde.py

ControlledSSLSDE + simulate_sde_paths

用法示例（伪代码，放在你的 MoFlow model 里）:

    from controlled_sde import ControlledSSLSDE, simulate_sde_paths

    class MyMoFlowWithSDE(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.z_dim = cfg.z_dim
            self.stim_dim = cfg.stim_dim
            self.num_regimes = cfg.num_regimes

            self.sde = ControlledSSLSDE(
                z_dim=self.z_dim,
                stim_dim=self.stim_dim,
                num_regimes=self.num_regimes,
                num_bases=16,
                hidden_dim=64,
            )
            # 其他 MoFlow 组件略...

        def forward(self, x_hist, u_seq, ...):
            # 假设 u_seq: [B, T, stim_dim]
            B, T, _ = u_seq.shape
            z0 = torch.zeros(B, self.z_dim, device=u_seq.device, dtype=u_seq.dtype)
            z_seq = simulate_sde_paths(self.sde, z0, u_seq, dt=1.0 / 30.0)
            # z_seq: [B, T, z_dim]
            # 然后将 z_seq 作为条件输入 MoFlow decoder 即可
"""

from typing import Optional

import math
import torch
from torch import nn
import torch.nn.functional as F

import os
from datetime import datetime

class RegimePartition(nn.Module):
    """
    状态依赖的分段权重 π(z)，对应 S 个 regime。
    π(z) = softmax( (phi(z) @ W) / tau )

    这里借鉴 Hu 等人 SSL kernel 的思想，但用 MLP 来构造 φ(z)。
    """

    def __init__(
        self,
        z_dim: int,
        num_bases: int,
        num_regimes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        assert num_regimes >= 2, "num_regimes 必须 >= 2"

        self.z_dim = z_dim
        self.num_bases = num_bases
        self.num_regimes = num_regimes
        self.hidden_dim = hidden_dim

        # φ(z) 映射网络: z -> R^{num_bases}
        self.feature_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_bases),
            nn.Tanh(),
        )
        # W: [num_bases, num_regimes-1]
        # 最后一个 regime 的 logit 通过拼 0 得到，类似 SoftmaxCentered
        self.W = nn.Parameter(torch.randn(num_bases, num_regimes - 1) * 0.1)
        # 平滑温度 log_tau
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, z_dim]
        return:
            pi: [B, num_regimes]，每行 softmax 为 1
        """
        # φ(z): [B, num_bases]
        phi = self.feature_net(z)

        # [B, num_regimes-1]
        logits_centered = torch.matmul(phi, self.W)

        # 最后一列拼一个 0，实现“居中”的 softmax（最后一类为基准）
        last_col = torch.zeros(
            logits_centered.size(0),
            1,
            device=logits_centered.device,
            dtype=logits_centered.dtype,
        )
        logits = torch.cat([logits_centered, last_col], dim=-1)  # [B, S]

        tau = torch.exp(self.log_tau)
        pi = F.softmax(logits / tau, dim=-1)  # [B, S]

        return pi


import torch
import torch.nn as nn
import torch.nn.functional as F

class CommandEncoder(nn.Module):
    """
    Input command per step: 7D
        0-3: onehot of {none, fwd, left, right}
        4  : strength (0 when none)
        5  : signed_strength (left<0, right>0, fwd/none=0)
        6  : time_since_last_cmd (steps since last valid cmd; if no history, accum from 0)

    Output: continuous u suitable for subtraction and for:
        - level term:  B_lvl * u
        - event term : B_evt * (u_t - u_{t-1}) / dt
    """

    def __init__(
        self,
        emb_dim: int = 8,
        out_dim: int = 16,              # must match your sde.stim_dim
        normalize_emb: bool = True,
        # time feature design
        time_scale: float = 30.0,       # steps -> roughly seconds if 30Hz; adjust to your FPS
        tau: float = 30.0,              # decay timescale in steps (e.g., 30 steps ~ 1s at 30Hz)
        use_time_decay: bool = True,
        use_time_log: bool = True,
        dataset_type: str = "rat"
    ):
        super().__init__()
        self.emb = nn.Embedding(4, emb_dim)  # {none, fwd, left, right}
        self.normalize_emb = normalize_emb

        self.time_scale = float(time_scale)
        self.tau = float(tau)
        self.use_time_decay = use_time_decay
        self.use_time_log = use_time_log

        self.dataset_type = dataset_type

        
        if self.dataset_type == "rat":
            # feature dims: emb + strength + signed_strength + time_feats
            time_feat_dim = 0
            if use_time_decay:
                time_feat_dim += 1          # exp(-t/tau)
            if use_time_log:
                time_feat_dim += 1          # log1p(t)/log1p(time_scale)
            
            in_dim = emb_dim + 1 + 1 + time_feat_dim
            self.proj = nn.Linear(in_dim, out_dim)
        
        elif self.dataset_type == "babel":
            self.word_emb = nn.Embedding(8192, emb_dim)
            hidden_dim = 128
            bit_dim = emb_dim // 2  # 你也可以设成 emb_dim
            self.bit_mlp = nn.Sequential(
                nn.Linear(13, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bit_dim),
            )
            
            self.proj = nn.Linear(emb_dim + bit_dim, out_dim)
        # ----------------------------- #
        self._init_weights()

    def _init_weights(self):
        # small init for stable Δu
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        
        if self.dataset_type == "babel":
            for m in self.bit_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, cmd7: torch.Tensor) -> torch.Tensor:
        """
        cmd7: [B, T, 7] or [*, 7]
        returns u: [B, T, out_dim] or [*, out_dim]
        """
        assert cmd7.size(-1) in [7, 13], f"Expected last dim=7, got {cmd7.size(-1)}"
        if cmd7.size(-1) == 7:
            onehot = cmd7[..., 0:4]                 # [..., 4]
            strength = cmd7[..., 4:5]               # [..., 1]
            signed_strength = cmd7[..., 5:6]        # [..., 1]
            tslc = cmd7[..., 6:7]                   # [..., 1], steps since last valid cmd

            # command id from onehot (robust even if it's soft-ish)
            cmd_id = torch.argmax(onehot, dim=-1)   # [...]

            # has_cmd: 1 if not "none" else 0
            # category index 0 is "none" by your definition
            has_cmd = (cmd_id != 0).to(cmd7.dtype).unsqueeze(-1)  # [..., 1]

            # category embedding
            e = self.emb(cmd_id)                    # [..., emb_dim]
            if self.normalize_emb:
                e = F.normalize(e, dim=-1)

            # IMPORTANT: force "none" embedding to 0, so subtraction is meaningful
            # and "no command" does not introduce an arbitrary category vector.
            e = e * has_cmd                          # [..., emb_dim]

            # time features (continuous, bounded, scale-controlled)
            time_feats = []
            if self.use_time_decay:
                # exp decay: recent command -> ~1, long ago -> ~0
                # clamp to avoid overflow in exp for very large tslc
                t = torch.clamp(tslc, min=0.0, max=1e6)
                decay = torch.exp(-t / max(self.tau, 1e-6))      # [..., 1]
                time_feats.append(decay)

            if self.use_time_log:
                # normalized log time: grows slowly with tslc, stable scale
                # divide by log1p(time_scale) to keep magnitude around [0, ~1] for typical ranges
                denom = torch.log1p(torch.tensor(self.time_scale, device=cmd7.device, dtype=cmd7.dtype))
                logt = torch.log1p(torch.clamp(tslc, min=0.0)) / (denom + 1e-8)   # [..., 1]
                time_feats.append(logt)

            if len(time_feats) > 0:
                tfeat = torch.cat(time_feats, dim=-1)            # [..., time_feat_dim]
            else:
                tfeat = None

            # Assemble features.
            # Strength terms are already 0 when none (per your data definition),
            # so they naturally vanish without extra masking.
            feats = [e, strength, signed_strength]
            if tfeat is not None:
                feats.append(tfeat)

            x = torch.cat(feats, dim=-1)                         # [..., in_dim]
            u = self.proj(x)
            return u
        elif cmd7.size(-1) == 13:
            # cmd_bits: [..., 13]  float/bool  (0/1 code)
            cmd_bits = cmd7[..., 0:13].to(cmd7.dtype)

            # ---- (1) bits -> word_id (0..8191) ----
            # robust binarization (in case bits are soft-ish)
            bits01 = (cmd_bits > 0.5).to(torch.long)  # [..., 13]

            # compute integer id: sum_{b=0..12} bits[b] * 2^b
            # NOTE: choose consistent bit order with your dataset definition.
            # Here we treat cmd_bits[...,0] as LSB.
            weights = (2 ** torch.arange(13, device=cmd7.device)).to(torch.long)  # [13]
            word_id = (bits01 * weights).sum(dim=-1)  # [...], in [0, 8191]

            # ---- (2) token embedding ----
            # requires: self.word_emb = nn.Embedding(8192, emb_dim) in __init__
            e_tok = self.word_emb(word_id)  # [..., emb_dim]
            if self.normalize_emb:
                e_tok = F.normalize(e_tok, dim=-1)

            # ---- (3) bit-level continuous features (optional but recommended) ----
            # requires: self.bit_mlp = nn.Sequential(nn.Linear(13, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, bit_dim))
            # This yields a smooth embedding that changes "locally" when bits flip.
            e_bit = self.bit_mlp(cmd_bits)  # [..., bit_dim]
            # (optional) normalize to keep scale stable for du
            e_bit = F.layer_norm(e_bit, (e_bit.shape[-1],))

            # ---- (4) assemble ----
            # No strength / signed_strength / tslc for BABEL token code
            # (If you later add extra scalar features, append them here.)
            x = torch.cat([e_tok, e_bit], dim=-1)  # [..., emb_dim + bit_dim]
            u = self.proj(x)
            return u    


class ControlledSSLSDE(nn.Module):
    """
    Controlled Smooth Switching Linear SDE:

        dz_t = [ Σ_i π_i(z_t) (A_i z_t + a_i + B_i u_t) ] dt
               + Σ(z_t) dW_t

    - z_t: 认知隐变量 [B, z_dim]
    - u_t: 控制 / 刺激指令 [B, stim_dim]
    - π_i(z_t): RegimePartition 给出的 softmax 权重
    """

    def __init__(
        self,
        z_dim: int,
        stim_dim: int,
        num_regimes: int = 3,
        num_bases: int = 16,
        hidden_dim: int = 64,
        init_scale: float = 0.1,
        dataset_type: str = 'rat'
    ):
        super().__init__()

        self.z_dim = z_dim
        self.stim_dim = stim_dim
        self.num_regimes = num_regimes

        # 每个 regime 的线性动力 A_i, a_i, B_i
        # A: [S, z_dim, z_dim]
        # a: [S, z_dim]
        # B: [S, z_dim, stim_dim]
        self.A = nn.Parameter(torch.zeros(num_regimes, z_dim, z_dim))
        self.a = nn.Parameter(torch.zeros(num_regimes, z_dim))
        self.B = nn.Parameter(torch.zeros(num_regimes, z_dim, stim_dim))
        self.B_lvl = nn.Parameter(torch.zeros(num_regimes, z_dim, stim_dim))
        self.B_evt = nn.Parameter(torch.zeros(num_regimes, z_dim, stim_dim))

        self.cmd_encoder = CommandEncoder(out_dim=stim_dim, dataset_type=dataset_type)
        # 噪声强度 log_sigma: [S, z_dim]，Sigma_i = diag(exp(log_sigma_i))
        self.log_sigma = nn.Parameter(torch.zeros(num_regimes, z_dim))

        self.sigma_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.sigma_scale = nn.Parameter(torch.zeros(num_regimes, z_dim))  # per-regime scaling
        self.sigma_bias = nn.Parameter(torch.zeros(num_regimes, z_dim))  # per-regime bias

        self.sigma_min = 0.02  # 经验值：提升多样性常用
        self.sigma_max = 0.5  # 视 dt/稳定性调整

        # 状态依赖的 regime 权重 π(z)
        self.partition = RegimePartition(
            z_dim=z_dim,
            num_bases=num_bases,
            num_regimes=num_regimes,
            hidden_dim=hidden_dim,
        )
        self.sigma_gain = nn.Parameter(torch.tensor(1.0))

        self._init_params(init_scale)

    def _init_params(self, scale: float):
        """
        参数初始化：A 初始为接近稳定的小值，B 略小，a 为 0。
        """
        with torch.no_grad():
            # A 初始化为略微收缩的对角阵 + 小噪声
            eye = torch.eye(self.z_dim)
            for s in range(self.num_regimes):
                self.A[s].copy_(0.1 * eye + scale * torch.randn_like(self.A[s]))
                self.B[s].copy_(scale * torch.randn_like(self.B[s]))
                self.B_lvl[s].copy_(scale * torch.randn_like(self.B_lvl[s]))
                self.B_evt[s].copy_(0.5 * scale * torch.randn_like(self.B_evt[s]))
                self.a[s].zero_()
                self.log_sigma[s].fill_(math.log(0.1))
            nn.init.constant_(self.sigma_scale, 0.1)

    # def drift(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    #     """
    #     漂移项 f(z, u):

    #     z: [B, z_dim]
    #     u: [B, stim_dim]
    #     return:
    #         drift: [B, z_dim]
    #     """
    #     B = z.size(0)
    #     assert u.size(0) == B, "z, u batch size 不一致"

    #     # π(z): [B, S]
    #     pi = self.partition(z)  # [B, S]

    #     # 计算各 regime 下的 A_i z_t + B_i u_t + a_i
    #     # Az: [B, S, z_dim]
    #     Az = torch.einsum("sij,bj->bsi", self.A, z)
    #     # Bu: [B, S, z_dim]
    #     Bu = torch.einsum("sik,bk->bsi", self.B, u)
    #     # a: [1, S, z_dim]
    #     a = self.a.unsqueeze(0)

    #     # [B, S, z_dim]
    #     drift_regime = Az + Bu + a

    #     # 按 π(z) 加权求和: [B, z_dim]
    #     drift = torch.einsum("bs,bsk->bk", pi, drift_regime)

    #     return drift

    def drift(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        *,
        dt: float | None = None,
        u_prev: torch.Tensor | None = None,
        du: torch.Tensor | None = None,
        # weights to balance sustained vs event effects
        w_level: float = 1.0,
        w_event: float = 1.0,
        # safety/stability options
        clip_udot: float | None = None,
        contract_lam: float = 0.0,
    ) -> torch.Tensor:
        """
        Drift with dual-channel control (Level + Delta):

            dz/dt = f0(z) + f_level(z, u_t) + f_event(z, (u_t - u_{t-1})/dt)

        where
            f0(z)      = sum_i pi_i(z) (A_i z + a_i)
            f_level    = sum_i pi_i(z) (B_lvl_i * u_t)
            f_event    = sum_i pi_i(z) (B_evt_i * u_dot)

        Notes:
        - u must be continuous (e.g., [Embedding(category), intensity]).
        - If (u_prev, dt) or du is not provided, event term is set to 0 (backward compatible).
        - If self.B_lvl / self.B_evt do not exist, falls back to self.B for level and (optionally) event.

        Args:
            z: [B, z_dim]
            u: [B, stim_dim]   (continuous control vector)
            dt: time step size (float)
            u_prev: previous control [B, stim_dim]
            du: precomputed delta u [B, stim_dim] (overrides u_prev)
            w_level: weight for sustained control
            w_event: weight for event (delta) control
            clip_udot: optional clipping value for u_dot magnitude (for numerical stability)
            contract_lam: optional global contraction coefficient

        Returns:
            drift: [B, z_dim]
        """
        Bsz = z.size(0)
        assert u.size(0) == Bsz, "z, u batch size mismatch"

        # ---- regime weights pi(z): [B, S]
        pi = self.partition(z)  # [B, S]

        # ---- baseline term f0(z): sum_i pi_i(z) (A_i z + a_i)
        Az = torch.einsum("sij,bj->bsi", self.A, z)      # [B, S, z_dim]
        a = self.a.unsqueeze(0)                          # [1, S, z_dim]
        f0_regime = Az + a                               # [B, S, z_dim]
        f0 = torch.einsum("bs,bsk->bk", pi, f0_regime)   # [B, z_dim]

        # ---- choose parameter matrices
        # Preferred: self.B_lvl, self.B_evt
        # Fallback: self.B for level; for event reuse self.B unless B_evt exists
        B_lvl = getattr(self, "B_lvl", None)
        if B_lvl is None:
            B_lvl = getattr(self, "B", None)
            if B_lvl is None:
                raise AttributeError("No control matrix found: expected self.B_lvl or self.B")

        B_evt = getattr(self, "B_evt", None)
        if B_evt is None:
            # fallback: reuse level matrix (works, but less expressive)
            B_evt = B_lvl

        # ---- level control term: f_level(z, u)
        # regime-wise: B_lvl_i * u
        Bu_lvl = torch.einsum("sik,bk->bsi", B_lvl, u)          # [B, S, z_dim]
        f_level = torch.einsum("bs,bsk->bk", pi, Bu_lvl)        # [B, z_dim]

        # ---- event control term: f_event(z, u_dot)
        f_event = torch.zeros_like(f_level)
        has_event = (du is not None) or ((u_prev is not None) and (dt is not None))
        if has_event and (w_event != 0.0):
            if du is None:
                du = u - u_prev                                 # [B, stim_dim]
            assert dt is not None and dt > 0.0, "dt must be provided and > 0 when using event term"
            u_dot = du / dt                                     # [B, stim_dim]

            # optional clipping for stability (prevents rare huge spikes)
            if (clip_udot is not None) and (clip_udot > 0.0):
                u_dot = torch.clamp(u_dot, min=-clip_udot, max=clip_udot)

            Bu_evt = torch.einsum("sik,bk->bsi", B_evt, u_dot)   # [B, S, z_dim]
            f_event = torch.einsum("bs,bsk->bk", pi, Bu_evt)     # [B, z_dim]

        # ---- combine
        drift = f0 + (w_level * f_level) + (w_event * f_event)
                
        # optional: global contraction for long-horizon stability
        if contract_lam and contract_lam > 0.0:
            drift = drift - contract_lam * z

        return drift


    # def diffusion(self, z: torch.Tensor) -> torch.Tensor:
    #     """
    #     扩散项 Σ(z)，这里简化为对角矩阵 diag(sigma_eff(z)):
    #         sigma_eff(z) = Σ_i π_i(z) * exp(log_sigma_i)
    #
    #     z: [B, z_dim]
    #     return:
    #         sigma_eff: [B, z_dim]
    #     """
    #     # π(z): [B, S]
    #     pi = self.partition(z)
    #     # sigma: [S, z_dim]
    #     sigma = torch.exp(self.log_sigma)
    #
    #     # sigma_eff: [B, z_dim] = pi @ sigma
    #     sigma_eff = torch.einsum("bs,sk->bk", pi, sigma)
    #
    #     return sigma_eff

    def diffusion(self, z: torch.Tensor) -> torch.Tensor:
        pi = self.partition(z)  # [B,S]
        base = torch.exp(self.log_sigma)  # [S,z]

        # state-dependent residual (shared), then per-regime affine
        h = self.sigma_mlp(z)  # [B,z]
        # expand to regimes: [B,S,z]
        h_reg = self.sigma_gain * (h.unsqueeze(1) * self.sigma_scale.unsqueeze(0) + self.sigma_bias.unsqueeze(0))
        # print(h_reg)
        # positive residual multiplier
        # mult = torch.nn.functional.softplus(h_reg) + 1e-4  # [B,S,z], >=0
        # mult = 1.0 + 0.1 * torch.tanh(h_reg)  # (0.9, 1.1) 左右，可调 0.1
        log_sigma_reg = self.log_sigma.unsqueeze(0) + h_reg  # [B,S,z]
        sigma_reg = torch.exp(log_sigma_reg)
        # print(sigma_reg.norm())

        # sigma_reg = base.unsqueeze(0) * mult  # [B,S,z]
        sigma_eff = torch.einsum("bs,bsk->bk", pi, sigma_reg)

        # clamp to keep SDE stable + avoid collapse
        sigma_eff = torch.clamp(sigma_eff, min=self.sigma_min, max=self.sigma_max)
        return sigma_eff


    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        方便与其他模块组合的接口：
        给定当前 z,u，返回 drift 和 diffusion 的组合形式 f(z,u)。
        主要用于 ODE 形式时；SDE 仿真建议用 drift()/diffusion() 分开。
        """
        return self.drift(z, u)


def simulate_sde_paths(
    sde: ControlledSSLSDE,
    z0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    u_seq_encoded: torch.Tensor | None = None,
    return_terms: bool = False,
    use_encoded_controls: bool = True,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    使用 Euler–Maruyama 方法前向仿真 SDE，生成隐变量轨迹 z_seq。

    参数:
        sde: ControlledSSLSDE 实例
        z0: 初始隐变量 [B, z_dim]
        u_seq: 控制 / 刺激序列 [B, T, stim_dim]
        dt: 时间步长（例如 1/30 表示 30Hz）

    返回:
        z_seq: [B, T, z_dim]，每个时间步的隐变量 z_t
    """
    assert isinstance(sde, ControlledSSLSDE)
    device = z0.device
    dtype = z0.dtype

    B, T, stim_dim = u_seq.shape
    _, z_dim = z0.shape

    z = z0
    states = []
    zs = []
    drifts = []
    sigmas = []
    
    sqrt_dt = math.sqrt(dt)
    if use_encoded_controls:
        if u_seq_encoded is None:
            u_seq_encoded = sde.cmd_encoder(u_seq)
        control_seq = u_seq_encoded
    else:
        control_seq = u_seq

    for t in range(T):
        u_t = control_seq[:, t, :]  # [B, stim_dim]
        u_prev = control_seq[:, t - 1, :] if t > 0 else u_t  # t=0 => du=0
        states.append(z)
        drift = sde.drift(z, u_t, dt=dt, u_prev=u_prev, clip_udot=10)      # [B, z_dim]
        sigma = sde.diffusion(z)       # [B, z_dim]
        noise = torch.randn(B, z_dim, device=device, dtype=dtype)  # dW_t ~ N(0, dt)
        z = z + drift * dt + sigma * sqrt_dt * noise

        zs.append(z)
        drifts.append(drift)
        sigmas.append(sigma)
        
    z_seq = torch.stack(zs, dim=1)  # [B, T, z_dim]
    if not return_terms:
        return z_seq

    return {
        "state_seq": torch.stack(states, dim=1),  # [B, T, z_dim], aligned with drift/sigma
        "z_seq": z_seq,  # [B, T, z_dim], post-step rollout for decoder conditioning
        "drift_seq": torch.stack(drifts, dim=1),  # [B, T, z_dim]
        "sigma_seq": torch.stack(sigmas, dim=1),  # [B, T, z_dim]
        "u_seq": control_seq,
        "u_seq_raw": u_seq,
    }
    

if __name__ == "__main__":
    pass
