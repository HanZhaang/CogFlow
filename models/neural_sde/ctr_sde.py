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

        # 噪声强度 log_sigma: [S, z_dim]，Sigma_i = diag(exp(log_sigma_i))
        self.log_sigma = nn.Parameter(torch.zeros(num_regimes, z_dim))

        # 状态依赖的 regime 权重 π(z)
        self.partition = RegimePartition(
            z_dim=z_dim,
            num_bases=num_bases,
            num_regimes=num_regimes,
            hidden_dim=hidden_dim,
        )

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
                self.a[s].zero_()
                self.log_sigma[s].fill_(math.log(0.1))

    def drift(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        漂移项 f(z, u):

        z: [B, z_dim]
        u: [B, stim_dim]
        return:
            drift: [B, z_dim]
        """
        B = z.size(0)
        assert u.size(0) == B, "z, u batch size 不一致"

        # π(z): [B, S]
        pi = self.partition(z)  # [B, S]

        # 计算各 regime 下的 A_i z_t + B_i u_t + a_i
        # Az: [B, S, z_dim]
        Az = torch.einsum("sij,bj->bsi", self.A, z)
        # Bu: [B, S, z_dim]
        Bu = torch.einsum("sik,bk->bsi", self.B, u)
        # a: [1, S, z_dim]
        a = self.a.unsqueeze(0)

        # [B, S, z_dim]
        drift_regime = Az + Bu + a

        # 按 π(z) 加权求和: [B, z_dim]
        drift = torch.einsum("bs,bsk->bk", pi, drift_regime)

        return drift

    def diffusion(self, z: torch.Tensor) -> torch.Tensor:
        """
        扩散项 Σ(z)，这里简化为对角矩阵 diag(sigma_eff(z)):
            sigma_eff(z) = Σ_i π_i(z) * exp(log_sigma_i)

        z: [B, z_dim]
        return:
            sigma_eff: [B, z_dim]
        """
        # π(z): [B, S]
        pi = self.partition(z)
        # sigma: [S, z_dim]
        sigma = torch.exp(self.log_sigma)

        # sigma_eff: [B, z_dim] = pi @ sigma
        sigma_eff = torch.einsum("bs,sk->bk", pi, sigma)

        return sigma_eff

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        方便与其他模块组合的接口：
        给定当前 z,u，返回 drift 和 diffusion 的组合形式 f(z,u)。
        主要用于 ODE 形式时；SDE 仿真建议用 drift()/diffusion() 分开。
        """
        return self.drift(z, u)


@torch.no_grad()
def simulate_sde_paths(
    sde: ControlledSSLSDE,
    z0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
) -> torch.Tensor:
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
    zs = []

    sqrt_dt = math.sqrt(dt)

    for t in range(T):
        u_t = u_seq[:, t, :]  # [B, stim_dim]

        drift = sde.drift(z, u_t)      # [B, z_dim]
        sigma = sde.diffusion(z)       # [B, z_dim]

        noise = torch.randn(B, z_dim, device=device, dtype=dtype)  # dW_t ~ N(0, dt)
        z = z + drift * dt + sigma * sqrt_dt * noise

        zs.append(z)

    z_seq = torch.stack(zs, dim=1)  # [B, T, z_dim]
    return z_seq

if __name__ == "__main__":
    pass