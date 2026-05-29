# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmaThetaCond(nn.Module):
    """
    Diagonal diffusion sigma_theta(Z, t, u) with optional conditioning.
    Z: (..., z_dim)
    t: (..., 1) or (...,)  (will be broadcast)
    u: (..., u_dim)
    """
    def __init__(
        self,
        z_dim: int,
        u_dim: int = 0,
        hidden_dim: int = 256,
        n_layers: int = 2,
        sigma_min: float = 1e-4,
        sigma_max: float = 5.0,
        softplus_beta: float = 1.0,
        use_time: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.use_time = use_time
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.softplus_beta = float(softplus_beta)

        in_dim = z_dim + (1 if use_time else 0) + (u_dim if u_dim > 0 else 0)

        layers = []
        d_in = in_dim
        for _ in range(max(n_layers, 1)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.SiLU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, z_dim))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Z: torch.Tensor, t: torch.Tensor | None = None, u: torch.Tensor | None = None) -> torch.Tensor:
        xs = [Z]
        if self.use_time:
            if t is None:
                raise ValueError("use_time=True but t is None")
            if t.dim() == Z.dim() - 1:
                t = t.unsqueeze(-1)
            t = t.expand(*Z.shape[:-1], 1)
            xs.append(t)
        if self.u_dim > 0:
            if u is None:
                raise ValueError("u_dim>0 but u is None")
            if u.dim() == Z.dim() - 1:
                u = u.unsqueeze(-1)
            u = u.expand(*Z.shape[:-1], self.u_dim)
            xs.append(u)

        inp = torch.cat(xs, dim=-1)
        sigma_logits = self.net(inp)
        sigma = F.softplus(sigma_logits, beta=self.softplus_beta)
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        return sigma
