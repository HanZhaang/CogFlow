from __future__ import annotations

from .common import gaussian_kl, gaussian_nll, reparameterize
from .latent_ar_gru import LatentARGRU
from .latent_ar_transformer import LatentARTransformer
from .rssm import RSSMDynamics


def build_latent_ar_dynamics(name: str, **kwargs):
    builders = {
        "gru": LatentARGRU,
        "transformer": LatentARTransformer,
    }
    if name not in builders:
        raise ValueError(f"Unsupported latent_ar dynamics '{name}'. Available: {sorted(builders)}")
    return builders[name](**kwargs)


__all__ = [
    "LatentARGRU",
    "LatentARTransformer",
    "build_latent_ar_dynamics",
    "RSSMDynamics",
    "gaussian_kl",
    "gaussian_nll",
    "reparameterize",
]
