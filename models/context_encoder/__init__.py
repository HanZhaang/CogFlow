# SPDX-License-Identifier: MIT
from .mtr_encoder import MTREncoder

__all__ = {
    'MTREncoder': MTREncoder,
}


def build_context_encoder(config, use_pre_norm, device):
    model = __all__[config.NAME](
        config=config,
        use_pre_norm=use_pre_norm,
        device=device
    ).to(device=device)

    return model
