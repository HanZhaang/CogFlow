from __future__ import annotations

from typing import Dict

import torch

from benchmark.registry import register_method
from benchmark_templates.adapter_template_base import ExternalMethodTemplateAdapter


# Remove this decorator until the file is copied into benchmark/adapters/.
@register_method("sld_hmp")
class SLDHMPTemplateAdapter(ExternalMethodTemplateAdapter):
    """
    Template for methods with a richer sample() path, e.g.:
    - context/history encoder
    - latent rollout
    - stochastic decoder
    - multi-sample generation
    """

    def __init__(self, cfg, args, logger):
        super().__init__(cfg=cfg, args=args, logger=logger)

        # TODO: replace with the real SLD-HMP model/trainer constructors.
        # Keep the highest-level callable you have, usually trainer.sample()
        # or model.predict().

    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO:
        # Call the real training entry for SLD-HMP and return a scalar loss.
        raise NotImplementedError("Fill in SLD-HMP train forward here.")

    @torch.no_grad()
    def inference_step(self, batch: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        # TODO:
        # This should cover the full public inference path, not a submodule.
        # Good:
        #   pred = self.model.predict(batch, num_samples=num_samples)
        #   pred = self.trainer.sample(batch, K=num_samples)
        #
        # Bad:
        #   latent = self.encoder(...)
        #   pred = self.decoder(latent)
        # if the real method normally does more than that.
        raise NotImplementedError("Fill in SLD-HMP inference here.")

    def get_inference_metadata(self, num_samples: int) -> Dict[str, str]:
        horizon = int(self.cfg.future_frames)

        # TODO:
        # Example for SDE-style rollout:
        #   f"{horizon} (SDE)" or f"{horizon}x{num_samples} (SDE)"
        return {
            "horizon": horizon,
            "steps_nfe": "TODO",
        }
