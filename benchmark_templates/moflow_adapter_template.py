from __future__ import annotations

from typing import Dict

import torch

from benchmark.registry import register_method
from benchmark_templates.adapter_template_base import ExternalMethodTemplateAdapter


# Remove this decorator until the file is copied into benchmark/adapters/.
@register_method("moflow")
class MoFlowTemplateAdapter(ExternalMethodTemplateAdapter):
    """
    Template for MoFlow-style methods.

    Use this for:
    - one-step student model
    - flow-matching teacher model
    - other variants with a direct sample() API
    """

    def __init__(self, cfg, args, logger):
        super().__init__(cfg=cfg, args=args, logger=logger)

        # TODO: replace with the actual MoFlow repo build path.
        # Decide whether this adapter targets teacher or student.
        # Keep them separate if their NFE and latency differ materially.

    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO:
        # - teacher: return FM training loss
        # - student: return IMLE/distillation loss
        raise NotImplementedError("Fill in MoFlow train forward here.")

    @torch.no_grad()
    def inference_step(self, batch: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        # TODO:
        # pred = self.model.sample(batch, num_trajs=num_samples)
        # or pred = self.model.predict(batch, num_samples=num_samples)
        # Return only the predicted samples tensor.
        raise NotImplementedError("Fill in MoFlow inference here.")

    def get_inference_metadata(self, num_samples: int) -> Dict[str, str]:
        horizon = int(self.cfg.future_frames)

        # TODO:
        # teacher example: f"{self.cfg.sampling_steps}"
        # student example: "1"
        # If sampling is repeated K times internally, reflect that clearly.
        return {
            "horizon": horizon,
            "steps_nfe": "TODO",
        }
