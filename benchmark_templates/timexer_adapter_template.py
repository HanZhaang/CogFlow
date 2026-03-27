from __future__ import annotations

from typing import Any, Dict

import torch

from benchmark.registry import register_method
from benchmark_templates.adapter_template_base import ExternalMethodTemplateAdapter


# Remove this decorator until the file is copied into benchmark/adapters/.
@register_method("timexer")
class TimeXerTemplateAdapter(ExternalMethodTemplateAdapter):
    """
    Template for deterministic or autoregressive time-series baselines.

    Typical mapping:
    - training: loss = trainer_step(batch) or criterion(model(...), target)
    - inference: pred = model(...)
    """

    def __init__(self, cfg, args, logger):
        super().__init__(cfg=cfg, args=args, logger=logger)

        # TODO: replace with the real TimeXer imports and builders.
        # Example:
        # from timexer_repo.entry import build_model_and_loaders
        # self.model, self.optimizer, self.train_loader, self.val_loader = ...

    def forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        # TODO: map the unified batch into TimeXer inputs.
        # Example:
        # x_enc = batch["past_traj"]
        # x_mark_enc = batch["hist_cond_cue"]
        # x_dec = ...
        # target = batch["fut_traj"]
        # pred = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # return self.criterion(pred, target)
        raise NotImplementedError("Fill in TimeXer train forward here.")

    @torch.no_grad()
    def inference_step(self, batch: Dict[str, Any], num_samples: int) -> torch.Tensor:
        # TODO: deterministic models should usually run one forward pass only.
        # Then expand to [B, K, ...] if you need interface alignment.
        # pred = self.model(...)
        # pred = pred.view(B, 1, A, T, D) or similar
        # return self._repeat_deterministic_output(pred, num_samples)
        raise NotImplementedError("Fill in TimeXer inference here.")

    def get_inference_metadata(self, num_samples: int) -> Dict[str, Any]:
        horizon = int(self.cfg.future_frames)

        # TODO:
        # - If TimeXer is plain one-shot forecasting, use "1" or "1 (single-pass)".
        # - If it is autoregressive over horizon, use f"{horizon} (AR)".
        # For deterministic methods, you usually should not multiply runtime by K.
        return {
            "horizon": horizon,
            "steps_nfe": f"{horizon} (AR)",
        }
