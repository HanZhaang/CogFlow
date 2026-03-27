from __future__ import annotations

from typing import Any, Dict

import torch

from benchmark.base_adapter import MethodAdapter
from benchmark.registry import register_method
from data.dataset_registry import build_data_loader
from models.model_registry import build_network
from trainer.trainer_registry import build_trainer


class _ForecastAdapter(MethodAdapter):
    def __init__(self, cfg, args, logger):
        super().__init__(cfg=cfg, args=args, logger=logger)
        self.train_loader, self.val_loader = build_data_loader(cfg, args)
        self.model = build_network(cfg, args, logger)
        self.trainer = build_trainer(
            cfg=cfg,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            tb_log=None,
            logger=logger,
        )

    @property
    def device(self) -> str:
        return self.trainer.device

    @property
    def raw_model(self):
        return self.trainer.accelerator.unwrap_model(self.trainer.denoiser)

    def get_batch(self, split: str, batch_index: int = 0) -> Dict[str, Any]:
        loader = self.trainer.train_loader if split == "train" else self.trainer.val_loader
        iterator = iter(loader)
        batch = None
        for _ in range(batch_index + 1):
            batch = next(iterator)
        if batch is None:
            raise RuntimeError(f"Unable to fetch batch {batch_index} from split='{split}'.")
        return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=torch.cuda.is_available())
            else:
                moved[key] = value
        return moved

    def prepare_train_step(self) -> None:
        self.trainer.denoiser.train()
        self.trainer.opt.zero_grad()

    def forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        with self.trainer.accelerator.autocast():
            loss_out = self.trainer.denoiser.training_step(batch, log_dict={})
        return loss_out.total

    def backward(self, loss: torch.Tensor) -> None:
        self.trainer.accelerator.backward(loss)

    def optimizer_step(self) -> None:
        self.trainer.accelerator.clip_grad_norm_(
            self.trainer.denoiser.parameters(), self.cfg.OPTIMIZATION.GRAD_NORM_CLIP
        )
        self.trainer.opt.step()
        self.trainer.opt.zero_grad()

    def prepare_inference(self) -> None:
        self.trainer.denoiser.eval()

    def inference_step(self, batch: Dict[str, Any], num_samples: int) -> torch.Tensor:
        pred = self.trainer.denoiser.predict(batch, num_samples=num_samples, return_trace=False)
        return pred.samples

    def get_inference_metadata(self, num_samples: int) -> Dict[str, Any]:
        horizon = int(self.cfg.future_frames)
        if self.cfg.METHOD.NAME == "cogflow":
            nfe = int(self.cfg.get("sampling_steps", horizon))
            label = f"{nfe} (FM)"
        else:
            nfe = horizon
            label = f"{nfe} (AR)"
        if num_samples > 1:
            label = f"{label} x {num_samples}"
        return {"horizon": horizon, "steps_nfe": label}


@register_method("cogflow")
class ForecastMethodAdapter(_ForecastAdapter):
    pass


@register_method("latent_ar")
class LatentARBenchmarkAdapter(_ForecastAdapter):
    pass


@register_method("rssm")
class RSSMBenchmarkAdapter(_ForecastAdapter):
    pass
