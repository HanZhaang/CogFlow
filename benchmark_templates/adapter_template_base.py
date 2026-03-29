from __future__ import annotations

from typing import Any, Dict

import torch

from benchmark.base_adapter import MethodAdapter


class ExternalMethodTemplateAdapter(MethodAdapter):
    """
    Shared template for integrating external baselines into the unified
    benchmark runner.

    This file is intentionally standalone and not registered by default.
    Copy it into benchmark/adapters/ and add @register_method(...) only after
    all TODOs are replaced with the real baseline code.
    """

    def __init__(self, cfg, args, logger):
        super().__init__(cfg=cfg, args=args, logger=logger)

        # TODO: build the baseline dataloaders or dataset handles.
        self.train_loader = None
        self.val_loader = None

        # TODO: build the model/trainer/wrapper here.
        self.model = None
        self.optimizer = None

    @property
    def device(self) -> str:
        return self.cfg.device

    @property
    def raw_model(self):
        # TODO: return the nn.Module whose parameters should be counted.
        return self.model

    def get_batch(self, split: str, batch_index: int = 0) -> Dict[str, Any]:
        loader = self.train_loader if split == "train" else self.val_loader
        if loader is None:
            raise RuntimeError("TODO: assign train_loader/val_loader before benchmarking.")

        iterator = iter(loader)
        batch = None
        for _ in range(batch_index + 1):
            batch = next(iterator)
        return self._to_cpu_dict(batch)

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=torch.cuda.is_available())
            else:
                moved[key] = value
        return moved

    def prepare_train_step(self) -> None:
        # TODO: switch the training object to train mode and clear grads.
        self.model.train()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        # TODO: run the training forward pass and return a scalar loss tensor.
        raise NotImplementedError

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("TODO: assign optimizer before benchmarking train mode.")
        self.optimizer.step()
        self.optimizer.zero_grad()

    def prepare_inference(self) -> None:
        self.model.eval()

    def inference_step(self, batch: Dict[str, Any], num_samples: int) -> torch.Tensor:
        # TODO: return predictions only. Do not compute metrics here.
        raise NotImplementedError

    def get_inference_metadata(self, num_samples: int) -> Dict[str, Any]:
        # TODO: fill with the actual horizon and NFE definition.
        return {
            "horizon": int(self.cfg.future_frames),
            "steps_nfe": f"TODO x {num_samples}",
        }

    def _to_cpu_dict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.detach().cpu()
            else:
                out[key] = value
        return out

    def _repeat_deterministic_output(self, pred: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        For deterministic baselines, you may return one prediction tensor and
        expand it on the sample dimension to match the stochastic interface.
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if pred.shape[1] == num_samples:
            return pred
        if pred.shape[1] != 1:
            raise ValueError("Expected deterministic output to have sample dim == 1.")
        return pred.expand(-1, num_samples, *pred.shape[2:])
