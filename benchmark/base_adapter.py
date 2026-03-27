from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class MethodAdapter(ABC):
    def __init__(self, cfg, args, logger):
        self.cfg = cfg
        self.args = args
        self.logger = logger

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, split: str, batch_index: int = 0) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_train_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def optimizer_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_inference(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def inference_step(self, batch: Dict[str, Any], num_samples: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_inference_metadata(self, num_samples: int) -> Dict[str, Any]:
        raise NotImplementedError

    def count_params(self) -> int:
        return sum(p.numel() for p in self.raw_model.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)

    def batch_size(self, batch: Dict[str, Any]) -> Optional[int]:
        for value in batch.values():
            if torch.is_tensor(value) and value.dim() > 0:
                return int(value.shape[0])
        return None
