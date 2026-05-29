# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Callable, Dict, Any, Optional

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    """Decorator to register a Model builder."""
    def wrapper(fn: Callable[..., Any]):
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn
    return wrapper


def get_model_builder(name: str) -> Callable[..., Any]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' is not registered. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def list_registered_models() -> dict:
    return {"models": sorted(_MODEL_REGISTRY.keys())}


def build_network(cfg, args, logger):
    """
    Unified entry point to build the denoising network.

    Required cfg fields:
      - cfg.model_name: backbone registry name (e.g., "motion_transformer")
      - cfg.denoising_method: denoiser registry name (e.g., "fm")
    """
    import models  # noqa: F401

    method_cfg = getattr(cfg, "METHOD", None)
    model_name = None
    if method_cfg is not None:
        model_name = getattr(method_cfg, "NAME", None)
    if model_name is None:
        model_name = getattr(cfg, "method_name", None)
    if model_name is None:
        model_name = getattr(cfg.MODEL, "NAME", None)
    if model_name is None:
        raise ValueError("cfg.METHOD.NAME or cfg.MODEL.NAME is required.")

    model_builder = get_model_builder(model_name)
    denoiser = model_builder(cfg=cfg, args=args, logger=logger)

    return denoiser
#     """
#     backbone_name = getattr(cfg, "model_name", None)
#     if backbone_name is None:
#         raise ValueError("cfg.model_name is required (e.g., 'motion_transformer').")

#     denoiser_name = getattr(cfg, "denoising_method", None)
#     if denoiser_name is None:
#         raise ValueError("cfg.denoising_method is required (e.g., 'fm').")

#     backbone_builder = get_backbone_builder(backbone_name)
#     model = backbone_builder(cfg=cfg, args=args, logger=logger)

#     denoiser_builder = get_denoiser_builder(denoiser_name)
#     denoiser = denoiser_builder(cfg=cfg, args=args, logger=logger, model=model)

#     return denoiser
