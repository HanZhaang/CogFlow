# model_registry.py
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



# # model_registry.py，以防未来需要把编码器和流匹配拆开，现在看来没啥用
# from __future__ import annotations

# from typing import Callable, Dict, Any, Optional

# _BACKBONE_REGISTRY: Dict[str, Callable[..., Any]] = {}
# _DENOISER_REGISTRY: Dict[str, Callable[..., Any]] = {}


# def register_backbone(name: str):
#     """Decorator to register a backbone builder."""
#     def wrapper(fn: Callable[..., Any]):
#         if name in _BACKBONE_REGISTRY:
#             raise KeyError(f"Backbone '{name}' already registered.")
#         _BACKBONE_REGISTRY[name] = fn
#         return fn
#     return wrapper


# def register_denoiser(name: str):
#     """Decorator to register a denoiser/wrapper builder."""
#     def wrapper(fn: Callable[..., Any]):
#         if name in _DENOISER_REGISTRY:
#             raise KeyError(f"Denoiser '{name}' already registered.")
#         _DENOISER_REGISTRY[name] = fn
#         return fn
#     return wrapper


# def get_backbone_builder(name: str) -> Callable[..., Any]:
#     if name not in _BACKBONE_REGISTRY:
#         raise KeyError(
#             f"Backbone '{name}' is not registered. "
#             f"Available: {list(_BACKBONE_REGISTRY.keys())}"
#         )
#     return _BACKBONE_REGISTRY[name]


# def get_denoiser_builder(name: str) -> Callable[..., Any]:
#     if name not in _DENOISER_REGISTRY:
#         raise KeyError(
#             f"Denoiser '{name}' is not registered. "
#             f"Available: {list(_DENOISER_REGISTRY.keys())}"
#         )
#     return _DENOISER_REGISTRY[name]


# def list_registered_models() -> dict:
#     return {
#         "backbones": sorted(_BACKBONE_REGISTRY.keys()),
#         "denoisers": sorted(_DENOISER_REGISTRY.keys()),
#     }


# def build_network(cfg, args, logger):
#     """
#     Unified entry point to build the denoising network.

#     Required cfg fields:
#       - cfg.model_name: backbone registry name (e.g., "motion_transformer")
#       - cfg.denoising_method: denoiser registry name (e.g., "fm")
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
