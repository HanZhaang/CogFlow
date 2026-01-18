import copy
from typing import Callable, Dict
from torch.utils.data import DataLoader

# =========================
# Dataset Builder Registry
# =========================

_DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset builder.
    """
    def wrapper(builder_fn):
        if name in _DATASET_REGISTRY:
            raise KeyError(f"Dataset '{name}' already registered.")
        _DATASET_REGISTRY[name] = builder_fn
        return builder_fn
    return wrapper


def get_dataset_builder(name: str) -> Callable:
    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' is not registered. "
            f"Available: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name]


# =========================
# Generic Entry Point
# =========================

def build_data_loader(cfg, args):
    """
    Unified dataset entry point.

    cfg.dataset_name must be specified.
    """
    builder = get_dataset_builder(cfg.dataset_name)
    return builder(cfg, args)
