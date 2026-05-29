# SPDX-License-Identifier: MIT
import copy
from typing import Callable, Dict
from torch.utils.data import DataLoader

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

def build_data_loader(cfg, args):
    """
    Unified dataset entry point.

    cfg.dataset_name must be specified.
    """
    import data  # noqa: F401

    dataset_name = getattr(cfg, "dataset_name", None)
    if dataset_name is None:
        dataset = getattr(cfg, "dataset", None)
        dataset_name_map = {
            "rat": "rat_dataset",
            "babel": "babel_dataset",
        }
        dataset_name = dataset_name_map.get(dataset)
        if dataset_name is None:
            raise ValueError("cfg.dataset_name or cfg.dataset is required.")
        cfg.dataset_name = dataset_name

    builder = get_dataset_builder(dataset_name)
    return builder(cfg, args)
