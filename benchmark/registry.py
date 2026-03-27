from __future__ import annotations

from typing import Dict, Type

from .base_adapter import MethodAdapter

METHOD_REGISTRY: Dict[str, Type[MethodAdapter]] = {}


def register_method(name: str):
    def decorator(cls: Type[MethodAdapter]) -> Type[MethodAdapter]:
        if name in METHOD_REGISTRY:
            raise KeyError(f"Benchmark adapter '{name}' already registered.")
        METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def build_adapter(name: str, cfg, args, logger) -> MethodAdapter:
    if name not in METHOD_REGISTRY:
        raise KeyError(
            f"Benchmark adapter '{name}' is not registered. Available: {sorted(METHOD_REGISTRY)}"
        )
    return METHOD_REGISTRY[name](cfg=cfg, args=args, logger=logger)


def list_registered_methods():
    return sorted(METHOD_REGISTRY)
