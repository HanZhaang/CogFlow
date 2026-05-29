# SPDX-License-Identifier: MIT
from typing import Callable, Dict, Any

_TRAINER_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_trainer(name: str):
    def wrapper(fn: Callable[..., Any]):
        if name in _TRAINER_REGISTRY:
            raise KeyError(f"Trainer '{name}' already registered.")
        _TRAINER_REGISTRY[name] = fn
        return fn
    return wrapper

def build_trainer(cfg, model, train_loader, val_loader, tb_log, logger):
    import trainer  # noqa: F401

    trainer_name = getattr(cfg, "trainer_name", None)
    if trainer_name is None and getattr(cfg, "METHOD", None) is not None:
        trainer_name = getattr(cfg.METHOD, "TRAINER", None)
    if trainer_name is None:
        trainer_name = getattr(cfg.MODEL, "NAME", None)
    if trainer_name is None:
        raise ValueError("cfg.trainer_name (or cfg.model_name) is required.")
    if trainer_name not in _TRAINER_REGISTRY:
        raise KeyError(f"Trainer '{trainer_name}' not registered. Available: {list(_TRAINER_REGISTRY)}")
    return _TRAINER_REGISTRY[trainer_name](
        cfg=cfg, model=model, 
        train_loader=train_loader, val_loader=val_loader,
        tb_log=tb_log, logger=logger,
    )
    
