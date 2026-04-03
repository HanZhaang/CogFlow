#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: torch. Please run this script inside the project Python environment."
    ) from exc
from easydict import EasyDict

from models.model_registry import build_network
from utils.config import Config


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cfg_path: str
    method: str
    variant: str | None = None
    decoder: str | None = None
    action_fusion: str | None = None


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="rat | RSSM + concat + moflow decoder",
        cfg_path="cfg/baselines/rat/rssm_moflow.yml",
        method="rssm",
        decoder="moflow_structured",
        action_fusion="none",
    ),
    ModelSpec(
        name="rat | RSSM + attention + moflow decoder",
        cfg_path="cfg/baselines/rat/rssm_moflow.yml",
        method="rssm",
        decoder="moflow_structured",
        action_fusion="cross_attention",
    ),
    ModelSpec(
        name="rat | latent AR (transformer) + concat + moflow decoder",
        cfg_path="cfg/baselines/rat/latent_ar_transformer_moflow.yml",
        method="latent_ar",
        variant="transformer",
        decoder="moflow_structured",
        action_fusion="none",
    ),
    ModelSpec(
        name="rat | latent AR (transformer) + attention + moflow decoder",
        cfg_path="cfg/baselines/rat/latent_ar_transformer_moflow.yml",
        method="latent_ar",
        variant="transformer",
        decoder="moflow_structured",
        action_fusion="cross_attention",
    ),
    ModelSpec(
        name="rat | cogflow (ours)",
        cfg_path="cfg/full_cfg/cor_rat_fm_mn.yml",
        method="cogflow",
    ),
    ModelSpec(
        name="babel | RSSM + concat + moflow decoder",
        cfg_path="cfg/baselines/babel/rssm_moflow.yml",
        method="rssm",
        decoder="moflow_structured",
        action_fusion="none",
    ),
    ModelSpec(
        name="babel | latent AR (transformer) + concat + moflow decoder",
        cfg_path="cfg/baselines/babel/latent_ar_transformer_moflow.yml",
        method="latent_ar",
        variant="transformer",
        decoder="moflow_structured",
        action_fusion="none",
    ),
    ModelSpec(
        name="babel | cogflow (ours)",
        cfg_path="cfg/full_cfg/cor_babel_fm_m1.yml",
        method="cogflow",
    ),
]


class NullLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate predefined CogFlow/baseline models and count parameters."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional substring filters applied to model names.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include top-level child module parameter counts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the rendered result.",
    )
    return parser.parse_args()


def _normalize_model_name(cfg) -> None:
    model_name = cfg.MODEL.get("NAME", None)
    if model_name is None and cfg.MODEL.get("Name", None) is not None:
        cfg.MODEL.NAME = cfg.MODEL.Name


def _ensure_easydict(value) -> EasyDict:
    if isinstance(value, EasyDict):
        return value
    return EasyDict(value)


def apply_runtime_overrides(cfg, spec: ModelSpec) -> None:
    dataset_name_map = {
        "rat": "rat_dataset",
        "babel": "babel_dataset",
        "nba": "nba_dataset",
        "eth_ucy": "eth_dataset",
        "sdd": "sdd_dataset",
    }
    if cfg.get("dataset_name", None) is None and cfg.get("dataset", None) in dataset_name_map:
        cfg.dataset_name = dataset_name_map[cfg.dataset]

    _normalize_model_name(cfg)

    method_cfg = _ensure_easydict(cfg.yml_dict.get("METHOD", EasyDict()))
    method_cfg.NAME = spec.method
    if spec.variant is not None:
        method_cfg.VARIANT = spec.variant
    else:
        method_cfg.VARIANT = method_cfg.get("VARIANT", "gru")
    if spec.decoder is not None:
        method_cfg.DECODER = spec.decoder
    else:
        method_cfg.DECODER = method_cfg.get("DECODER", "moflow_structured")
    if spec.action_fusion is not None:
        method_cfg.ACTION_FUSION = spec.action_fusion
    else:
        method_cfg.ACTION_FUSION = method_cfg.get("ACTION_FUSION", "none")
    method_cfg.TRAINER = "cogflow" if spec.method == "cogflow" else "forecast"
    cfg.yml_dict["METHOD"] = method_cfg

    action_fusion_cfg = _ensure_easydict(cfg.MODEL.get("ACTION_FUSION", EasyDict()))
    action_fusion_cfg.NAME = method_cfg.ACTION_FUSION
    action_fusion_cfg.D_MODEL = int(
        action_fusion_cfg.get("D_MODEL", cfg.MODEL.CONTEXT_ENCODER.D_MODEL)
    )
    action_fusion_cfg.NUM_HEADS = int(action_fusion_cfg.get("NUM_HEADS", 4))
    action_fusion_cfg.NUM_LAYERS = int(action_fusion_cfg.get("NUM_LAYERS", 1))
    action_fusion_cfg.DROPOUT = float(action_fusion_cfg.get("DROPOUT", 0.1))
    action_fusion_cfg.MAX_SEQ_LEN = int(
        action_fusion_cfg.get("MAX_SEQ_LEN", cfg.future_frames * 2)
    )
    action_fusion_cfg.INCLUDE_HISTORY = bool(action_fusion_cfg.get("INCLUDE_HISTORY", True))
    action_fusion_cfg.USE_RAW_CTRL_RESIDUAL = bool(
        action_fusion_cfg.get("USE_RAW_CTRL_RESIDUAL", True)
    )
    cfg.MODEL.ACTION_FUSION = action_fusion_cfg

    cfg.MODEL.NAME = spec.method

    if spec.method == "cogflow":
        cfg.trainer_name = "cogflow"
        cfg.denoising_method = cfg.get("denoising_method", "fm")
    else:
        cfg.trainer_name = "forecast"
        cfg.denoising_method = spec.method
        cfg.MODEL.LATENT_DIM = cfg.MODEL.get("LATENT_DIM", cfg.MODEL.get("COG_D_Z", 64))
        cfg.MODEL.LATENT_AR_HIDDEN_DIM = cfg.MODEL.get(
            "LATENT_AR_HIDDEN_DIM", cfg.MODEL.CONTEXT_ENCODER.D_MODEL
        )
        cfg.MODEL.RSSM_STOCH_DIM = cfg.MODEL.get("RSSM_STOCH_DIM", cfg.MODEL.get("COG_D_Z", 64))
        cfg.MODEL.RSSM_DET_DIM = cfg.MODEL.get(
            "RSSM_DET_DIM", cfg.MODEL.CONTEXT_ENCODER.D_MODEL
        )
        cfg.MODEL.RSSM_OBS_DIM = cfg.MODEL.get("RSSM_OBS_DIM", cfg.MODEL.get("COG_D_Z", 64))
        cfg.MODEL.RSSM_DECODER_LATENT_DIM = cfg.MODEL.get(
            "RSSM_DECODER_LATENT_DIM", cfg.MODEL.get("COG_D_Z", 64)
        )
        if cfg.get("BASELINE_LOSS_WEIGHTS", None) is None:
            cfg.BASELINE_LOSS_WEIGHTS = EasyDict({"recon": 1.0, "latent_nll": 0.1})
        if cfg.get("RSSM_KL_BETA", None) is None:
            cfg.RSSM_KL_BETA = 0.1


def load_cfg(spec: ModelSpec):
    cfg = Config(str(ROOT / spec.cfg_path), tag="param_count", train_mode=False)
    cfg.device = "cpu"
    if cfg.get("stats", None) is None:
        cfg.stats = {}
    apply_runtime_overrides(cfg, spec)
    return cfg


def count_model_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_child = []
    for child_name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        if child_total == 0:
            continue
        by_child.append(
            {
                "name": child_name,
                "total_params": child_total,
                "trainable_params": sum(p.numel() for p in child.parameters() if p.requires_grad),
            }
        )
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "by_child": by_child,
    }


def format_int(value: int) -> str:
    return f"{value:,}"


def collect_results(specs: Iterable[ModelSpec], include_details: bool) -> list[dict]:
    results = []
    logger = NullLogger()
    args = EasyDict()
    for spec in specs:
        cfg = load_cfg(spec)
        model = build_network(cfg=cfg, args=args, logger=logger)
        counts = count_model_parameters(model)
        row = {
            "name": spec.name,
            "cfg_path": spec.cfg_path,
            "method": spec.method,
            "variant": spec.variant,
            "decoder": spec.decoder,
            "action_fusion": spec.action_fusion,
            "total_params": counts["total_params"],
            "trainable_params": counts["trainable_params"],
            "non_trainable_params": counts["non_trainable_params"],
        }
        if include_details:
            row["by_child"] = counts["by_child"]
        results.append(row)
    return results


def render_markdown(results: list[dict], include_details: bool) -> str:
    lines = [
        "| Model | Total Params | Trainable Params | Config |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {format_int(row['total_params'])} | "
            f"{format_int(row['trainable_params'])} | `{row['cfg_path']}` |"
        )
    if include_details:
        lines.append("")
        lines.append("Top-level child module breakdown:")
        for row in results:
            lines.append(f"- {row['name']}")
            for child in row.get("by_child", []):
                lines.append(
                    f"  - {child['name']}: total={format_int(child['total_params'])}, "
                    f"trainable={format_int(child['trainable_params'])}"
                )
    return "\n".join(lines)


def render_csv(results: list[dict]) -> str:
    fieldnames = [
        "name",
        "cfg_path",
        "method",
        "variant",
        "decoder",
        "action_fusion",
        "total_params",
        "trainable_params",
        "non_trainable_params",
    ]
    rows = []
    for row in results:
        rows.append({key: row.get(key) for key in fieldnames})
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().rstrip()


def render_json(results: list[dict]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)


def filter_specs(specs: Iterable[ModelSpec], filters: list[str] | None) -> list[ModelSpec]:
    if not filters:
        return list(specs)
    lowered = [item.lower() for item in filters]
    return [
        spec for spec in specs
        if any(token in spec.name.lower() for token in lowered)
    ]


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    selected_specs = filter_specs(SPECS, args.only)
    if not selected_specs:
        raise SystemExit("No model specs matched --only filters.")

    results = collect_results(selected_specs, include_details=args.details)
    if args.format == "markdown":
        output = render_markdown(results, include_details=args.details)
    elif args.format == "csv":
        output = render_csv(results)
    else:
        output = render_json(results)

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()
