from __future__ import annotations

import copy
import csv
import json
import logging
import re
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from utils.config import Config
from train import apply_runtime_overrides


def create_benchmark_logger(name: str = "benchmark.cost") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_runtime_config(args):
    cfg = Config(args.cfg, tag="benchmark", train_mode=True)
    apply_runtime_overrides(cfg, args)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime_root = Path(tempfile.gettempdir()) / "cogflow_benchmark" / cfg.cfg_name / args.method
    runtime_root.mkdir(parents=True, exist_ok=True)
    cfg.cfg_dir = str(runtime_root)
    cfg.model_dir = str(runtime_root / "models")
    cfg.log_dir = str(runtime_root / "log")
    cfg.npz_dir = str(runtime_root / "npz")
    cfg.sample_dir = str(runtime_root / "samples")
    return cfg


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def save_batch(batch: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clone_batch(batch), path_obj)


def load_batch(path: str) -> Dict[str, Any]:
    batch = torch.load(path, map_location="cpu")
    return clone_batch(batch)


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def max_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def serialize_data(obj: Any):
    if is_dataclass(obj):
        return serialize_data(asdict(obj))
    if isinstance(obj, dict):
        return {key: serialize_data(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_data(value) for value in obj]
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def write_json(data: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(serialize_data(data), f, indent=2, ensure_ascii=False)


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path_obj.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def flatten_train_row(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "method": result["method"],
        "params_m": round(result["params"] / 1e6, 4),
        "gpu": result["env"]["gpu"],
        "batch": result["batch_size"],
    }
    train_stats = result.get("train", {})
    summary = result.get("training_summary", {})
    row.update(
        {
            "h2d_ms": train_stats.get("h2d_time_ms"),
            "forward_ms": train_stats.get("forward_time_ms"),
            "backward_ms": train_stats.get("backward_time_ms"),
            "optim_ms": train_stats.get("optimizer_time_ms"),
            "step_ms": train_stats.get("step_time_ms"),
            "peak_mem_gb": train_stats.get("peak_mem_gb"),
            "time_to_best_h": summary.get("time_to_best_h"),
            "total_time_h": summary.get("total_time_h"),
            "gpu_hours": summary.get("gpu_hours"),
        }
    )
    return row


def flatten_infer_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for infer_stats in result.get("infer", []):
        rows.append(
            {
                "method": result["method"],
                "gpu": result["env"]["gpu"],
                "batch": result["batch_size"],
                "k": infer_stats.get("k"),
                "horizon": infer_stats.get("horizon"),
                "steps_nfe": infer_stats.get("steps_nfe"),
                "latency_per_sample_ms": infer_stats.get("latency_per_sample_ms"),
                "latency_per_batch_ms": infer_stats.get("latency_per_batch_ms"),
                "throughput_seq_s": infer_stats.get("throughput_seq_s"),
                "peak_mem_gb": infer_stats.get("peak_mem_gb"),
            }
        )
    return rows


def render_markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_markdown_value(row.get(col)) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def format_markdown_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)")


def _parse_timestamp(line: str) -> Optional[datetime]:
    match = _TIMESTAMP_RE.match(line.strip())
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def _file_mtime(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime)


def parse_training_artifacts(
    experiment_dir: Optional[str], num_gpus: int = 1
) -> Dict[str, Optional[float]]:
    if not experiment_dir:
        return {"time_to_best_h": None, "total_time_h": None, "gpu_hours": None}

    exp_dir = Path(experiment_dir)
    log_path = exp_dir / "log" / "log.txt"
    model_dir = exp_dir / "models"

    start_ts = None
    best_log_ts = None
    complete_ts = None

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                ts = _parse_timestamp(line)
                if ts is None:
                    continue
                if "training start" in line and start_ts is None:
                    start_ts = ts
                if "Current best ADE_MIN" in line:
                    best_log_ts = ts
                if "training complete" in line:
                    complete_ts = ts

    best_ckpt_ts = _file_mtime(model_dir / "checkpoint_best.pt")
    last_ckpt_ts = _file_mtime(model_dir / "checkpoint_last.pt")

    best_ts = best_ckpt_ts or best_log_ts
    end_ts = last_ckpt_ts or complete_ts

    if start_ts is None or end_ts is None:
        return {"time_to_best_h": None, "total_time_h": None, "gpu_hours": None}

    total_time_h = max((end_ts - start_ts).total_seconds() / 3600.0, 0.0)
    time_to_best_h = None
    if best_ts is not None:
        time_to_best_h = max((best_ts - start_ts).total_seconds() / 3600.0, 0.0)

    gpu_hours = total_time_h * max(int(num_gpus), 1)
    return {
        "time_to_best_h": round(time_to_best_h, 4) if time_to_best_h is not None else None,
        "total_time_h": round(total_time_h, 4),
        "gpu_hours": round(gpu_hours, 4),
    }


def normalize_method_list(method: str, available: Iterable[str]) -> List[str]:
    if method == "all":
        return list(available)
    return [method]
