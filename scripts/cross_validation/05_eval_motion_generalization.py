#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_registry import build_network
from scripts.cross_validation.ver2_dataset import VER2_CMD_DIR, default_cross_subject_path, parse_cmd_txt
from utils.config import Config
from utils.normalization import unnormalize_mean_std


XY_KEYS = ("xy", "keypoints", "kp", "kp_seq", "traj", "trajectory", "pose")
CMD_PATH_COLUMNS = ("cmd_path", "stim_path", "cue_path", "instruction_path", "command_path")
POSE_PATH_COLUMNS = ("pose_path", "source_npz_path")
CMD_ARRAY_KEYS = (
    "cmd",
    "stim",
    "cue",
    "instruction",
    "instructions",
    "command",
    "commands",
    "control",
    "controls",
)
CMD_ID_KEYS = ("instr_id", "instruction_id", "cmd_id", "stim_id", "command_id")
CMD_STRENGTH_KEYS = (
    "instr_strength",
    "instruction_strength",
    "cmd_strength",
    "stim_strength",
    "command_strength",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_splits(split_dir: Path, split_name: str = "") -> List[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    if str(split_name).strip():
        p = split_dir / str(split_name).strip()
        if not p.exists():
            raise FileNotFoundError(f"split not found: {p}")
        return [p]
    out = [p for p in sorted(split_dir.glob("loro_*")) if p.is_dir()]
    if not out:
        raise RuntimeError(f"no split found in {split_dir}")
    return out


def build_logger() -> logging.Logger:
    logger = logging.getLogger("cross_validation.motion")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def resolve_existing(paths: Sequence[Path], what: str) -> Path:
    for path in paths:
        if path.exists():
            return path.resolve()
    tried = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(f"{what} not found. tried:\n{tried}")


def resolve_checkpoint_path(split_name: str, model_root: Path, checkpoint_name: str) -> Path:
    candidates = [
        model_root / split_name / "models" / checkpoint_name,
        model_root / split_name / checkpoint_name,
        model_root / checkpoint_name,
    ]
    return resolve_existing(candidates, f"checkpoint for split={split_name}")


def resolve_cfg_path(split_name: str, checkpoint_path: Path, cfg_arg: str) -> Path:
    cfg_arg = str(cfg_arg).strip()
    if cfg_arg and cfg_arg.lower() != "auto":
        cfg_candidate = Path(cfg_arg.format(split=split_name)).expanduser()
        if not cfg_candidate.is_absolute():
            cfg_candidate = ROOT / cfg_candidate
        return cfg_candidate.resolve()

    experiment_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "models" else checkpoint_path.parent
    yaml_candidates = sorted(experiment_dir.glob("*_updated.yml")) + sorted(experiment_dir.glob("*.yml"))
    if not yaml_candidates:
        raise FileNotFoundError(f"no config yaml found under {experiment_dir}")
    return yaml_candidates[0].resolve()


def resolve_pose_path(row: pd.Series) -> Path:
    for col in POSE_PATH_COLUMNS:
        raw = str(row.get(col, "")).strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path.resolve()
    raise FileNotFoundError(
        f"cannot resolve pose path for trial_id={row.get('trial_id', '')}; "
        f"checked columns={POSE_PATH_COLUMNS}"
    )


def resolve_cmd_path(row: pd.Series, cmd_dir: Optional[Path]) -> Optional[Path]:
    for col in CMD_PATH_COLUMNS:
        raw = str(row.get(col, "")).strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path.resolve()

    if cmd_dir is None:
        return None

    trial_id = str(row.get("trial_id", "")).strip()
    if not trial_id:
        return None

    candidates = [
        cmd_dir / f"{trial_id}.npy",
        cmd_dir / f"{trial_id}.npz",
        cmd_dir / f"{trial_id}_cmd.npy",
        cmd_dir / f"{trial_id}_cmd.npz",
        cmd_dir / f"{trial_id}_stim.npy",
        cmd_dir / f"{trial_id}_stim.npz",
        cmd_dir / f"cmd_{trial_id}.npy",
        cmd_dir / f"cmd_{trial_id}.npz",
        cmd_dir / f"stim_{trial_id}.npy",
        cmd_dir / f"stim_{trial_id}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def load_pose_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"pose file not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            arr = None
            for key in XY_KEYS:
                if key in data:
                    arr = np.asarray(data[key], dtype=np.float32)
                    break
            if arr is None:
                raise KeyError(f"pose key not found in {path}; available={list(data.files)}")
    else:
        raise ValueError(f"unsupported pose suffix: {path.suffix}")

    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"pose must be (T,A,2), got {arr.shape} at {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"pose has NaN/Inf: {path}")
    return arr.astype(np.float32, copy=False)


def compute_cue_features(instr_id: np.ndarray, instr_strength: np.ndarray) -> np.ndarray:
    instr_id = np.asarray(instr_id).reshape(-1).astype(np.int64)
    instr_strength = np.asarray(instr_strength).reshape(-1).astype(np.float32)
    if instr_id.shape[0] != instr_strength.shape[0]:
        raise ValueError(f"instruction id/strength length mismatch: {instr_id.shape} vs {instr_strength.shape}")

    T = int(instr_id.shape[0])
    instr_id = np.clip(instr_id, 0, 3)

    onehot = np.zeros((T, 4), dtype=np.float32)
    onehot[np.arange(T), instr_id] = 1.0

    has_cmd = instr_id > 0
    strength = np.where(has_cmd, instr_strength, 0.0).astype(np.float32)

    signed_strength = np.zeros((T,), dtype=np.float32)
    signed_strength[instr_id == 2] = -strength[instr_id == 2]
    signed_strength[instr_id == 3] = strength[instr_id == 3]

    event_mask = np.logical_and(has_cmd, instr_strength > 0)
    time_since = np.zeros((T,), dtype=np.float32)
    last_idx = -1
    for idx in range(T):
        if bool(event_mask[idx]):
            last_idx = idx
        time_since[idx] = float(idx - last_idx) if last_idx >= 0 else float(idx)

    return np.concatenate(
        [
            onehot,
            strength[:, None],
            signed_strength[:, None],
            time_since[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def load_cmd_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"command file not found: {path}")

    if path.suffix.lower() == ".txt":
        return parse_cmd_txt(path)

    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        return normalize_cmd_representation(arr, source=str(path))

    if path.suffix.lower() != ".npz":
        raise ValueError(f"unsupported command suffix: {path.suffix}")

    with np.load(path, allow_pickle=True) as data:
        for key in CMD_ARRAY_KEYS:
            if key in data:
                return normalize_cmd_representation(data[key], source=f"{path}:{key}")

        instr_id = None
        instr_strength = None
        for key in CMD_ID_KEYS:
            if key in data:
                instr_id = np.asarray(data[key]).reshape(-1)
                break
        for key in CMD_STRENGTH_KEYS:
            if key in data:
                instr_strength = np.asarray(data[key]).reshape(-1)
                break
        if instr_id is not None:
            if instr_strength is None:
                instr_strength = np.zeros_like(instr_id, dtype=np.float32)
            return compute_cue_features(instr_id=instr_id, instr_strength=instr_strength)

        raise KeyError(f"command keys not found in {path}; available={list(data.files)}")


def normalize_cmd_representation(arr: np.ndarray, source: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return compute_cue_features(instr_id=arr.astype(np.int64), instr_strength=np.zeros_like(arr, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"command array must be 1D or 2D, got {arr.shape} from {source}")
    if arr.shape[-1] == 7:
        return np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] >= 2:
        return compute_cue_features(instr_id=arr[:, 0], instr_strength=arr[:, 1])
    raise ValueError(f"unsupported command shape {arr.shape} from {source}")


def load_trial_cmd(row: pd.Series, pose_path: Path, cmd_dir: Optional[Path]) -> np.ndarray:
    cmd_path = resolve_cmd_path(row, cmd_dir=cmd_dir)
    if cmd_path is not None:
        return load_cmd_array(cmd_path)

    if pose_path.suffix.lower() == ".npz":
        return load_cmd_array(pose_path)

    source_npz = str(row.get("source_npz_path", "")).strip()
    if source_npz:
        source_path = Path(source_npz).expanduser()
        if source_path.exists() and source_path.suffix.lower() == ".npz":
            return load_cmd_array(source_path)

    trial_id = str(row.get("trial_id", "")).strip()
    raise FileNotFoundError(
        f"command sequence not found for trial_id={trial_id or '<unknown>'}. "
        f"Provide one of {CMD_PATH_COLUMNS} in the manifest or pass --cmd-dir."
    )


def list_window_starts(T: int, t_h: int, t_p: int, stride: int) -> List[int]:
    if T < t_h + t_p:
        return []
    return list(range(0, int(T - t_h - t_p + 1), max(1, int(stride))))


def compute_velocities(traj: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(traj, dtype=np.float32)
    if traj.shape[2] > 1:
        vel[:, :, :-1, :] = traj[:, :, 1:, :] - traj[:, :, :-1, :]
    return vel


def _update_stats(acc: Dict[str, np.ndarray], key: str, values: np.ndarray) -> None:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
    acc[f"{key}_sum"] += flat.sum(axis=0)
    acc[f"{key}_sq_sum"] += np.square(flat).sum(axis=0)
    acc[f"{key}_count"] += flat.shape[0]


def initialize_stats_accumulator() -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in ("abs", "rel", "vel", "fut"):
        out[f"{key}_sum"] = np.zeros((2,), dtype=np.float64)
        out[f"{key}_sq_sum"] = np.zeros((2,), dtype=np.float64)
        out[f"{key}_count"] = np.array(0.0, dtype=np.float64)
    return out


def finalize_stats(acc: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    stats: Dict[str, np.ndarray] = {}
    for key in ("abs", "rel", "vel", "fut"):
        count = float(acc[f"{key}_count"])
        if count <= 0:
            raise RuntimeError(f"no samples collected for {key} normalization stats")
        mean = acc[f"{key}_sum"] / count
        var = np.maximum(acc[f"{key}_sq_sum"] / count - np.square(mean), 1e-6)
        stats[f"{key}_mean"] = mean.astype(np.float32)
        stats[f"{key}_std"] = np.sqrt(var).astype(np.float32)
    return stats


def accumulate_train_stats(
    manifest: pd.DataFrame,
    t_h: int,
    t_p: int,
    stride: int,
    cmd_dir: Optional[Path],
) -> Dict[str, np.ndarray]:
    acc = initialize_stats_accumulator()
    total_windows = 0

    for _, row in manifest.iterrows():
        pose_path = resolve_pose_path(row)
        pose = load_pose_array(pose_path)
        cmd = load_trial_cmd(row, pose_path=pose_path, cmd_dir=cmd_dir)
        if cmd.shape[0] != pose.shape[0]:
            raise ValueError(
                f"pose/cmd length mismatch for trial_id={row.get('trial_id', '')}: "
                f"pose={pose.shape[0]}, cmd={cmd.shape[0]}"
            )

        starts = list_window_starts(int(pose.shape[0]), t_h=t_h, t_p=t_p, stride=stride)
        if not starts:
            continue

        hist = np.stack([pose[s : s + t_h] for s in starts], axis=0).transpose(0, 2, 1, 3)
        fut = np.stack([pose[s + t_h : s + t_h + t_p] for s in starts], axis=0).transpose(0, 2, 1, 3)
        init = hist[:, :, -1:, :]
        rel = hist - init
        fut_rel = fut - init
        vel = compute_velocities(rel)

        _update_stats(acc, "abs", hist)
        _update_stats(acc, "rel", rel)
        _update_stats(acc, "vel", vel)
        _update_stats(acc, "fut", fut_rel)
        total_windows += len(starts)

    if total_windows <= 0:
        raise RuntimeError("no training windows available to compute normalization stats")

    stats = finalize_stats(acc)
    stats["n_windows"] = np.array(total_windows, dtype=np.int64)
    return stats


def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean_ = mean.reshape(*([1] * (x.ndim - 1)), -1)
    std_ = std.reshape(*([1] * (x.ndim - 1)), -1)
    return ((x - mean_) / std_).astype(np.float32)


def build_window_batch(
    pose: np.ndarray,
    cue: np.ndarray,
    starts: Sequence[int],
    t_h: int,
    t_p: int,
    stats: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor | int]:
    if pose.shape[1] <= 0:
        raise ValueError(f"invalid agent dimension in pose: {pose.shape}")
    hist = np.stack([pose[s : s + t_h] for s in starts], axis=0).transpose(0, 2, 1, 3)
    fut = np.stack([pose[s + t_h : s + t_h + t_p] for s in starts], axis=0).transpose(0, 2, 1, 3)
    hist_cue = np.stack([cue[s : s + t_h] for s in starts], axis=0)
    fut_cue = np.stack([cue[s + t_h : s + t_h + t_p] for s in starts], axis=0)

    init = hist[:, :, -1:, :]
    rel = hist - init
    fut_rel = fut - init
    vel = compute_velocities(rel)
    fut_vel = compute_velocities(fut_rel)

    past_orig = np.concatenate([hist, rel, vel], axis=-1).astype(np.float32)
    past_norm = np.concatenate(
        [
            zscore(hist, stats["abs_mean"], stats["abs_std"]),
            zscore(rel, stats["rel_mean"], stats["rel_std"]),
            zscore(vel, stats["vel_mean"], stats["vel_std"]),
        ],
        axis=-1,
    ).astype(np.float32)
    fut_norm = zscore(fut_rel, stats["fut_mean"], stats["fut_std"]).astype(np.float32)

    batch = {
        "batch_size": int(len(starts)),
        "past_traj": torch.from_numpy(past_norm).to(device=device),
        "fut_traj": torch.from_numpy(fut_norm).to(device=device),
        "past_traj_original_scale": torch.from_numpy(past_orig).to(device=device),
        "fut_traj_original_scale": torch.from_numpy(fut_rel.astype(np.float32)).to(device=device),
        "fut_traj_vel": torch.from_numpy(fut_vel.astype(np.float32)).to(device=device),
        "hist_cond_cue": torch.from_numpy(hist_cue.astype(np.float32)).to(device=device),
        "fut_cond_cue": torch.from_numpy(fut_cue.astype(np.float32)).to(device=device),
    }
    return batch


def sample_future_rel(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | int],
    cfg: Config,
) -> torch.Tensor:
    fut_mean = torch.as_tensor(cfg.stats["fut_mean"], device=cfg.device, dtype=torch.float32).view(1, 1, 1, 1, -1)
    fut_std = torch.as_tensor(cfg.stats["fut_std"], device=cfg.device, dtype=torch.float32).view(1, 1, 1, 1, -1)

    with torch.no_grad():
        pred_flat, _, _, _, _ = model.sample(batch, num_trajs=cfg.denoising_head_preds, return_all_states=False)
    pred = pred_flat.view(
        batch["past_traj"].shape[0],
        cfg.denoising_head_preds,
        cfg.agents,
        cfg.future_frames,
        -1,
    )[..., :2]
    pred = unnormalize_mean_std(pred, fut_mean, fut_std)
    return pred


@dataclass
class MetricAccumulator:
    n_windows: int = 0
    sum_ade: float = 0.0
    sum_fde: float = 0.0
    sum_min_ade: float = 0.0
    sum_min_fde: float = 0.0
    sum_avg_ade: float = 0.0
    sum_avg_fde: float = 0.0

    def update(self, ade: np.ndarray, fde: np.ndarray) -> None:
        self.n_windows += int(ade.shape[0])
        self.sum_ade += float(ade[:, 0].sum())
        self.sum_fde += float(fde[:, 0].sum())
        self.sum_min_ade += float(ade.min(axis=1).sum())
        self.sum_min_fde += float(fde.min(axis=1).sum())
        self.sum_avg_ade += float(ade.mean(axis=1).sum())
        self.sum_avg_fde += float(fde.mean(axis=1).sum())

    def to_dict(self) -> Dict[str, float]:
        if self.n_windows <= 0:
            return {
                "n_windows": 0,
                "ade": float("nan"),
                "fde": float("nan"),
                "avg_ade": float("nan"),
                "avg_fde": float("nan"),
                "min_ade": float("nan"),
                "min_fde": float("nan"),
            }
        denom = float(self.n_windows)
        return {
            "n_windows": int(self.n_windows),
            "ade": self.sum_ade / denom,
            "fde": self.sum_fde / denom,
            "avg_ade": self.sum_avg_ade / denom,
            "avg_fde": self.sum_avg_fde / denom,
            "min_ade": self.sum_min_ade / denom,
            "min_fde": self.sum_min_fde / denom,
        }


def evaluate_trials(
    model: torch.nn.Module,
    trial_df: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    cfg: Config,
    t_h: int,
    t_p: int,
    stride: int,
    batch_size: int,
    cmd_dir: Optional[Path],
    device: torch.device,
) -> Dict[str, float]:
    metrics = MetricAccumulator()

    for _, row in trial_df.iterrows():
        pose_path = resolve_pose_path(row)
        pose = load_pose_array(pose_path)
        cue = load_trial_cmd(row, pose_path=pose_path, cmd_dir=cmd_dir)

        if cue.shape[0] != pose.shape[0]:
            raise ValueError(
                f"pose/cmd length mismatch for trial_id={row.get('trial_id', '')}: "
                f"pose={pose.shape[0]}, cmd={cue.shape[0]}"
            )

        starts = list_window_starts(int(pose.shape[0]), t_h=t_h, t_p=t_p, stride=stride)
        if not starts:
            continue

        for idx in range(0, len(starts), max(1, int(batch_size))):
            batch_starts = starts[idx : idx + max(1, int(batch_size))]
            batch = build_window_batch(
                pose=pose,
                cue=cue,
                starts=batch_starts,
                t_h=t_h,
                t_p=t_p,
                stats=stats,
                device=device,
            )
            pred_rel = sample_future_rel(model=model, batch=batch, cfg=cfg)
            gt_rel = batch["fut_traj_original_scale"]
            dist = torch.linalg.norm(pred_rel - gt_rel[:, None, :, :, :], dim=-1)
            ade = dist.mean(dim=(2, 3)).detach().cpu().numpy()
            fde = dist[:, :, :, -1].mean(dim=2).detach().cpu().numpy()
            metrics.update(ade=ade, fde=fde)

    return metrics.to_dict()


def load_model_for_split(
    split_name: str,
    model_root: Path,
    checkpoint_name: str,
    cfg_arg: str,
    n_samples: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, Config, Path, Path]:
    ckpt_path = resolve_checkpoint_path(split_name=split_name, model_root=model_root, checkpoint_name=checkpoint_name)
    cfg_path = resolve_cfg_path(split_name=split_name, checkpoint_path=ckpt_path, cfg_arg=cfg_arg)

    cfg = Config(str(cfg_path), tag=f"heldout_eval_{split_name}", train_mode=False)
    if getattr(cfg.MODEL, "NAME", None) is None and getattr(cfg.MODEL, "Name", None) is not None:
        cfg.MODEL.NAME = str(cfg.MODEL.Name).lower()
    cfg.device = str(device)
    cfg.denoising_head_preds = int(n_samples)
    cfg.k_preds = int(n_samples)
    if getattr(cfg, "sampling_steps", None) is None:
        cfg.sampling_steps = 10

    dummy_args = SimpleNamespace(
        method=None,
        variant=None,
        decoder=None,
        action_fusion=None,
        enable_dissipativity=False,
        dissipativity_weight=None,
    )
    model = build_network(cfg=cfg, args=dummy_args, logger=logger).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, cfg, ckpt_path, cfg_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("05_eval_motion_generalization")
    p.add_argument("--split-dir", type=str, default=str(default_cross_subject_path("splits")))
    p.add_argument("--split-name", type=str, default="")
    p.add_argument(
        "--model-root",
        type=str,
        default=str(ROOT / "results_rat"),
        help="Root containing per-split experiment folders or checkpoints.",
    )
    p.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    p.add_argument(
        "--cfg",
        type=str,
        default="auto",
        help="Config yaml path. Use {split} placeholder if needed. Default auto-detect from checkpoint dir.",
    )
    p.add_argument(
        "--cmd-dir",
        type=str,
        default=str(VER2_CMD_DIR),
        help="Optional directory containing per-trial command files.",
    )
    p.add_argument("--out-dir", type=str, default=str(default_cross_subject_path("summary")))
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-stride", type=int, default=1)
    p.add_argument("--stats-stride", type=int, default=1)
    p.add_argument("--device", type=str, default="", help="cpu/cuda; empty for auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logger = build_logger()

    split_dir = Path(args.split_dir).expanduser().resolve()
    model_root = Path(args.model_root).expanduser().resolve()
    cmd_dir = Path(args.cmd_dir).expanduser().resolve() if str(args.cmd_dir).strip() else None
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    per_split_dir = ensure_dir(out_dir / "motion_generalization")
    device = torch.device(args.device if str(args.device).strip() else ("cuda" if torch.cuda.is_available() else "cpu"))

    rows: List[Dict[str, object]] = []
    splits = discover_splits(split_dir=split_dir, split_name=str(args.split_name).strip())

    for split_path in splits:
        train_manifest = split_path / "train_manifest.csv"
        test_manifest = split_path / "test_manifest.csv"
        if not train_manifest.exists() or not test_manifest.exists():
            raise FileNotFoundError(f"split missing train/test manifest: {split_path}")

        train_df = pd.read_csv(train_manifest)
        test_df = pd.read_csv(test_manifest)
        if train_df.empty or test_df.empty:
            raise ValueError(f"empty train/test manifest under {split_path}")

        model, cfg, ckpt_path, cfg_path = load_model_for_split(
            split_name=split_path.name,
            model_root=model_root,
            checkpoint_name=str(args.checkpoint_name),
            cfg_arg=str(args.cfg),
            n_samples=int(args.n_samples),
            device=device,
            logger=logger,
        )
        t_h = int(cfg.past_frames)
        t_p = int(cfg.future_frames)
        if int(cfg.agents) <= 0:
            raise ValueError(f"invalid cfg.agents={cfg.agents} for split={split_path.name}")
        sample_pose = load_pose_array(resolve_pose_path(train_df.iloc[0]))
        if int(sample_pose.shape[1]) != int(cfg.agents):
            raise ValueError(
                f"cfg.agents={cfg.agents} mismatches pose agents={sample_pose.shape[1]} "
                f"for split={split_path.name}"
            )

        stats = accumulate_train_stats(
            manifest=train_df,
            t_h=t_h,
            t_p=t_p,
            stride=int(args.stats_stride),
            cmd_dir=cmd_dir,
        )
        cfg.stats = {
            "abs_mean": torch.tensor(stats["abs_mean"], dtype=torch.float32, device=device),
            "abs_std": torch.tensor(stats["abs_std"], dtype=torch.float32, device=device),
            "rel_mean": torch.tensor(stats["rel_mean"], dtype=torch.float32, device=device),
            "rel_std": torch.tensor(stats["rel_std"], dtype=torch.float32, device=device),
            "vel_mean": torch.tensor(stats["vel_mean"], dtype=torch.float32, device=device),
            "vel_std": torch.tensor(stats["vel_std"], dtype=torch.float32, device=device),
            "fut_mean": torch.tensor(stats["fut_mean"], dtype=torch.float32, device=device),
            "fut_std": torch.tensor(stats["fut_std"], dtype=torch.float32, device=device),
        }

        split_rows: List[Dict[str, object]] = []
        train_rats = sorted(train_df["rat_id"].astype(str).dropna().unique().tolist()) if "rat_id" in train_df.columns else []
        rat_groups = (
            test_df.groupby("rat_id", sort=True)
            if "rat_id" in test_df.columns
            else [("heldout", test_df)]
        )

        for rat_id, rat_df in rat_groups:
            metric_row = evaluate_trials(
                model=model,
                trial_df=rat_df,
                stats=stats,
                cfg=cfg,
                t_h=t_h,
                t_p=t_p,
                stride=int(args.eval_stride),
                batch_size=int(args.eval_batch_size),
                cmd_dir=cmd_dir,
                device=device,
            )
            row = {
                "split": split_path.name,
                "test_rat": str(rat_id),
                "train_rats": ",".join(train_rats),
                "model": "codsde",
                "checkpoint": str(ckpt_path),
                "config": str(cfg_path),
                "n_test_trials": int(len(rat_df)),
                "n_test_frames": int(pd.to_numeric(rat_df.get("n_frames", 0), errors="coerce").fillna(0).sum()),
                "n_train_windows": int(stats["n_windows"]),
                "n_samples": int(args.n_samples),
                "T_h": int(t_h),
                "T_p": int(t_p),
                **metric_row,
            }
            split_rows.append(row)
            rows.append(row)

        split_out = ensure_dir(per_split_dir / split_path.name)
        pd.DataFrame(split_rows).to_csv(split_out / "motion_metrics.csv", index=False)
        with (split_out / "train_stats.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split_path.name,
                    "t_h": t_h,
                    "t_p": t_p,
                    "stats_stride": int(args.stats_stride),
                    "n_train_windows": int(stats["n_windows"]),
                    "stats": {k: np.asarray(v).astype(float).tolist() for k, v in stats.items() if k != "n_windows"},
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        logger.info(f"[05] done split={split_path.name}, rows={len(split_rows)}")

    table = pd.DataFrame(rows)
    table_path = per_split_dir / "table_motion_generalization.csv"
    table.to_csv(table_path, index=False)

    if not table.empty:
        agg_rows: List[Dict[str, object]] = []
        numeric_cols = [c for c in ["ade", "fde", "avg_ade", "avg_fde", "min_ade", "min_fde", "n_windows"] if c in table.columns]
        row: Dict[str, object] = {"model": "codsde", "n_splits": int(table["split"].nunique()), "n_rats": int(len(table))}
        for col in numeric_cols:
            values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            row[f"{col}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"{col}_std"] = float(values.std()) if values.size else np.nan
        agg_rows.append(row)
        pd.DataFrame(agg_rows).to_csv(per_split_dir / "table_motion_generalization_agg.csv", index=False)

    with (per_split_dir / "eval_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)
    logger.info(f"[05] table: {table_path}")


if __name__ == "__main__":
    main()
