#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validation.ver2_dataset import default_cross_subject_path


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
        raise RuntimeError(f"no split found under {split_dir}")
    return out


def load_pose_cmd_from_manifest_row(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    pose_path = Path(str(row.get("pose_path", ""))).expanduser()
    cmd_path = Path(str(row.get("cmd_path", ""))).expanduser()
    if not pose_path.exists():
        raise FileNotFoundError(f"pose_path missing: {pose_path}")
    if not cmd_path.exists():
        raise FileNotFoundError(f"cmd_path missing: {cmd_path}")

    pose = np.asarray(np.load(pose_path, allow_pickle=True), dtype=np.float32)
    cmd = np.asarray(np.load(cmd_path, allow_pickle=True), dtype=np.float32)

    if pose.ndim != 3 or pose.shape[-1] != 2:
        raise ValueError(f"pose must be (T,A,2), got {pose.shape} at {pose_path}")
    if cmd.ndim != 2:
        raise ValueError(f"cmd must be (T,C), got {cmd.shape} at {cmd_path}")
    if pose.shape[0] != cmd.shape[0]:
        raise ValueError(f"pose/cmd length mismatch: pose={pose.shape[0]} cmd={cmd.shape[0]} for {pose_path}")
    return pose, normalize_cmd_for_rat_loader(cmd)


def normalize_cmd_for_rat_loader(cmd: np.ndarray) -> np.ndarray:
    cmd = np.asarray(cmd, dtype=np.float32)
    if cmd.ndim != 2:
        raise ValueError(f"cmd must be 2D, got {cmd.shape}")
    if cmd.shape[1] == 2:
        return cmd.astype(np.float32, copy=False)
    if cmd.shape[1] >= 7:
        action = np.argmax(cmd[:, :4], axis=1).astype(np.float32)
        strength = cmd[:, 4].astype(np.float32)
        return np.stack([action, strength], axis=1).astype(np.float32)
    raise ValueError(f"unsupported cmd shape for rat loader: {cmd.shape}")


def window_starts(length: int, seq_len: int, stride: int) -> List[int]:
    if length < seq_len:
        return []
    return list(range(0, int(length - seq_len + 1), max(1, int(stride))))


def build_windows_from_manifest(manifest_path: Path, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    df = pd.read_csv(manifest_path)
    if df.empty:
        raise ValueError(f"manifest is empty: {manifest_path}")

    pose_windows: List[np.ndarray] = []
    cmd_windows: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        pose, cmd = load_pose_cmd_from_manifest_row(row)
        starts = window_starts(length=int(pose.shape[0]), seq_len=seq_len, stride=stride)
        trial_id = str(row.get("trial_id", ""))
        rat_id = str(row.get("rat_id", ""))
        for start in starts:
            end = start + seq_len
            pose_windows.append(pose[start:end])
            cmd_windows.append(cmd[start:end])
            meta_rows.append(
                {
                    "trial_id": trial_id,
                    "rat_id": rat_id,
                    "start": int(start),
                    "end": int(end),
                    "source_pose_path": str(row.get("pose_path", "")),
                    "source_cmd_path": str(row.get("cmd_path", "")),
                }
            )

    if not pose_windows:
        raise RuntimeError(f"no windows generated from {manifest_path}; check seq_len/stride")

    return (
        np.stack(pose_windows, axis=0).astype(np.float32),
        np.stack(cmd_windows, axis=0).astype(np.float32),
        meta_rows,
    )


def materialize_split_dataset(split_path: Path, dataset_root: Path, past_frames: int, future_frames: int, stride: int) -> Dict[str, object]:
    seq_len = int(past_frames) + int(future_frames)
    train_manifest = split_path / "train_manifest.csv"
    val_manifest = split_path / "val_manifest.csv"
    test_manifest = split_path / "test_manifest.csv"
    if not train_manifest.exists() or not val_manifest.exists() or not test_manifest.exists():
        raise FileNotFoundError(
            f"split is missing train/val/test manifests: {split_path}. "
            "Rerun 01_make_loro_splits.py to regenerate the held-out split."
        )

    train_pose, train_cmd, train_meta = build_windows_from_manifest(train_manifest, seq_len=seq_len, stride=stride)
    val_pose, val_cmd, val_meta = build_windows_from_manifest(val_manifest, seq_len=seq_len, stride=stride)

    target_dir = ensure_dir(dataset_root / "rat_ver2_smooth_3060")
    train_pose_path = target_dir / "rat_pose_train.npy"
    train_cmd_path = target_dir / "rat_stim_train.npy"
    val_pose_path = target_dir / "rat_pose_test.npy"
    val_cmd_path = target_dir / "rat_stim_test.npy"

    np.save(train_pose_path, train_pose)
    np.save(train_cmd_path, train_cmd)
    np.save(val_pose_path, val_pose)
    np.save(val_cmd_path, val_cmd)

    pd.DataFrame(train_meta).to_csv(dataset_root / "train_windows.csv", index=False)
    pd.DataFrame(val_meta).to_csv(dataset_root / "val_windows.csv", index=False)

    summary = {
        "split": split_path.name,
        "dataset_root": str(dataset_root.resolve()),
        "seq_len": int(seq_len),
        "past_frames": int(past_frames),
        "future_frames": int(future_frames),
        "train_windows": int(train_pose.shape[0]),
        "val_windows": int(val_pose.shape[0]),
        "agents": int(train_pose.shape[2]),
        "train_pose_path": str(train_pose_path.resolve()),
        "train_cmd_path": str(train_cmd_path.resolve()),
        "val_pose_path": str(val_pose_path.resolve()),
        "val_cmd_path": str(val_cmd_path.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "heldout_test_manifest": str(test_manifest.resolve()),
    }
    with (dataset_root / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    return summary


def write_split_cfg(
    split_name: str,
    cfg_template: Path,
    cfg_out_path: Path,
    results_root_dir: Path,
    dataset_root: Path,
    n_train: int,
    n_test: int,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    epochs: Optional[int],
    lr: Optional[float],
    past_frames: int,
    future_frames: int,
) -> Dict[str, object]:
    with cfg_template.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["results_root_dir"] = str(results_root_dir.resolve())
    cfg["data_dir"] = str(dataset_root.resolve())
    cfg["n_train"] = int(n_train)
    cfg["n_test"] = int(n_test)
    cfg["past_frames"] = int(past_frames)
    cfg["future_frames"] = int(future_frames)
    cfg["train_batch_size"] = int(train_batch_size)
    cfg["test_batch_size"] = int(test_batch_size)
    cfg["num_workers"] = int(num_workers)
    cfg["dataset"] = "rat"
    cfg["dataset_name"] = "rat_dataset"
    cfg["MODEL"]["NAME"] = "cogflow"
    cfg["MODEL"]["MODEL_OUT_DIM"] = int(future_frames) * 2
    cfg["notes"] = f"LORO CogFlow split={split_name}"

    if epochs is not None:
        cfg["OPTIMIZATION"]["NUM_EPOCHS"] = int(epochs)
    if lr is not None:
        cfg["OPTIMIZATION"]["LR"] = float(lr)

    ensure_dir(cfg_out_path.parent)
    with cfg_out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def expected_run_dir(results_root_dir: Path, cfg_path: Path, exp_name: str) -> Path:
    cfg_name = cfg_path.stem
    return results_root_dir / cfg_name / f"{exp_name}_"


def canonical_checkpoint_dir(motion_out_dir: Path, split_name: str) -> Path:
    return motion_out_dir / split_name / "models"


def canonical_checkpoint_path(motion_out_dir: Path, split_name: str) -> Path:
    return canonical_checkpoint_dir(motion_out_dir, split_name) / "checkpoint_best.pt"


def resolve_existing_motion_best(split_name: str, args: argparse.Namespace) -> Optional[Path]:
    candidates = [
        canonical_checkpoint_path(Path(args.motion_out_dir).resolve(), split_name),
    ]
    for raw in str(args.motion_existing_roots).split(","):
        raw = raw.strip()
        if not raw:
            continue
        root = Path(raw)
        root = root if root.is_absolute() else (ROOT / root)
        candidates.append(root.resolve() / split_name / "models" / "checkpoint_best.pt")
        candidates.append(root.resolve() / split_name / "checkpoint_best.pt")
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def archive_run_artifacts(run_dir: Path, cfg_path: Path, motion_out_dir: Path, split_name: str) -> Dict[str, str]:
    ckpt_src = run_dir / "models" / "checkpoint_best.pt"
    if not ckpt_src.exists():
        raise FileNotFoundError(f"training finished but checkpoint_best.pt not found: {ckpt_src}")

    split_root = ensure_dir(Path(motion_out_dir) / split_name)
    split_model_dir = ensure_dir(split_root / "models")
    split_cfg_path = split_root / cfg_path.name
    split_meta_path = split_root / "train_run_meta.json"
    ckpt_dst = split_model_dir / "checkpoint_best.pt"

    shutil.copy2(ckpt_src, ckpt_dst)
    shutil.copy2(cfg_path, split_cfg_path)
    meta = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_src": str(ckpt_src.resolve()),
        "checkpoint_dst": str(ckpt_dst.resolve()),
        "config_path": str(split_cfg_path.resolve()),
    }
    with split_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)
    return meta


def run_cogflow_train_for_split(split_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    split_name = split_path.name
    prep_root = ensure_dir(Path(args.prep_root).resolve())
    dataset_root = ensure_dir(prep_root / split_name / "dataset")
    cfg_out_path = prep_root / split_name / "cfg" / f"{split_name}.yml"
    results_root_dir = ensure_dir(Path(args.motion_out_dir).resolve() / "_train_runs")

    dataset_info = materialize_split_dataset(
        split_path=split_path,
        dataset_root=dataset_root,
        past_frames=int(args.past_frames),
        future_frames=int(args.future_frames),
        stride=int(args.window_stride),
    )
    write_split_cfg(
        split_name=split_name,
        cfg_template=Path(args.cfg_template).resolve(),
        cfg_out_path=cfg_out_path,
        results_root_dir=results_root_dir,
        dataset_root=dataset_root,
        n_train=int(dataset_info["train_windows"]),
        n_test=int(dataset_info["val_windows"]),
        train_batch_size=int(args.train_batch_size),
        test_batch_size=int(args.test_batch_size),
        num_workers=int(args.num_workers),
        epochs=int(args.epochs) if args.epochs > 0 else None,
        lr=float(args.lr) if args.lr > 0 else None,
        past_frames=int(args.past_frames),
        future_frames=int(args.future_frames),
    )

    exp_name = split_name
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--cfg",
        str(cfg_out_path.resolve()),
        "--exp",
        exp_name,
        "--method",
        "cogflow",
    ]
    print(f"[02] run cogflow train: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    run_dir = expected_run_dir(results_root_dir=results_root_dir, cfg_path=cfg_out_path, exp_name=exp_name)

    out: Dict[str, object] = {
        "split": split_name,
        "train_returncode": int(proc.returncode),
        "generated_cfg": str(cfg_out_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        **dataset_info,
    }
    if proc.returncode == 0:
        out.update(archive_run_artifacts(run_dir=run_dir, cfg_path=cfg_out_path, motion_out_dir=Path(args.motion_out_dir).resolve(), split_name=split_name))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("02_fit_loro_models")
    p.add_argument("--split-dir", type=str, default=str(default_cross_subject_path("splits")))
    p.add_argument("--split-name", type=str, default="", help="optional one split, e.g. loro_2023-2-16")
    p.add_argument("--prep-root", type=str, default=str(default_cross_subject_path("prepared_cogflow")))
    p.add_argument("--summary-out-dir", type=str, default=str(default_cross_subject_path("models")))
    p.add_argument("--motion-out-dir", type=str, default=str(ROOT / "results_rat" / "cross_validation"))
    p.add_argument("--motion-existing-roots", type=str, default=str(ROOT / "results_rat" / "cross_validation"))
    p.add_argument("--cfg-template", type=str, default=str(ROOT / "cfg" / "full_cfg" / "cor_rat_fm_mn.yml"))
    p.add_argument("--skip-motion-if-best-exists", action="store_true")

    p.add_argument("--fit-motion", action="store_true", help="Compatibility flag; CogFlow motion training is the default behavior.")
    p.add_argument("--fit-state", action="store_true", help="Unsupported. State-model training has been removed from this script.")

    p.add_argument("--past-frames", type=int, default=30)
    p.add_argument("--future-frames", type=int, default=60)
    p.add_argument("--window-stride", type=int, default=15)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--test-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=600, help="<=0 keeps template value")
    p.add_argument("--lr", type=float, default=0.001, help="<=0 keeps template value")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.fit_state:
        raise ValueError("02_fit_loro_models no longer trains state models. Use CogFlow motion training only.")

    split_paths = discover_splits(Path(args.split_dir).resolve(), split_name=str(args.split_name).strip())
    summary_out_dir = ensure_dir(Path(args.summary_out_dir).resolve())
    rows: List[Dict[str, object]] = []

    for split_path in split_paths:
        print(f"[02] processing split: {split_path.name}")
        existing_best = resolve_existing_motion_best(split_name=split_path.name, args=args)
        if args.skip_motion_if_best_exists and existing_best is not None:
            row = {
                "split": split_path.name,
                "motion_status": "skip_exists",
                "checkpoint_dst": str(existing_best),
            }
            rows.append(row)
            print(f"[02] skip motion train for {split_path.name}: found {existing_best}")
            continue

        result = run_cogflow_train_for_split(split_path=split_path, args=args)
        result["motion_status"] = "ok" if int(result["train_returncode"]) == 0 else f"failed({result['train_returncode']})"
        rows.append(result)

    summary_path = summary_out_dir / "fit_loro_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    with (summary_out_dir / "fit_loro_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)
    print(f"[02] summary: {summary_path}")


if __name__ == "__main__":
    main()
