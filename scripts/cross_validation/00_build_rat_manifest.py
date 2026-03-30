#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validation.ver2_dataset import (
    VER2_CMD_DIR,
    VER2_SINGLE_KP_DIR,
    VER2_VALID_DATA_RANGES,
    build_ver2_concat,
    default_cross_subject_path,
    derive_rat_id_from_stem,
    dump_json,
    ensure_dir,
    iter_trimmed_ver2_trials,
)


XY_KEYS = ("xy", "keypoints", "kp", "kp_seq", "traj", "trajectory")
STATE_KEYS = ("state", "states", "label", "labels")


def _safe_name(text: str) -> str:
    import re

    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return s.strip("_") or "unknown"


def _load_trial_npz(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trial npz not found: {p}")

    with np.load(p, allow_pickle=True) as z:
        xy = None
        for k in XY_KEYS:
            if k in z:
                xy = np.asarray(z[k], dtype=np.float32)
                break
        if xy is None:
            raise KeyError(f"xy key not found in {p}. available={list(z.files)}")
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"xy must be (T,N,2), got {xy.shape} in {p}")
        if not np.isfinite(xy).all():
            raise ValueError(f"xy has NaN/Inf in {p}")

        state = None
        for k in STATE_KEYS:
            if k in z:
                state = np.asarray(z[k]).reshape(-1).astype(np.int32)
                break
        if state is None:
            state = np.full((xy.shape[0],), -1, dtype=np.int32)
        elif state.shape[0] != xy.shape[0]:
            raise ValueError(f"state length mismatch in {p}: state={state.shape[0]} vs T={xy.shape[0]}")

        fps = 30.0
        if "fps" in z:
            fps_raw = np.asarray(z["fps"]).reshape(-1)
            if fps_raw.size > 0 and np.isfinite(fps_raw[0]):
                fps = float(fps_raw[0])

    return xy, state, fps


def _resolve_unique_trial_id(trial_id: str, used: Dict[str, int]) -> str:
    key = str(trial_id)
    if key not in used:
        used[key] = 1
        return key
    n = used[key]
    used[key] += 1
    return f"{key}_dup{n}"


def _validate_manifest(df: pd.DataFrame) -> None:
    required = [
        "trial_id",
        "rat_id",
        "scene",
        "task",
        "pose_path",
        "state_path",
        "meta_path",
        "fps",
        "n_frames",
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"manifest missing columns: {miss}")

    if df["trial_id"].duplicated().any():
        dup = df[df["trial_id"].duplicated()]["trial_id"].astype(str).tolist()[:5]
        raise ValueError(f"duplicate trial_id in manifest: {dup}")

    if (df["rat_id"].astype(str).str.strip() == "").any():
        raise ValueError("manifest has empty rat_id")

    for _, row in df.iterrows():
        for key in ["pose_path", "state_path", "meta_path"]:
            path = Path(str(row[key]))
            if not path.exists():
                raise FileNotFoundError(f"{key} missing: {path}")
        cmd_path = str(row.get("cmd_path", "")).strip()
        if cmd_path and not Path(cmd_path).exists():
            raise FileNotFoundError(f"cmd_path missing: {cmd_path}")


def _export_flow_by_rat(manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    by_rat_dir = ensure_dir(out_dir / "flow_by_rat")
    summary_rows: List[Dict[str, object]] = []

    for rat_id, gdf in manifest.groupby("rat_id", sort=True):
        gdf = gdf.sort_values(["trial_id"]).reset_index(drop=True)
        chunks_xy: List[np.ndarray] = []
        chunks_state: List[np.ndarray] = []
        chunks_cmd: List[np.ndarray] = []
        ranges_rows: List[Dict[str, object]] = []
        cursor = 0
        n_joints = -1

        for _, row in gdf.iterrows():
            xy = np.asarray(np.load(str(row["pose_path"]), allow_pickle=True), dtype=np.float32)
            st = np.asarray(np.load(str(row["state_path"]), allow_pickle=True)).reshape(-1).astype(np.int32)
            if xy.ndim != 3 or xy.shape[-1] != 2:
                raise ValueError(f"pose array must be (T,N,2): {row['pose_path']}, got {xy.shape}")
            if st.shape[0] != xy.shape[0]:
                raise ValueError(f"state/xy mismatch for trial={row['trial_id']}: state={st.shape[0]}, frames={xy.shape[0]}")
            cmd = None
            if str(row.get("cmd_path", "")).strip():
                cmd = np.asarray(np.load(str(row["cmd_path"]), allow_pickle=True), dtype=np.float32)
                if cmd.shape[0] != xy.shape[0]:
                    raise ValueError(f"cmd/xy mismatch for trial={row['trial_id']}: cmd={cmd.shape[0]}, frames={xy.shape[0]}")

            T = int(xy.shape[0])
            n_joints = int(xy.shape[1])
            chunks_xy.append(xy)
            chunks_state.append(st)
            if cmd is not None:
                chunks_cmd.append(cmd)

            rg_start = cursor
            rg_end = cursor + T
            ranges_rows.append(
                {
                    "rat_id": rat_id,
                    "trial_id": row["trial_id"],
                    "rat_flow_start": int(rg_start),
                    "rat_flow_end": int(rg_end),
                    "n_frames": int(T),
                    "source_pose_path": row["pose_path"],
                    "source_state_path": row["state_path"],
                    "source_cmd_path": row.get("cmd_path", ""),
                }
            )
            cursor += T

        xy_all = np.concatenate(chunks_xy, axis=0) if chunks_xy else np.zeros((0, 0, 2), dtype=np.float32)
        st_all = np.concatenate(chunks_state, axis=0) if chunks_state else np.zeros((0,), dtype=np.int32)
        cmd_all = np.concatenate(chunks_cmd, axis=0) if chunks_cmd else None

        rat_safe = _safe_name(rat_id)
        flow_path = by_rat_dir / f"flow_{rat_safe}.npy"
        state_path = by_rat_dir / f"state_{rat_safe}.npy"
        ranges_path = by_rat_dir / f"flow_{rat_safe}_ranges.csv"
        cmd_path = by_rat_dir / f"cmd_{rat_safe}.npy"

        np.save(flow_path, xy_all)
        np.save(state_path, st_all)
        if cmd_all is not None:
            np.save(cmd_path, cmd_all)
        pd.DataFrame(ranges_rows).to_csv(ranges_path, index=False)

        summary_rows.append(
            {
                "rat_id": rat_id,
                "flow_path": str(flow_path.resolve()),
                "state_path": str(state_path.resolve()),
                "cmd_path": str(cmd_path.resolve()) if cmd_all is not None else "",
                "ranges_csv": str(ranges_path.resolve()),
                "n_trials": int(len(gdf)),
                "n_frames": int(xy_all.shape[0]),
                "n_joints": int(n_joints),
            }
        )

    out = pd.DataFrame(summary_rows).sort_values("rat_id").reset_index(drop=True)
    out_path = by_rat_dir / "rat_flow_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"[00] flow-by-rat summary: {out_path}")
    return out


def _build_manifest_from_trial_index(args: argparse.Namespace) -> pd.DataFrame:
    trial_index_path = Path(args.trial_index)
    if not trial_index_path.exists():
        raise FileNotFoundError(f"trial index not found: {trial_index_path}")

    idx = pd.read_csv(trial_index_path)
    for col in [args.npz_path_col]:
        if col not in idx.columns:
            raise ValueError(f"trial index missing required column: {col}")

    if not bool(args.include_sim):
        if "is_real" not in idx.columns:
            raise ValueError("trial index missing is_real; use --include-sim to bypass filter")
        idx = idx[pd.to_numeric(idx["is_real"], errors="coerce").fillna(0).astype(int) == 1].copy()

    if idx.empty:
        raise ValueError("no rows left after filtering trial index")

    out_dir = ensure_dir(Path(args.out_dir))
    trials_dir = ensure_dir(out_dir / "trials")
    meta_dir = ensure_dir(out_dir / "meta")

    used_trial_ids: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []

    for row_id, row in idx.reset_index(drop=True).iterrows():
        source_npz = str(row.get(args.npz_path_col, "")).strip()
        if not source_npz:
            raise ValueError(f"empty npz path at row={row_id}")

        xy, state, fps_from_npz = _load_trial_npz(source_npz)
        n_frames = int(xy.shape[0])

        trial_id_raw = str(row.get(args.trial_id_col, "")).strip()
        if not trial_id_raw:
            trial_id_raw = f"{args.trial_prefix}_{row_id:05d}"
        trial_id = _resolve_unique_trial_id(trial_id_raw, used_trial_ids)

        if bool(args.use_source_rat_id):
            rat_raw = str(row.get(args.rat_col, "")).strip()
            rat_id = rat_raw if rat_raw else str(args.default_rat_id)
        else:
            rat_id = str(args.default_rat_id)

        scene_raw = str(row.get(args.scene_col, "")).strip() if args.scene_col in idx.columns else ""
        task_raw = str(row.get(args.task_col, "")).strip() if args.task_col in idx.columns else ""
        split_raw = str(row.get(args.split_group_col, "")).strip() if args.split_group_col in idx.columns else ""
        scene = scene_raw if scene_raw else str(args.default_scene)
        task = task_raw if task_raw else str(args.default_task)
        split_group = split_raw if split_raw else ""

        fps = float(args.default_fps)
        if args.fps_col in idx.columns:
            fps_csv = pd.to_numeric(pd.Series([row.get(args.fps_col)]), errors="coerce").iloc[0]
            if pd.notna(fps_csv):
                fps = float(fps_csv)
        elif np.isfinite(fps_from_npz):
            fps = float(fps_from_npz)

        pose_path = trials_dir / f"{trial_id}_pose.npy"
        state_path = trials_dir / f"{trial_id}_state.npy"
        np.save(pose_path, xy.astype(np.float32))
        np.save(state_path, state.astype(np.int32))

        meta = {
            "trial_id": trial_id,
            "rat_id": rat_id,
            "scene": scene,
            "task": task,
            "split_group": split_group,
            "source_trial_index": str(trial_index_path.resolve()),
            "source_npz_path": str(Path(source_npz).resolve()),
            "source_row": int(row_id),
        }
        meta_path = meta_dir / f"{trial_id}.json"
        dump_json(meta_path, meta)

        rows.append(
            {
                "trial_id": trial_id,
                "rat_id": rat_id,
                "scene": scene,
                "task": task,
                "split_group": split_group,
                "pose_path": str(pose_path.resolve()),
                "state_path": str(state_path.resolve()),
                "meta_path": str(meta_path.resolve()),
                "fps": float(fps),
                "n_frames": int(n_frames),
                "source_trial_index": str(trial_index_path.resolve()),
                "source_npz_path": str(Path(source_npz).resolve()),
            }
        )

    manifest = pd.DataFrame(rows)
    _validate_manifest(manifest)
    return manifest


def _build_manifest_from_ver2(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = ensure_dir(Path(args.out_dir))
    trials_dir = ensure_dir(out_dir / "trials")
    meta_dir = ensure_dir(out_dir / "meta")

    rows: List[Dict[str, object]] = []
    for item in iter_trimmed_ver2_trials(
        kp_dir=Path(args.ver2_kp_dir),
        cmd_dir=Path(args.ver2_cmd_dir),
        valid_ranges=VER2_VALID_DATA_RANGES,
    ):
        stem = str(item["stem"])
        trial_id = stem
        rat_id = derive_rat_id_from_stem(stem=stem, mode=str(args.ver2_group_by))
        scene = rat_id
        task = str(args.default_task)
        split_group = rat_id

        pose_path = trials_dir / f"{trial_id}_pose.npy"
        state_path = trials_dir / f"{trial_id}_state.npy"
        cmd_path = trials_dir / f"{trial_id}_cmd.npy"

        xy = np.asarray(item["xy_trim"], dtype=np.float32)
        cmd = np.asarray(item["cmd_trim"], dtype=np.float32)
        state = np.full((xy.shape[0],), -1, dtype=np.int32)

        np.save(pose_path, xy)
        np.save(state_path, state)
        np.save(cmd_path, cmd)

        meta = {
            "trial_id": trial_id,
            "rat_id": rat_id,
            "scene": scene,
            "task": task,
            "split_group": split_group,
            "source_pose_path": item["source_pose_path"],
            "source_cmd_path": item["source_cmd_path"],
            "global_range_start": int(item["global_start"]),
            "global_range_end": int(item["global_end"]),
            "local_range_start": int(item["local_start"]),
            "local_range_end": int(item["local_end"]),
        }
        meta_path = meta_dir / f"{trial_id}.json"
        dump_json(meta_path, meta)

        rows.append(
            {
                "trial_id": trial_id,
                "rat_id": rat_id,
                "scene": scene,
                "task": task,
                "split_group": split_group,
                "pose_path": str(pose_path.resolve()),
                "state_path": str(state_path.resolve()),
                "cmd_path": str(cmd_path.resolve()),
                "meta_path": str(meta_path.resolve()),
                "fps": float(args.default_fps),
                "n_frames": int(xy.shape[0]),
                "source_trial_index": "",
                "source_npz_path": item["source_pose_path"],
                "source_cmd_path": item["source_cmd_path"],
                "global_range_start": int(item["global_start"]),
                "global_range_end": int(item["global_end"]),
                "local_range_start": int(item["local_start"]),
                "local_range_end": int(item["local_end"]),
            }
        )

    manifest = pd.DataFrame(rows)
    _validate_manifest(manifest)

    if bool(args.export_full_concat):
        xy_all, cmd_all, ranges = build_ver2_concat(kp_dir=Path(args.ver2_kp_dir), cmd_dir=Path(args.ver2_cmd_dir))
        np.save(out_dir / "flow_11030.npy", xy_all)
        np.save(out_dir / "cmd_11030.npy", cmd_all)
        pd.DataFrame(ranges).to_csv(out_dir / "flow_11030_ranges.csv", index=False)

    return manifest


def _run_export_from_manifest(manifest_input: str, out_dir: str) -> None:
    p = Path(manifest_input)
    if not p.exists():
        raise FileNotFoundError(f"manifest input not found: {p}")
    manifest = pd.read_csv(p)
    _validate_manifest(manifest)
    _export_flow_by_rat(manifest, out_dir=Path(out_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("00_build_rat_manifest")
    p.add_argument("--source", type=str, default="ver2", choices=["ver2", "trial_index"])
    p.add_argument("--trial-index", type=str, default="")
    p.add_argument("--npz-path-col", type=str, default="path")
    p.add_argument("--trial-id-col", type=str, default="trial_id")
    p.add_argument("--rat-col", type=str, default="rat_id")
    p.add_argument("--scene-col", type=str, default="env_id")
    p.add_argument("--task-col", type=str, default="task_id")
    p.add_argument("--split-group-col", type=str, default="group_id")
    p.add_argument("--fps-col", type=str, default="fps")
    p.add_argument("--include-sim", action="store_true")

    p.add_argument("--default-rat-id", type=str, default="0")
    p.add_argument("--use-source-rat-id", action="store_true")
    p.add_argument("--default-scene", type=str, default="ver2")
    p.add_argument("--default-task", type=str, default="motion")
    p.add_argument("--default-fps", type=float, default=30.0)
    p.add_argument("--trial-prefix", type=str, default="trial")

    p.add_argument("--ver2-kp-dir", type=str, default=str(VER2_SINGLE_KP_DIR))
    p.add_argument("--ver2-cmd-dir", type=str, default=str(VER2_CMD_DIR))
    p.add_argument("--ver2-group-by", type=str, default="date", choices=["date", "stem", "constant"])
    p.add_argument("--export-full-concat", action="store_true")

    p.add_argument("--out-dir", type=str, default=str(default_cross_subject_path()))
    p.add_argument("--split-flow-by-rat", action="store_true")
    p.add_argument(
        "--manifest-input",
        type=str,
        default="",
        help="If provided, skip source parsing and export flow_by_rat from this manifest.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if str(args.manifest_input).strip():
        if not bool(args.split_flow_by_rat):
            raise ValueError("--manifest-input requires --split-flow-by-rat")
        _run_export_from_manifest(manifest_input=args.manifest_input, out_dir=args.out_dir)
        return

    if str(args.source) == "ver2":
        manifest = _build_manifest_from_ver2(args)
    else:
        if not str(args.trial_index).strip():
            raise ValueError("--trial-index is required when --source trial_index")
        manifest = _build_manifest_from_trial_index(args)

    out_dir = ensure_dir(Path(args.out_dir))
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"[00] manifest: {manifest_path}")
    print(f"[00] source: {args.source}")
    print(f"[00] trials: {len(manifest)}")
    stat = (
        manifest.groupby("rat_id", sort=True)["n_frames"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "n_trials", "sum": "n_frames"})
    )
    for _, row in stat.iterrows():
        print(f"[00] {row['rat_id']}: {int(row['n_trials'])} trials, {int(row['n_frames'])} frames")
    print(f"[00] total: {int(len(manifest))} trials, {int(manifest['n_frames'].sum())} frames")

    if bool(args.split_flow_by_rat):
        _export_flow_by_rat(manifest, out_dir=out_dir)


if __name__ == "__main__":
    main()
