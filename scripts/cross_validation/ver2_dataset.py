from __future__ import annotations

import ast
import json
import re
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VER2_ROOT = ROOT / "scripts" / "cross_validation" / "ver2"
VER2_SINGLE_KP_DIR = VER2_ROOT / "key_points" / "single_kp"
VER2_CMD_DIR = VER2_ROOT / "cmd"
VER2_FLOW_PATH = VER2_ROOT / "key_points" / "flow_11030.npy"
VER2_CMD_NPY_PATH = VER2_ROOT / "key_points" / "cmd_11030.npy"

VER2_VALID_DATA_RANGES: List[Tuple[int, int]] = [
    (270, 4509),
    (4779, 8520),
    (8790, 11351),
    (11621, 14709),
    (14979, 16797),
    (17067, 19899),
    (20169, 23437),
    (23707, 28388),
    (28658, 31119),
    (31389, 33793),
    (34063, 38505),
    (38775, 42586),
    (42856 + 90, 46223),
    (46493, 48408),
    (48678, 51373),
    (51643, 53895),
    (54165, 57262),
    (57532, 60665),
    (60935, 63529),
    (63799, 65144),
    (65414, 68826),
    (69096, 73127),
    (73397 + 270, 75690),
    (75960 + 270, 77964 - 144),
    (78234 + 270, 80402 - 54),
    (80672 + 270, 83027),
    (83297 + 270, 84907),
    (85177 + 270, 87303),
    (87573, 89142),
    (89412, 90217),
    (90487, 91971),
    (92241 + 270, 96903),
    (97173 + 270, 100611 - 126),
    (100881 + 360, 106107 - 90),
    (106377, 110168 - 234),
    (110438 + 270, 112807 - 162),
]

ACTION_MAP = {"None": 0, "go_ahead": 1, "turn_left": 2, "turn_right": 3}


def default_cross_subject_path(*parts: str) -> Path:
    return ROOT.joinpath("outputs", "cross_subject", *parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return s.strip("_") or "unknown"


def npy_header(path: Path) -> Dict[str, object]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a npy file: {path}")
        major, minor = struct.unpack("BB", f.read(2))
        if (major, minor) == (1, 0):
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = f.read(header_len).decode("latin1")
    return dict(ast.literal_eval(header))


def sorted_ver2_trial_stems(kp_dir: Path = VER2_SINGLE_KP_DIR) -> List[str]:
    stems = []
    for path in sorted(kp_dir.glob("*_origin.npy")):
        stems.append(path.stem)
    if len(stems) != len(VER2_VALID_DATA_RANGES):
        raise ValueError(
            f"ver2 trial count mismatch: files={len(stems)} ranges={len(VER2_VALID_DATA_RANGES)}"
        )
    return stems


def derive_rat_id_from_stem(stem: str, mode: str = "date") -> str:
    if mode == "constant":
        return "11030"
    if mode == "stem":
        return stem
    if mode == "date":
        parts = stem.split("_")
        return parts[0] if parts else stem
    raise ValueError(f"unsupported rat grouping mode: {mode}")


def load_single_kp(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"single_kp must be 3D, got {arr.shape} at {path}")
    if arr.shape[-1] == 2:
        xy = arr
    elif arr.shape[1] == 2:
        xy = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(f"unsupported single_kp shape: {arr.shape} at {path}")
    if xy.shape[-1] != 2:
        raise ValueError(f"expected last dim=2 after transpose, got {xy.shape} at {path}")
    if not np.isfinite(xy).all():
        raise ValueError(f"NaN/Inf found in {path}")
    return xy.astype(np.float32, copy=False)


def parse_cmd_txt(path: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw.startswith("{") or "frame_id" not in raw:
                continue
            try:
                item = ast.literal_eval(raw)
            except Exception:
                continue
            if not isinstance(item, dict) or "frame_id" not in item:
                continue
            final_action = str(item.get("final_action", "None"))
            tar_action = str(item.get("tar_action", "None"))
            voltage = item.get("voltage", 0)
            if voltage in ("None", None):
                voltage = 0.0
            action = 0
            strength = 0.0
            if tar_action != "None":
                action = int(ACTION_MAP.get(final_action, 0))
                strength = float(voltage)
            rows.append(np.array([action, strength], dtype=np.float32))
    if not rows:
        raise ValueError(f"no frame command rows parsed from {path}")
    return np.stack(rows, axis=0).astype(np.float32)


def load_ver2_trial_pair(stem: str, kp_dir: Path = VER2_SINGLE_KP_DIR, cmd_dir: Path = VER2_CMD_DIR) -> Tuple[np.ndarray, np.ndarray]:
    kp_path = kp_dir / f"{stem}.npy"
    cmd_path = cmd_dir / f"{stem}.txt"
    if not kp_path.exists():
        raise FileNotFoundError(f"ver2 keypoint file not found: {kp_path}")
    if not cmd_path.exists():
        raise FileNotFoundError(f"ver2 command file not found: {cmd_path}")
    xy = load_single_kp(kp_path)
    cmd = parse_cmd_txt(cmd_path)
    if xy.shape[0] != cmd.shape[0]:
        raise ValueError(f"pose/cmd length mismatch for {stem}: pose={xy.shape[0]}, cmd={cmd.shape[0]}")
    return xy, cmd


def iter_trimmed_ver2_trials(
    kp_dir: Path = VER2_SINGLE_KP_DIR,
    cmd_dir: Path = VER2_CMD_DIR,
    valid_ranges: Sequence[Tuple[int, int]] = VER2_VALID_DATA_RANGES,
) -> Iterable[Dict[str, object]]:
    cursor = 0
    stems = sorted_ver2_trial_stems(kp_dir=kp_dir)
    for idx, stem in enumerate(stems):
        xy_full, cmd_full = load_ver2_trial_pair(stem=stem, kp_dir=kp_dir, cmd_dir=cmd_dir)
        global_start, global_end = valid_ranges[idx]
        local_start = int(global_start - cursor)
        local_end = int(global_end - cursor)
        if local_start < 0 or local_end > int(xy_full.shape[0]) or local_start >= local_end:
            raise ValueError(
                f"invalid ver2 trim for {stem}: global=({global_start},{global_end}) "
                f"cursor={cursor} local=({local_start},{local_end}) len={xy_full.shape[0]}"
            )

        xy_trim = xy_full[local_start:local_end].astype(np.float32, copy=False)
        cmd_trim = cmd_full[local_start:local_end].astype(np.float32, copy=False)
        yield {
            "index": idx,
            "stem": stem,
            "xy_full": xy_full,
            "cmd_full": cmd_full,
            "xy_trim": xy_trim,
            "cmd_trim": cmd_trim,
            "global_start": int(global_start),
            "global_end": int(global_end),
            "local_start": int(local_start),
            "local_end": int(local_end),
            "source_pose_path": str((kp_dir / f"{stem}.npy").resolve()),
            "source_cmd_path": str((cmd_dir / f"{stem}.txt").resolve()),
        }
        cursor += int(xy_full.shape[0])


def build_ver2_concat(
    kp_dir: Path = VER2_SINGLE_KP_DIR,
    cmd_dir: Path = VER2_CMD_DIR,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    xy_chunks: List[np.ndarray] = []
    cmd_chunks: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    cursor = 0
    for item in iter_trimmed_ver2_trials(kp_dir=kp_dir, cmd_dir=cmd_dir, valid_ranges=VER2_VALID_DATA_RANGES):
        xy_full = np.asarray(item["xy_full"], dtype=np.float32)
        cmd_full = np.asarray(item["cmd_full"], dtype=np.float32)
        xy_chunks.append(xy_full)
        cmd_chunks.append(cmd_full)
        rows.append(
            {
                "stem": item["stem"],
                "global_start": cursor,
                "global_end": cursor + int(xy_full.shape[0]),
                "valid_global_start": int(item["global_start"]),
                "valid_global_end": int(item["global_end"]),
            }
        )
        cursor += int(xy_full.shape[0])
    return np.concatenate(xy_chunks, axis=0), np.concatenate(cmd_chunks, axis=0), rows


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
