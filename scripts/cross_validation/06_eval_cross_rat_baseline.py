#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validation.ver2_dataset import default_cross_subject_path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def js_divergence_1d(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    pa = pa.astype(np.float64) + 1e-12
    pb = pb.astype(np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    js = 0.5 * (np.sum(pa * np.log(pa / m)) + np.sum(pb * np.log(pb / m)))
    return float(js)


def _js_discrete(pa: np.ndarray, pb: np.ndarray) -> float:
    pa = np.asarray(pa, dtype=np.float64) + 1e-12
    pb = np.asarray(pb, dtype=np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    return float(0.5 * (np.sum(pa * np.log(pa / m)) + np.sum(pb * np.log(pb / m))))


def _aggregate_transition_matrix(seqs: List[np.ndarray], n_states: int, laplace: float) -> np.ndarray:
    counts = np.full((n_states, n_states), float(laplace), dtype=np.float64)
    for seq in seqs:
        seq = np.asarray(seq, dtype=np.int32).reshape(-1)
        if seq.size < 2:
            continue
        src = seq[:-1]
        dst = seq[1:]
        valid = np.logical_and.reduce((src >= 0, src < n_states, dst >= 0, dst < n_states))
        src = src[valid]
        dst = dst[valid]
        if src.size == 0:
            continue
        np.add.at(counts, (src, dst), 1.0)
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return counts / row_sum


def transition_js(
    real_sequences: List[np.ndarray],
    sim_sequences: List[np.ndarray],
    n_states: int,
    laplace: float = 1e-6,
) -> Dict[str, float]:
    p = _aggregate_transition_matrix(real_sequences, n_states=n_states, laplace=laplace)
    q = _aggregate_transition_matrix(sim_sequences, n_states=n_states, laplace=laplace)
    row_js = [_js_discrete(p[i], q[i]) for i in range(n_states)]
    return {
        "transition_js": _js_discrete(p.reshape(-1), q.reshape(-1)),
        "transition_js_rowavg": float(np.mean(row_js)) if row_js else float("nan"),
    }


def _dwell_lengths(seqs: List[np.ndarray], n_states: int) -> List[List[int]]:
    out: List[List[int]] = [[] for _ in range(n_states)]
    for seq in seqs:
        seq = np.asarray(seq, dtype=np.int32).reshape(-1)
        if seq.size == 0:
            continue
        cur = int(seq[0])
        run = 1
        for value in seq[1:]:
            value = int(value)
            if value == cur:
                run += 1
                continue
            if 0 <= cur < n_states:
                out[cur].append(run)
            cur = value
            run = 1
        if 0 <= cur < n_states:
            out[cur].append(run)
    return out


def dwell_ks(real_sequences: List[np.ndarray], sim_sequences: List[np.ndarray], n_states: int) -> Dict[str, object]:
    real = _dwell_lengths(real_sequences, n_states=n_states)
    sim = _dwell_lengths(sim_sequences, n_states=n_states)
    per_state: List[Dict[str, object]] = []
    ks_vals: List[float] = []
    weights: List[int] = []
    for state in range(n_states):
        a = np.asarray(real[state], dtype=np.float64)
        b = np.asarray(sim[state], dtype=np.float64)
        if a.size == 0 or b.size == 0:
            ks = float("nan")
        else:
            ks = float(ks_2samp(a, b).statistic)
        per_state.append(
            {
                "state": int(state),
                "ks": ks,
                "n_real": int(a.size),
                "n_sim": int(b.size),
            }
        )
        if np.isfinite(ks):
            ks_vals.append(ks)
            weights.append(int(a.size))
    macro = float(np.mean(ks_vals)) if ks_vals else float("nan")
    weighted = float(np.average(ks_vals, weights=weights)) if ks_vals and sum(weights) > 0 else float("nan")
    return {
        "dwell_ks_macro": macro,
        "dwell_ks_weighted": weighted,
        "per_state": per_state,
    }


def extract_motion_features(xy: np.ndarray, fps: float) -> Dict[str, np.ndarray]:
    # xy: (T,V,2)
    centers = np.asarray(xy, dtype=np.float64).mean(axis=1)  # (T,2)
    dxy = np.diff(centers, axis=0, prepend=centers[:1])
    step = np.linalg.norm(dxy, axis=1)
    speed = step * max(float(fps), 1e-6)
    heading = np.arctan2(dxy[:, 1], dxy[:, 0])
    turning = wrap_angle(np.diff(heading, prepend=heading[:1]))
    curvature = np.abs(turning) / (step + 1e-6)
    return {
        "speed": speed.astype(np.float64),
        "turning_angle": turning.astype(np.float64),
        "curvature": curvature.astype(np.float64),
    }


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    for c in ["trial_id", "rat_id", "pose_path", "state_path"]:
        if c not in df.columns:
            raise ValueError(f"manifest missing required column: {c}")
    if df.empty:
        raise ValueError("manifest is empty")
    return df


def infer_n_states_from_sequences(seqs_by_rat: Dict[str, List[np.ndarray]]) -> int:
    mx = -1
    for seqs in seqs_by_rat.values():
        for s in seqs:
            if s.size == 0:
                continue
            mx = max(mx, int(np.max(s)))
    if mx < 0:
        raise ValueError("cannot infer n_states from empty state sequences")
    return int(mx + 1)


def summarize_metric(metric: str, values: List[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"metric": metric, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n_pairs": 0}
    return {
        "metric": metric,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_pairs": int(arr.size),
    }


def main() -> None:
    p = argparse.ArgumentParser("06_eval_cross_rat_baseline")
    p.add_argument("--manifest", type=str, default=str(default_cross_subject_path("manifest.csv")))
    p.add_argument("--out-dir", type=str, default=str(default_cross_subject_path("summary", "baseline")))
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--n-states", type=int, default=-1, help="<=0 means infer from data")
    p.add_argument("--transition-laplace", type=float, default=1e-6)
    p.add_argument("--features", type=str, default="speed,turning_angle,curvature")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--skip-motion", action="store_true")
    p.add_argument("--fps-col", type=str, default="fps")
    args = p.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    df = load_manifest(Path(args.manifest))
    features = parse_csv_list(args.features)
    rats = sorted(df["rat_id"].astype(str).dropna().unique().tolist())
    if len(rats) < 2:
        raise RuntimeError(f"need >=2 rats to compute cross-rat baseline, got {rats}")

    # Aggregate raw data by rat.
    seqs_by_rat: Dict[str, List[np.ndarray]] = {r: [] for r in rats}
    motion_by_rat: Dict[str, Dict[str, List[np.ndarray]]] = {
        r: {f: [] for f in features} for r in rats
    }

    for _, row in df.iterrows():
        rat = str(row["rat_id"])
        pose_path = Path(str(row["pose_path"]))
        state_path = Path(str(row["state_path"]))
        if not pose_path.exists():
            continue
        xy = np.asarray(np.load(pose_path, allow_pickle=True), dtype=np.float32)
        if xy.ndim != 3 or xy.shape[-1] != 2:
            continue
        fps = float(row[args.fps_col]) if args.fps_col in df.columns and pd.notna(row[args.fps_col]) else 30.0

        if state_path.exists():
            s = np.asarray(np.load(state_path, allow_pickle=True)).reshape(-1).astype(np.int32)
            s = s[s >= 0]
            if s.size >= int(args.min_len):
                seqs_by_rat[rat].append(s)

        if not args.skip_motion:
            feats = extract_motion_features(xy=xy, fps=fps)
            for f in features:
                if f in feats:
                    motion_by_rat[rat][f].append(feats[f])

    if int(args.n_states) > 0:
        n_states = int(args.n_states)
    else:
        n_states = infer_n_states_from_sequences(seqs_by_rat)

    transition_rows: List[Dict[str, object]] = []
    dwell_rows: List[Dict[str, object]] = []
    dwell_state_rows: List[Dict[str, object]] = []
    motion_rows: List[Dict[str, object]] = []

    for rat_i, rat_j in itertools.combinations(rats, 2):
        seq_i = seqs_by_rat[rat_i]
        seq_j = seqs_by_rat[rat_j]
        if not seq_i or not seq_j:
            continue

        trans = transition_js(
            real_sequences=seq_i,
            sim_sequences=seq_j,
            n_states=n_states,
            laplace=float(args.transition_laplace),
        )
        transition_rows.append(
            {
                "rat_i": rat_i,
                "rat_j": rat_j,
                "transition_js": float(trans["transition_js"]),
                "transition_js_rowavg": float(trans["transition_js_rowavg"]),
                "n_seq_i": int(len(seq_i)),
                "n_seq_j": int(len(seq_j)),
                "n_trans_i": int(sum(max(0, len(s) - 1) for s in seq_i)),
                "n_trans_j": int(sum(max(0, len(s) - 1) for s in seq_j)),
            }
        )

        dwell = dwell_ks(seq_i, seq_j, n_states=n_states)
        dwell_rows.append(
            {
                "rat_i": rat_i,
                "rat_j": rat_j,
                "dwell_ks_macro": float(dwell["dwell_ks_macro"]),
                "dwell_ks_weighted": float(dwell["dwell_ks_weighted"]),
            }
        )
        for item in dwell["per_state"]:
            dwell_state_rows.append(
                {
                    "rat_i": rat_i,
                    "rat_j": rat_j,
                    "state": int(item["state"]),
                    "ks": float(item["ks"]),
                    "n_real": int(item["n_real"]),
                    "n_sim": int(item["n_sim"]),
                }
            )

        if not args.skip_motion:
            for feat in features:
                ai = np.concatenate(motion_by_rat[rat_i][feat], axis=0) if motion_by_rat[rat_i][feat] else np.array([])
                aj = np.concatenate(motion_by_rat[rat_j][feat], axis=0) if motion_by_rat[rat_j][feat] else np.array([])
                ai = ai[np.isfinite(ai)]
                aj = aj[np.isfinite(aj)]
                w1 = float(wasserstein_distance(ai, aj)) if ai.size > 0 and aj.size > 0 else float("nan")
                js = js_divergence_1d(ai, aj, bins=int(args.bins))
                motion_rows.append(
                    {
                        "rat_i": rat_i,
                        "rat_j": rat_j,
                        "feature": feat,
                        "n_i": int(ai.size),
                        "n_j": int(aj.size),
                        "W1": w1,
                        "JS": js,
                    }
                )

    trans_df = pd.DataFrame(transition_rows)
    dwell_df = pd.DataFrame(dwell_rows)
    dwell_state_df = pd.DataFrame(dwell_state_rows)
    motion_df = pd.DataFrame(motion_rows)

    trans_path = out_dir / "cross_rat_transition.csv"
    dwell_path = out_dir / "cross_rat_dwell.csv"
    dwell_state_path = out_dir / "cross_rat_dwell_per_state.csv"
    motion_path = out_dir / "cross_rat_motion.csv"
    trans_df.to_csv(trans_path, index=False)
    dwell_df.to_csv(dwell_path, index=False)
    dwell_state_df.to_csv(dwell_state_path, index=False)
    motion_df.to_csv(motion_path, index=False)

    # Aggregate baseline summary table.
    summary_rows: List[Dict[str, object]] = []
    if not trans_df.empty:
        summary_rows.append(summarize_metric("real_rat_transition_js", trans_df["transition_js"].tolist()))
        summary_rows.append(summarize_metric("real_rat_transition_js_rowavg", trans_df["transition_js_rowavg"].tolist()))
    if not dwell_df.empty:
        summary_rows.append(summarize_metric("real_rat_dwell_ks_macro", dwell_df["dwell_ks_macro"].tolist()))
        summary_rows.append(summarize_metric("real_rat_dwell_ks_weighted", dwell_df["dwell_ks_weighted"].tolist()))
    if not motion_df.empty:
        for feat in sorted(motion_df["feature"].astype(str).unique().tolist()):
            sub = motion_df[motion_df["feature"].astype(str) == feat]
            summary_rows.append(summarize_metric(f"real_rat_{feat}_JS", sub["JS"].tolist()))
            summary_rows.append(summarize_metric(f"real_rat_{feat}_W1", sub["W1"].tolist()))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "table_cross_rat_baseline.csv"
    summary_df.to_csv(summary_path, index=False)

    with (out_dir / "baseline_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    print(f"[06] transition: {trans_path}")
    print(f"[06] dwell: {dwell_path}")
    print(f"[06] motion: {motion_path}")
    print(f"[06] summary: {summary_path}")


if __name__ == "__main__":
    main()
