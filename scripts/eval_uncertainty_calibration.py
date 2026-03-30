#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty calibration from saved prediction samples."
    )
    parser.add_argument("--pred_npz", required=True, help="Path to saved prediction npz.")
    parser.add_argument("--out_dir", required=True, help="Directory to save CSVs and plots.")
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=10,
        help="Number of quantile buckets for uncertainty binning.",
    )
    parser.add_argument(
        "--expected_k",
        type=int,
        default=20,
        help="Expected number of stochastic samples K. Set <=0 to disable the check.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for saved figures.",
    )
    return parser.parse_args()


def _maybe_parse_meta(data: np.lib.npyio.NpzFile) -> Optional[Dict[str, object]]:
    if "meta" not in data.files:
        return None
    raw_meta = data["meta"]
    if isinstance(raw_meta, np.ndarray) and raw_meta.ndim == 0:
        raw_meta = raw_meta.item()
    try:
        return json.loads(str(raw_meta))
    except json.JSONDecodeError:
        return {"raw_meta": str(raw_meta)}


def _normalize_repo_shapes(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pred.ndim < 5 or gt.ndim < 4:
        raise ValueError(
            f"Repo-format arrays require pred.ndim>=5 and gt.ndim>=4, got pred={pred.shape}, gt={gt.shape}"
        )
    if pred.ndim != gt.ndim + 1:
        raise ValueError(f"Expected pred.ndim = gt.ndim + 1, got pred={pred.shape}, gt={gt.shape}")
    if tuple(pred.shape[:-4]) != tuple(gt.shape[:-3]):
        raise ValueError(f"Leading dims mismatch: pred={pred.shape}, gt={gt.shape}")
    if tuple(pred.shape[-3:]) != tuple(gt.shape[-3:]):
        raise ValueError(f"Tail dims mismatch: pred={pred.shape}, gt={gt.shape}")

    leading_shape = pred.shape[:-4]
    num_samples = int(np.prod(leading_shape)) if leading_shape else 1
    k_dim, joint_dim, time_dim, coord_dim = pred.shape[-4:]

    pred_std = pred.reshape(num_samples, k_dim, joint_dim, time_dim, coord_dim).transpose(0, 1, 3, 2, 4)
    gt_std = gt.reshape(num_samples, joint_dim, time_dim, coord_dim).transpose(0, 2, 1, 3)
    return pred_std.astype(np.float32, copy=False), gt_std.astype(np.float32, copy=False)


def load_prediction_bundle(npz_path: Path, expected_k: int) -> Dict[str, object]:
    data = np.load(npz_path, allow_pickle=False)
    meta = _maybe_parse_meta(data)
    cmd = data["cmd"] if "cmd" in data.files else None

    if {"pred_samples", "gt"}.issubset(data.files):
        pred_samples = data["pred_samples"].astype(np.float32, copy=False)
        gt = data["gt"].astype(np.float32, copy=False)
    elif {"pred", "fut"}.issubset(data.files):
        pred_samples, gt = _normalize_repo_shapes(
            data["pred"].astype(np.float32, copy=False),
            data["fut"].astype(np.float32, copy=False),
        )
    else:
        raise KeyError(
            f"Unsupported npz keys: {data.files}. Expected either (pred_samples, gt) or (pred, fut)."
        )

    if pred_samples.ndim != 5:
        raise ValueError(f"pred_samples must be 5D [N,K,T,J,D], got {pred_samples.shape}")
    if gt.ndim != 4:
        raise ValueError(f"gt must be 4D [N,T,J,D], got {gt.shape}")
    if expected_k > 0 and pred_samples.shape[1] != expected_k:
        raise ValueError(f"Expected K={expected_k}, got K={pred_samples.shape[1]}")
    if pred_samples.shape[0] != gt.shape[0]:
        raise ValueError(f"N mismatch: pred_samples={pred_samples.shape}, gt={gt.shape}")
    if pred_samples.shape[2:] != gt.shape[1:]:
        raise ValueError(f"Trajectory shape mismatch: pred_samples={pred_samples.shape}, gt={gt.shape}")
    if not np.isfinite(pred_samples).all() or not np.isfinite(gt).all():
        raise ValueError("pred_samples/gt contain non-finite values.")

    return {
        "pred_samples": pred_samples,
        "gt": gt,
        "cmd": cmd,
        "meta": meta,
    }


def compute_sample_mean(pred_samples: np.ndarray) -> np.ndarray:
    return pred_samples.mean(axis=1)


def compute_uncertainty_timestep(pred_samples: np.ndarray) -> np.ndarray:
    pred_mean = compute_sample_mean(pred_samples)
    sq_dev = np.square(pred_samples - pred_mean[:, None, ...], dtype=np.float32)
    return np.sqrt(sq_dev.mean(axis=(1, 3, 4), dtype=np.float64)).astype(np.float32)


def compute_uncertainty_traj(pred_samples: np.ndarray) -> np.ndarray:
    pred_mean = compute_sample_mean(pred_samples)
    sq_dev = np.square(pred_samples - pred_mean[:, None, ...], dtype=np.float32)
    return np.sqrt(sq_dev.mean(axis=(1, 2, 3, 4), dtype=np.float64)).astype(np.float32)


def compute_ade_timestep(pred_mean: np.ndarray, gt: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(pred_mean - gt, axis=-1)
    return dist.mean(axis=-1).astype(np.float32)


def compute_ade(pred_mean: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return compute_ade_timestep(pred_mean, gt).mean(axis=1)


def compute_fde(pred_mean: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return compute_ade_timestep(pred_mean, gt)[:, -1]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    result = {
        "n": int(x.size),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return result

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    result.update(
        {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        }
    )
    return result


def build_correlation_rows(
    uncertainty_traj: np.ndarray,
    uncertainty_timestep: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    ade_timestep: np.ndarray,
) -> List[Dict[str, object]]:
    pairs = [
        ("trajectory", "ADE", uncertainty_traj, ade),
        ("trajectory", "FDE", uncertainty_traj, fde),
        ("trajectory", "FDE_last_uncertainty", uncertainty_timestep[:, -1], fde),
        ("timestep", "ADE_timestep", uncertainty_timestep.reshape(-1), ade_timestep.reshape(-1)),
    ]

    rows: List[Dict[str, object]] = []
    for granularity, target, x, y in pairs:
        row = {
            "granularity": granularity,
            "target": target,
        }
        row.update(_safe_corr(x, y))
        rows.append(row)
    return rows


def _quantile_edges(values: np.ndarray, num_buckets: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.linspace(0.0, 1.0, max(num_buckets, 1) + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size == 1:
        eps = 1e-6 if edges[0] == 0 else abs(edges[0]) * 1e-6
        edges = np.array([edges[0], edges[0] + eps], dtype=np.float64)
    return edges


def build_bucket_rows(
    uncertainty_traj: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    num_buckets: int,
) -> List[Dict[str, object]]:
    edges = _quantile_edges(uncertainty_traj, num_buckets)
    bucket_ids = np.digitize(uncertainty_traj, edges[1:-1], right=False)
    total = int(uncertainty_traj.shape[0])
    rows: List[Dict[str, object]] = []
    for bucket_idx in range(edges.size - 1):
        mask = bucket_ids == bucket_idx
        if not np.any(mask):
            continue
        unc = uncertainty_traj[mask]
        ade_bucket = ade[mask]
        fde_bucket = fde[mask]
        rows.append(
            {
                "bucket_id": bucket_idx,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "uncertainty_left": float(edges[bucket_idx]),
                "uncertainty_right": float(edges[bucket_idx + 1]),
                "uncertainty_mean": float(unc.mean()),
                "uncertainty_std": float(unc.std(ddof=0)),
                "ade_mean": float(ade_bucket.mean()),
                "ade_std": float(ade_bucket.std(ddof=0)),
                "fde_mean": float(fde_bucket.mean()),
                "fde_std": float(fde_bucket.std(ddof=0)),
                "mean_error_gap": float(ade_bucket.mean() - unc.mean()),
                "bucket_rank": f"{bucket_idx + 1}/{edges.size - 1}",
                "total_samples": total,
            }
        )
    return rows


def _collapse_cmd_sequence(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return seq.astype(np.int64, copy=False)
    collapsed = np.zeros(seq.shape[0], dtype=np.int64)
    for idx, row in enumerate(seq):
        row = np.asarray(row).reshape(-1)
        non_zero = row[row != 0]
        if non_zero.size > 0:
            collapsed[idx] = int(non_zero[-1])
        elif row.size > 0:
            collapsed[idx] = int(row[-1])
    return collapsed


def derive_regime_labels(cmd: Optional[np.ndarray], num_samples: int, time_steps: int) -> Optional[np.ndarray]:
    if cmd is None:
        return None
    cmd = np.asarray(cmd)
    if cmd.shape[0] != num_samples:
        return None

    if cmd.ndim == 1:
        regime_ids = cmd.astype(np.int64, copy=False)
    elif cmd.ndim == 2:
        if cmd.shape[1] == time_steps:
            regime_ids = _collapse_cmd_sequence(cmd)
        elif cmd.shape[1] == 7:
            regime_ids = np.argmax(cmd[:, :4], axis=-1).astype(np.int64)
        else:
            regime_ids = np.argmax(cmd, axis=-1).astype(np.int64)
    else:
        if cmd.shape[1] != time_steps:
            return None
        if cmd.shape[-1] == 7:
            step_ids = np.argmax(cmd[..., :4], axis=-1)
        elif cmd.shape[-1] == 13:
            bits = (cmd > 0.5).astype(np.int64)
            weights = (1 << np.arange(bits.shape[-1], dtype=np.int64))
            step_ids = np.tensordot(bits, weights, axes=([-1], [0]))
        else:
            step_ids = np.argmax(cmd, axis=-1)
        regime_ids = _collapse_cmd_sequence(step_ids)

    return np.array([f"cmd_{int(regime_id)}" for regime_id in regime_ids], dtype=object)


def build_regime_rows(
    regime_labels: Optional[np.ndarray],
    uncertainty_traj: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
) -> List[Dict[str, object]]:
    if regime_labels is None:
        return []
    rows: List[Dict[str, object]] = []
    unique_labels = sorted({str(label) for label in regime_labels})
    for label in unique_labels:
        mask = regime_labels == label
        if not np.any(mask):
            continue
        rows.append(
            {
                "regime": label,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "uncertainty_mean": float(uncertainty_traj[mask].mean()),
                "uncertainty_std": float(uncertainty_traj[mask].std(ddof=0)),
                "ade_mean": float(ade[mask].mean()),
                "ade_std": float(ade[mask].std(ddof=0)),
                "fde_mean": float(fde[mask].mean()),
                "fde_std": float(fde[mask].std(ddof=0)),
            }
        )
    return rows


def build_sample_rows(
    uncertainty_traj: np.ndarray,
    uncertainty_timestep: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    ade_timestep: np.ndarray,
    regime_labels: Optional[np.ndarray],
) -> List[Dict[str, object]]:
    num_samples, time_steps = uncertainty_timestep.shape
    rows: List[Dict[str, object]] = []
    for sample_idx in range(num_samples):
        row: Dict[str, object] = {
            "sample_index": sample_idx,
            "uncertainty_traj": float(uncertainty_traj[sample_idx]),
            "uncertainty_last": float(uncertainty_timestep[sample_idx, -1]),
            "ade": float(ade[sample_idx]),
            "fde": float(fde[sample_idx]),
        }
        if regime_labels is not None:
            row["regime"] = str(regime_labels[sample_idx])
        for time_idx in range(time_steps):
            row[f"uncertainty_t{time_idx:03d}"] = float(uncertainty_timestep[sample_idx, time_idx])
            row[f"error_t{time_idx:03d}"] = float(ade_timestep[sample_idx, time_idx])
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(
    uncertainty_traj: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    correlation_rows: Sequence[Dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    corr_map = {(row["granularity"], row["target"]): row for row in correlation_rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_specs = [
        (axes[0], ade, "ADE", corr_map.get(("trajectory", "ADE"))),
        (axes[1], fde, "FDE", corr_map.get(("trajectory", "FDE"))),
    ]

    for axis, y_values, label, corr in plot_specs:
        axis.scatter(uncertainty_traj, y_values, s=14, alpha=0.45, edgecolors="none")
        if uncertainty_traj.size >= 2 and np.unique(uncertainty_traj).size >= 2:
            slope, intercept = np.polyfit(uncertainty_traj, y_values, deg=1)
            x_line = np.linspace(uncertainty_traj.min(), uncertainty_traj.max(), 100)
            axis.plot(x_line, slope * x_line + intercept, color="tab:red", linewidth=1.5)
        axis.set_xlabel("Uncertainty")
        axis.set_ylabel(label)
        axis.set_title(f"Uncertainty vs {label}")
        if corr:
            axis.text(
                0.03,
                0.97,
                "Pearson={:.4f}\nSpearman={:.4f}".format(
                    corr["pearson_r"],
                    corr["spearman_rho"],
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )
        axis.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_barplot(bucket_rows: Sequence[Dict[str, object]], output_path: Path, dpi: int) -> None:
    if not bucket_rows:
        return

    x = np.arange(len(bucket_rows))
    width = 0.35
    ade_mean = np.array([row["ade_mean"] for row in bucket_rows], dtype=np.float64)
    fde_mean = np.array([row["fde_mean"] for row in bucket_rows], dtype=np.float64)
    uncertainty_mean = np.array([row["uncertainty_mean"] for row in bucket_rows], dtype=np.float64)
    labels = [row["bucket_rank"] for row in bucket_rows]

    fig, axis_left = plt.subplots(figsize=(max(8.0, len(bucket_rows) * 1.2), 5.2))
    axis_left.bar(x - width / 2, ade_mean, width=width, label="ADE", color="tab:blue", alpha=0.8)
    axis_left.bar(x + width / 2, fde_mean, width=width, label="FDE", color="tab:orange", alpha=0.8)
    axis_left.set_xlabel("Uncertainty Bucket")
    axis_left.set_ylabel("Error")
    axis_left.set_xticks(x)
    axis_left.set_xticklabels(labels)
    axis_left.grid(axis="y", alpha=0.25, linestyle="--")

    axis_right = axis_left.twinx()
    axis_right.plot(x, uncertainty_mean, color="black", marker="o", linewidth=1.5, label="Mean uncertainty")
    axis_right.set_ylabel("Mean Uncertainty")

    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(
    output_path: Path,
    npz_path: Path,
    pred_samples: np.ndarray,
    gt: np.ndarray,
    correlation_rows: Sequence[Dict[str, object]],
    bucket_rows: Sequence[Dict[str, object]],
    regime_rows: Sequence[Dict[str, object]],
) -> None:
    summary = {
        "pred_npz": str(npz_path),
        "num_samples": int(pred_samples.shape[0]),
        "num_modes": int(pred_samples.shape[1]),
        "time_steps": int(pred_samples.shape[2]),
        "joints": int(pred_samples.shape[3]),
        "coord_dim": int(pred_samples.shape[4]),
        "gt_shape": list(gt.shape),
        "correlations": list(correlation_rows),
        "num_buckets": len(bucket_rows),
        "num_regimes": len(regime_rows),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    npz_path = Path(args.pred_npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_prediction_bundle(npz_path, expected_k=args.expected_k)
    pred_samples = bundle["pred_samples"]
    gt = bundle["gt"]
    cmd = bundle["cmd"]

    pred_mean = compute_sample_mean(pred_samples)
    uncertainty_traj = compute_uncertainty_traj(pred_samples)
    uncertainty_timestep = compute_uncertainty_timestep(pred_samples)
    ade_timestep = compute_ade_timestep(pred_mean, gt)
    ade = compute_ade(pred_mean, gt)
    fde = compute_fde(pred_mean, gt)

    regime_labels = derive_regime_labels(cmd, num_samples=pred_samples.shape[0], time_steps=pred_samples.shape[2])

    correlation_rows = build_correlation_rows(uncertainty_traj, uncertainty_timestep, ade, fde, ade_timestep)
    bucket_rows = build_bucket_rows(uncertainty_traj, ade, fde, args.num_buckets)
    sample_rows = build_sample_rows(uncertainty_traj, uncertainty_timestep, ade, fde, ade_timestep, regime_labels)
    regime_rows = build_regime_rows(regime_labels, uncertainty_traj, ade, fde)

    write_csv(correlation_rows, out_dir / "correlation_metrics.csv")
    write_csv(bucket_rows, out_dir / "bucket_metrics.csv")
    write_csv(sample_rows, out_dir / "sample_metrics.csv")
    if regime_rows:
        write_csv(regime_rows, out_dir / "regime_metrics.csv")
    else:
        write_csv([], out_dir / "regime_metrics.csv")

    plot_scatter(
        uncertainty_traj=uncertainty_traj,
        ade=ade,
        fde=fde,
        correlation_rows=correlation_rows,
        output_path=out_dir / "scatter_uncertainty_vs_error.png",
        dpi=args.dpi,
    )
    plot_bucket_barplot(bucket_rows, out_dir / "bucket_barplot.png", dpi=args.dpi)
    write_summary_json(
        out_dir / "summary.json",
        npz_path=npz_path,
        pred_samples=pred_samples,
        gt=gt,
        correlation_rows=correlation_rows,
        bucket_rows=bucket_rows,
        regime_rows=regime_rows,
    )

    print(f"Saved uncertainty calibration outputs to {out_dir}")
    print(f"Samples={pred_samples.shape[0]} K={pred_samples.shape[1]} T={pred_samples.shape[2]}")


if __name__ == "__main__":
    main()
