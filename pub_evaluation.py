# SPDX-License-Identifier: MIT
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from utils.pred_future_io import load_pred_fut

ArrayLike = Union[np.ndarray]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")


def evaluate_predictions(
    pred: ArrayLike,
    fut: ArrayLike,
    horizons: Optional[Sequence[int]] = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    pred_np = _to_numpy(pred).astype(np.float64, copy=False)
    fut_np = _to_numpy(fut).astype(np.float64, copy=False)

    bsz, k_modes, n_agents, future_frames, coord_dim = pred_np.shape
    if fut_np.shape != (bsz, n_agents, future_frames, coord_dim):
        raise ValueError(f"Shape mismatch: pred {pred_np.shape} vs fut {fut_np.shape}")

    if horizons is None:
        horizons = [future_frames]
    else:
        horizons = list(horizons)
        for horizon in horizons:
            if not (1 <= horizon <= future_frames):
                raise ValueError(
                    f"horizon must be in [1, {future_frames}]. Got {horizon}"
                )

    fut_expanded = fut_np[:, None, :, :, :]
    dist = np.sqrt(np.maximum(np.sum((pred_np - fut_expanded) ** 2, axis=-1), eps))
    dist_flat = dist.transpose(0, 2, 1, 3).reshape(bsz * n_agents, k_modes, future_frames)

    def _metrics_at_horizon(horizon: int):
        ade_k = dist_flat[:, :, :horizon].mean(axis=-1)
        fde_k = dist_flat[:, :, horizon - 1]

        ade_min = ade_k.min(axis=1).mean()
        fde_min = fde_k.min(axis=1).mean()
        ade_avg = ade_k.mean(axis=1).mean()
        fde_avg = fde_k.mean(axis=1).mean()

        pred_end = pred_np[:, :, :, horizon - 1, :].transpose(0, 2, 1, 3).reshape(
            bsz * n_agents, k_modes, coord_dim
        )
        if k_modes <= 1:
            diversity = 0.0
        else:
            x2 = np.sum(pred_end * pred_end, axis=-1, keepdims=True)
            gram = pred_end @ pred_end.transpose(0, 2, 1)
            d2 = np.maximum(x2 + x2.transpose(0, 2, 1) - 2.0 * gram, 0.0)
            triu = np.triu_indices(k_modes, k=1)
            diversity = np.sqrt(d2[:, triu[0], triu[1]] + eps).mean()

        return ade_min, fde_min, ade_avg, fde_avg, diversity

    ade_min_ls, fde_min_ls, ade_avg_ls, fde_avg_ls, div_ls = [], [], [], [], []
    for horizon in horizons:
        ade_min, fde_min, ade_avg, fde_avg, diversity = _metrics_at_horizon(horizon)
        ade_min_ls.append(ade_min)
        fde_min_ls.append(fde_min)
        ade_avg_ls.append(ade_avg)
        fde_avg_ls.append(fde_avg)
        div_ls.append(diversity)

    return {
        "ADE_min": np.array(ade_min_ls, dtype=np.float64),
        "FDE_min": np.array(fde_min_ls, dtype=np.float64),
        "ADE_avg": np.array(ade_avg_ls, dtype=np.float64),
        "FDE_avg": np.array(fde_avg_ls, dtype=np.float64),
        "Diversity": np.array(div_ls, dtype=np.float64),
        "num_trajs": np.array(bsz * n_agents, dtype=np.int64),
        "K": np.array(k_modes, dtype=np.int64),
        "F": np.array(future_frames, dtype=np.int64),
        "D": np.array(coord_dim, dtype=np.int64),
        "horizons": np.array(horizons, dtype=np.int64),
    }


def _default_horizons(total_frames: int) -> Sequence[int]:
    if total_frames >= 10 and total_frames % 10 == 0:
        return list(range(10, total_frames + 1, 10))
    return [total_frames]


def performance_to_serializable(
    performance: Dict[str, np.ndarray]
) -> Dict[str, Union[int, float, list]]:
    out = {}
    for key, value in performance.items():
        if isinstance(value, np.ndarray):
            out[key] = value.item() if value.ndim == 0 else value.tolist()
        else:
            out[key] = value
    return out


def print_performance(performance: Dict[str, np.ndarray]) -> None:
    horizons = performance["horizons"].tolist()
    ade_min = performance["ADE_min"].tolist()
    fde_min = performance["FDE_min"].tolist()
    ade_avg = performance["ADE_avg"].tolist()
    fde_avg = performance["FDE_avg"].tolist()
    diversity = performance["Diversity"].tolist()

    print(
        f"num_trajs={int(performance['num_trajs'])} "
        f"K={int(performance['K'])} F={int(performance['F'])} D={int(performance['D'])}"
    )
    for horizon, ade_m, fde_m, ade_a, fde_a, div in zip(
        horizons, ade_min, fde_min, ade_avg, fde_avg, diversity
    ):
        print(
            "horizon={:>3d} | ADE_min={:.6f} | FDE_min={:.6f} | "
            "ADE_avg={:.6f} | FDE_avg={:.6f} | Diversity={:.6f}".format(
                int(horizon), ade_m, fde_m, ade_a, fde_a, div
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline public evaluation for saved prediction npz files."
    )
    parser.add_argument("--npz_path", required=True, help="Path to the saved prediction npz.")
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Evaluation horizons in frames. Default: auto infer as 10,20,...,F when possible.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to dump metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_pred_fut(args.npz_path)
    pred = bundle["pred"]
    fut = bundle["fut"]
    total_frames = pred.shape[-2]
    horizons = args.horizons if args.horizons else _default_horizons(total_frames)
    performance = evaluate_predictions(pred, fut, horizons=horizons)
    print_performance(performance)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(performance_to_serializable(performance), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
