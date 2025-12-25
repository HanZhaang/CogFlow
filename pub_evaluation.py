import numpy as np
from utils.pred_future_io import load_pred_fut
from typing import Dict, Optional, Union, Sequence

ArrayLike = Union[np.ndarray]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Accept numpy or torch tensor and return numpy."""
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor support (without importing torch explicitly)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")


def evaluate_predictions(
    pred: ArrayLike,  # (N, B, K, A, F, D)
    fut: ArrayLike,   # (N, B, A, F, D)
    horizons: Optional[Sequence[int]] = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Unified evaluator (offline).

    Args:
        pred: predicted trajectories, shape (N, B, K, A, F, D)
        fut:  ground-truth future trajectories, shape (N, B, A, F, D)
        horizons: optional list of frame indices (1..F) to report ADE/FDE at multiple horizons.
                  If None, will report only full-horizon ADE and final-step FDE.
                  Example: horizons=[5, 10, 20, 30] for F=30.
        eps: numerical stability.

    Returns:
        performance: dict of metrics. Each value is a numpy array.
            - "ADE_min": (H,) or (1,)    best-of-K ADE over frames up to horizon
            - "FDE_min": (H,) or (1,)    best-of-K final displacement at horizon
            - "ADE_avg": (H,) or (1,)    average over K of ADE
            - "FDE_avg": (H,) or (1,)    average over K of FDE
            - "Diversity": (H,) or (1,)  mean pairwise distance among K predictions (optional but useful)
            - "num_trajs": scalar        total evaluated trajectories count = N*B*A
            - "K": scalar                number of modes
            - "F": scalar                total frames
            - "D": scalar                coord dimension
    Notes:
        - Distance is L2 over last dimension D.
        - Aggregation is mean over all (N,B,A) trajectories.
    """
    pred_np = _to_numpy(pred).astype(np.float64, copy=False)
    fut_np = _to_numpy(fut).astype(np.float64, copy=False)

    # if pred_np.ndim != 6:
    #     raise ValueError(f"pred must be 6D (N,B,K,A,F,D). Got {pred_np.shape}")
    # if fut_np.ndim != 5:
    #     raise ValueError(f"fut must be 5D (N,B,A,F,D). Got {fut_np.shape}")

    B, K, A, F, D = pred_np.shape
    if fut_np.shape != (B, A, F, D):
        raise ValueError(f"Shape mismatch: pred {pred_np.shape} vs fut {fut_np.shape}")

    if horizons is None:
        horizons = [F]
    else:
        horizons = list(horizons)
        for h in horizons:
            if not (1 <= h <= F):
                raise ValueError(f"horizon must be in [1, F]. Got {h} with F={F}")

    # Expand fut to align with pred modes: (N,B,1,A,F,D) -> broadcast along K
    fut_exp = fut_np[:, None, :, :, :]  # (N,B,1,A,F,D)

    # Per-frame L2 error: (N,B,K,A,F)
    diff = pred_np - fut_exp
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), eps))

    # Flatten trajectories for stable aggregation: T = N*B*A
    # dist_flat: (T, K, F)
    dist_flat = dist.transpose(0, 2, 1, 3).reshape(B * A, K, F)

    # Helper to compute metrics at a given horizon h (1..F)
    def _metrics_at_h(h: int):
        # ADE per mode: mean over frames [0:h)
        ade_k = dist_flat[:, :, :h].mean(axis=-1)  # (T,K)

        # FDE per mode: frame (h-1)
        fde_k = dist_flat[:, :, h - 1]            # (T,K)

        # best-of-K
        ade_min = ade_k.min(axis=1).mean()        # scalar
        fde_min = fde_k.min(axis=1).mean()

        # average over K
        ade_avg = ade_k.mean(axis=1).mean()
        fde_avg = fde_k.mean(axis=1).mean()

        # Diversity: mean pairwise distance among K predicted endpoints at horizon h
        # pred_end: (T,K,D)
        pred_end = pred_np[:, :, :, h - 1, :].transpose(0, 2, 1, 3).reshape(B * A, K, D)
        if K <= 1:
            diversity = 0.0
        else:
            # compute mean pairwise L2 efficiently
            # ||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
            x2 = np.sum(pred_end * pred_end, axis=-1, keepdims=True)   # (T,K,1)
            gram = pred_end @ pred_end.transpose(0, 2, 1)              # (T,K,K)
            d2 = x2 + x2.transpose(0, 2, 1) - 2.0 * gram               # (T,K,K)
            d2 = np.maximum(d2, 0.0)
            # take upper triangle mean (exclude diag)
            triu = np.triu_indices(K, k=1)
            diversity = np.sqrt(d2[:, triu[0], triu[1]] + eps).mean()

        return ade_min, fde_min, ade_avg, fde_avg, diversity

    ADE_min, FDE_min, ADE_avg, FDE_avg, DIV = [], [], [], [], []
    for h in horizons:
        a_min, f_min, a_avg, f_avg, div = _metrics_at_h(h)
        ADE_min.append(a_min)
        FDE_min.append(f_min)
        ADE_avg.append(a_avg)
        FDE_avg.append(f_avg)
        DIV.append(div)

    performance = {
        "ADE_min": np.array(ADE_min, dtype=np.float64),
        "FDE_min": np.array(FDE_min, dtype=np.float64),
        "ADE_avg": np.array(ADE_avg, dtype=np.float64),
        "FDE_avg": np.array(FDE_avg, dtype=np.float64),
        "Diversity": np.array(DIV, dtype=np.float64),
        "num_trajs": np.array(B * A, dtype=np.int64),
        "K": np.array(K, dtype=np.int64),
        "F": np.array(F, dtype=np.int64),
        "D": np.array(D, dtype=np.int64),
        "horizons": np.array(horizons, dtype=np.int64),
    }
    return performance

if __name__ == '__main__':
    bundle = load_pred_fut("/root/CogFlow/cfg/full_cfg/npz/rat_cogflow.npz")
    perf = evaluate_predictions(bundle["pred"], bundle["fut"], horizons=[10, 20, 30])
    print(perf["ADE_min"], perf["FDE_min"], perf["Diversity"])