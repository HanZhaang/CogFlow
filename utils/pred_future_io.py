import os
import json
import numpy as np
from typing import Any, Dict, Optional, Union

ArrayLike = Union[np.ndarray]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Accept numpy or torch tensor and return numpy."""
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor support (without importing torch explicitly)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")

def save_pred_fut(
    save_path: str,
    pred: ArrayLike,   # (N,B,K,A,F,D)
    fut: ArrayLike,    # (N,B,A,F,D)
    extra: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> str:
    """
    Save offline predictions and GT for fair evaluation.

    Args:
        save_path: path ending with .npz (recommended) or .npy (not recommended here)
        pred: (N,B,K,A,F,D)
        fut:  (N,B,A,F,D)
        extra: optional metadata, e.g. {"dataset":"rat", "model":"cogflow_fm", "seed":0, ...}
               Note: will be JSON-serialized.
        compress: if True, use np.savez_compressed

    Returns:
        save_path
    """
    pred_np = _to_numpy(pred)
    fut_np = _to_numpy(fut)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    meta = {
        "pred_shape": list(pred_np.shape),
        "fut_shape": list(fut_np.shape),
        "pred_dtype": str(pred_np.dtype),
        "fut_dtype": str(fut_np.dtype),
    }
    if extra is not None:
        meta["extra"] = extra

    meta_json = json.dumps(meta, ensure_ascii=False)

    if not save_path.endswith(".npz"):
        # Enforce npz for multi-array
        save_path = save_path + ".npz"

    if compress:
        np.savez_compressed(save_path, pred=pred_np, fut=fut_np, meta=meta_json)
    else:
        np.savez(save_path, pred=pred_np, fut=fut_np, meta=meta_json)

    return save_path


def load_pred_fut(npz_path: str) -> Dict[str, Any]:
    """
    Load saved predictions.

    Returns:
        dict with keys: pred, fut, meta (parsed dict)
    """
    data = np.load(npz_path, allow_pickle=False)
    pred = data["pred"]
    fut = data["fut"]
    meta = json.loads(str(data["meta"]))
    return {"pred": pred, "fut": fut, "meta": meta}
