#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validation.ver2_dataset import default_cross_subject_path
try:
    from simactrat.state_models import DiscreteHMMModel, MarkovChainModel, SemiMarkovModel
except ImportError as exc:
    DiscreteHMMModel = None
    MarkovChainModel = None
    SemiMarkovModel = None
    _SIMACTRAT_IMPORT_ERROR = exc
else:
    _SIMACTRAT_IMPORT_ERROR = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


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


def load_sequences(manifest_path: Path, min_len: int) -> tuple[List[np.ndarray], int]:
    df = pd.read_csv(manifest_path)
    if "state_path" not in df.columns:
        raise ValueError(f"manifest missing state_path: {manifest_path}")
    seqs: List[np.ndarray] = []
    total_frames = 0
    for _, row in df.iterrows():
        path = Path(str(row["state_path"]))
        if not path.exists():
            continue
        s = np.asarray(np.load(path, allow_pickle=True)).reshape(-1).astype(np.int32)
        s = s[s >= 0]
        if s.size < int(min_len):
            continue
        seqs.append(s)
        total_frames += int(s.size)
    return seqs, total_frames


def build_state_model(name: str, args: argparse.Namespace):
    if _SIMACTRAT_IMPORT_ERROR is not None:
        raise ImportError(
            "state-model fitting requires simactrat to be installed; "
            f"original import error: {_SIMACTRAT_IMPORT_ERROR}"
        )
    n_states = None if int(args.n_states) <= 0 else int(args.n_states)
    if name == "mc":
        return MarkovChainModel(n_states=n_states, laplace=float(args.laplace))
    if name == "hmm":
        n_hidden = int(args.hmm_n_hidden) if int(args.hmm_n_hidden) > 0 else None
        return DiscreteHMMModel(
            n_states=n_states,
            n_hidden=n_hidden,
            n_iter=int(args.hmm_n_iter),
            tol=float(args.hmm_tol),
            laplace=float(args.hmm_laplace),
            random_state=int(args.seed),
        )
    if name == "smp":
        return SemiMarkovModel(
            n_states=n_states,
            max_dwell=int(args.max_dwell),
            laplace=float(args.laplace),
            dwell_smoothing=float(args.dwell_smoothing),
        )
    raise ValueError(f"unsupported state model: {name}")


def fit_state_models_for_split(split_path: Path, out_root: Path, args: argparse.Namespace) -> Dict[str, object]:
    train_manifest = split_path / "train_manifest.csv"
    test_manifest = split_path / "test_manifest.csv"
    if not train_manifest.exists() or not test_manifest.exists():
        raise FileNotFoundError(f"split missing train/test manifest: {split_path}")

    train_seqs, train_frames = load_sequences(train_manifest, min_len=int(args.min_len))
    test_seqs, test_frames = load_sequences(test_manifest, min_len=int(args.min_len))
    if not train_seqs:
        raise RuntimeError(f"{split_path.name}: no valid train sequences after filtering")

    model_dir = ensure_dir(out_root / split_path.name / "state")
    rows: List[Dict[str, object]] = []
    for name in parse_csv_list(args.state_models):
        model = build_state_model(name, args)
        model.fit(train_seqs)
        path = model_dir / f"{name}.pkl"
        model.save(path)
        rows.append(
            {
                "split": split_path.name,
                "model": name,
                "model_path": str(path.resolve()),
                "n_states": int(getattr(model, "n_states", -1)),
                "n_train_trials": int(len(train_seqs)),
                "n_train_frames": int(train_frames),
                "n_test_trials": int(len(test_seqs)),
                "n_test_frames": int(test_frames),
            }
        )

    out_table = model_dir / "fit_summary.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)
    return {
        "split": split_path.name,
        "train_manifest": str(train_manifest.resolve()),
        "test_manifest": str(test_manifest.resolve()),
        "state_model_dir": str(model_dir.resolve()),
        "n_train_seq": int(len(train_seqs)),
        "n_test_seq": int(len(test_seqs)),
    }


def run_motion_train_for_split(split_path: Path, args: argparse.Namespace) -> int:
    legacy_train = ROOT / "algorithm" / "train.py"
    if not legacy_train.exists():
        raise FileNotFoundError(
            "legacy DiffSTG trainer not found at "
            f"{legacy_train}. Use existing CogFlow/CodSDE checkpoints with "
            "--skip-motion-if-best-exists and --motion-existing-roots."
        )
    cmd = [
        sys.executable,
        str(legacy_train),
        "--run-loro",
        "--split-dir",
        str(Path(args.split_dir).resolve()),
        "--split-name",
        split_path.name,
        "--flow-dir",
        str(Path(args.flow_dir).resolve()),
        "--out-dir",
        str(Path(args.motion_out_dir).resolve()),
        "--adj-path",
        str(Path(args.adj_path).resolve()),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--early-stop",
        str(args.early_stop),
        "--T_h",
        str(args.T_h),
        "--T_p",
        str(args.T_p),
        "--N",
        str(args.N),
        "--hidden-size",
        str(args.hidden_size),
        "--beta-schedule",
        str(args.beta_schedule),
        "--beta-end",
        str(args.beta_end),
        "--sample-steps",
        str(args.sample_steps),
        "--sample-strategy",
        str(args.sample_strategy),
        "--seed",
        str(args.seed),
    ]
    if float(args.mask_ratio) > 0:
        cmd.extend(["--mask-ratio", str(args.mask_ratio)])
    if int(args.num_workers) > 0:
        cmd.extend(["--num-workers", str(args.num_workers)])
    print(f"[02] run motion train: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def resolve_existing_motion_best(split_name: str, args: argparse.Namespace) -> Optional[Path]:
    best_name = str(args.motion_best_name).strip() or "best.pt"
    candidates: List[Path] = []

    # Primary output location used by algorithm/train.py
    candidates.append(Path(args.motion_out_dir).resolve() / split_name / best_name)

    # Optional extra roots for existing checkpoints (e.g., checkpoints/loro_*/best.pt)
    for raw in parse_csv_list(args.motion_existing_roots):
        root = Path(raw)
        root = root if root.is_absolute() else (ROOT / root)
        candidates.append(root.resolve() / split_name / best_name)

    for p in candidates:
        if p.exists():
            return p
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("02_fit_loro_models")
    p.add_argument("--split-dir", type=str, default=str(default_cross_subject_path("splits")))
    p.add_argument("--split-name", type=str, default="", help="optional one split, e.g. loro_RAT_13-2")
    p.add_argument("--out-dir", type=str, default=str(default_cross_subject_path("models")))

    p.add_argument("--fit-state", action="store_true")
    p.add_argument("--fit-motion", action="store_true")

    p.add_argument("--state-models", type=str, default="mc,hmm,smp")
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--n-states", type=int, default=-1)
    p.add_argument("--laplace", type=float, default=1.0)
    p.add_argument("--dwell-smoothing", type=float, default=1.0)
    p.add_argument("--max-dwell", type=int, default=200)
    p.add_argument("--hmm-n-hidden", type=int, default=-1)
    p.add_argument("--hmm-n-iter", type=int, default=50)
    p.add_argument("--hmm-tol", type=float, default=1e-4)
    p.add_argument("--hmm-laplace", type=float, default=1e-2)

    p.add_argument("--flow-dir", type=str, default=str(default_cross_subject_path("flow_by_rat")))
    p.add_argument("--adj-path", type=str, default=str(ROOT / "data" / "rat" / "hist10pred20" / "adj.npy"))
    p.add_argument("--motion-out-dir", type=str, default=str(ROOT / "results_rat"))
    p.add_argument("--skip-motion-if-best-exists", action="store_true")
    p.add_argument("--motion-best-name", type=str, default="best.pt")
    p.add_argument(
        "--motion-existing-roots",
        type=str,
        default=f"{ROOT / 'results_rat'}",
        help="Comma-separated roots to check for existing <split>/<best_name>",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--early-stop", type=int, default=12)
    p.add_argument("--mask-ratio", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--beta-schedule", type=str, default="quad")
    p.add_argument("--beta-end", type=float, default=0.1)
    p.add_argument("--sample-steps", type=int, default=200)
    p.add_argument("--sample-strategy", type=str, default="ddpm")
    p.add_argument("--T_h", type=int, default=18)
    p.add_argument("--T_p", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.fit_state and not args.fit_motion:
        raise ValueError("At least one of --fit-state / --fit-motion is required")

    split_paths = discover_splits(Path(args.split_dir), split_name=str(args.split_name).strip())
    out_root = ensure_dir(Path(args.out_dir))
    rows: List[Dict[str, object]] = []

    for split_path in split_paths:
        print(f"[02] processing split: {split_path.name}")
        item: Dict[str, object] = {"split": split_path.name}

        if args.fit_state:
            state_info = fit_state_models_for_split(split_path, out_root=out_root, args=args)
            item.update(state_info)
            item["state_status"] = "ok"
        else:
            item["state_status"] = "skip"

        if args.fit_motion:
            existing_best = resolve_existing_motion_best(split_name=split_path.name, args=args)
            if args.skip_motion_if_best_exists and existing_best is not None:
                item["motion_status"] = "skip_exists"
                item["motion_existing_best"] = str(existing_best.resolve())
                item["motion_run_dir"] = str((Path(args.motion_out_dir) / split_path.name).resolve())
                print(f"[02] skip motion train for {split_path.name}: found {existing_best}")
            else:
                rc = run_motion_train_for_split(split_path, args=args)
                item["motion_status"] = "ok" if rc == 0 else f"failed({rc})"
                item["motion_run_dir"] = str((Path(args.motion_out_dir) / split_path.name).resolve())
                if existing_best is not None:
                    item["motion_existing_best"] = str(existing_best.resolve())
        else:
            item["motion_status"] = "skip"

        rows.append(item)

    summary_path = out_root / "fit_loro_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    with (out_root / "fit_loro_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)
    print(f"[02] summary: {summary_path}")


if __name__ == "__main__":
    main()
