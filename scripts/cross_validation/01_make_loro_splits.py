#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validation.ver2_dataset import default_cross_subject_path


def _safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return s.strip("_") or "unknown"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_trial_list(path: Path, trial_ids: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in trial_ids:
            f.write(f"{t}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("01_make_loro_splits")
    p.add_argument("--manifest", type=str, default=str(default_cross_subject_path("manifest.csv")))
    p.add_argument("--out-dir", type=str, default=str(default_cross_subject_path("splits")))
    p.add_argument("--rat-col", type=str, default="rat_id")
    p.add_argument("--trial-col", type=str, default="trial_id")
    p.add_argument("--min-trials-per-rat", type=int, default=1)
    p.add_argument("--only-rats", type=str, default="", help="Comma-separated rat_id subset")
    p.add_argument("--strict", action="store_true", help="Raise if any selected rat has < min trials")
    return p


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    for c in [args.rat_col, args.trial_col]:
        if c not in df.columns:
            raise ValueError(f"manifest missing required column: {c}")

    if df.empty:
        raise ValueError("manifest is empty")

    selected = [x.strip() for x in str(args.only_rats).split(",") if x.strip()]
    if selected:
        df = df[df[args.rat_col].astype(str).isin(set(selected))].copy()
        if df.empty:
            raise ValueError(f"no rows left after --only-rats={selected}")

    counts = df.groupby(args.rat_col).size().to_dict()
    valid_rats: List[str] = []
    skipped_rats: List[str] = []
    for rat_id, n in sorted(counts.items(), key=lambda x: str(x[0])):
        if int(n) >= int(args.min_trials_per_rat):
            valid_rats.append(str(rat_id))
        else:
            skipped_rats.append(str(rat_id))

    if args.strict and skipped_rats:
        raise ValueError(
            f"rats below min-trials ({args.min_trials_per_rat}): {skipped_rats}"
        )

    if len(valid_rats) < 2:
        raise ValueError(
            f"need at least 2 rats for LORO, got valid={valid_rats}, skipped={skipped_rats}"
        )

    out_dir = _ensure_dir(Path(args.out_dir))
    summary_rows: List[Dict[str, object]] = []

    for heldout in valid_rats:
        split_name = f"loro_{_safe_name(heldout)}"
        split_dir = _ensure_dir(out_dir / split_name)

        test_df = df[df[args.rat_col].astype(str) == heldout].copy()
        train_df = df[df[args.rat_col].astype(str) != heldout].copy()

        if test_df.empty or train_df.empty:
            raise ValueError(f"invalid split {split_name}: empty train/test")

        train_manifest = split_dir / "train_manifest.csv"
        test_manifest = split_dir / "test_manifest.csv"
        train_df.to_csv(train_manifest, index=False)
        test_df.to_csv(test_manifest, index=False)

        train_trials = [str(x) for x in train_df[args.trial_col].astype(str).tolist()]
        test_trials = [str(x) for x in test_df[args.trial_col].astype(str).tolist()]
        _write_trial_list(split_dir / "train_trials.txt", train_trials)
        _write_trial_list(split_dir / "test_trials.txt", test_trials)

        train_rats = sorted(train_df[args.rat_col].astype(str).unique().tolist())
        split_meta = {
            "split": split_name,
            "heldout_rat": heldout,
            "train_rats": train_rats,
            "n_train_trials": int(len(train_df)),
            "n_test_trials": int(len(test_df)),
            "n_train_frames": int(pd.to_numeric(train_df.get("n_frames", 0), errors="coerce").fillna(0).sum()),
            "n_test_frames": int(pd.to_numeric(test_df.get("n_frames", 0), errors="coerce").fillna(0).sum()),
            "train_manifest": str(train_manifest.resolve()),
            "test_manifest": str(test_manifest.resolve()),
        }
        with (split_dir / "split_meta.json").open("w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2, ensure_ascii=True)

        summary_rows.append(
            {
                "split": split_name,
                "heldout_rat": heldout,
                "train_rats": ",".join(train_rats),
                "n_train_trials": int(len(train_df)),
                "n_test_trials": int(len(test_df)),
                "n_train_frames": split_meta["n_train_frames"],
                "n_test_frames": split_meta["n_test_frames"],
                "train_manifest": str(train_manifest.resolve()),
                "test_manifest": str(test_manifest.resolve()),
            }
        )

        print(
            f"[01] {split_name}: train_rats={train_rats}, "
            f"train_trials={len(train_df)}, test_trials={len(test_df)}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("split").reset_index(drop=True)
    summary_path = out_dir / "loro_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[01] summary: {summary_path}")

    if skipped_rats:
        print(f"[01] skipped rats (<{args.min_trials_per_rat} trials): {skipped_rats}")


if __name__ == "__main__":
    main()
