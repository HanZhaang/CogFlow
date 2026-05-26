#!/usr/bin/env python
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validation.ver2_dataset import default_cross_subject_path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{name} is empty: {path}")
    return df


def read_csv_optional(path: Path, name: str) -> pd.DataFrame:
    if not str(path).strip() or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser("07_merge_cross_subject_reports")
    p.add_argument(
        "--state-table",
        type=str,
        default=str(default_cross_subject_path("summary", "state_generalization", "table_state_generalization.csv")),
    )
    p.add_argument(
        "--motion-table",
        type=str,
        default=str(default_cross_subject_path("summary", "motion_generalization", "table_motion_generalization.csv")),
    )
    p.add_argument(
        "--baseline-table",
        type=str,
        default=str(default_cross_subject_path("summary", "baseline", "table_cross_rat_baseline.csv")),
    )
    p.add_argument("--out-dir", type=str, default=str(default_cross_subject_path("summary")))
    p.add_argument("--state-model-order", type=str, default="smp,mc,hmm")
    p.add_argument("--motion-model-name", type=str, default="codsde")
    args = p.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    state_df = read_csv_optional(Path(args.state_table), "state table")
    motion_df = read_csv_required(Path(args.motion_table), "motion table")
    baseline_df = read_csv_optional(Path(args.baseline_table), "baseline table")

    if not state_df.empty:
        for c in ["split", "test_rat", "model"]:
            if c not in state_df.columns:
                raise ValueError(f"state table missing column: {c}")
    for c in ["split", "test_rat", "ade", "fde"]:
        if c not in motion_df.columns:
            raise ValueError(f"motion table missing column: {c}")

    # one motion row per split+test_rat
    motion_cols = [
        "split",
        "test_rat",
        "ade",
        "fde",
        "avg_ade",
        "avg_fde",
        "min_ade",
        "min_fde",
        "n_windows",
    ]
    existing_motion_cols = [c for c in motion_cols if c in motion_df.columns]
    motion_key_df = motion_df[existing_motion_cols].copy()
    motion_key_df = motion_key_df.drop_duplicates(subset=["split", "test_rat"], keep="first")

    if state_df.empty:
        merged = motion_key_df.copy()
        merged = merged.rename(columns={"test_rat": "heldout_rat"})
        merged.insert(2, "motion_model", str(args.motion_model_name))
        merged.insert(3, "state_model", "none")
    else:
        merged = state_df.merge(motion_key_df, on=["split", "test_rat"], how="left")
        merged = merged.rename(columns={"model": "state_model", "test_rat": "heldout_rat"})
        merged.insert(2, "motion_model", str(args.motion_model_name))

    desired_cols = [
        "split",
        "heldout_rat",
        "motion_model",
        "state_model",
        "ade",
        "fde",
        "avg_ade",
        "avg_fde",
        "min_ade",
        "min_fde",
        "state_nll_frame",
        "state_nll_segment",
        "transition_js",
        "transition_js_rowavg",
        "dwell_ks_macro",
        "dwell_ks_weighted",
        "n_windows",
        "n_test_trials",
        "n_test_frames",
    ]
    use_cols = [c for c in desired_cols if c in merged.columns]
    main_table = merged[use_cols].copy()

    # Sort rows for readability.
    order = parse_csv_list(args.state_model_order)
    if order:
        rank = {name: i for i, name in enumerate(order)}
        main_table["_rank"] = main_table["state_model"].map(lambda x: rank.get(str(x), 999))
        main_table = main_table.sort_values(["_rank", "split", "heldout_rat"]).drop(columns=["_rank"])
    else:
        main_table = main_table.sort_values(["state_model", "split", "heldout_rat"])

    main_path = out_dir / "table_cross_subject_main.csv"
    main_table.to_csv(main_path, index=False)

    # Aggregate mean/std by state model.
    numeric_cols = [
        c
        for c in [
            "ade",
            "fde",
            "avg_ade",
            "avg_fde",
            "min_ade",
            "min_fde",
            "state_nll_frame",
            "state_nll_segment",
            "transition_js",
            "transition_js_rowavg",
            "dwell_ks_macro",
            "dwell_ks_weighted",
        ]
        if c in main_table.columns
    ]
    agg_rows: List[Dict[str, object]] = []
    for model_name, gdf in main_table.groupby("state_model", sort=False):
        row: Dict[str, object] = {"state_model": model_name, "n_splits": int(gdf["split"].nunique())}
        for c in numeric_cols:
            vals = pd.to_numeric(gdf[c], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{c}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{c}_std"] = float(np.std(vals)) if vals.size else np.nan
        agg_rows.append(row)
    agg_table = pd.DataFrame(agg_rows)
    agg_path = out_dir / "table_cross_subject_agg.csv"
    agg_table.to_csv(agg_path, index=False)

    # Baseline copy (normalized output location).
    baseline_out = out_dir / "table_cross_rat_baseline.csv"
    baseline_df.to_csv(baseline_out, index=False)

    # Plot-friendly long table.
    long_rows: List[Dict[str, object]] = []
    plot_metrics = [c for c in ["ade", "fde", "state_nll_frame", "transition_js", "dwell_ks_macro"] if c in main_table.columns]
    for _, row in main_table.iterrows():
        for metric in plot_metrics:
            val = pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]
            if pd.isna(val):
                continue
            long_rows.append(
                {
                    "split": row.get("split", ""),
                    "heldout_rat": row.get("heldout_rat", ""),
                    "motion_model": row.get("motion_model", ""),
                    "state_model": row.get("state_model", ""),
                    "metric": metric,
                    "value": float(val),
                }
            )
    fig_table = pd.DataFrame(long_rows)
    fig_path = out_dir / "figure_cross_subject_barplot.csv"
    fig_table.to_csv(fig_path, index=False)

    # Text summary for response letter drafting.
    smp_row = agg_table[agg_table["state_model"].astype(str) == "smp"]
    baseline_map = {str(r["metric"]): r for _, r in baseline_df.iterrows()} if "metric" in baseline_df.columns else {}

    lines: List[str] = []
    lines.append("LORO summary")
    lines.append(f"- Raw table: {main_path}")
    lines.append(f"- Aggregated table: {agg_path}")
    if not smp_row.empty:
        r = smp_row.iloc[0]
        if "ade_mean" in r.index and "fde_mean" in r.index:
            lines.append(
                f"- {str(args.motion_model_name)}+SMP ADE/FDE mean: "
                f"{r.get('ade_mean', np.nan):.4f} / {r.get('fde_mean', np.nan):.4f}"
            )
        if "transition_js_mean" in r.index and "dwell_ks_macro_mean" in r.index:
            lines.append(
                f"- {str(args.motion_model_name)}+SMP transition JS / dwell KS mean: "
                f"{r.get('transition_js_mean', np.nan):.4f} / {r.get('dwell_ks_macro_mean', np.nan):.4f}"
            )
    if "real_rat_transition_js" in baseline_map:
        b = baseline_map["real_rat_transition_js"]
        lines.append(
            f"- Real cross-rat transition JS (mean±std): "
            f"{float(b.get('mean', np.nan)):.4f} ± {float(b.get('std', np.nan)):.4f}"
        )
    if "real_rat_dwell_ks_macro" in baseline_map:
        b = baseline_map["real_rat_dwell_ks_macro"]
        lines.append(
            f"- Real cross-rat dwell KS macro (mean±std): "
            f"{float(b.get('mean', np.nan)):.4f} ± {float(b.get('std', np.nan)):.4f}"
        )
    summary_txt = "\n".join(lines) + "\n"
    summary_path = out_dir / "loro_summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")

    with (out_dir / "merge_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    print(f"[07] main: {main_path}")
    print(f"[07] agg: {agg_path}")
    print(f"[07] baseline: {baseline_out}")
    print(f"[07] figure table: {fig_path}")
    print(f"[07] summary: {summary_path}")


if __name__ == "__main__":
    main()
