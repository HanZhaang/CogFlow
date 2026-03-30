#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch uncertainty calibration evaluation for prediction npz files under results directories."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Results root to scan recursively for npz/*.npz. Default: <repo_root>/results_rat",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="CogFlow repository root.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch uncertainty evaluation.",
    )
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="uncertainty_eval",
        help="Per-experiment output directory name created under each npz directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose correlation_metrics.csv already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=10,
        help="Number of uncertainty buckets passed to the single-file evaluator.",
    )
    parser.add_argument(
        "--expected-k",
        type=int,
        default=20,
        help="Expected K passed to the single-file evaluator.",
    )
    parser.add_argument(
        "--summary-csv-name",
        type=str,
        default="batch_uncertainty_summary.csv",
        help="Summary CSV filename under results root.",
    )
    parser.add_argument(
        "--summary-md-name",
        type=str,
        default="batch_uncertainty_summary.md",
        help="Summary markdown filename under results root.",
    )
    return parser.parse_args()


def resolve_results_root(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.results_root:
        return Path(args.results_root).expanduser().resolve()
    return repo_root / "results_rat"


def discover_latest_npz(results_root: Path) -> List[Tuple[Path, Path]]:
    latest_by_exp: Dict[Path, Path] = {}
    for npz_path in sorted(results_root.rglob("npz/*.npz")):
        exp_dir = npz_path.parent.parent
        current = latest_by_exp.get(exp_dir)
        if current is None or npz_path.stat().st_mtime > current.stat().st_mtime:
            latest_by_exp[exp_dir] = npz_path
    return sorted(latest_by_exp.items(), key=lambda item: str(item[0]))


def run_command(cmd: Sequence[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def build_eval_cmd(args: argparse.Namespace, repo_root: Path, npz_path: Path, out_dir: Path) -> List[str]:
    return [
        args.python,
        str(repo_root / "scripts" / "eval_uncertainty_calibration.py"),
        "--pred_npz",
        str(npz_path),
        "--out_dir",
        str(out_dir),
        "--num_buckets",
        str(args.num_buckets),
        "--expected_k",
        str(args.expected_k),
    ]


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        return list(reader)


def correlation_lookup(rows: Sequence[Dict[str, str]], granularity: str, target: str) -> Dict[str, str]:
    for row in rows:
        if row.get("granularity") == granularity and row.get("target") == target:
            return row
    return {}


def build_summary_row(
    exp_dir: Path,
    npz_path: Path,
    out_dir: Path,
    correlation_rows: Sequence[Dict[str, str]],
    regime_exists: bool,
) -> Dict[str, object]:
    ade_row = correlation_lookup(correlation_rows, "trajectory", "ADE")
    fde_row = correlation_lookup(correlation_rows, "trajectory", "FDE")
    timestep_row = correlation_lookup(correlation_rows, "timestep", "ADE_timestep")
    return {
        "experiment": exp_dir.name,
        "experiment_dir": str(exp_dir),
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "num_samples": int(float(ade_row.get("n", 0) or 0)),
        "traj_ade_pearson": float(ade_row.get("pearson_r", "nan")),
        "traj_ade_spearman": float(ade_row.get("spearman_rho", "nan")),
        "traj_fde_pearson": float(fde_row.get("pearson_r", "nan")),
        "traj_fde_spearman": float(fde_row.get("spearman_rho", "nan")),
        "timestep_ade_pearson": float(timestep_row.get("pearson_r", "nan")),
        "timestep_ade_spearman": float(timestep_row.get("spearman_rho", "nan")),
        "has_regime_metrics": regime_exists,
    }


def write_summary_csv(rows: Sequence[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(results_root: Path, summary_rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# Batch Uncertainty Calibration Summary (rat)",
        "",
        f"Results root: `{results_root}`",
        "",
        "| Experiment | Samples | ADE Pearson | ADE Spearman | FDE Pearson | FDE Spearman | Regime | NPZ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in summary_rows:
        lines.append(
            "| {experiment} | {num_samples} | {traj_ade_pearson:.6f} | {traj_ade_spearman:.6f} | "
            "{traj_fde_pearson:.6f} | {traj_fde_spearman:.6f} | {has_regime_metrics} | `{npz_path}` |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    results_root = resolve_results_root(args)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    experiments = discover_latest_npz(results_root)
    if not experiments:
        raise FileNotFoundError(
            f"No npz files found under {results_root}. Expected paths like <exp>/npz/*.npz"
        )

    summary_rows: List[Dict[str, object]] = []
    for exp_dir, npz_path in experiments:
        out_dir = npz_path.parent / args.out_dir_name
        correlation_csv = out_dir / "correlation_metrics.csv"
        regime_csv = out_dir / "regime_metrics.csv"

        if not (args.skip_existing and correlation_csv.exists()):
            run_command(build_eval_cmd(args, repo_root, npz_path, out_dir), cwd=repo_root, dry_run=args.dry_run)
        if args.dry_run:
            continue
        if not correlation_csv.exists():
            raise FileNotFoundError(f"Missing correlation metrics after evaluation: {correlation_csv}")

        correlation_rows = read_csv_rows(correlation_csv)
        summary_rows.append(
            build_summary_row(
                exp_dir=exp_dir,
                npz_path=npz_path,
                out_dir=out_dir,
                correlation_rows=correlation_rows,
                regime_exists=regime_csv.exists() and regime_csv.stat().st_size > 0,
            )
        )

    if args.dry_run:
        return

    summary_csv = results_root / args.summary_csv_name
    summary_md = results_root / args.summary_md_name
    write_summary_csv(summary_rows, summary_csv)
    summary_md.write_text(render_markdown(results_root, summary_rows), encoding="utf-8")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD:  {summary_md}")


if __name__ == "__main__":
    main()
