#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_RAT_RESULTS_ROOT = "/home/zhanghan/01_code/CogFlow/results_rat/cor_rat_fm_mn"
DEFAULT_BABEL_RESULTS_ROOT = "/home/zhanghan/01_code/CogFlow/results_babel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch export prediction npz files and run pub_evaluation.py for each checkpoint."
    )
    parser.add_argument("--dataset", required=True, choices=["rat", "babel"], help="Dataset to evaluate.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Root directory whose immediate children contain models/checkpoint_best.pt.",
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
        help="Python executable used to launch eval and pub evaluation.",
    )
    parser.add_argument("--cfg", type=str, default="auto", help="Config passed to eval entry.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch size override.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional FM sampling steps override.")
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "lin_poly"], help="Optional FM solver.")
    parser.add_argument("--lin-poly-p", type=int, default=None, help="Optional lin_poly degree override.")
    parser.add_argument(
        "--lin-poly-long-step",
        type=int,
        default=None,
        help="Optional lin_poly long step override.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument("--eval-on-train", action="store_true", help="Evaluate on training set instead of test set.")
<<<<<<< HEAD
    parser.add_argument("--skip-existing", action="store_true", help="Skip eval if metrics json already exists.")
=======
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments when metrics json already exists.")
>>>>>>> origin/optimize_time_mem
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Optional horizons passed to pub_evaluation.py.",
    )
    parser.add_argument(
<<<<<<< HEAD
        "--summary-json-name",
        type=str,
        default="batch_pub_eval_summary.json",
        help="Filename for machine-readable summary under results root.",
    )
    parser.add_argument(
        "--summary-md-name",
        type=str,
        default="batch_pub_eval_summary.md",
        help="Filename for markdown summary under results root.",
=======
        "--summary-name",
        type=str,
        default="batch_pub_eval_summary",
        help="Base filename used for the summary json/csv under results root.",
>>>>>>> origin/optimize_time_mem
    )
    return parser.parse_args()


def resolve_results_root(args: argparse.Namespace) -> Path:
    if args.results_root:
        return Path(args.results_root).expanduser().resolve()
    default_root = DEFAULT_RAT_RESULTS_ROOT if args.dataset == "rat" else DEFAULT_BABEL_RESULTS_ROOT
    return Path(default_root).expanduser().resolve()


def eval_entry(dataset: str, repo_root: Path) -> Path:
    return repo_root / ("eval_rat.py" if dataset == "rat" else "eval.py")


def discover_experiments(results_root: Path) -> List[Path]:
    checkpoints = sorted(results_root.rglob("models/checkpoint_best.pt"))
    return sorted({ckpt.parent.parent for ckpt in checkpoints})


def latest_npz(npz_dir: Path) -> Optional[Path]:
    candidates = sorted(npz_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_command(cmd: Sequence[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def build_eval_cmd(args: argparse.Namespace, repo_root: Path, ckpt_path: Path) -> List[str]:
    cmd = [
        args.python,
        str(eval_entry(args.dataset, repo_root)),
        "--cfg",
        args.cfg,
        "--ckpt_path",
        str(ckpt_path),
        "--save_samples",
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.sampling_steps is not None:
        cmd.extend(["--sampling_steps", str(args.sampling_steps)])
    if args.solver is not None:
        cmd.extend(["--solver", args.solver])
    if args.lin_poly_p is not None:
        cmd.extend(["--lin_poly_p", str(args.lin_poly_p)])
    if args.lin_poly_long_step is not None:
        cmd.extend(["--lin_poly_long_step", str(args.lin_poly_long_step)])
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    if args.eval_on_train:
        cmd.append("--eval_on_train")
<<<<<<< HEAD
    print(cmd)
=======
>>>>>>> origin/optimize_time_mem
    return cmd


def build_pub_cmd(args: argparse.Namespace, repo_root: Path, npz_path: Path, output_json: Path) -> List[str]:
    cmd = [
        args.python,
        str(repo_root / "pub_evaluation.py"),
        "--npz_path",
        str(npz_path),
        "--output_json",
        str(output_json),
    ]
    if args.horizons:
        cmd.append("--horizons")
        cmd.extend(str(horizon) for horizon in args.horizons)
    return cmd


def read_metrics(metrics_path: Path) -> Dict[str, object]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


<<<<<<< HEAD
def best_final_metric(metrics: Dict[str, object], key: str) -> Optional[float]:
    values = metrics.get(key)
    if isinstance(values, list) and values:
        return float(values[-1])
    return None


def render_markdown(results_root: Path, dataset: str, rows: List[Dict[str, object]]) -> str:
    lines = [
        f"# Batch Public Evaluation Summary ({dataset})",
        "",
        f"Results root: `{results_root}`",
        "",
        "| Experiment | Final Horizon | ADE_min | FDE_min | ADE_avg | FDE_avg | Diversity | NPZ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        metrics = row["metrics"]
        horizons = metrics.get("horizons", [])
        final_horizon = horizons[-1] if isinstance(horizons, list) and horizons else "-"
        npz_path = row["npz_path"]
        lines.append(
            "| {experiment} | {final_horizon} | {ade_min:.6f} | {fde_min:.6f} | "
            "{ade_avg:.6f} | {fde_avg:.6f} | {diversity:.6f} | `{npz_path}` |".format(
                experiment=row["experiment"],
                final_horizon=final_horizon,
                ade_min=best_final_metric(metrics, "ADE_min") or 0.0,
                fde_min=best_final_metric(metrics, "FDE_min") or 0.0,
                ade_avg=best_final_metric(metrics, "ADE_avg") or 0.0,
                fde_avg=best_final_metric(metrics, "FDE_avg") or 0.0,
                diversity=best_final_metric(metrics, "Diversity") or 0.0,
                npz_path=npz_path,
            )
        )

    for row in rows:
        metrics = row["metrics"]
        lines.extend(
            [
                "",
                f"## {row['experiment']}",
                "",
                f"- Checkpoint: `{row['checkpoint']}`",
                f"- NPZ: `{row['npz_path']}`",
                f"- Metrics JSON: `{row['metrics_json']}`",
                "",
                "| Horizon | ADE_min | FDE_min | ADE_avg | FDE_avg | Diversity |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        horizons = metrics.get("horizons", [])
        ade_min = metrics.get("ADE_min", [])
        fde_min = metrics.get("FDE_min", [])
        ade_avg = metrics.get("ADE_avg", [])
        fde_avg = metrics.get("FDE_avg", [])
        diversity = metrics.get("Diversity", [])
        for horizon, a_min, f_min, a_avg, f_avg, div in zip(
            horizons, ade_min, fde_min, ade_avg, fde_avg, diversity
        ):
            lines.append(
                f"| {horizon} | {a_min:.6f} | {f_min:.6f} | {a_avg:.6f} | {f_avg:.6f} | {div:.6f} |"
            )

    lines.append("")
    return "\n".join(lines)
=======
def flatten_metrics_row(exp_name: str, ckpt_path: Path, npz_path: Path, metrics: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "experiment": exp_name,
        "checkpoint": str(ckpt_path),
        "npz_path": str(npz_path),
        "num_trajs": metrics.get("num_trajs"),
        "K": metrics.get("K"),
        "F": metrics.get("F"),
        "D": metrics.get("D"),
    }
    for metric_name in ["ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "Diversity"]:
        values = metrics.get(metric_name, [])
        if isinstance(values, list):
            for idx, value in enumerate(values):
                row[f"{metric_name}[{idx}]"] = value
    horizons = metrics.get("horizons", [])
    if isinstance(horizons, list):
        for idx, value in enumerate(horizons):
            row[f"horizon[{idx}]"] = value
    return row


def write_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    all_keys = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
>>>>>>> origin/optimize_time_mem


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    results_root = resolve_results_root(args)

    experiments = discover_experiments(results_root)
    if not experiments:
        raise FileNotFoundError(f"No checkpoint_best.pt found under {results_root}")

    summary_rows: List[Dict[str, object]] = []
<<<<<<< HEAD
    for exp_dir in experiments:
        ckpt_path = exp_dir / "models" / "checkpoint_best.pt"
        npz_dir = exp_dir / "npz"
        metrics_json = npz_dir / f"{args.dataset}_pub_eval.json"
        exp_name = exp_dir.name

        if args.skip_existing and metrics_json.exists():
            npz_path = latest_npz(npz_dir)
            if npz_path is None:
                raise FileNotFoundError(f"Metrics exists but no npz found under {npz_dir}")
            metrics = read_metrics(metrics_json)
        else:
            before_npz = set(npz_dir.glob("*.npz")) if npz_dir.exists() else set()
            print("npz_dir = {}".format(npz_dir))
            run_command(build_eval_cmd(args, repo_root, ckpt_path), cwd=repo_root, dry_run=args.dry_run)
            if args.dry_run:
                continue

            after_npz = set(npz_dir.glob("*.npz")) if npz_dir.exists() else set()
            new_npz = sorted(after_npz - before_npz, key=lambda path: path.stat().st_mtime, reverse=True)
            npz_path = new_npz[0] if new_npz else latest_npz(npz_dir)
            if npz_path is None:
                raise FileNotFoundError(f"No npz generated under {npz_dir} after evaluating {ckpt_path}")

            run_command(build_pub_cmd(args, repo_root, npz_path, metrics_json), cwd=repo_root, dry_run=args.dry_run)
            metrics = read_metrics(metrics_json)

        summary_rows.append(
=======
    summary_payload = []

    for exp_dir in experiments:
        exp_name = exp_dir.name
        ckpt_path = exp_dir / "models" / "checkpoint_best.pt"
        npz_dir = exp_dir / "npz"
        metrics_json = npz_dir / f"{args.dataset}_pub_eval.json"

        if args.skip_existing and metrics_json.exists():
            metrics = read_metrics(metrics_json)
            npz_path = latest_npz(npz_dir)
            if npz_path is None:
                raise FileNotFoundError(f"Metrics exists but no npz found under {npz_dir}")
            summary_rows.append(flatten_metrics_row(exp_name, ckpt_path, npz_path, metrics))
            summary_payload.append(
                {
                    "experiment": exp_name,
                    "checkpoint": str(ckpt_path),
                    "npz_path": str(npz_path),
                    "metrics_json": str(metrics_json),
                    "metrics": metrics,
                    "skipped": True,
                }
            )
            continue

        before_npz = set(npz_dir.glob("*.npz")) if npz_dir.exists() else set()
        run_command(build_eval_cmd(args, repo_root, ckpt_path), cwd=repo_root, dry_run=args.dry_run)

        if args.dry_run:
            continue

        after_npz = set(npz_dir.glob("*.npz")) if npz_dir.exists() else set()
        new_npz = sorted(after_npz - before_npz, key=lambda path: path.stat().st_mtime, reverse=True)
        npz_path = new_npz[0] if new_npz else latest_npz(npz_dir)
        if npz_path is None:
            raise FileNotFoundError(f"No npz generated under {npz_dir} after evaluating {ckpt_path}")

        run_command(
            build_pub_cmd(args, repo_root, npz_path, metrics_json),
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        metrics = read_metrics(metrics_json)
        summary_rows.append(flatten_metrics_row(exp_name, ckpt_path, npz_path, metrics))
        summary_payload.append(
>>>>>>> origin/optimize_time_mem
            {
                "experiment": exp_name,
                "checkpoint": str(ckpt_path),
                "npz_path": str(npz_path),
                "metrics_json": str(metrics_json),
                "metrics": metrics,
<<<<<<< HEAD
=======
                "skipped": False,
>>>>>>> origin/optimize_time_mem
            }
        )

    if args.dry_run:
        return

<<<<<<< HEAD
    summary_json = results_root / args.summary_json_name
    summary_md = results_root / args.summary_md_name
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md.write_text(render_markdown(results_root, args.dataset, summary_rows), encoding="utf-8")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD:   {summary_md}")
=======
    summary_json = results_root / f"{args.summary_name}.json"
    summary_csv = results_root / f"{args.summary_name}.csv"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(summary_rows, summary_csv)
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")
>>>>>>> origin/optimize_time_mem


if __name__ == "__main__":
    main()
