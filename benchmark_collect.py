from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from benchmark.utils import (
    build_result_label,
    create_benchmark_logger,
    flatten_infer_rows,
    flatten_train_row,
    render_markdown_table,
    resolve_result_files,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect multiple benchmark JSON files into train/infer tables."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Benchmark JSON files, directories, or glob patterns.",
    )
    parser.add_argument("--output-train-csv", type=str, default=None, help="Optional training CSV path.")
    parser.add_argument("--output-infer-csv", type=str, default=None, help="Optional inference CSV path.")
    parser.add_argument("--output-markdown", type=str, default=None, help="Optional markdown report path.")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="label",
        choices=["label", "method", "step_ms", "latency_per_sample_ms"],
        help="Primary sort key for the output tables.",
    )
    parser.add_argument("--print-markdown", action="store_true", help="Print markdown tables to stdout.")
    return parser.parse_args()


def load_results(files: List[Path]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results", [])
        for item in results:
            result = dict(item)
            result["_source_file"] = str(path)
            result["label"] = build_result_label(result, fallback_label=path.stem)
            collected.append(result)
    return collected


def sort_rows(rows: List[Dict[str, Any]], sort_by: str, infer: bool = False) -> List[Dict[str, Any]]:
    def key_fn(row: Dict[str, Any]):
        primary = row.get(sort_by)
        if infer:
            return (str(primary), row.get("label", ""), int(row.get("k", 0)))
        return (str(primary), row.get("label", ""))

    if sort_by in {"step_ms", "latency_per_sample_ms"}:
        return sorted(
            rows,
            key=lambda row: (
                float("inf") if row.get(sort_by) is None else float(row.get(sort_by)),
                row.get("label", ""),
                int(row.get("k", 0)) if infer else 0,
            ),
        )
    return sorted(rows, key=key_fn)


def render_report(train_rows: List[Dict[str, Any]], infer_rows: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    if train_rows:
        sections.append("Training Cost")
        sections.append(
            render_markdown_table(
                train_rows,
                [
                    "label",
                    "params_m",
                    "gpu",
                    "batch",
                    "step_ms",
                    "peak_mem_gb",
                    "time_to_best_h",
                    "total_time_h",
                    "gpu_hours",
                ],
            )
        )
    if infer_rows:
        sections.append("Inference Cost")
        sections.append(
            render_markdown_table(
                infer_rows,
                [
                    "label",
                    "k",
                    "horizon",
                    "steps_nfe",
                    "latency_per_sample_ms",
                    "latency_per_batch_ms",
                    "throughput_seq_s",
                    "peak_mem_gb",
                ],
            )
        )
    return "\n\n".join(section for section in sections if section)


def main():
    args = parse_args()
    logger = create_benchmark_logger("benchmark.collect")
    files = resolve_result_files(args.inputs)
    if not files:
        raise FileNotFoundError("No benchmark JSON files matched the provided inputs.")

    logger.info("Collecting %d benchmark JSON files", len(files))
    results = load_results(files)
    if not results:
        raise ValueError("No benchmark results found in the provided JSON files.")

    train_rows = [flatten_train_row(item) for item in results if "train" in item]
    infer_rows: List[Dict[str, Any]] = []
    for item in results:
        infer_rows.extend(flatten_infer_rows(item))

    train_rows = sort_rows(train_rows, sort_by=args.sort_by, infer=False)
    infer_rows = sort_rows(infer_rows, sort_by=args.sort_by, infer=True)

    if args.output_train_csv:
        write_csv(train_rows, args.output_train_csv)
    if args.output_infer_csv:
        write_csv(infer_rows, args.output_infer_csv)

    report = render_report(train_rows, infer_rows)
    if args.output_markdown:
        output_path = Path(args.output_markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")

    if args.print_markdown or not args.output_markdown:
        print(report)


if __name__ == "__main__":
    main()
