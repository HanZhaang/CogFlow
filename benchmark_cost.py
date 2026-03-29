from __future__ import annotations

import argparse
from pathlib import Path

import benchmark.adapters  # noqa: F401

from benchmark.registry import list_registered_methods
from benchmark.runner import BenchmarkRunner
from benchmark.utils import (
    build_result_label,
    create_benchmark_logger,
    flatten_infer_rows,
    flatten_train_row,
    gpu_name,
    load_batch,
    load_runtime_config,
    normalize_method_list,
    parse_training_artifacts,
    render_markdown_table,
    save_batch,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training/inference cost benchmark.")
    parser.add_argument("--cfg", required=True, type=str, help="Config file path.")
    parser.add_argument("--exp", default="benchmark", type=str, help="Runtime tag.")
    parser.add_argument("--method", default="all", type=str, help="cogflow | latent_ar | rssm | all")
    parser.add_argument("--variant", type=str, help="Method variant override.")
    parser.add_argument("--decoder", type=str, help="Decoder override.")
    parser.add_argument("--enable_dissipativity", action="store_true", help="Enable dissipativity constraint.")
    parser.add_argument("--dissipativity_weight", type=float, default=None, help="Override dissipativity weight.")
    parser.add_argument("--mode", nargs="+", default=["train", "infer"], choices=["train", "infer"], help="Benchmark modes.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Benchmark batch source split.")
    parser.add_argument("--batch-index", default=0, type=int, help="Which batch to benchmark.")
    parser.add_argument("--batch-cache", type=str, default=None, help="Optional fixed benchmark batch path.")
    parser.add_argument("--warmup", default=10, type=int, help="Warmup iterations.")
    parser.add_argument("--repeat", default=50, type=int, help="Measured iterations.")
    parser.add_argument("--k", nargs="+", default=[1, 20], type=int, help="Inference sample counts.")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Existing training result dir for time summary.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Used for GPU-hours estimation.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-train-csv", type=str, default=None, help="Optional training CSV output path.")
    parser.add_argument("--output-infer-csv", type=str, default=None, help="Optional inference CSV output path.")
    parser.add_argument("--print-markdown", action="store_true", help="Print markdown tables.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = create_benchmark_logger()
    methods = normalize_method_list(args.method, list_registered_methods())
    runner = BenchmarkRunner(warmup=args.warmup, repeat=args.repeat)

    cached_batch = load_batch(args.batch_cache) if args.batch_cache and Path(args.batch_cache).exists() else None
    all_results = []

    for method in methods:
        args.method = method
        cfg = load_runtime_config(args)
        logger.info("Building adapter for method=%s", method)
        from benchmark.registry import build_adapter

        adapter = build_adapter(method, cfg=cfg, args=args, logger=logger)

        if cached_batch is None:
            cached_batch = adapter.get_batch(split=args.split, batch_index=args.batch_index)
            if args.batch_cache:
                save_batch(cached_batch, args.batch_cache)

        result = {
            "method": method,
            "variant": getattr(cfg.METHOD, "VARIANT", None),
            "decoder": getattr(cfg.METHOD, "DECODER", None),
            "params": adapter.count_params(),
            "trainable_params": adapter.count_trainable_params(),
            "batch_size": adapter.batch_size(cached_batch),
            "env": {"gpu": gpu_name()},
            "config": {"cfg": args.cfg, "split": args.split, "batch_index": args.batch_index},
        }
        result["label"] = build_result_label(result)

        if "train" in args.mode:
            logger.info("Running training benchmark for method=%s", method)
            result["train"] = runner.benchmark_train(adapter, cached_batch)

        if "infer" in args.mode:
            logger.info("Running inference benchmark for method=%s", method)
            result["infer"] = [runner.benchmark_inference(adapter, cached_batch, num_samples=k) for k in args.k]

        result["training_summary"] = parse_training_artifacts(
            experiment_dir=args.experiment_dir,
            num_gpus=args.num_gpus,
        )
        all_results.append(result)

    payload = {"results": all_results}

    if args.output_json:
        write_json(payload, args.output_json)

    train_rows = [flatten_train_row(item) for item in all_results if "train" in item]
    infer_rows = []
    for item in all_results:
        infer_rows.extend(flatten_infer_rows(item))

    if args.output_train_csv:
        write_csv(train_rows, args.output_train_csv)
    if args.output_infer_csv:
        write_csv(infer_rows, args.output_infer_csv)

    if args.print_markdown:
        if train_rows:
            print("Training Cost")
            print(
                render_markdown_table(
                    train_rows,
                    [
                        "method",
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
            print()
        if infer_rows:
            print("Inference Cost")
            print(
                render_markdown_table(
                    infer_rows,
                    [
                        "method",
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


if __name__ == "__main__":
    main()
