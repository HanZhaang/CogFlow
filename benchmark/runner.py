from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List

import torch

from .utils import max_memory_gb, reset_peak_memory, sync_device


class BenchmarkRunner:
    def __init__(self, warmup: int = 10, repeat: int = 50):
        self.warmup = warmup
        self.repeat = repeat

    def benchmark_train(self, adapter, batch_cpu: Dict[str, Any]) -> Dict[str, Any]:
        for _ in range(self.warmup):
            self._run_train_once(adapter, batch_cpu)
        sync_device()

        h2d_times: List[float] = []
        forward_times: List[float] = []
        backward_times: List[float] = []
        optimizer_times: List[float] = []
        total_times: List[float] = []
        peak_memories: List[float] = []

        for _ in range(self.repeat):
            reset_peak_memory()
            stats = self._run_train_once(adapter, batch_cpu)
            h2d_times.append(stats["h2d_time_ms"])
            forward_times.append(stats["forward_time_ms"])
            backward_times.append(stats["backward_time_ms"])
            optimizer_times.append(stats["optimizer_time_ms"])
            total_times.append(stats["step_time_ms"])
            if stats["peak_mem_gb"] is not None:
                peak_memories.append(stats["peak_mem_gb"])

        return {
            "h2d_time_ms": self._mean(h2d_times),
            "forward_time_ms": self._mean(forward_times),
            "backward_time_ms": self._mean(backward_times),
            "optimizer_time_ms": self._mean(optimizer_times),
            "step_time_ms": self._mean(total_times),
            "step_time_std_ms": self._std(total_times),
            "peak_mem_gb": max(peak_memories) if peak_memories else None,
        }

    @torch.no_grad()
    def benchmark_inference(self, adapter, batch_cpu: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
        for _ in range(self.warmup):
            self._run_inference_once(adapter, batch_cpu, num_samples)
        sync_device()

        latencies: List[float] = []
        peak_memories: List[float] = []

        for _ in range(self.repeat):
            reset_peak_memory()
            stats = self._run_inference_once(adapter, batch_cpu, num_samples)
            latencies.append(stats["latency_per_batch_ms"])
            if stats["peak_mem_gb"] is not None:
                peak_memories.append(stats["peak_mem_gb"])

        meta = adapter.get_inference_metadata(num_samples=num_samples)
        batch_size = stats["batch_size"]
        latency_batch = self._mean(latencies)
        latency_sample = latency_batch / batch_size if batch_size else None
        throughput = None
        if batch_size and latency_batch and latency_batch > 0:
            throughput = batch_size / (latency_batch / 1000.0)

        return {
            "k": num_samples,
            "horizon": meta.get("horizon"),
            "steps_nfe": meta.get("steps_nfe"),
            "latency_per_sample_ms": round(latency_sample, 4) if latency_sample is not None else None,
            "latency_per_batch_ms": round(latency_batch, 4),
            "latency_std_ms": self._std(latencies),
            "throughput_seq_s": round(throughput, 4) if throughput is not None else None,
            "peak_mem_gb": max(peak_memories) if peak_memories else None,
        }

    def _run_train_once(self, adapter, batch_cpu: Dict[str, Any]) -> Dict[str, Any]:
        adapter.prepare_train_step()

        sync_device()
        t0 = time.perf_counter()
        batch_gpu = adapter.move_batch_to_device(batch_cpu)
        sync_device()
        t1 = time.perf_counter()

        sync_device()
        t1_forward = time.perf_counter()
        loss = self._measure(lambda: adapter.forward_loss(batch_gpu))
        sync_device()
        t2 = time.perf_counter()
        forward_ms = (t2 - t1_forward) * 1000.0

        sync_device()
        t3 = time.perf_counter()
        adapter.backward(loss)
        sync_device()
        t4 = time.perf_counter()

        sync_device()
        t5 = time.perf_counter()
        adapter.optimizer_step()
        sync_device()
        t6 = time.perf_counter()

        return {
            "h2d_time_ms": (t1 - t0) * 1000.0,
            "forward_time_ms": forward_ms,
            "backward_time_ms": (t4 - t3) * 1000.0,
            "optimizer_time_ms": (t6 - t5) * 1000.0,
            "step_time_ms": (t6 - t0) * 1000.0,
            "peak_mem_gb": max_memory_gb(),
        }

    def _run_inference_once(self, adapter, batch_cpu: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
        adapter.prepare_inference()
        batch_gpu = adapter.move_batch_to_device(batch_cpu)

        sync_device()
        t0 = time.perf_counter()
        output = adapter.inference_step(batch_gpu, num_samples=num_samples)
        _ = output
        sync_device()
        t1 = time.perf_counter()
        return {
            "latency_per_batch_ms": (t1 - t0) * 1000.0,
            "peak_mem_gb": max_memory_gb(),
            "batch_size": adapter.batch_size(batch_cpu),
        }

    @staticmethod
    def _measure(fn):
        return fn()

    @staticmethod
    def _mean(values: List[float]):
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    @staticmethod
    def _std(values: List[float]):
        if len(values) <= 1:
            return 0.0
        return round(statistics.pstdev(values), 4)
