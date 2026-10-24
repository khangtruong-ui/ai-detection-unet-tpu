"""Hardware utilization and training bottleneck profiling utilities for SID-UNet."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import jax
import numpy as np

logger = logging.getLogger("sid_unet.profiler")


def get_gpu_utilization() -> Dict[str, Any]:
    """Query current GPU utilization and memory statistics if running on an NVIDIA GPU."""
    info: Dict[str, Any] = {
        "is_gpu": False,
        "gpu_utilization_percent": None,
        "memory_utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
        "jax_bytes_in_use": 0,
        "jax_peak_bytes_in_use": 0,
    }

    # 1. Check JAX Device Backend
    devices = jax.local_devices()
    is_gpu = any(d.platform == "gpu" for d in devices)
    info["is_gpu"] = is_gpu

    if is_gpu:
        try:
            mem_stats = devices[0].memory_stats()
            if mem_stats:
                info["jax_bytes_in_use"] = mem_stats.get("bytes_in_use", 0)
                info["jax_peak_bytes_in_use"] = mem_stats.get("peak_bytes_in_use", 0)
                info["jax_bytes_limit"] = mem_stats.get("bytes_limit", 0)
        except Exception as e:
            logger.debug(f"Could not read JAX memory_stats: {e}")

    # 2. Query nvidia-smi if available
    if shutil.which("nvidia-smi"):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ]
            output = subprocess.check_output(cmd, timeout=5).decode("utf-8").strip()
            if output:
                lines = output.splitlines()
                first_gpu = [x.strip() for x in lines[0].split(",")]
                if len(first_gpu) >= 4:
                    info["gpu_utilization_percent"] = float(first_gpu[0])
                    info["memory_utilization_percent"] = float(first_gpu[1])
                    info["memory_used_mb"] = float(first_gpu[2])
                    info["memory_total_mb"] = float(first_gpu[3])
                    info["is_gpu"] = True
        except Exception as e:
            logger.debug(f"nvidia-smi query failed: {e}")

    return info


class TrainingProfiler:
    """Profiles training loop throughput, hardware utilization, and pipeline bottlenecks.

    Differentiates data loading overhead from forward/backward execution time to diagnose
    bottlenecks (e.g. streaming I/O bound vs compute/GPU bound).
    """

    def __init__(self, warmup_steps: int = 2):
        self.warmup_steps = warmup_steps
        self.step_times: List[float] = []
        self.data_times: List[float] = []
        self.compute_times: List[float] = []
        self.batch_sizes: List[int] = []
        self.step_count = 0

    def record_step(self, data_time: float, compute_time: float, batch_size: int = 1) -> None:
        """Record timing breakdown for an individual training step."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            return

        self.data_times.append(float(data_time))
        self.compute_times.append(float(compute_time))
        self.step_times.append(float(data_time + compute_time))
        self.batch_sizes.append(int(batch_size))

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Compute average timings, sample throughput, bottleneck ratios, and hardware metrics."""
        if not self.step_times:
            return {
                "steps_profiled": 0,
                "avg_data_time_ms": 0.0,
                "avg_compute_time_ms": 0.0,
                "avg_step_time_ms": 0.0,
                "throughput_samples_per_sec": 0.0,
                "data_bottleneck_ratio": 0.0,
                "compute_bottleneck_ratio": 0.0,
                "primary_bottleneck": "none",
                "gpu_metrics": get_gpu_utilization(),
            }

        avg_data = float(np.mean(self.data_times))
        avg_comp = float(np.mean(self.compute_times))
        avg_step = float(np.mean(self.step_times))
        total_samples = sum(self.batch_sizes)
        total_time = sum(self.step_times)
        throughput = (total_samples / total_time) if total_time > 0 else 0.0

        data_ratio = (avg_data / avg_step) if avg_step > 0 else 0.0
        comp_ratio = (avg_comp / avg_step) if avg_step > 0 else 0.0

        if data_ratio > 0.5:
            bottleneck = "data_loading (I/O or augmentation bound)"
        elif comp_ratio > 0.5:
            bottleneck = "compute (GPU / TPU compute bound)"
        else:
            bottleneck = "balanced"

        return {
            "steps_profiled": len(self.step_times),
            "avg_data_time_ms": avg_data * 1000.0,
            "avg_compute_time_ms": avg_comp * 1000.0,
            "avg_step_time_ms": avg_step * 1000.0,
            "throughput_samples_per_sec": throughput,
            "data_bottleneck_ratio": data_ratio,
            "compute_bottleneck_ratio": comp_ratio,
            "primary_bottleneck": bottleneck,
            "gpu_metrics": get_gpu_utilization(),
        }
