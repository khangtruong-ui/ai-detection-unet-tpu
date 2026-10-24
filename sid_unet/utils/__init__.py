"""Utilities package for SID-UNet JAX/TPU."""

from sid_unet.utils.logger import setup_logger, get_logger, MetricLogger
from sid_unet.utils.config import load_config, save_config, ConfigDict
from sid_unet.utils.checkpoint import save_checkpoint, load_checkpoint, find_auto_resume_checkpoint
from sid_unet.utils.distributed import (
    get_backend,
    is_tpu,
    is_gpu,
    get_device_count,
    is_multi_device,
    get_devices,
    get_device_info,
    replicate_params,
    unreplicate_params,
    shard_batch,
    unshard_batch,
)
from sid_unet.utils.detach import maybe_detach_cli, should_detach, is_detached_worker
from sid_unet.utils.profiler import get_gpu_utilization, TrainingProfiler

__all__ = [
    "setup_logger",
    "get_logger",
    "MetricLogger",
    "load_config",
    "save_config",
    "ConfigDict",
    "save_checkpoint",
    "load_checkpoint",
    "find_auto_resume_checkpoint",
    "get_backend",
    "is_tpu",
    "is_gpu",
    "get_device_count",
    "is_multi_device",
    "get_devices",
    "get_device_info",
    "replicate_params",
    "unreplicate_params",
    "shard_batch",
    "unshard_batch",
    "maybe_detach_cli",
    "should_detach",
    "is_detached_worker",
    "get_gpu_utilization",
    "TrainingProfiler",
]
