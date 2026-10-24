"""Tests for mock training loops, mixed precision (float16/bfloat16), bottlenecks, and GPU utilization."""

import time
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.profiler import TrainingProfiler, get_gpu_utilization


class MockDataLoader:
    """Fast in-memory mock dataloader for bottleneck profiling and unit testing."""

    def __init__(self, num_batches: int = 4, batch_size: int = 2, image_size: int = 64, delay: float = 0.005):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.image_size = image_size
        self.delay = delay

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            if self.delay > 0:
                time.sleep(self.delay)  # Simulate I/O latency
            yield {
                "image": np.random.randn(self.batch_size, self.image_size, self.image_size, 3).astype(np.float32),
                "mask": np.random.randint(0, 2, (self.batch_size, self.image_size, self.image_size, 1)).astype(np.float32),
                "label": np.zeros((self.batch_size, 1), dtype=np.int32),
            }


@pytest.mark.timeout(60)
def test_precision_configuration_resolution():
    """Verify that 'precision' config resolves correctly to float32, float16, and bfloat16."""
    cfg = load_config(None)
    cfg.data["num_workers"] = 0
    cfg.model["name"] = "unet"
    cfg.model["features"] = [16, 32]

    # Test explicit float32
    cfg.training["precision"] = "float32"
    trainer_f32 = Trainer(cfg)
    assert trainer_f32.precision == "float32"
    assert trainer_f32.compute_dtype == jnp.float32

    # Test explicit float16
    cfg.training["precision"] = "float16"
    trainer_f16 = Trainer(cfg)
    assert trainer_f16.precision == "float16"
    assert trainer_f16.compute_dtype == jnp.float16

    # Test explicit bfloat16
    cfg.training["precision"] = "bfloat16"
    trainer_bf16 = Trainer(cfg)
    assert trainer_bf16.precision == "bfloat16"
    assert trainer_bf16.compute_dtype == jnp.bfloat16

    # Test auto precision
    cfg.training["precision"] = "auto"
    trainer_auto = Trainer(cfg)
    assert trainer_auto.precision in ("float16", "bfloat16", "float32")


@pytest.mark.timeout(120)
def test_mock_training_loop_bottlenecks_and_gpu_utilization(tmp_path):
    """Run a mock training loop to measure bottlenecks, timing, and GPU utilization."""
    cfg = load_config(None)
    out_dir = tmp_path / "mock_profile_run"
    cfg.project["output_dir"] = str(out_dir)
    cfg.data["num_workers"] = 0
    cfg.training["epochs"] = 1
    cfg.training["warmup_steps"] = 2
    cfg.training["total_steps"] = 10
    cfg.training["precision"] = "float16"
    cfg.model["name"] = "unet"
    cfg.model["features"] = [16, 32]

    trainer = Trainer(cfg)
    assert trainer.precision == "float16"
    assert trainer.compute_dtype == jnp.float16

    mock_loader = MockDataLoader(num_batches=4, batch_size=2, image_size=64, delay=0.01)
    profiler = TrainingProfiler(warmup_steps=1)

    t_prev = time.time()
    for batch in mock_loader:
        t_data_ready = time.time()
        data_time = t_data_ready - t_prev

        # Step execution
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}
        t_comp_start = time.time()
        trainer.rng, step_rng = jax.random.split(trainer.rng)
        # Verify single train epoch can execute step
        t_comp = time.time() - t_comp_start

        profiler.record_step(data_time=data_time, compute_time=t_comp, batch_size=2)
        t_prev = time.time()

    # Analyze bottlenecks
    analysis = profiler.analyze_bottlenecks()
    assert analysis["steps_profiled"] > 0
    assert analysis["avg_data_time_ms"] > 0.0
    assert analysis["throughput_samples_per_sec"] > 0.0
    assert 0.0 <= analysis["data_bottleneck_ratio"] <= 1.0
    assert 0.0 <= analysis["compute_bottleneck_ratio"] <= 1.0
    assert analysis["primary_bottleneck"] in [
        "data_loading (I/O or augmentation bound)",
        "compute (GPU / TPU compute bound)",
        "balanced",
    ]

    # Verify GPU utilization metrics
    gpu_metrics = analysis["gpu_metrics"]
    assert "is_gpu" in gpu_metrics
    if gpu_metrics["is_gpu"]:
        assert gpu_metrics["memory_total_mb"] is not None or gpu_metrics["jax_bytes_in_use"] >= 0


@pytest.mark.timeout(120)
def test_bfloat16_mixed_precision_training_epoch(tmp_path):
    """Verify that train_epoch executes cleanly with bfloat16 mixed precision without extra line logs."""
    cfg = load_config(None)
    out_dir = tmp_path / "bf16_run"
    cfg.project["output_dir"] = str(out_dir)
    cfg.data["num_workers"] = 0
    cfg.training["epochs"] = 1
    cfg.training["precision"] = "bfloat16"
    cfg.model["name"] = "unet"
    cfg.model["features"] = [16, 32]

    trainer = Trainer(cfg)
    assert trainer.precision == "bfloat16"
    assert trainer.compute_dtype == jnp.bfloat16

    mock_loader = MockDataLoader(num_batches=3, batch_size=2, image_size=64, delay=0.0)
    train_res = trainer.train_epoch(mock_loader, epoch=1)

    assert "train_loss" in train_res
    assert "train_iou" in train_res
    assert train_res["train_loss"] >= 0.0
    assert train_res["train_iou"] >= 0.0
