# ai-detection-unet-tpu

**SID-UNet TPU / JAX**: High-performance, hardware-accelerated deep learning framework for AI-generated synthetic image forensics detection and manipulation segmentation.

Built on **JAX**, **Flax Linen**, and **Optax**, `ai-detection-unet-tpu` is engineered to run seamlessly across:
- **NVIDIA GPUs** (e.g., Tesla T4, A100, H100)
- **Google Cloud TPU v5e-1** (single-core edge/inference/training)
- **Google Cloud TPU v5e-8** (8-core data-parallel pod slices via `jax.pmap` and tensor sharding)

---

## 🌟 Key Features

1. **JAX/Flax Native Architecture & Full Multi-GPU / TPU Utilization**:
   - High-throughput execution with XLA compilation (`jax.jit`).
   - Seamless compatibility with modern JAX (>=0.11) and Flax (>=0.12) with built-in tracer and replication compatibility shims.
   - **Per-Device Batch Size Configuration**: `data.batch_size` in configuration files or CLI overrides is interpreted as **per device**. The global batch size is automatically computed across all detected devices (`global_batch_size = per_device_batch_size * num_devices`), guaranteeing even distribution across dual GPUs or 8 TPU cores.
   - **Multi-Device Data Parallelism via `jax.pmap`**: Automatically utilizes all available GPUs (e.g. dual Tesla T4 GPUs) or TPU cores. Divides microbatches across devices (`shard_batch`), replicates parameters (`replicate_params`), synchronizes gradients and batch normalization running statistics using all-reduce `jax.lax.pmean(axis_name="devices")`, ensuring 100% hardware utilization across all devices.
   - Proper Linen state management: updates `mutable=['batch_stats']` for batch normalization and threads `rngs={'dropout': ...}` PRNG keys across devices.
   - Asynchronous threaded prefetching in `DataLoaderIterator` to keep GPUs/TPUs fully utilized during streaming dataset training.
   - Native support for `bfloat16` (TPU-native precision) and `float32`.

2. **Diffusion-Diff Series Forensics Backbones**:
   - Integrated with [`jax-model-zoo`](https://github.com/khangtruong-ui/jax-model-zoo) (specified in `pyproject.toml`).
   - **`DiffusionDiff`**: Multi-timestep latent feature decoder with sinusoidal timestep and noise schedule embeddings.
   - **`DiffusionDiffV2`**: Perpendicular skip connection fusion injecting parallel frozen VAE decoder skip features directly into trainable upsampling stages.
   - **`DiffusionDiffMinimized`**: Strictly adheres to 2 lines of computation:
     - *Line 1*: Real image clean latent $z_0$ (4 channels)
     - *Line 2*: Single chosen noisy latent from the diffusion model ($t=250$, 80 channels)
     - Total representation $Z = 84$ channels with parallel VAE perpendicular skips.

3. **Periodic Checkpointing & Hugging Face Hub Sync**:
   - **Periodic Checkpointing (`checkpoint_period`)**: Supports real-time periodic checkpointing during long training runs (e.g. `checkpoint_period: "2m"`, `"1h"`, or `3600`). Saves `checkpoint_periodic.pt` and updates `checkpoint_latest.pt` without waiting for epoch completion. The checkpoint timer starts **strictly after the second batch has been consumed**, ensuring that initial XLA / JIT graph compilation delays do not prematurely trigger early checkpoint saves and that the checkpoint timer accurately accounts for real training speed (including when resuming).
   - **Hugging Face Hub Integration (`--hf-repo`, `hf_repo_id`, `push_to_hub`)**: Automatically uploads checkpoints (`checkpoint_latest.pt`, `checkpoint_best.pt`, `checkpoint_periodic.pt`) to Hugging Face model repositories (e.g. `KhangTruong/diffusion-diff-minimized-512-fpn`). Directly pass `--hf-repo <repo_id>` on the CLI or specify in config.
   - **Hugging Face Auto-Resume**: Directly pass Hugging Face model identifiers or URIs (e.g. `--resume KhangTruong/diffusion-diff-minimized-512` or `hf://...`) to download and resume checkpoints automatically.

4. **100% Checkpoint Interoperability with `khangtruong-ui/ai-detection-unet-gpu`**:
   - Model weights default to saving and loading in the exact PyTorch `state_dict` format expected by `ai-detection-unet-gpu`.
   - Seamlessly loads `.pt` and `.safetensors` checkpoints created by PyTorch trainers, converting them into Flax parameter PyTrees on the fly.
   - Normalized layer mappings for auxiliary classifiers (`classifier_head.fc.1` / `fc.4` and `fc1` / `fc2`).
   - Saved checkpoints contain `model_state_dict` with canonical PyTorch tensor keys (`z_norm.weight`, `decoder.init_conv.0.weight`, etc.) for instant cross-framework deployment.
   - Rigorously tested by loading checkpoints generated from the `khangtruong-ui/ai-detection-unet-gpu` repository.

5. **Detached Terminal Execution Logging to `.logs.txt` with Clean TQDM Progress**:
   - All primary CLI commands (`sid-train`, `sid-cache`, `sid-kill`) default to **detaching from the controlling terminal session** into background mode (`setsid`).
   - Standard output and standard error are automatically streamed into `.logs.txt`.
   - Dedicated, clean `tqdm` progress tracking with live step metric monitoring (`loss`, `iou`) without interleaved step logs, combined with comprehensive epoch-level metric summaries.
   - Immune to terminal window closes, SSH disconnects, and hangup signals (`SIGHUP`, `SIGPIPE`).
   - Use `--foreground` or `--no-detach` anytime for interactive debugging.

6. **Mixed Precision Support & Pipeline Bottleneck Profiling**:
   - Configurable compute precision via `--precision auto|float16|bfloat16|float32` (or `training.precision` in YAML config). Automatically selects `bfloat16` for TPU cores and `float16` for GPUs with Tensor Cores, keeping float32 master weights for numerical stability.
   - Built-in `TrainingProfiler` and `get_gpu_utilization` to detect data loading vs compute bottlenecks, throughput (samples/sec), and GPU memory/utilization.

7. **Integrated Process Termination (`sid-kill`)**:
   - Automatically discovers running or orphaned `sid_unet` processes.
   - Performs two-stage termination: graceful shutdown (`SIGINT`/`SIGTERM`) followed by timeout escalation to `SIGKILL`.

8. **Default Configuration for Real Forensics Datasets**:
   - Default dataset set to [`KhangTruong/COCO-inpainted`](https://huggingface.co/datasets/KhangTruong/COCO-inpainted).
   - Standard forensics resolution `image_size: [512, 512]`.
   - Automatic multi-worker data loading defaulting to full CPU count (`os.cpu_count()`).

9. **High-Performance `parquet-dataset-loader` Integration & DataLoader API**:
   - Integrated with [`parquet-dataset-loader`](https://github.com/khangtruong-ui/parquet-dataset-loader) for non-blocking index streaming (`use_parquet_loader: true`).
   - Seeks directly to row groups using HTTP range requests without downloading entire multi-gigabyte datasets (such as 62 GB of `COCO-inpainted`).
   - Features `ParquetForensicsDataset` sequence adapter for on-the-fly sample preprocessing, decoding compressed PNG/JPEG bytes, and joint spatial transformations.
   - Automatically maintains row-group and file cache locality via `BlockShuffledSampler`, eliminating multi-worker cache thrashing, remote connection resets, and DataLoader timeout errors.
   - Bounded on-disk forward file caching (`cache_files: true`, `max_cached_files: 4`, `max_cache_gb: 6.0`) with background prefetching, eliminating Hugging Face HTTP 429 rate limit errors during multi-worker streaming iterations.
   - Powered by `parquet_dataset_loader.DataLoader` with `numpy_collate` for pure NumPy batch collation, preventing PyTorch tensor conversion issues in JAX/Flax pipelines.
   - Built-in data speed benchmarking via `loader.benchmark(num_batches=50)` and bottleneck diagnostics via `loader.view_bottleneck()`.

---

## 🚀 Quick Start

### Installation

Clone the repository and install in editable mode:
```bash
git clone https://github.com/khangtruong-ui/ai-detection-unet-tpu.git
cd ai-detection-unet-tpu
pip install -e .
```

To install on Google Cloud TPU:
```bash
pip install -e .
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## 🛠️ CLI Utilities

### 1. `sid-train` (Model Training)

By default, `sid-train` detaches into the background and streams logs to `.logs.txt`:
```bash
# Default background detached mode
sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml

# Interactive foreground mode with Hugging Face Hub checkpoint upload
sid-train --config configs/experiments/diffusion_diff_minimized/default_512.yaml --hf-repo KhangTruong/diffusion-diff-minimized-512-fpn --foreground

# Override training hyperparameters on the command line (e.g. multi-device batch size and periodic checkpoint interval)
sid-train --config configs/experiments/diffusion_diff_minimized/default_512.yaml --override data.batch_size=4 training.checkpoint_period=30m
```

### 2. `sid-cache` (Latent Feature Caching)

Extracts high-dimensional diffusion forensics representations $Z$ and stores them in compressed shards for up to $50\times$ faster training:
```bash
# Default background detached caching
sid-cache --config configs/experiments/diffusion_diff_minimized/default.yaml --max-samples 1000

# Foreground execution
sid-cache --config configs/experiments/diffusion_diff_minimized/default.yaml --foreground --batch-size 16
```

### 3. `sid-kill` (Process Cleanup)

Discovers and safely terminates background training and caching tasks:
```bash
# Dry run: view matching processes without killing
sid-kill --dry-run

# Gracefully terminate all background sid_unet processes
sid-kill

# Force kill stubborn tasks immediately with SIGKILL
sid-kill --force
```

### 4. `sid-eval` & `sid-predict` (Evaluation & Inference)

```bash
# Evaluate checkpoint on validation set
sid-eval --config configs/default.yaml --checkpoint outputs/default_run/checkpoints/checkpoint_best.pt

# Generate binary manipulation masks
sid-predict --config configs/default.yaml --checkpoint outputs/default_run/checkpoints/checkpoint_best.pt --input sample.png
```

---

## 📁 Repository Structure

```text
ai-detection-unet-tpu/
├── pyproject.toml              # Build metadata, scripts, dependencies
├── setup.py                    # Setuptools integration
├── README.md                   # Complete documentation
├── configs/                    # YAML configuration files (fully aligned with ai-detection-unet-gpu)
│   ├── default.yaml
│   ├── train_streaming.yaml
│   ├── train_non_streaming.yaml
│   ├── evaluate.yaml
│   ├── test_smoke.yaml
│   ├── test_quick.yaml
│   ├── cross-eval/             # Benchmark evaluation configs (diffseg30k, casia, cocoglide, etc.)
│   └── experiments/
│       ├── diffusion_diff/
│       ├── diffusion_diff_v2/
│       ├── diffusion_diff_minimized/
│       └── unet_scratch/
├── sid_unet/
│   ├── __init__.py             # Package init & JAX/Flax compatibility shims
│   ├── train.py                # sid-train CLI entrypoint
│   ├── kill.py                 # sid-kill CLI entrypoint
│   ├── evaluate.py             # sid-eval CLI entrypoint
│   ├── predict.py              # sid-predict CLI entrypoint
│   ├── cache/                  # Latent feature caching pipeline
│   │   ├── cli.py              # sid-cache CLI entrypoint
│   │   ├── extractor.py        # Model cache feature extractors
│   │   ├── dataset.py          # Cached tensor shard loader
│   │   └── manager.py          # Progressive shard manager
│   ├── dataset/                # Data loader and spatial joint transforms
│   ├── losses/                 # BCE, Soft Dice, Focal, Combined mask loss
│   ├── metrics/                # IoU, F1, AUROC, Accuracy accumulators
│   ├── models/                 # DiffusionDiff series & UNet in Flax Linen
│   ├── training/               # Flax/Optax Trainer & CheckpointManager
│   └── utils/                  # Checkpoint compatibility, detach, signals, distributed
└── tests/                      # Comprehensive pytest test suite with timeout guards
```

---

## 🧪 Testing

Every test includes strict timeout guards (`timeout = 240s`) to ensure no runaway processes:
```bash
pytest -v tests/
```

---

## 📜 License

Licensed under the [Apache-2.0 License](LICENSE).
