# ai-detection-unet-tpu

**SID-UNet TPU / JAX**: High-performance, hardware-accelerated deep learning framework for AI-generated synthetic image forensics detection and manipulation segmentation.

Built on **JAX**, **Flax Linen**, and **Optax**, `ai-detection-unet-tpu` is engineered to run seamlessly across:
- **NVIDIA GPUs** (e.g., Tesla T4, A100, H100)
- **Google Cloud TPU v5e-1** (single-core edge/inference/training)
- **Google Cloud TPU v5e-8** (8-core data-parallel pod slices via `jax.pmap` and tensor sharding)

---

## 🌟 Key Features

1. **JAX/Flax Native Architecture & GPU/TPU Utilization**:
   - High-throughput execution with XLA compilation (`jax.jit`).
   - Proper Linen state management: updates `mutable=['batch_stats']` for batch normalization and threads `rngs={'dropout': ...}` PRNG keys, preventing fallback loops and maximizing hardware utilization.
   - Asynchronous threaded prefetching in `DataLoaderIterator` to keep GPUs/TPUs fully utilized during streaming dataset training.
   - Multi-device single-host scaling across 8 TPU v5e cores or multi-GPU environments using `jax.pmap` and `axis_name='devices'`.
   - Native support for `bfloat16` (TPU-native precision) and `float32`.

2. **Diffusion-Diff Series Forensics Backbones**:
   - Integrated with [`jax-model-zoo`](https://github.com/khangtruong-ui/jax-model-zoo) (specified in `pyproject.toml`).
   - **`DiffusionDiff`**: Multi-timestep latent feature decoder with sinusoidal timestep and noise schedule embeddings.
   - **`DiffusionDiffV2`**: Perpendicular skip connection fusion injecting parallel frozen VAE decoder skip features directly into trainable upsampling stages.
   - **`DiffusionDiffMinimized`**: Strictly adheres to 2 lines of computation:
     - *Line 1*: Real image clean latent $z_0$ (4 channels)
     - *Line 2*: Single chosen noisy latent from the diffusion model ($t=250$, 80 channels)
     - Total representation $Z = 84$ channels with parallel VAE perpendicular skips.

3. **Flexible Epoch Sizing & Robust Batch Handling**:
   - Dynamic sample resolution via `resolve_sample_limit`: easily configure `steps_per_epoch` or `train_samples_per_epoch` directly from CLI overrides or YAML configs.
   - Open-ended streaming datasets (setting limit to `-1` or `None`) stream continuously without being artificially clamped to 1000 steps.
   - Robust batch dimension handling: loss formulations (segmentation mask losses and auxiliary classification cross-entropy) natively handle arbitrary batch dimensions including single-sample executions (`batch_size=1`) for memory-constrained environments.

4. **100% Checkpoint Interoperability with `ai-detection-unet-gpu`**:
   - Model weights default to saving and loading in the exact PyTorch `state_dict` format expected by `ai-detection-unet-gpu`.
   - Seamlessly loads `.pt` and `.safetensors` checkpoints created by PyTorch trainers, converting them into Flax parameter PyTrees on the fly.
   - Saved checkpoints contain `model_state_dict` with canonical PyTorch tensor keys (`z_norm.weight`, `decoder.init_conv.0.weight`, etc.) for instant cross-framework deployment.

5. **Detached Terminal Execution Logging to `.logs.txt` with TQDM Progress**:
   - All primary CLI commands (`sid-train`, `sid-cache`, `sid-kill`) default to **detaching from the controlling terminal session** into background mode (`setsid`).
   - Standard output and standard error are automatically streamed into `.logs.txt`.
   - Rich `tqdm` progress tracking with live metric monitoring (`loss`, `iou`, lr, steps, epochs) during both training and validation, as well as periodic step-level and epoch-level metric logs.
   - Immune to terminal window closes, SSH disconnects, and hangup signals (`SIGHUP`, `SIGPIPE`).
   - Use `--foreground` or `--no-detach` anytime for interactive debugging.

6. **Integrated Process Termination (`sid-kill`)**:
   - Automatically discovers running or orphaned `sid_unet` processes.
   - Performs two-stage termination: graceful shutdown (`SIGINT`/`SIGTERM`) followed by timeout escalation to `SIGKILL`.

7. **Default Configuration for Real Forensics Datasets**:
   - Default dataset set to [`KhangTruong/COCO-inpainted`](https://huggingface.co/datasets/KhangTruong/COCO-inpainted).
   - Standard forensics resolution `image_size: [512, 512]`.
   - Automatic multi-worker data loading defaulting to full CPU count (`os.cpu_count()`).

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

# Interactive foreground mode
sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml --foreground

# Override training hyperparameters on the command line
sid-train --config configs/default.yaml --override training.batch_size=16 training.learning_rate=0.0002
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
│   ├── __init__.py
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
