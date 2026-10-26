"""Unit tests for dataset transforms, loaders, and cache shard management."""

import os
import pytest
import numpy as np
from PIL import Image

from sid_unet.dataset.transforms import SpatialJointTransform
from sid_unet.dataset.loader import (
    DataLoader,
    DataLoaderIterator,
    MockDataset,
    ParquetForensicsDataset,
    create_dataloaders,
    numpy_collate,
    process_raw_sample,
    resolve_num_workers,
    resolve_sample_limit,
    safe_dataloader_len,
)
from sid_unet.cache.manager import DatasetCacheManager
from sid_unet.cache.dataset import CachedShardDataset
from sid_unet.models.unet import UNet
from sid_unet.utils.config import load_config


@pytest.mark.timeout(60)
def test_resolve_num_workers():
    expected_cpu = max(1, os.cpu_count() or 4)
    assert resolve_num_workers(-1) == expected_cpu
    assert resolve_num_workers(None) == expected_cpu
    assert resolve_num_workers("auto") == expected_cpu
    assert resolve_num_workers(4) == 4
    assert resolve_num_workers(1) == 1


@pytest.mark.timeout(60)
def test_process_raw_sample_coco_format():
    # Simulate a raw sample matching KhangTruong/COCO-inpainted
    raw_img = Image.new("RGB", (640, 480), color=(200, 100, 50))
    raw_mask = Image.new("L", (640, 480), color=0)
    # Put an inpainted tampering box
    for y in range(100, 200):
        for x in range(100, 200):
            raw_mask.putpixel((x, y), 255)

    sample = {"image": raw_img, "mask": raw_mask}
    processed = process_raw_sample(sample, target_image_size=(512, 512))

    assert processed["image"].shape == (512, 512, 3)
    assert processed["mask"].shape == (512, 512, 1)
    # Check normalization to [-1, 1]
    assert np.all(processed["image"] >= -1.0) and np.all(processed["image"] <= 1.0)
    # Check label is 2 (partially tampered / inpainted)
    assert processed["label"][0] == 2


@pytest.mark.timeout(60)
def test_process_raw_sample_labels():
    # Real image (all black mask)
    sample_real = {"image": Image.new("RGB", (100, 100)), "mask": Image.new("L", (100, 100), color=0)}
    proc_real = process_raw_sample(sample_real, target_image_size=(64, 64))
    assert proc_real["label"][0] == 0

    # Fully synthetic image (all white mask)
    sample_synth = {"image": Image.new("RGB", (100, 100)), "mask": Image.new("L", (100, 100), color=255)}
    proc_synth = process_raw_sample(sample_synth, target_image_size=(64, 64))
    assert proc_synth["label"][0] == 1


@pytest.mark.timeout(60)
def test_spatial_joint_transform():
    transform = SpatialJointTransform(horizontal_flip=1.0, vertical_flip=1.0, random_rotate90=1.0)
    img = np.arange(16).reshape((4, 4, 1)).astype(np.float32)
    mask = np.arange(16).reshape((4, 4, 1)).astype(np.float32)

    t_img, t_mask = transform(img, mask)
    assert t_img.shape == (4, 4, 1)
    assert t_mask.shape == (4, 4, 1)
    np.testing.assert_array_equal(t_img, t_mask)


@pytest.mark.timeout(60)
def test_mock_dataset_and_loader():
    ds = MockDataset(num_samples=16, image_size=(512, 512), is_cached=False)
    assert len(ds) == 16
    sample = ds[0]
    assert sample["image"].shape == (512, 512, 3)
    assert sample["mask"].shape == (512, 512, 1)

    loader = DataLoaderIterator(ds, batch_size=4, shuffle=False, num_workers=2)
    assert len(loader) == 4
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 512, 512, 3)
    assert batch["mask"].shape == (4, 512, 512, 1)


@pytest.mark.timeout(60)
def test_cache_manager_and_shard_dataset(tmp_path):
    dummy_model = UNet(out_channels=1, features=[8, 16])
    dummy_params = {}

    manager = DatasetCacheManager(
        model=dummy_model,
        params=dummy_params,
        source_dataset_name="mock",
        output_dir=str(tmp_path),
        batch_size=4,
        samples_per_shard=8,
        fp16=True,
    )

    shards = manager.cache_split("train", max_samples=16)
    assert len(shards) == 2
    for s in shards:
        assert (tmp_path / "train" / s).exists()

    cached_ds = CachedShardDataset(shards)
    assert len(cached_ds) == 16
    item = cached_ds[0]
    assert "latent" in item
    assert "mask" in item
    assert "label" in item


@pytest.mark.timeout(60)
def test_resolve_sample_limit():
    # steps specified takes precedence and multiplies by batch_size
    assert resolve_sample_limit(samples_val=100, steps_val=50, batch_size=4) == 200
    # negative steps or samples resolves to None (unbounded streaming)
    assert resolve_sample_limit(samples_val=100, steps_val=-1, batch_size=4) is None
    assert resolve_sample_limit(samples_val=-1, steps_val=None, batch_size=4) is None
    # sample count used when steps is None
    assert resolve_sample_limit(samples_val=128, steps_val=None, batch_size=4) == 128
    # fallback to default_samples
    assert resolve_sample_limit(samples_val=None, steps_val=None, batch_size=4, default_samples=64) == 64
    assert resolve_sample_limit(samples_val=None, steps_val=None, batch_size=4, default_samples=None) is None


@pytest.mark.timeout(60)
def test_streaming_dataloader_iterator_prefetch():
    # Iterable dataset that does not support indexing
    class SimpleIterableDataset:
        def __init__(self, n=12):
            self.n = n

        def __iter__(self):
            for i in range(self.n):
                yield {
                    "image": np.full((32, 32, 3), i, dtype=np.float32),
                    "mask": np.zeros((32, 32, 1), dtype=np.float32),
                    "label": np.array([0], dtype=np.int32),
                }

    ds = SimpleIterableDataset(n=10)
    loader = DataLoaderIterator(ds, batch_size=2, shuffle=False, prefetch_batches=4)
    # Length of unindexed streaming dataset with no max_samples raises TypeError
    with pytest.raises(TypeError):
        _ = len(loader)
    assert safe_dataloader_len(loader) is None

    batches = list(loader)
    assert len(batches) == 5
    assert batches[0]["image"].shape == (2, 32, 32, 3)
    assert np.all(batches[0]["image"][0] == 0)
    assert np.all(batches[0]["image"][1] == 1)


@pytest.mark.timeout(60)
def test_create_dataloaders_steps_per_epoch_override():
    cfg = load_config(
        "configs/test_quick.yaml",
        overrides=["data.dataset_name=mock", "data.steps_per_epoch=5", "data.batch_size=4"],
    )
    train_loader, val_loader, _ = create_dataloaders(cfg)
    assert safe_dataloader_len(train_loader) == 5


@pytest.mark.timeout(30)
def test_create_dataloaders_per_device_batch_size():
    """Verify create_dataloaders treats data.batch_size as per-device and computes global batch size."""
    from sid_unet.utils.distributed import get_device_count
    n_dev = get_device_count()
    cfg = load_config(
        "configs/test_quick.yaml",
        overrides=["data.dataset_name=mock", "data.batch_size=3"],
    )
    train_loader, val_loader, _ = create_dataloaders(cfg)
    assert train_loader.per_device_batch_size == 3
    assert train_loader.batch_size == 3 * n_dev
    assert train_loader.global_batch_size == 3 * n_dev


@pytest.mark.timeout(60)
def test_process_raw_sample_bytes_and_dict():
    """Verify process_raw_sample handles raw byte buffers and dict payloads from Parquet."""
    import io
    import torch

    img = Image.new("RGB", (64, 64), color=(50, 150, 250))
    mask = Image.new("L", (64, 64), color=255)

    buf_img = io.BytesIO()
    img.save(buf_img, format="PNG")
    img_bytes = buf_img.getvalue()

    buf_mask = io.BytesIO()
    mask.save(buf_mask, format="PNG")
    mask_bytes = buf_mask.getvalue()

    # Dictionary containing bytes (standard Hugging Face Parquet struct format)
    sample_dict = {
        "image": {"bytes": img_bytes, "path": "test.png"},
        "mask": {"bytes": mask_bytes, "path": "mask.png"},
    }
    res_dict = process_raw_sample(sample_dict, target_image_size=(32, 32))
    assert res_dict["image"].shape == (32, 32, 3)
    assert res_dict["mask"].shape == (32, 32, 1)
    assert res_dict["label"][0] == 1  # All white mask -> synthetic (1)
    assert isinstance(res_dict["image"], np.ndarray)

    # Raw bytes directly
    sample_raw = {"image": img_bytes, "mask": mask_bytes}
    res_raw = process_raw_sample(sample_raw, target_image_size=(32, 32))
    assert res_raw["image"].shape == (32, 32, 3)
    assert res_raw["mask"].shape == (32, 32, 1)


@pytest.mark.timeout(60)
def test_parquet_forensics_dataset_wrapper():
    """Verify ParquetForensicsDataset wraps indexable records with on-the-fly transform."""
    mock_parquet_rows = [
        {"image": Image.new("RGB", (32, 32), color=(i, i, i)), "mask": Image.new("L", (32, 32), color=0)}
        for i in range(10)
    ]

    pds = ParquetForensicsDataset(
        mock_parquet_rows,
        image_size=(64, 64),
        max_samples=6,
    )
    assert len(pds) == 6

    item0 = pds[0]
    assert item0["image"].shape == (64, 64, 3)
    assert item0["mask"].shape == (64, 64, 1)
    assert item0["label"][0] == 0

    # Slicing
    slice_items = pds[1:4]
    assert len(slice_items) == 3
    assert slice_items[0]["image"].shape == (64, 64, 3)


@pytest.mark.timeout(60)
def test_parquet_dataloader_api_features():
    """Verify DataLoader supports benchmark, bottleneck diagnostics, and pure NumPy collation."""
    import torch

    ds = MockDataset(num_samples=16, image_size=(64, 64), is_cached=False)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn="numpy", num_workers=2)

    # Verify collation returns NumPy arrays, not torch.Tensor
    batch = next(iter(loader))
    assert isinstance(batch["image"], np.ndarray)
    assert not isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape == (4, 64, 64, 3)

    # Verify benchmarking API
    bench_result = loader.benchmark(num_batches=2, warmup=0, show_summary=False)
    assert "batches_per_sec" in bench_result
    assert bench_result["total_batches"] == 2

    # Verify debug & bottleneck diagnosis API
    loader.enable_debug()
    for _ in loader:
        pass
    diagnosis = loader.view_bottleneck()
    assert "primary_bottleneck" in diagnosis
    loader.disable_debug()


@pytest.mark.timeout(60)
def test_parquet_forensics_dataset_row_groups_and_dataloader_block_shuffle():
    """Verify ParquetForensicsDataset exposes row group indices and DataLoader uses BlockShuffledSampler."""
    from parquet_dataset_loader.dataset import BlockShuffledSampler

    class MockInnerDataset:
        def __len__(self):
            return 20
        def __getitem__(self, idx):
            return {"image": Image.new("RGB", (16, 16)), "mask": Image.new("L", (16, 16))}
        def get_row_group_indices(self):
            return [list(range(0, 10)), list(range(10, 20))]

    inner = MockInnerDataset()
    pds = ParquetForensicsDataset(inner, max_samples=15)
    assert pds.pdl_dataset is inner
    groups = pds.get_row_group_indices()
    assert len(groups) == 2
    assert groups[0] == list(range(0, 10))
    assert groups[1] == list(range(10, 15))

    loader = DataLoaderIterator(pds, batch_size=5, shuffle=True, seed=42, num_workers=2)
    assert isinstance(loader.sampler, BlockShuffledSampler)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0]["image"].shape == (5, 512, 512, 3)


@pytest.mark.timeout(60)
def test_parquet_forensics_dataset_file_indices_and_forward_cache_config():
    """Verify ParquetForensicsDataset exposes file indices and configuration supports forward caching."""
    from parquet_dataset_loader.dataset import BlockShuffledSampler

    class MockMultiFileInnerDataset:
        def __len__(self):
            return 30
        def __getitem__(self, idx):
            return {"image": Image.new("RGB", (16, 16)), "mask": Image.new("L", (16, 16))}
        def get_row_group_indices(self):
            return [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
        def get_row_group_file_indices(self):
            return [0, 1, 2]

    inner = MockMultiFileInnerDataset()
    pds = ParquetForensicsDataset(inner)
    assert pds.get_row_group_file_indices() == [0, 1, 2]

    sampler = BlockShuffledSampler(pds, seed=42, shuffle=True)
    assert sampler.file_indices == [0, 1, 2]
    indices = list(sampler)
    assert len(indices) == 30
    assert set(indices) == set(range(30))

    # Verify config loading with forward caching options
    cfg = load_config(
        "configs/experiments/diffusion_diff_minimized/default_512.yaml",
        overrides=["data.dataset_name=mock", "data.train_samples_per_epoch=16", "data.val_samples=8"],
    )
    assert cfg.data.cache_files is True
    assert cfg.data.max_cached_files == 4
    assert cfg.data.max_cache_gb == 6.0
    train_loader, val_loader, _ = create_dataloaders(cfg)
    assert len(train_loader) == 16 // train_loader.batch_size

