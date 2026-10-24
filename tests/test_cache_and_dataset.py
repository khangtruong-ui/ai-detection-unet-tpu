"""Unit tests for dataset transforms, loaders, and cache shard management."""

import os
import pytest
import numpy as np
from PIL import Image

from sid_unet.dataset.transforms import SpatialJointTransform
from sid_unet.dataset.loader import (
    MockDataset,
    DataLoaderIterator,
    resolve_num_workers,
    process_raw_sample,
    create_dataloaders,
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
