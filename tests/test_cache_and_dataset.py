"""Unit tests for dataset transforms, loaders, and cache shard management."""

import pytest
import numpy as np

from sid_unet.dataset.transforms import SpatialJointTransform
from sid_unet.dataset.loader import MockDataset, DataLoaderIterator
from sid_unet.cache.manager import DatasetCacheManager
from sid_unet.cache.dataset import CachedShardDataset
from sid_unet.models.unet import UNet


@pytest.mark.timeout(60)
def test_spatial_joint_transform():
    transform = SpatialJointTransform(horizontal_flip=1.0, vertical_flip=1.0, random_rotate90=1.0)
    img = np.arange(16).reshape((4, 4, 1)).astype(np.float32)
    mask = np.arange(16).reshape((4, 4, 1)).astype(np.float32)

    t_img, t_mask = transform(img, mask)
    assert t_img.shape == (4, 4, 1)
    assert t_mask.shape == (4, 4, 1)
    # Both transformations applied congruently
    np.testing.assert_array_equal(t_img, t_mask)


@pytest.mark.timeout(60)
def test_mock_dataset_and_loader():
    ds = MockDataset(num_samples=16, image_size=(32, 32), is_cached=False)
    assert len(ds) == 16
    sample = ds[0]
    assert sample["image"].shape == (32, 32, 3)
    assert sample["mask"].shape == (32, 32, 1)

    loader = DataLoaderIterator(ds, batch_size=4, shuffle=False)
    assert len(loader) == 4
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 32, 32, 3)
    assert batch["mask"].shape == (4, 32, 32, 1)


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

    # Read back through CachedShardDataset
    cached_ds = CachedShardDataset(shards)
    assert len(cached_ds) == 16
    item = cached_ds[0]
    assert "latent" in item
    assert "mask" in item
    assert "label" in item
