"""Tests for the LIDC Dataset and DataLoaders.

Most tests run against a small synthetic npz written into ``tmp_path`` so the suite
stays fast; the few that check the real numbers skip cleanly when
``data/processed/lidc.npz`` is absent.

The properties that matter here are the ones that fail silently: the grader pairing
must vary across epochs yet replay from the seed, empty masks must be drawn rather
than skipped, dtypes must be exactly what each consumer needs, and no patch may leak
between splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from probunet.data.lidc import (
    DEFAULT_PAIRING_SEED,
    DataConfig,
    LidcArrays,
    LidcDataset,
    build_data,
    group_by_bucket,
    nonempty_grader_counts,
)
from probunet.data.splits import SPLIT_NAMES, generate_split, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_NPZ = REPO_ROOT / "data" / "processed" / "lidc.npz"
REAL_SPLIT = REPO_ROOT / "data" / "splits" / "split.json"
N_GRADERS = 4
SIZE = 8


def write_npz(
    path: Path,
    sizes: list[int],
    seed: int = 0,
    nonempty_plan: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a synthetic lidc.npz.

    Args:
        path: Destination file.
        sizes: Patch count per synthetic series.
        seed: Seed for the random content.
        nonempty_plan: Optional map from patch row to the exact number of non-empty
            grader masks that patch should have. Rows not listed get a random count
            in ``1..4``, matching the real data where no patch has zero.

    Returns:
        The ``(images, masks)`` arrays that were written.
    """
    uids = []
    for index, size in enumerate(sizes):
        uids.extend([f"series-{index:04d}"] * size)
    series = np.array(uids, dtype=np.str_)
    n_patches = series.size

    rng = np.random.default_rng(seed)
    images = rng.random((n_patches, SIZE, SIZE), dtype=np.float32)
    masks = np.zeros((n_patches, N_GRADERS, SIZE, SIZE), dtype=np.uint8)
    for row in range(n_patches):
        planned = (nonempty_plan or {}).get(row)
        count = planned if planned is not None else int(rng.integers(1, N_GRADERS + 1))
        for slot in range(count):
            # A distinct, non-trivial footprint per (row, slot).
            masks[row, slot, : 1 + slot, : 1 + (row % 3)] = 1
    keys = np.array([f"{uid}_slice{i}" for i, uid in enumerate(uids)], dtype=np.str_)
    np.savez_compressed(path, images=images, masks=masks, series_uid=series, keys=keys)
    return images, masks


@pytest.fixture
def tiny(tmp_path: Path) -> DataConfig:
    """A synthetic dataset plus a generated split, wired into a DataConfig."""
    npz = tmp_path / "lidc.npz"
    write_npz(npz, [7, 5, 3, 11, 2, 9, 4, 6, 8, 1])
    split = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=split)
    return DataConfig(npz_path=npz, split_path=split, batch_size=4)


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def test_load_and_validate(tiny: DataConfig) -> None:
    """A well-formed file loads and validates."""
    arrays = LidcArrays.load(tiny.npz_path)
    assert len(arrays) == 56
    assert arrays.spatial_shape == (SIZE, SIZE)
    assert arrays.images.dtype == np.float32
    assert arrays.masks.dtype == np.uint8
    assert arrays.keys is not None


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    """A missing dataset points at the converter rather than failing obscurely."""
    with pytest.raises(FileNotFoundError, match="convert_data.py"):
        LidcArrays.load(tmp_path / "absent.npz")


def test_missing_array_is_an_error(tmp_path: Path) -> None:
    """A file without the required arrays names what is missing."""
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, images=np.zeros((2, SIZE, SIZE), dtype=np.float32))
    with pytest.raises(ValueError, match="missing required array"):
        LidcArrays.load(path)


def test_non_binary_masks_are_rejected(tmp_path: Path) -> None:
    """A stray value above 1 -- e.g. a 0/255 encoding -- fails loudly."""
    npz = tmp_path / "bad.npz"
    images, masks = write_npz(npz, [4])
    masks[0, 0, 0, 0] = 255
    np.savez_compressed(
        npz,
        images=images,
        masks=masks,
        series_uid=np.array(["s"] * images.shape[0], dtype=np.str_),
    )
    with pytest.raises(ValueError, match="strictly binary"):
        LidcArrays.load(npz)


def test_float_masks_are_rejected(tmp_path: Path) -> None:
    """Soft labels are not supported by the baseline and must not slip through."""
    npz = tmp_path / "soft.npz"
    images, masks = write_npz(npz, [4])
    np.savez_compressed(
        npz,
        images=images,
        masks=masks.astype(np.float32),
        series_uid=np.array(["s"] * images.shape[0], dtype=np.str_),
    )
    with pytest.raises(ValueError, match="integer dtype"):
        LidcArrays.load(npz)


def test_images_outside_unit_range_are_rejected(tmp_path: Path) -> None:
    """Images must already be in [0, 1]; anything else signals a bad conversion."""
    npz = tmp_path / "range.npz"
    images, masks = write_npz(npz, [4])
    images[0, 0, 0] = 1.5
    np.savez_compressed(
        npz,
        images=images,
        masks=masks,
        series_uid=np.array(["s"] * images.shape[0], dtype=np.str_),
    )
    with pytest.raises(ValueError, match=r"outside \[0.0, 1.0\]"):
        LidcArrays.load(npz)


def test_validation_can_be_skipped(tmp_path: Path) -> None:
    """validate=False loads without checking, for deliberate experimentation."""
    npz = tmp_path / "bad.npz"
    images, masks = write_npz(npz, [4])
    masks[0, 0, 0, 0] = 255
    np.savez_compressed(
        npz,
        images=images,
        masks=masks,
        series_uid=np.array(["s"] * images.shape[0], dtype=np.str_),
    )
    assert LidcArrays.load(npz, validate=False).masks.max() == 255


# --------------------------------------------------------------------------- #
# Config hooks
# --------------------------------------------------------------------------- #
def test_normalization_hook_raises() -> None:
    """The normalization hook refuses to silently no-op."""
    with pytest.raises(NotImplementedError, match="normalization"):
        DataConfig(normalization="standardize")


def test_augment_hook_raises() -> None:
    """The augmentation hook refuses to silently no-op."""
    with pytest.raises(NotImplementedError, match="augmentation"):
        DataConfig(augment=True)


def test_config_validates_numbers() -> None:
    """Nonsense loader settings are rejected."""
    with pytest.raises(ValueError, match="batch_size"):
        DataConfig(batch_size=0)
    with pytest.raises(ValueError, match="num_workers"):
        DataConfig(num_workers=-1)


def test_pairing_seed_default_is_distinct_from_split_seed() -> None:
    """The pairing seed is logically independent of the frozen split seed."""
    from probunet.data.splits import DEFAULT_SEED

    assert DEFAULT_PAIRING_SEED == 2018
    assert DEFAULT_PAIRING_SEED != DEFAULT_SEED


# --------------------------------------------------------------------------- #
# Dtypes and shapes
# --------------------------------------------------------------------------- #
def test_train_sample_dtypes_and_shapes(tiny: DataConfig) -> None:
    """Train samples carry an int64 CE target and a float32 image."""
    data = build_data(tiny)
    sample = data.datasets["train"][0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (1, SIZE, SIZE)
    assert sample["mask"].dtype == torch.int64
    assert sample["mask"].shape == (SIZE, SIZE)
    assert sample["grader"].dtype == torch.int64
    assert sample["index"].dtype == torch.int64
    assert set(torch.unique(sample["mask"]).tolist()) <= {0, 1}


def test_eval_sample_returns_all_four_masks(tiny: DataConfig) -> None:
    """Eval samples keep every grader mask, as uint8."""
    data = build_data(tiny)
    sample = data.datasets["val"][0]
    assert "mask" not in sample
    assert sample["masks"].dtype == torch.uint8
    assert sample["masks"].shape == (N_GRADERS, SIZE, SIZE)


def test_eval_masks_match_the_source(tiny: DataConfig) -> None:
    """Eval masks are exactly the stored rows."""
    data = build_data(tiny)
    dataset = data.datasets["test"]
    for position in range(min(5, len(dataset))):
        sample = dataset[position]
        row = int(sample["index"])
        assert np.array_equal(sample["masks"].numpy(), data.arrays.masks[row])


def test_train_mask_is_the_selected_grader(tiny: DataConfig) -> None:
    """The emitted target is the mask of the grader that was drawn."""
    data = build_data(tiny)
    dataset = data.datasets["train"]
    for position in range(len(dataset)):
        sample = dataset[position]
        row = int(sample["index"])
        grader = int(sample["grader"])
        expected = data.arrays.masks[row, grader].astype(np.int64)
        assert np.array_equal(sample["mask"].numpy(), expected)


def test_posterior_accepts_the_emitted_mask(tiny: DataConfig) -> None:
    """The int64 target concatenates cleanly onto the float32 image."""
    from probunet.model import PosteriorNet

    data = build_data(tiny)
    batch = next(iter(data.loaders["train"]))
    posterior = PosteriorNet(image_channels=1, mask_channels=1, latent_dim=6)
    assembled = posterior.assemble_input(batch["image"], batch["mask"])
    assert assembled.dtype == torch.float32
    assert assembled.shape == (batch["image"].shape[0], 2, SIZE, SIZE)


# --------------------------------------------------------------------------- #
# No normalization
# --------------------------------------------------------------------------- #
def test_images_are_passed_through_unchanged(tiny: DataConfig) -> None:
    """No normalization is applied: the image is bit-identical to the stored row."""
    data = build_data(tiny)
    dataset = data.datasets["train"]
    for position in range(min(10, len(dataset))):
        sample = dataset[position]
        row = int(sample["index"])
        assert np.array_equal(
            sample["image"].numpy()[0], data.arrays.images[row]
        ), "image was modified; the baseline applies no normalization"


def test_sample_does_not_alias_the_shared_arrays(tiny: DataConfig) -> None:
    """Mutating a sample must not corrupt the dataset for other splits."""
    data = build_data(tiny)
    sample = data.datasets["train"][0]
    row = int(sample["index"])
    original = data.arrays.images[row].copy()
    sample["image"] += 1.0
    assert np.array_equal(data.arrays.images[row], original)


# --------------------------------------------------------------------------- #
# Random grader pairing
# --------------------------------------------------------------------------- #
def test_pairing_varies_across_epochs(tiny: DataConfig) -> None:
    """Two epochs give different pairings."""
    dataset = build_data(tiny).datasets["train"]
    dataset.set_epoch(0)
    first = dataset.graders.copy()
    dataset.set_epoch(1)
    assert not np.array_equal(first, dataset.graders)


def test_pairing_is_reproducible_from_the_seed(tiny: DataConfig) -> None:
    """The same seed replays the same sequence of epochs."""
    first = build_data(tiny).datasets["train"]
    second = build_data(tiny).datasets["train"]
    for epoch in range(4):
        first.set_epoch(epoch)
        second.set_epoch(epoch)
        assert np.array_equal(first.graders, second.graders)


def test_different_pairing_seeds_differ(tiny: DataConfig) -> None:
    """A different run seed gives a different pairing."""
    from dataclasses import replace

    first = build_data(tiny).datasets["train"]
    second = build_data(replace(tiny, pairing_seed=999)).datasets["train"]
    first.set_epoch(0)
    second.set_epoch(0)
    assert not np.array_equal(first.graders, second.graders)


def test_pairing_is_uniform_over_all_four_graders(tiny: DataConfig) -> None:
    """Every grader is drawn about a quarter of the time."""
    dataset = build_data(tiny).datasets["train"]
    counts = np.zeros(N_GRADERS, dtype=np.int64)
    epochs = 400
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        counts += np.bincount(dataset.graders, minlength=N_GRADERS)
    expected = counts.sum() / N_GRADERS
    assert np.all(np.abs(counts - expected) < 0.1 * expected), counts.tolist()


def test_empty_masks_are_drawn_not_skipped(tmp_path: Path) -> None:
    """A patch with one non-empty grader yields an empty target ~3 epochs in 4.

    Skipping empty masks would remove exactly the lesion-absence cases the model has
    to learn to reproduce, so this is the test that guards the sampling policy.
    """
    npz = tmp_path / "lidc.npz"
    # Every patch has exactly one non-empty grader mask (slot 0).
    write_npz(npz, [6, 6, 6, 6], nonempty_plan={row: 1 for row in range(24)})
    split = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=split)
    dataset = build_data(DataConfig(npz_path=npz, split_path=split)).datasets["train"]

    empty = 0
    total = 0
    for epoch in range(200):
        dataset.set_epoch(epoch)
        for position in range(len(dataset)):
            total += 1
            if dataset[position]["mask"].sum() == 0:
                empty += 1
    fraction = empty / total
    assert 0.65 < fraction < 0.85, f"empty-target fraction {fraction:.3f}, expected ~0.75"


def test_set_epoch_rejected_on_eval_dataset(tiny: DataConfig) -> None:
    """Evaluation datasets have no pairing to redraw."""
    data = build_data(tiny)
    with pytest.raises(ValueError, match="mode='train'"):
        data.datasets["val"].set_epoch(1)
    with pytest.raises(ValueError, match="evaluation dataset"):
        _ = data.datasets["val"].graders


def test_lidc_data_set_epoch_fans_out(tiny: DataConfig) -> None:
    """LidcData.set_epoch updates the training dataset."""
    data = build_data(tiny)
    data.set_epoch(5)
    assert data.datasets["train"].epoch == 5


def test_unknown_mode_rejected(tiny: DataConfig) -> None:
    """An unknown mode is an error."""
    arrays = LidcArrays.load(tiny.npz_path)
    with pytest.raises(ValueError, match="mode must be"):
        LidcDataset(arrays, np.array([0, 1]), mode="both")  # type: ignore[arg-type]


def test_out_of_range_indices_rejected(tiny: DataConfig) -> None:
    """Indices beyond the dataset are caught at construction."""
    arrays = LidcArrays.load(tiny.npz_path)
    with pytest.raises(ValueError, match="out of range"):
        LidcDataset(arrays, np.array([0, len(arrays)]), mode="eval")


# --------------------------------------------------------------------------- #
# No leakage, complete coverage
# --------------------------------------------------------------------------- #
def test_no_patch_leaks_across_splits(tiny: DataConfig) -> None:
    """Split index sets are disjoint and cover every patch exactly once."""
    data = build_data(tiny)
    seen: set[int] = set()
    for name in SPLIT_NAMES:
        indices = set(data.datasets[name].indices.tolist())
        assert not (indices & seen), f"{name} overlaps an earlier split"
        seen |= indices
    assert seen == set(range(len(data.arrays)))


def test_no_series_leaks_across_splits(tiny: DataConfig) -> None:
    """No CT series contributes patches to two splits."""
    data = build_data(tiny)
    owners: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for row in data.datasets[name].indices:
            uid = str(data.arrays.series_uid[row])
            assert owners.setdefault(uid, name) == name, f"{uid} spans splits"


def test_nothing_is_filtered(tiny: DataConfig) -> None:
    """Dataset lengths equal the split sizes: no patch is dropped."""
    data = build_data(tiny)
    split = load_split(tiny.split_path)
    for name in SPLIT_NAMES:
        assert len(data.datasets[name]) == split.indices[name].size


# --------------------------------------------------------------------------- #
# Ambiguity buckets
# --------------------------------------------------------------------------- #
def test_nonempty_grader_counts(tmp_path: Path) -> None:
    """Counting non-empty graders works for all patches and for a subset."""
    npz = tmp_path / "lidc.npz"
    _, masks = write_npz(npz, [4], nonempty_plan={0: 1, 1: 2, 2: 3, 3: 4})
    counts = nonempty_grader_counts(masks)
    assert counts.tolist() == [1, 2, 3, 4]
    assert nonempty_grader_counts(masks, np.array([3, 0])).tolist() == [4, 1]


def test_group_by_bucket() -> None:
    """Bucketing returns all five buckets, empty ones included."""
    counts = np.array([1, 4, 1, 2])
    buckets = group_by_bucket(counts)
    assert sorted(buckets) == [0, 1, 2, 3, 4]
    assert buckets[1].tolist() == [0, 2]
    assert buckets[4].tolist() == [1]
    assert buckets[0].size == 0

    indices = np.array([10, 11, 12, 13])
    assert group_by_bucket(counts, indices)[1].tolist() == [10, 12]


def test_dataset_buckets_use_global_indices(tiny: DataConfig) -> None:
    """A dataset's buckets map to global patch indices, ready for reporting."""
    data = build_data(tiny)
    dataset = data.datasets["val"]
    buckets = dataset.buckets()
    combined = np.concatenate([buckets[k] for k in range(N_GRADERS + 1)])
    assert sorted(combined.tolist()) == sorted(dataset.indices.tolist())


def test_arrays_nonempty_counts_are_cached(tiny: DataConfig) -> None:
    """Repeated bucket queries reuse one computation."""
    arrays = LidcArrays.load(tiny.npz_path)
    first = arrays.nonempty_counts()
    assert arrays.nonempty_counts() is first


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def test_loader_batch_shapes(tiny: DataConfig) -> None:
    """Collated batches have the expected shapes and dtypes."""
    data = build_data(tiny)
    train_batch = next(iter(data.loaders["train"]))
    assert train_batch["image"].shape[1:] == (1, SIZE, SIZE)
    assert train_batch["mask"].shape[1:] == (SIZE, SIZE)
    assert train_batch["mask"].dtype == torch.int64

    eval_batch = next(iter(data.loaders["val"]))
    assert eval_batch["masks"].shape[1:] == (N_GRADERS, SIZE, SIZE)
    assert eval_batch["masks"].dtype == torch.uint8


def test_eval_loader_is_deterministic_and_unshuffled(tiny: DataConfig) -> None:
    """Eval order is exactly the split's index order, on every pass."""
    data = build_data(tiny)
    expected = data.datasets["test"].indices.tolist()
    for _ in range(2):
        seen = [
            int(index)
            for batch in data.loaders["test"]
            for index in batch["index"]
        ]
        assert seen == expected


def test_train_loader_shuffles_reproducibly(tiny: DataConfig) -> None:
    """Training order is shuffled, differs per epoch, and replays from the seed."""

    def order(config: DataConfig, epochs: int) -> list[list[int]]:
        loader = build_data(config).loaders["train"]
        return [
            [int(i) for batch in loader for i in batch["index"]] for _ in range(epochs)
        ]

    first = order(tiny, 2)
    assert first[0] != first[1], "every epoch used the same order"
    assert first[0] != sorted(first[0]), "training order was not shuffled"
    assert order(tiny, 2) == first, "training order did not replay from the seed"


def test_loaders_never_use_persistent_workers(tiny: DataConfig) -> None:
    """Persistent workers would freeze a stale grader pairing."""
    data = build_data(tiny)
    for name in SPLIT_NAMES:
        assert data.loaders[name].persistent_workers is False


def test_set_epoch_changes_what_the_loader_yields(tiny: DataConfig) -> None:
    """The pairing redraw is visible through the DataLoader."""
    data = build_data(tiny)
    data.set_epoch(0)
    first = torch.cat([batch["grader"] for batch in data.loaders["train"]])
    data.set_epoch(1)
    second = torch.cat([batch["grader"] for batch in data.loaders["train"]])
    assert not torch.equal(first, second)


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_SPLIT.exists()), reason="converted dataset absent"
)
def test_real_split_counts_match_split_json() -> None:
    """Dataset lengths equal the counts recorded in split.json."""
    data = build_data(DataConfig())
    recorded = json.loads(REAL_SPLIT.read_text())["achieved"]
    for name in SPLIT_NAMES:
        assert len(data.datasets[name]) == recorded[name]["n_patches"]
    assert len(data.arrays) == 15_096
    assert data.arrays.spatial_shape == (128, 128)


@pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_SPLIT.exists()), reason="converted dataset absent"
)
def test_real_empty_mask_distribution_per_split() -> None:
    """The per-split ambiguity distribution matches the documented figures."""
    data = build_data(DataConfig())
    expected = {
        "train": [0, 2978, 1653, 1575, 2850],
        "val": [0, 993, 552, 526, 950],
        "test": [0, 992, 551, 525, 951],
    }
    for name in SPLIT_NAMES:
        counts = data.datasets[name].nonempty_counts()
        observed = [int((counts == k).sum()) for k in range(N_GRADERS + 1)]
        assert observed == expected[name], name