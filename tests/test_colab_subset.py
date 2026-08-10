"""Tests for the Colab demo subset builder, on a synthetic stand-in dataset.

Mac-runnable, no GPU, no ``lidc.npz``. The fixture below builds a small dataset with the
same shapes, dtypes and grader layout as the real one and a matching ``split.json``, which
is enough to exercise everything that can go wrong in this script: test contamination,
index re-basing, stratification, and format compatibility with the real loader.

The one thing a synthetic fixture makes *easy* rather than harder is the re-basing check.
Every synthetic patch is a constant plane whose value encodes its row, so "does subset row
``r`` hold the patch that full-dataset index ``source_index[r]`` holds" is answerable **by
content**, not by shape -- which is the only form of that check worth having, since a
re-basing bug produces perfectly well-shaped arrays holding the wrong patches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probunet.data.colab_subset import (
    DEMO_BUCKETS,
    MEASURED_BYTES_PER_PATCH,
    PROVENANCE_KEY,
    PROVENANCE_REQUIRED_KEYS,
    SIZE_TARGET_BYTES,
    assert_no_test_indices,
    build_demo_split_document,
    buckets_in_pool,
    draw_stratified,
    plan_per_bucket,
    read_provenance,
    rebase,
    select_demo_patches,
    sha256_file,
    size_arithmetic,
    write_demo_split,
    write_demo_subset,
)
from probunet.data.lidc import DataConfig, LidcArrays, build_data
from probunet.data.splits import SCHEMA_VERSION, load_split

PATCH_SIZE = 16
"""Small enough to keep the fixture instant; the loader does not care about H and W."""

N_SERIES = 30
PATCHES_PER_SERIES = 4
N_PATCHES = N_SERIES * PATCHES_PER_SERIES  # 120


# ---------------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------------


def _bucket_of(index: int) -> int:
    """Cycle patches through buckets 1..4 so every stratum is populated."""
    return 1 + index % 4


def build_synthetic_arrays() -> dict[str, np.ndarray]:
    """Build a dataset with the real format and a content fingerprint per patch.

    Returns:
        Arrays in ``lidc.npz`` layout. ``images[i]`` is a constant plane of value
        ``i / N_PATCHES``, so a patch's identity is readable from its pixels and a
        re-basing error cannot hide behind a correct shape.
    """
    images = np.stack(
        [
            np.full((PATCH_SIZE, PATCH_SIZE), index / N_PATCHES, dtype=np.float32)
            for index in range(N_PATCHES)
        ]
    )
    masks = np.zeros((N_PATCHES, 4, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    for index in range(N_PATCHES):
        for grader in range(_bucket_of(index)):
            masks[index, grader, 2:6, 2:6] = 1
    series_uid = np.array(
        [f"series-{index // PATCHES_PER_SERIES:03d}" for index in range(N_PATCHES)],
        dtype="<U64",
    )
    return {"images": images, "masks": masks, "series_uid": series_uid}


def write_synthetic_dataset(directory: Path) -> tuple[Path, Path]:
    """Write a synthetic ``lidc.npz`` and a matching ``split.json``.

    The split is series-grouped, like the real one: series 0-17 train, 18-23 val, 24-29
    test. Every split therefore carries all four buckets, and no series spans two splits.

    Args:
        directory: Where to write the pair.

    Returns:
        ``(npz_path, split_path)``.
    """
    arrays = build_synthetic_arrays()
    npz_path = directory / "lidc.npz"
    np.savez_compressed(npz_path, **arrays)

    boundaries = {"train": (0, 18), "val": (18, 24), "test": (24, 30)}
    indices, series = {}, {}
    for name, (low, high) in boundaries.items():
        rows = [
            index
            for index in range(N_PATCHES)
            if low <= index // PATCHES_PER_SERIES < high
        ]
        indices[name] = rows
        series[name] = sorted({str(arrays["series_uid"][row]) for row in rows})

    split_path = directory / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": 1806,
                "indices": indices,
                "series_uid": series,
                "ratios_requested": {"train": 0.6, "val": 0.2, "test": 0.2},
                "achieved": {
                    name: {"n_patches": len(rows)} for name, rows in indices.items()
                },
                "source_npz": {
                    "path": str(npz_path),
                    "n_patches": N_PATCHES,
                    "sha256": sha256_file(npz_path),
                },
            },
            indent=2,
        )
    )
    return npz_path, split_path


@pytest.fixture()
def synthetic(tmp_path: Path) -> dict[str, object]:
    """A synthetic dataset, its split, and the loaded pieces the tests need."""
    npz_path, split_path = write_synthetic_dataset(tmp_path)
    arrays = LidcArrays.load(npz_path)
    split = load_split(split_path, expected_n_patches=len(arrays))
    return {
        "dir": tmp_path,
        "npz_path": npz_path,
        "split_path": split_path,
        "arrays": arrays,
        "split": split,
        "counts": arrays.nonempty_counts(),
    }


def make_subset(
    synthetic: dict[str, object], train_patches: int = 16, val_patches: int = 8, seed: int = 2018
) -> dict[str, object]:
    """Run the whole selection and write both output files.

    Args:
        synthetic: The fixture.
        train_patches: Demo train patches.
        val_patches: Demo val patches.
        seed: Draw seed.

    Returns:
        The selection, the provenance, and the two written paths.
    """
    arrays, split = synthetic["arrays"], synthetic["split"]
    selection, exclusion = select_demo_patches(
        nonempty_counts=synthetic["counts"],
        train_pool=split.indices["train"],
        val_pool=split.indices["val"],
        test_indices=split.indices["test"],
        train_patches=train_patches,
        val_patches=val_patches,
        seed=seed,
    )
    rows = selection.rows
    provenance = {
        "schema_version": 1,
        "generated_at": "2026-08-10T00:00:00+00:00",
        "git_revision": "abc1234",
        "source_npz_path": str(synthetic["npz_path"]),
        "source_npz_sha256": sha256_file(synthetic["npz_path"]),
        "source_n_patches": len(arrays),
        "source_split_path": str(synthetic["split_path"]),
        "source_split_seed": int(split.seed),
        "seed": seed,
        "n_patches": len(selection),
        "original_indices": {
            "all": [int(i) for i in rows],
            "train": [int(i) for i in selection.train_global],
            "val": [int(i) for i in selection.val_global],
        },
        "per_bucket": {
            pool: {str(b): c for b, c in counts.items()}
            for pool, counts in selection.per_bucket.items()
        },
        "available_per_bucket": {
            pool: {str(b): c for b, c in counts.items()}
            for pool, counts in selection.available_per_bucket.items()
        },
        "test_split_exclusion": exclusion,
        "size": size_arithmetic(train_patches, val_patches),
        "demo_split_path": str(synthetic["dir"] / "colab_demo_split.json"),
    }
    out_npz = synthetic["dir"] / "lidc_colab_demo.npz"
    size = write_demo_subset(
        path=out_npz,
        images=arrays.images[rows],
        masks=arrays.masks[rows],
        series_uid=arrays.series_uid[rows],
        source_index=rows.astype(np.int64),
        provenance=provenance,
    )
    out_split = synthetic["dir"] / "colab_demo_split.json"
    write_demo_split(
        out_split,
        build_demo_split_document(
            selection=selection,
            series_uid=arrays.series_uid,
            seed=seed,
            subset_npz_path=out_npz,
            subset_npz_sha256=sha256_file(out_npz),
            source_split_path=synthetic["split_path"],
            source_split_seed=int(split.seed),
            exclusion=exclusion,
        ),
    )
    return {
        "selection": selection,
        "exclusion": exclusion,
        "provenance": provenance,
        "npz": out_npz,
        "split": out_split,
        "size": size,
    }


# ---------------------------------------------------------------------------------
# Test contamination
# ---------------------------------------------------------------------------------


def test_test_exclusion_fires_when_a_test_index_is_injected() -> None:
    """A single test index among the selection must raise, naming it."""
    selected = np.array([1, 2, 3, 99, 4], dtype=np.int64)
    test_indices = np.array([90, 99, 110], dtype=np.int64)

    with pytest.raises(ValueError) as raised:
        assert_no_test_indices(selected, test_indices)

    message = str(raised.value)
    assert "TEST CONTAMINATION" in message
    assert "99" in message
    assert "do not relax this check" in message


def test_test_exclusion_passes_and_records_the_evidence() -> None:
    """A clean selection returns the record the provenance block carries."""
    record = assert_no_test_indices(np.arange(10), np.arange(50, 60))
    assert record["verified"] is True
    assert record["intersection_size"] == 0
    assert record["n_selected"] == 10
    assert "FINAL selected indices" in record["statement"]
    assert "intersect1d" in record["assertion"]


def test_selection_rejects_a_pool_contaminated_with_a_test_index(
    synthetic: dict[str, object],
) -> None:
    """The check runs on the FINAL selection, so a poisoned pool cannot slip through.

    The train pool here wrongly contains one test patch, and the request is sized so that
    patch must be drawn. A pool-level check would have to reason about what *might* be
    picked; the final-selection check simply catches it.
    """
    split = synthetic["split"]
    test_index = int(split.indices["test"][0])
    # One bucket's worth of train patches, all of bucket 1, plus the intruder -- which is
    # in the same bucket, so the draw has to take it.
    counts = synthetic["counts"]
    bucket = int(counts[test_index])
    same_bucket_train = split.indices["train"][counts[split.indices["train"]] == bucket]
    poisoned = np.concatenate([same_bucket_train[:3], [test_index]])

    with pytest.raises(ValueError, match="TEST CONTAMINATION"):
        select_demo_patches(
            nonempty_counts=counts,
            train_pool=poisoned,
            val_pool=split.indices["val"],
            test_indices=split.indices["test"],
            # Ask for exactly the four available in that bucket, and nothing elsewhere.
            train_patches=4,
            val_patches=0,
            seed=0,
            buckets=(bucket,),
        )


def test_emitted_demo_split_has_an_empty_test_list(synthetic: dict[str, object]) -> None:
    """The demo split's test list is empty by construction, and load_split accepts that."""
    built = make_subset(synthetic)
    document = json.loads(Path(built["split"]).read_text())
    assert document["indices"]["test"] == []
    assert document["series_uid"]["test"] == []

    arrays = LidcArrays.load(built["npz"])
    loaded = load_split(built["split"], expected_n_patches=len(arrays))
    assert loaded.indices["test"].size == 0


# ---------------------------------------------------------------------------------
# Index re-basing
# ---------------------------------------------------------------------------------


def test_rebased_indices_address_the_correct_subset_rows_by_content(
    synthetic: dict[str, object],
) -> None:
    """The core re-basing check, verified by PIXELS rather than by shape.

    A re-basing bug produces perfectly well-shaped arrays holding the wrong patches, so a
    shape assertion would pass on it. Each synthetic patch is a constant plane encoding its
    full-dataset row, which makes "is this the patch it claims to be" directly checkable.
    """
    built = make_subset(synthetic)
    full = synthetic["arrays"]
    subset = LidcArrays.load(built["npz"])
    loaded = load_split(built["split"], expected_n_patches=len(subset))
    selection = built["selection"]

    for name, originals in (
        ("train", selection.train_global),
        ("val", selection.val_global),
    ):
        rebased = loaded.indices[name]
        assert rebased.size == originals.size
        for position, subset_row in enumerate(rebased):
            original = int(originals[position])
            # 1. The subset row claims to come from the right place...
            assert int(subset.source_index[subset_row]) == original, name
            # 2. ...and it actually holds that patch's pixels.
            assert np.array_equal(
                subset.images[subset_row], full.images[original]
            ), f"{name} row {subset_row} does not hold full-dataset patch {original}"
            assert np.array_equal(subset.masks[subset_row], full.masks[original])


def test_rebased_indices_partition_the_subset(synthetic: dict[str, object]) -> None:
    """Every subset row belongs to exactly one demo split -- load_split requires it."""
    built = make_subset(synthetic)
    selection = built["selection"]
    covered = np.sort(np.concatenate([selection.train_rows, selection.val_rows]))
    assert np.array_equal(covered, np.arange(len(selection)))


def test_rebase_rejects_overlapping_or_duplicated_selections() -> None:
    """A patch in both splits, or listed twice, is refused rather than silently kept."""
    with pytest.raises(ValueError, match="overlap"):
        rebase(np.array([1, 2, 3]), np.array([3, 4]))
    with pytest.raises(ValueError, match="duplicates"):
        rebase(np.array([1, 1, 2]), np.array([3, 4]))


def test_the_main_split_is_not_valid_for_the_subset(synthetic: dict[str, object]) -> None:
    """The trap this whole design exists around, made explicit.

    The main split covers 120 patches; the subset has 24. Loading one against the other
    must fail, which is what makes a separate demo split necessary rather than tidy.
    """
    built = make_subset(synthetic)
    subset = LidcArrays.load(built["npz"])
    with pytest.raises(ValueError, match="stale"):
        load_split(synthetic["split_path"], expected_n_patches=len(subset))


# ---------------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------------


def test_plan_per_bucket_spreads_evenly_and_places_the_remainder() -> None:
    """The plan is deterministic, including how an indivisible total is distributed."""
    assert plan_per_bucket(16) == {1: 4, 2: 4, 3: 4, 4: 4}
    assert plan_per_bucket(18) == {1: 5, 2: 5, 3: 4, 4: 4}
    assert plan_per_bucket(0) == {1: 0, 2: 0, 3: 0, 4: 0}
    assert sum(plan_per_bucket(255).values()) == 255
    with pytest.raises(ValueError, match="non-negative"):
        plan_per_bucket(-1)


def test_stratification_produces_the_requested_per_bucket_counts(
    synthetic: dict[str, object],
) -> None:
    """The demo is not accidentally all easy 1-grader patches."""
    built = make_subset(synthetic, train_patches=16, val_patches=8)
    selection = built["selection"]

    assert selection.per_bucket["train"] == {1: 4, 2: 4, 3: 4, 4: 4}
    assert selection.per_bucket["val"] == {1: 2, 2: 2, 3: 2, 4: 2}

    # And the counts are true of the written file, not merely of the bookkeeping.
    subset = LidcArrays.load(built["npz"])
    counts = subset.nonempty_counts()
    for bucket in DEMO_BUCKETS:
        assert int((counts == bucket).sum()) == 6


def test_a_short_bucket_fails_loudly_instead_of_being_topped_up(
    synthetic: dict[str, object],
) -> None:
    """The reason this does not reuse diagnostics.stratified_indices.

    That function tops up from other buckets when one runs short, which would silently
    skew the demo. Here it is an error, and the message says which bucket and by how much.
    """
    split, counts = synthetic["split"], synthetic["counts"]
    with pytest.raises(ValueError) as raised:
        select_demo_patches(
            nonempty_counts=counts,
            train_pool=split.indices["train"],
            val_pool=split.indices["val"],
            test_indices=split.indices["test"],
            # 18 patches per bucket exist in the synthetic train split; ask for 25.
            train_patches=100,
            val_patches=8,
            seed=2018,
        )
    message = str(raised.value)
    assert "train pool cannot fill the stratified plan" in message
    assert "wanted 25, only 18 available" in message
    assert "bucket-1-heavy" in message


def test_draw_stratified_is_deterministic_under_the_seed() -> None:
    """The same seed reproduces the same subset exactly."""
    available = {bucket: np.arange(bucket * 100, bucket * 100 + 50) for bucket in DEMO_BUCKETS}
    wanted = plan_per_bucket(8)
    first = draw_stratified(available, wanted, np.random.default_rng(7), "train")
    second = draw_stratified(available, wanted, np.random.default_rng(7), "train")
    other = draw_stratified(available, wanted, np.random.default_rng(8), "train")

    for bucket in DEMO_BUCKETS:
        assert np.array_equal(first[bucket], second[bucket])
    assert any(not np.array_equal(first[b], other[b]) for b in DEMO_BUCKETS)


def test_buckets_in_pool_groups_only_the_pool(synthetic: dict[str, object]) -> None:
    """Grouping is restricted to the pool it was handed, never the whole dataset."""
    split = synthetic["split"]
    grouped = buckets_in_pool(synthetic["counts"], split.indices["val"])
    everything = np.concatenate([grouped[b] for b in DEMO_BUCKETS])
    assert np.array_equal(np.sort(everything), np.sort(split.indices["val"]))


# ---------------------------------------------------------------------------------
# Format compatibility with the real loader
# ---------------------------------------------------------------------------------


def test_emitted_npz_loads_through_the_real_dataset_class(
    synthetic: dict[str, object],
) -> None:
    """The subset must work with ZERO loader changes: same keys, dtypes, layout."""
    built = make_subset(synthetic, train_patches=16, val_patches=8)

    config = DataConfig(
        npz_path=built["npz"], split_path=built["split"], batch_size=4, num_workers=0
    )
    data = build_data(config)

    assert len(data.datasets["train"]) == 16
    assert len(data.datasets["val"]) == 8
    assert len(data.datasets["test"]) == 0

    train_batch = next(iter(data.loaders["train"]))
    assert train_batch["image"].shape == (4, 1, PATCH_SIZE, PATCH_SIZE)
    assert train_batch["image"].dtype.is_floating_point
    assert train_batch["mask"].shape == (4, PATCH_SIZE, PATCH_SIZE)
    assert str(train_batch["mask"].dtype) == "torch.int64"

    val_batch = next(iter(data.loaders["val"]))
    assert val_batch["masks"].shape == (4, 4, PATCH_SIZE, PATCH_SIZE)
    assert str(val_batch["masks"].dtype) == "torch.uint8"

    # An empty test loader yields nothing rather than raising -- which is what lets the
    # demo split have no test entry at all.
    assert list(data.loaders["test"]) == []


def test_emitted_npz_presents_as_a_subset_export(synthetic: dict[str, object]) -> None:
    """source_index is what makes the file auditable back to the full dataset."""
    built = make_subset(synthetic)
    subset = LidcArrays.load(built["npz"])
    assert subset.is_subset
    assert subset.source_index.dtype == np.int64
    assert np.array_equal(subset.source_index, np.sort(subset.source_index))
    # resolve_indices round-trips: full-dataset indices map back to their subset rows.
    assert np.array_equal(
        subset.resolve_indices(subset.source_index), np.arange(len(subset))
    )


def test_dtypes_match_the_full_dataset_and_downcasting_is_refused(
    synthetic: dict[str, object], tmp_path: Path
) -> None:
    """Nothing is downcast to save space; a float16 image array is an error, not a saving."""
    arrays = synthetic["arrays"]
    rows = np.arange(8, dtype=np.int64)
    provenance = {key: "x" for key in PROVENANCE_REQUIRED_KEYS}

    with pytest.raises(ValueError, match="do not downcast"):
        write_demo_subset(
            path=tmp_path / "bad.npz",
            images=arrays.images[rows].astype(np.float16),
            masks=arrays.masks[rows],
            series_uid=arrays.series_uid[rows],
            source_index=rows,
            provenance=provenance,
        )

    subset_path = tmp_path / "good.npz"
    write_demo_subset(
        path=subset_path,
        images=arrays.images[rows],
        masks=arrays.masks[rows],
        series_uid=arrays.series_uid[rows],
        source_index=rows,
        provenance=provenance,
    )
    with np.load(subset_path, allow_pickle=False) as handle:
        assert handle["images"].dtype == arrays.images.dtype
        assert handle["masks"].dtype == arrays.masks.dtype
        assert handle["masks"].shape[1:] == arrays.masks.shape[1:]


def test_write_refuses_an_incomplete_provenance_block(
    synthetic: dict[str, object], tmp_path: Path
) -> None:
    """The provenance schema is enforced where it can still be fixed."""
    arrays = synthetic["arrays"]
    rows = np.arange(4, dtype=np.int64)
    incomplete = {key: "x" for key in PROVENANCE_REQUIRED_KEYS if key != "seed"}
    with pytest.raises(ValueError, match="missing"):
        write_demo_subset(
            path=tmp_path / "x.npz",
            images=arrays.images[rows],
            masks=arrays.masks[rows],
            series_uid=arrays.series_uid[rows],
            source_index=rows,
            provenance=incomplete,
        )


# ---------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------


def test_provenance_round_trips_through_the_npz(synthetic: dict[str, object]) -> None:
    """Everything the file must record survives the write and reads back unchanged."""
    built = make_subset(synthetic)
    recovered = read_provenance(built["npz"])

    assert recovered == built["provenance"]
    for key in PROVENANCE_REQUIRED_KEYS:
        assert key in recovered

    assert recovered["source_npz_sha256"] == sha256_file(synthetic["npz_path"])
    assert recovered["source_n_patches"] == N_PATCHES
    assert recovered["source_split_seed"] == 1806, "the SPLIT seed, not the draw seed"
    assert recovered["seed"] == 2018, "the draw seed, recorded separately"
    assert recovered["original_indices"]["all"] == [
        int(i) for i in built["selection"].rows
    ]
    assert recovered["per_bucket"]["train"] == {"1": 4, "2": 4, "3": 4, "4": 4}
    assert recovered["test_split_exclusion"]["verified"] is True
    assert recovered["test_split_exclusion"]["intersection_size"] == 0

    # The provenance key is additive: it cannot change how the loader reads the file.
    with np.load(built["npz"], allow_pickle=False) as handle:
        assert PROVENANCE_KEY in handle.files
    assert LidcArrays.load(built["npz"]).is_subset


def test_read_provenance_fails_loudly_on_a_foreign_file(tmp_path: Path) -> None:
    """A subset without provenance is refused rather than half-trusted."""
    foreign = tmp_path / "foreign.npz"
    np.savez_compressed(foreign, images=np.zeros((2, 4, 4), dtype=np.float32))
    with pytest.raises(KeyError, match=PROVENANCE_KEY):
        read_provenance(foreign)
    with pytest.raises(FileNotFoundError):
        read_provenance(tmp_path / "absent.npz")


def test_demo_split_document_records_both_index_spaces(
    synthetic: dict[str, object],
) -> None:
    """The split file states which array its indices address, and carries the mapping."""
    built = make_subset(synthetic)
    document = json.loads(Path(built["split"]).read_text())

    assert "NOT data/processed/lidc.npz" in document["note"]
    assert document["source_split"]["seed"] == 1806
    assert document["seed"] == 2018
    assert document["indices"]["train"] == [int(i) for i in built["selection"].train_rows]
    assert document["original_indices"]["train"] == [
        int(i) for i in built["selection"].train_global
    ]
    assert document["test_split_exclusion"]["verified"] is True
    assert set(document["series_uid"]["train"]) & set(document["series_uid"]["val"]) == set()


# ---------------------------------------------------------------------------------
# Size budget
# ---------------------------------------------------------------------------------


def test_the_default_patch_count_fits_the_stated_budget() -> None:
    """The default is chosen from measured arithmetic, not guessed.

    30,349 bytes/patch measured on data/processed/lidc_subset.npz (2,154,808 B / 71
    patches). 256 + 32 = 288 patches project to 8.34 MiB, inside the 10 MiB target.
    """
    projection = size_arithmetic()
    assert projection["n_patches"] == 288
    assert projection["bytes_per_patch_measured"] == MEASURED_BYTES_PER_PATCH
    assert projection["projected_bytes"] == 288 * MEASURED_BYTES_PER_PATCH
    assert projection["within_target"] is True
    assert projection["projected_bytes"] < SIZE_TARGET_BYTES
    assert projection["max_patches_at_target"] == 345
    assert projection["max_patches_at_hard_stop"] == 691