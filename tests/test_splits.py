"""Tests for the grouped, seeded, ambiguity-stratified train/val/test split.

The properties that matter are: no CT series is shared between two splits, the
splits partition every patch exactly once, and the whole thing is reproducible from
the seed. Everything else (overall ratio accuracy, per-stratum balance) is a quality
check with tolerances.

Most tests run on small synthetic series arrays so they stay fast. The tests that
need the real dataset skip cleanly when ``data/processed/lidc.npz`` is absent, so
the suite passes on a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probunet.data.splits import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    N_GRADERS,
    SPLIT_NAMES,
    assign_series_to_splits,
    generate_split,
    indices_for_groups,
    load_split,
    mean_pairwise_iou_nonempty,
    nonempty_masks_per_patch,
    summarize_shape_agreement,
    summarize_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_NPZ = REPO_ROOT / "data" / "processed" / "lidc.npz"


def make_series(sizes: list[int]) -> np.ndarray:
    """Build a per-patch series array from a list of series sizes.

    Args:
        sizes: Patch count for each synthetic series.

    Returns:
        Array of length ``sum(sizes)`` of series UID strings.
    """
    uids = []
    for index, size in enumerate(sizes):
        uids.extend([f"series-{index:04d}"] * size)
    return np.array(uids, dtype=np.str_)


def make_strata(n_patches: int, seed: int = 0) -> np.ndarray:
    """Build per-patch stratum labels in ``1..4`` (LIDC never has a 0 stratum).

    Args:
        n_patches: Number of patches.
        seed: Seed for the label draw.

    Returns:
        Integer array of stratum labels.
    """
    return np.random.default_rng(seed).integers(1, N_GRADERS + 1, size=n_patches)


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """A LIDC-like synthetic dataset: uneven series sizes plus stratum labels."""
    sizes = np.random.default_rng(0).integers(1, 109, size=200).tolist()
    series = make_series(sizes)
    return series, make_strata(series.size, seed=1)


def test_no_series_appears_in_two_splits(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Each series UID is assigned to exactly one split."""
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    seen: dict[str, str] = {}
    for name, uids in groups.items():
        for uid in uids:
            assert uid not in seen, f"{uid} in both {seen.get(uid)} and {name}"
            seen[uid] = name
    assert set(seen) == set(np.unique(series).tolist())


def test_indices_partition_the_full_index(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """The union of split indices is exactly 0..N-1, with no duplicates."""
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)
    combined = np.concatenate([indices[name] for name in SPLIT_NAMES])
    assert combined.size == series.size
    assert np.array_equal(np.sort(combined), np.arange(series.size))


def test_patches_follow_their_series(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Every patch lands in the split its series was assigned to."""
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)
    for name in SPLIT_NAMES:
        assigned = set(groups[name])
        for row in indices[name]:
            assert str(series[row]) in assigned


def test_reproducible_from_seed(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """The same seed reproduces the split exactly; a different seed changes it."""
    series, strata = dataset
    first = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    second = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    assert first == second

    other = assign_series_to_splits(series, strata, seed=DEFAULT_SEED + 1)
    assert other != first, "different seeds produced an identical assignment"


def test_no_dependence_on_input_patch_order(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """The assignment depends on series content, not on row order."""
    series, strata = dataset
    permutation = np.random.default_rng(123).permutation(series.size)
    shuffled = assign_series_to_splits(series[permutation], strata[permutation], seed=DEFAULT_SEED)
    assert shuffled == assign_series_to_splits(series, strata, seed=DEFAULT_SEED)


def test_overall_ratios_close_to_target(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Achieved patch ratios are within 2 percentage points of the target."""
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)
    for name in SPLIT_NAMES:
        achieved = indices[name].size / series.size
        assert abs(achieved - DEFAULT_RATIOS[name]) < 0.02, f"{name}: {achieved:.4f}"


def test_no_split_is_starved(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Every split receives patches and series.

    Regression test: an earlier stratified objective minimized squared distance to
    the final per-stratum target, which -- with all splits empty at the start --
    always favoured the split with the smallest target and left train with zero
    patches.
    """
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)
    for name in SPLIT_NAMES:
        assert len(groups[name]) > 0, f"{name} received no series"
        assert indices[name].size > 0, f"{name} received no patches"


def test_largest_target_receives_most_patches(
    dataset: tuple[np.ndarray, np.ndarray]
) -> None:
    """Split sizes are ordered like their target ratios."""
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)
    assert indices["train"].size > indices["val"].size
    assert indices["train"].size > indices["test"].size


def test_each_stratum_is_balanced(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Every stratum is split near 60/20/20, not just the totals.

    This is the property the stratified objective exists for: a split that hits the
    overall ratio while concentrating the ambiguous cases in one split would pass
    the previous test and fail this one.
    """
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)
    for label in np.unique(strata):
        total = int((strata == label).sum())
        for name in SPLIT_NAMES:
            rows = indices[name]
            share = int((strata[rows] == label).sum()) / total
            assert abs(share - DEFAULT_RATIOS[name]) < 0.05, (
                f"stratum {label}, {name}: {share:.4f}"
            )


def test_stratification_beats_unstratified_balance(
    dataset: tuple[np.ndarray, np.ndarray]
) -> None:
    """Stratifying reduces the spread in mean stratum label across splits.

    Compared against a deliberately unstratified baseline: a single constant
    stratum, which reduces the objective to total patch count only.
    """
    series, strata = dataset
    constant = np.zeros_like(strata)

    def spread(labels: np.ndarray) -> float:
        groups = assign_series_to_splits(series, labels, seed=DEFAULT_SEED)
        indices = indices_for_groups(series, groups)
        means = [float(strata[indices[name]].mean()) for name in SPLIT_NAMES]
        return max(means) - min(means)

    assert spread(strata) < spread(constant)


def test_ratios_are_validated(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Malformed ratio dictionaries are rejected rather than silently normalized."""
    series, strata = dataset
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_series_to_splits(series, strata, ratios={"train": 0.5, "val": 0.2, "test": 0.2})
    with pytest.raises(ValueError, match="exactly"):
        assign_series_to_splits(series, strata, ratios={"train": 0.6, "val": 0.4})
    with pytest.raises(ValueError, match="non-negative"):
        assign_series_to_splits(
            series, strata, ratios={"train": 1.2, "val": -0.2, "test": 0.0}
        )


def test_strata_are_validated(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Mismatched, non-integer or negative strata are rejected."""
    series, strata = dataset
    with pytest.raises(ValueError, match="strata shape"):
        assign_series_to_splits(series, strata[:-1])
    with pytest.raises(ValueError, match="integer labels"):
        assign_series_to_splits(series, strata.astype(np.float64))
    with pytest.raises(ValueError, match="non-negative"):
        assign_series_to_splits(series, np.full_like(strata, -1))


def test_empty_series_rejected() -> None:
    """An empty dataset is an error, not an empty split."""
    with pytest.raises(ValueError, match="empty"):
        assign_series_to_splits(np.array([], dtype=np.str_), np.array([], dtype=np.int64))


def test_overlapping_groups_rejected(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """A hand-corrupted assignment with a shared series is caught."""
    series, strata = dataset
    groups = assign_series_to_splits(series, strata, seed=DEFAULT_SEED)
    groups["val"] = list(groups["val"]) + [groups["train"][0]]
    with pytest.raises(ValueError, match="assigned to both"):
        indices_for_groups(series, groups)


def test_nonempty_masks_per_patch_counts_graders() -> None:
    """Non-empty mask counting is per grader slot and tolerates all-empty masks."""
    masks = np.zeros((3, N_GRADERS, 4, 4), dtype=np.uint8)
    masks[0, 0, 0, 0] = 1
    masks[1, :, 1, 1] = 1
    assert nonempty_masks_per_patch(masks).tolist() == [1, 4, 0]

    with pytest.raises(ValueError, match="expected masks"):
        nonempty_masks_per_patch(np.zeros((3, 2, 4, 4), dtype=np.uint8))


def test_mean_pairwise_iou_nonempty() -> None:
    """Mean pairwise IoU uses only non-empty pairs and needs at least two."""
    masks = np.zeros((4, N_GRADERS, 2, 2), dtype=np.uint8)
    # Patch 0: two non-empty masks overlapping in 1 of 2 pixels -> IoU 0.5.
    masks[0, 0] = [[1, 0], [0, 0]]
    masks[0, 1] = [[1, 1], [0, 0]]
    # Patch 1: three identical non-empty masks -> IoU 1.0 for all three pairs.
    masks[1, 0] = masks[1, 1] = masks[1, 2] = [[1, 1], [1, 1]]
    # Patch 2: a single non-empty mask -> no pair, undefined.
    masks[2, 3] = [[1, 0], [0, 0]]
    # Patch 3: all empty -> undefined.

    values, valid = mean_pairwise_iou_nonempty(masks)
    assert valid.tolist() == [True, True, False, False]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert np.isnan(values[2]) and np.isnan(values[3])

    with pytest.raises(ValueError, match="expected masks"):
        mean_pairwise_iou_nonempty(np.zeros((2, 3, 2, 2), dtype=np.uint8))


def test_mean_pairwise_iou_disjoint_masks_is_zero() -> None:
    """Two non-overlapping non-empty masks score 0, not NaN."""
    masks = np.zeros((1, N_GRADERS, 2, 2), dtype=np.uint8)
    masks[0, 0] = [[1, 0], [0, 0]]
    masks[0, 1] = [[0, 0], [0, 1]]
    values, valid = mean_pairwise_iou_nonempty(masks)
    assert valid[0]
    assert values[0] == pytest.approx(0.0)


def test_summarize_shape_agreement_excludes_ineligible() -> None:
    """Only patches with >= 2 non-empty masks enter the shape-agreement summary."""
    series = make_series([2, 2])
    groups = {"train": ["series-0000"], "val": ["series-0001"], "test": []}
    indices = indices_for_groups(series, groups)
    values = np.array([0.5, np.nan, 0.25, 0.75])
    valid = np.array([True, False, True, True])

    summary = summarize_shape_agreement(indices, values, valid)
    assert summary["stratified_on"] is False
    assert summary["train"]["n_patches_evaluated"] == 1
    assert summary["train"]["mean_iou"] == pytest.approx(0.5)
    assert summary["val"]["mean_iou"] == pytest.approx(0.5)
    assert summary["test"]["n_patches_evaluated"] == 0
    assert summary["test"]["mean_iou"] is None


def test_summarize_split_reports_ambiguity() -> None:
    """The summary reports per-split ambiguity, not just sizes."""
    series = make_series([2, 2, 2])
    groups = {"train": ["series-0000"], "val": ["series-0001"], "test": ["series-0002"]}
    indices = indices_for_groups(series, groups)
    nonempty = np.array([1, 1, 4, 4, 2, 2], dtype=np.int64)
    summary = summarize_split(indices, groups, nonempty, DEFAULT_RATIOS)
    assert summary["train"]["mean_nonempty_masks"] == 1.0
    assert summary["val"]["nonempty_masks_per_patch"]["4"] == 2
    assert summary["totals"]["n_patches"] == 6


def write_synthetic_npz(path: Path, sizes: list[int], seed: int = 0) -> np.ndarray:
    """Write a minimal npz with the fields the split generator reads.

    Args:
        path: Destination ``.npz``.
        sizes: Patch count per synthetic series.
        seed: Seed for the random mask contents.

    Returns:
        The per-patch series array that was written.
    """
    series = make_series(sizes)
    rng = np.random.default_rng(seed)
    masks = np.zeros((series.size, N_GRADERS, 8, 8), dtype=np.uint8)
    for row in range(series.size):
        for slot in range(rng.integers(1, N_GRADERS + 1)):
            masks[row, slot, : rng.integers(1, 8), : rng.integers(1, 8)] = 1
    np.savez_compressed(path, series_uid=series, masks=masks)
    return series


def test_generate_and_load_round_trip(tmp_path: Path) -> None:
    """A generated split file loads back with matching indices and metadata."""
    npz = tmp_path / "mini.npz"
    series = write_synthetic_npz(npz, [7, 5, 3, 11, 2, 9, 4, 6, 8, 1])
    out = tmp_path / "split.json"

    document = generate_split(npz_path=npz, out_path=out, seed=DEFAULT_SEED)
    loaded = load_split(out, expected_n_patches=series.size, verify_source=npz)

    assert loaded.seed == DEFAULT_SEED
    assert loaded.n_patches == series.size
    for name in SPLIT_NAMES:
        assert loaded.indices[name].tolist() == document["indices"][name]
        assert loaded.series_uid[name] == document["series_uid"][name]
    assert loaded["train"].size == len(document["indices"]["train"])


def test_generated_document_records_stratification(tmp_path: Path) -> None:
    """The split file documents the seed and the stratification scheme."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [6, 6, 6, 6])
    document = generate_split(npz_path=npz, out_path=tmp_path / "split.json")

    assert document["seed"] == DEFAULT_SEED
    assert document["stratification"]["scheme"] == "nonempty_mask_count"
    assert document["stratification"]["n_strata"] >= 1
    assert sum(document["stratification"]["stratum_totals"].values()) == 24
    assert document["secondary_diagnostic_shape_agreement"]["stratified_on"] is False


def test_generate_refuses_to_overwrite(tmp_path: Path) -> None:
    """The split is generated once; regenerating requires an explicit override."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [4, 4, 4, 4])
    out = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=out)

    with pytest.raises(FileExistsError, match="generated once"):
        generate_split(npz_path=npz, out_path=out)

    generate_split(npz_path=npz, out_path=out, overwrite=True)


def test_generate_is_byte_identical_for_same_seed(tmp_path: Path) -> None:
    """Regenerating with the same seed reproduces the file content exactly."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [7, 5, 3, 11, 2, 9])
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    generate_split(npz_path=npz, out_path=first, seed=99)
    generate_split(npz_path=npz, out_path=second, seed=99)
    assert first.read_text() == second.read_text()


def test_load_split_missing_file_is_an_error(tmp_path: Path) -> None:
    """A missing split file never triggers silent regeneration."""
    with pytest.raises(FileNotFoundError, match="Generate it once"):
        load_split(tmp_path / "absent.json")


def test_load_split_detects_overlap(tmp_path: Path) -> None:
    """A tampered file whose splits share an index is rejected."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [4, 4, 4, 4])
    out = tmp_path / "split.json"
    document = generate_split(npz_path=npz, out_path=out)

    document["indices"]["val"] = list(document["indices"]["val"]) + [
        document["indices"]["train"][0]
    ]
    out.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="cover|duplicate"):
        load_split(out)


def test_load_split_detects_incomplete_coverage(tmp_path: Path) -> None:
    """A tampered file that drops a patch is rejected."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [4, 4, 4, 4])
    out = tmp_path / "split.json"
    document = generate_split(npz_path=npz, out_path=out)

    document["indices"]["train"] = list(document["indices"]["train"])[:-1]
    out.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="cover"):
        load_split(out)


def test_load_split_rejects_old_schema(tmp_path: Path) -> None:
    """A split file from an earlier schema version is rejected, not guessed at."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [4, 4, 4, 4])
    out = tmp_path / "split.json"
    document = generate_split(npz_path=npz, out_path=out)

    document["schema_version"] = 1
    out.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="schema_version"):
        load_split(out)


def test_load_split_detects_stale_source(tmp_path: Path) -> None:
    """A split generated against a different npz is rejected on request."""
    npz = tmp_path / "mini.npz"
    write_synthetic_npz(npz, [4, 4, 4, 4])
    out = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=out)

    write_synthetic_npz(npz, [4, 4, 4, 4], seed=7)  # same shape, different content
    with pytest.raises(ValueError, match="different"):
        load_split(out, verify_source=npz)


@pytest.mark.skipif(not REAL_NPZ.exists(), reason="converted dataset not present")
def test_real_dataset_split_properties() -> None:
    """On the real dataset: series are disjoint, all patches covered, strata balanced."""
    with np.load(REAL_NPZ, allow_pickle=False) as handle:
        series = handle["series_uid"]
        nonempty = nonempty_masks_per_patch(handle["masks"])

    groups = assign_series_to_splits(series, nonempty, seed=DEFAULT_SEED)
    indices = indices_for_groups(series, groups)

    combined = np.concatenate([indices[name] for name in SPLIT_NAMES])
    assert np.array_equal(np.sort(combined), np.arange(series.size))

    assert not set(groups["train"]) & set(groups["val"])
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["val"]) & set(groups["test"])

    # No patch in one split may share a series with a patch in another.
    for name in SPLIT_NAMES:
        others = set()
        for other in SPLIT_NAMES:
            if other != name:
                others |= {str(series[row]) for row in indices[other]}
        assert not {str(series[row]) for row in indices[name]} & others

    for name in SPLIT_NAMES:
        achieved = indices[name].size / series.size
        assert abs(achieved - DEFAULT_RATIOS[name]) < 0.01, f"{name}: {achieved:.4f}"

    # Every stratum present in the data is itself split near the target ratio.
    for label in np.unique(nonempty):
        total = int((nonempty == label).sum())
        for name in SPLIT_NAMES:
            share = int((nonempty[indices[name]] == label).sum()) / total
            assert abs(share - DEFAULT_RATIOS[name]) < 0.02, (
                f"stratum {label}, {name}: {share:.4f}"
            )


@pytest.mark.skipif(not REAL_NPZ.exists(), reason="converted dataset not present")
def test_committed_split_file_is_reproducible() -> None:
    """The split file in the repo is exactly what the recorded seed regenerates.

    This is the guard against the file on disk drifting away from the algorithm --
    the failure mode that would silently invalidate every reported comparison.
    """
    split_path = REPO_ROOT / "data" / "splits" / "split.json"
    if not split_path.exists():
        pytest.skip("split.json not generated yet")
    loaded = load_split(split_path)

    with np.load(REAL_NPZ, allow_pickle=False) as handle:
        series = handle["series_uid"]
        nonempty = nonempty_masks_per_patch(handle["masks"])

    groups = assign_series_to_splits(series, nonempty, seed=loaded.seed)
    indices = indices_for_groups(series, groups)
    for name in SPLIT_NAMES:
        assert indices[name].tolist() == loaded.indices[name].tolist()
        assert sorted(groups[name]) == sorted(loaded.series_uid[name])