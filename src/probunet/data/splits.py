"""Generate and load the fixed train/validation/test split of the LIDC data.

The split is **grouped by ``series_uid``**: every patch belonging to one CT series
lands in exactly one split. LIDC contributes many adjacent slices per nodule and
several nodules per scan, so a patch-level split would put near-duplicate crops on
both sides of the train/test boundary and inflate the reported scores.

Because series sizes are very uneven (1 to 108 patches, median 12), the requested
60/20/20 patch ratio cannot be hit exactly while keeping series intact. The
assignment is therefore a seeded, deterministic *greedy* one, **stratified over the
number of non-empty grader masks per patch** (0 to 4) so the splits are comparable
in ambiguity and not merely in size:

1. Shuffle the unique series with the given seed.
2. Sort them by patch count, largest first (a longest-processing-time heuristic:
   placing the big, awkward groups while all splits still have room is what keeps
   the final ratios tight).
3. Walk the series in that order. Each series carries a *profile*: how many of its
   patches fall in each stratum. Assign it to the split with the largest **relative
   shortfall** across the strata that series actually contains -- the weighted mean
   of ``(target - assigned) / stratum_total``, weighted by the series' own stratum
   composition. Ties break toward train, then val, then test.

With a single stratum this reduces exactly to "assign to the split furthest below
its target patch count", and because a split's total is the sum of its per-stratum
counts the overall 60/20/20 ratio stays tight without a separate term for it.

The result is written **once** to ``data/splits/split.json`` and loaded from that
file forever after. :func:`load_split` never regenerates: every baseline
comparison in the report depends on the split being byte-for-byte identical
across runs and machines.

Command line::

    python -m probunet.data.splits                    # generate (refuses to overwrite)
    python -m probunet.data.splits --overwrite        # regenerate deliberately
    python -m probunet.data.splits --seed 7 --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
"""Canonical split order. Ties in the greedy assignment break toward the front."""

DEFAULT_RATIOS: dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
"""Target share of *patches* per split, as agreed for this project."""

DEFAULT_SEED: int = 1806
"""Split seed. Arbitrary by nature but fixed and recorded; 1806 nods to arXiv:1806.05034."""

DEFAULT_NPZ_PATH = Path("data/processed/lidc.npz")
DEFAULT_SPLIT_PATH = Path("data/splits/split.json")

SCHEMA_VERSION = 2
N_GRADERS = 4
_HASH_CHUNK_BYTES = 8 * 1024 * 1024

STRATIFICATION_SCHEME = "nonempty_mask_count"
"""Name of the stratification variable recorded in the split file.

Each patch is labelled by how many of its four grader masks are non-empty (0..4).
This captures ambiguity about lesion *presence*. Shape disagreement among graders
is reported as a secondary diagnostic but is deliberately **not** stratified on --
stratifying on a continuous shape-agreement statistic would start shaping the
splits around the very quantity the extension is evaluated on.
"""


@dataclass(frozen=True)
class LoadedSplit:
    """A split loaded from disk, validated and ready to index the dataset.

    Attributes:
        indices: Split name to the patch row indices belonging to it. These index
            the row-aligned arrays in ``lidc.npz`` directly.
        series_uid: Split name to the DICOM series UIDs assigned to it.
        seed: The seed the split was generated with.
        ratios_requested: The target patch ratios that were asked for.
        achieved: Per-split achieved counts and ratios, as recorded at generation.
        source_npz_sha256: sha256 of the ``.npz`` the split was generated against.
        n_patches: Total number of patches the split covers.
    """

    indices: dict[str, np.ndarray]
    series_uid: dict[str, list[str]]
    seed: int
    ratios_requested: dict[str, float]
    achieved: dict[str, Any]
    source_npz_sha256: str
    n_patches: int

    def __getitem__(self, name: str) -> np.ndarray:
        """Return the patch indices for one split.

        Args:
            name: One of :data:`SPLIT_NAMES`.

        Returns:
            The row indices for that split.
        """
        return self.indices[name]


def _sha256_file(path: Path) -> str:
    """Compute the sha256 of a file in chunks.

    Duplicated deliberately from ``scratch/convert_data.py``: the package must not
    import anything from ``scratch/``.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Check that ratios cover exactly the known splits and sum to one.

    Args:
        ratios: Split name to target patch share.

    Returns:
        The ratios as a plain dict of floats.

    Raises:
        ValueError: If a split is missing or extra, a ratio is negative, or the
            ratios do not sum to 1.
    """
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must cover exactly {SPLIT_NAMES}, got {sorted(ratios)}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError(f"ratios must be non-negative, got {dict(ratios)}")
    total = float(sum(ratios.values()))
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    return {name: float(ratios[name]) for name in SPLIT_NAMES}


def assign_series_to_splits(
    series_uid: np.ndarray,
    strata: np.ndarray,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Assign whole CT series to splits, stratified over a per-patch label.

    Pure function of ``(series_uid, strata, ratios, seed)``: no file access, no
    global random state. Identical inputs always produce an identical assignment,
    which is what makes the split reproducible from the seed alone.

    Args:
        series_uid: Per-patch series UID array of length N (the ``series_uid``
            array from ``lidc.npz``).
        strata: Per-patch integer stratum label of length N. For this project it is
            the number of non-empty grader masks (0..4), see
            :func:`nonempty_masks_per_patch`.
        ratios: Target share of patches per split. Must sum to 1.
        seed: Seed for the initial shuffle of series.

    Returns:
        Split name to the sorted list of series UIDs assigned to it.

    Raises:
        ValueError: If ``ratios`` is malformed, ``series_uid`` is empty, or
            ``strata`` has a different length or holds negative labels.
    """
    ratios = _validate_ratios(ratios)
    series_uid = np.asarray(series_uid)
    strata = np.asarray(strata)
    if series_uid.size == 0:
        raise ValueError("series_uid is empty; nothing to split")
    if strata.shape != series_uid.shape:
        raise ValueError(
            f"strata shape {strata.shape} != series_uid shape {series_uid.shape}"
        )
    if not np.issubdtype(strata.dtype, np.integer):
        raise ValueError(f"strata must be integer labels, got dtype {strata.dtype}")
    if strata.min() < 0:
        raise ValueError("strata labels must be non-negative")

    unique, inverse, counts = np.unique(series_uid, return_inverse=True, return_counts=True)
    n_strata = int(strata.max()) + 1

    # profiles[i, s] = number of patches of series i that fall in stratum s.
    profiles = np.zeros((unique.size, n_strata), dtype=np.int64)
    np.add.at(profiles, (inverse, strata), 1)
    stratum_totals = profiles.sum(axis=0).astype(np.float64)

    # Dividing by the stratum total turns a shortfall into a *relative* shortfall,
    # so a large stratum cannot dominate a small one. Empty strata contribute
    # nothing (guard the division; their weight is zero anyway).
    scale = np.where(stratum_totals > 0, stratum_totals, 1.0)
    targets = {name: ratios[name] * stratum_totals for name in SPLIT_NAMES}

    # Seeded shuffle, then a stable sort by descending size. The shuffle is what
    # the seed controls; the sort places the big, awkward series first.
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.size)
    order = order[np.argsort(-counts[order], kind="stable")]

    assigned = {name: np.zeros(n_strata, dtype=np.float64) for name in SPLIT_NAMES}
    groups: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    rank = {name: position for position, name in enumerate(SPLIT_NAMES)}

    for index in order:
        uid = str(unique[index])
        profile = profiles[index].astype(np.float64)
        # Weight the strata by this series' own composition, so a series is placed
        # according to where *it* would help, not by strata it has no patches in.
        weights = profile / profile.sum()

        def shortfall(name: str, profile: np.ndarray = profile, weights: np.ndarray = weights) -> float:
            """Weighted mean relative shortfall of ``name`` over this series' strata."""
            return float(np.sum(weights * (targets[name] - assigned[name]) / scale))

        # Largest relative shortfall wins. Note this must *maximize* remaining need
        # rather than minimize distance to the final target: with every split empty
        # at the start, the split with the smallest target is always the closest to
        # it, which would starve train completely.
        chosen = max(SPLIT_NAMES, key=lambda name: (shortfall(name), -rank[name]))
        groups[chosen].append(uid)
        assigned[chosen] += profile

    return {name: sorted(groups[name]) for name in SPLIT_NAMES}


def indices_for_groups(
    series_uid: np.ndarray, groups: Mapping[str, Sequence[str]]
) -> dict[str, np.ndarray]:
    """Expand a series-level assignment into per-split patch indices.

    Args:
        series_uid: Per-patch series UID array of length N.
        groups: Split name to the series UIDs assigned to it.

    Returns:
        Split name to a sorted array of patch row indices.

    Raises:
        ValueError: If a series appears in two splits, or if some patch's series
            was never assigned.
    """
    owner: dict[str, str] = {}
    for name, uids in groups.items():
        for uid in uids:
            if uid in owner:
                raise ValueError(f"series {uid} assigned to both {owner[uid]} and {name}")
            owner[uid] = name

    series_uid = np.asarray(series_uid)
    buckets: dict[str, list[int]] = {name: [] for name in groups}
    for row, uid in enumerate(series_uid):
        key = str(uid)
        if key not in owner:
            raise ValueError(f"patch {row} has unassigned series {key}")
        buckets[owner[key]].append(row)
    return {name: np.array(rows, dtype=np.int64) for name, rows in buckets.items()}


def summarize_split(
    indices: Mapping[str, np.ndarray],
    groups: Mapping[str, Sequence[str]],
    nonempty_per_patch: np.ndarray,
    ratios: Mapping[str, float],
) -> dict[str, Any]:
    """Describe what the split achieved, including ambiguity balance.

    Size parity alone is not enough: the splits should also be comparable in how
    ambiguous their cases are, since a split richer in unanimous or in
    single-grader cases would make GED and oracle numbers incomparable.

    Args:
        indices: Split name to patch row indices.
        groups: Split name to assigned series UIDs.
        nonempty_per_patch: Per-patch count of non-empty grader masks (0 to 4).
        ratios: The requested target ratios.

    Returns:
        A JSON-serializable per-split summary.
    """
    total_patches = int(sum(len(rows) for rows in indices.values()))
    summary: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        rows = np.asarray(indices[name], dtype=np.int64)
        counts = nonempty_per_patch[rows] if rows.size else np.empty(0, dtype=np.int64)
        distribution = {
            str(k): int((counts == k).sum()) for k in range(N_GRADERS + 1)
        }
        summary[name] = {
            "n_series": len(groups[name]),
            "n_patches": int(rows.size),
            "patch_ratio": float(rows.size / total_patches) if total_patches else 0.0,
            "patch_ratio_requested": float(ratios[name]),
            "patch_ratio_error": (
                float(rows.size / total_patches - ratios[name]) if total_patches else 0.0
            ),
            "nonempty_masks_per_patch": distribution,
            "nonempty_masks_per_patch_fraction": {
                key: (float(value / rows.size) if rows.size else 0.0)
                for key, value in distribution.items()
            },
            "mean_nonempty_masks": float(counts.mean()) if counts.size else 0.0,
        }
    summary["totals"] = {
        "n_patches": total_patches,
        "n_series": int(sum(len(groups[name]) for name in SPLIT_NAMES)),
    }
    return summary


def nonempty_masks_per_patch(masks: np.ndarray) -> np.ndarray:
    """Count how many of the four grader masks are non-empty, per patch.

    Args:
        masks: Array of shape ``(N, 4, H, W)``.

    Returns:
        Integer array of shape ``(N,)`` with values in ``0..4``.

    Raises:
        ValueError: If ``masks`` does not have the expected rank or grader count.
    """
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[1] != N_GRADERS:
        raise ValueError(f"expected masks of shape (N, {N_GRADERS}, H, W), got {masks.shape}")
    return masks.reshape(masks.shape[0], N_GRADERS, -1).any(axis=2).sum(axis=1).astype(np.int64)


def mean_pairwise_iou_nonempty(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean pairwise IoU among the non-empty grader masks of each patch.

    This measures disagreement about lesion *shape*, which the stratification
    variable (non-empty mask count) does not capture: four graders can all agree a
    lesion is present and still outline it very differently. Patches with fewer
    than two non-empty masks have no pair to compare and are excluded.

    Both masks in a pair are non-empty by construction, so the union is always
    positive and no zero-division guard is needed.

    Args:
        masks: Array of shape ``(N, 4, H, W)`` with binary values.

    Returns:
        A ``(values, valid)`` pair. ``values[i]`` is the mean pairwise IoU for
        patch ``i`` and is NaN where ``valid[i]`` is False; ``valid[i]`` is True
        iff patch ``i`` has at least two non-empty masks.

    Raises:
        ValueError: If ``masks`` does not have the expected rank or grader count.
    """
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[1] != N_GRADERS:
        raise ValueError(f"expected masks of shape (N, {N_GRADERS}, H, W), got {masks.shape}")

    n_patches = masks.shape[0]
    # Keep the uint8 view: bitwise ops on 0/1 uint8 behave like booleans and this
    # avoids allocating a second copy of a ~1 GB array.
    flat = masks.reshape(n_patches, N_GRADERS, -1)
    nonempty = flat.any(axis=2)

    iou_sums = np.zeros(n_patches, dtype=np.float64)
    pair_counts = np.zeros(n_patches, dtype=np.int64)
    for first in range(N_GRADERS):
        for second in range(first + 1, N_GRADERS):
            both = nonempty[:, first] & nonempty[:, second]
            if not both.any():
                continue
            left = flat[both, first]
            right = flat[both, second]
            intersection = np.count_nonzero(left & right, axis=1)
            union = np.count_nonzero(left | right, axis=1)
            iou_sums[both] += intersection / union
            pair_counts[both] += 1

    valid = pair_counts > 0
    values = np.full(n_patches, np.nan, dtype=np.float64)
    values[valid] = iou_sums[valid] / pair_counts[valid]
    return values, valid


def summarize_shape_agreement(
    indices: Mapping[str, np.ndarray], shape_agreement: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    """Summarize per-split shape agreement among graders.

    Reported only. The splits are **not** stratified on this quantity: doing so
    would shape the splits around the same grader-agreement signal the consensus
    selection head is evaluated against.

    Args:
        indices: Split name to patch row indices.
        shape_agreement: Per-patch mean pairwise IoU, NaN where undefined.
        valid: Per-patch flag marking patches with at least two non-empty masks.

    Returns:
        Per-split count, mean and median of the mean pairwise IoU.
    """
    summary: dict[str, Any] = {
        "measure": "mean pairwise IoU among non-empty grader masks",
        "eligibility": "patches with >= 2 non-empty grader masks",
        "stratified_on": False,
    }
    for name in SPLIT_NAMES:
        rows = np.asarray(indices[name], dtype=np.int64)
        eligible = rows[valid[rows]] if rows.size else rows
        values = shape_agreement[eligible]
        summary[name] = {
            "n_patches_evaluated": int(eligible.size),
            "fraction_of_split": float(eligible.size / rows.size) if rows.size else 0.0,
            "mean_iou": float(values.mean()) if values.size else None,
            "median_iou": float(np.median(values)) if values.size else None,
        }
    means = [summary[name]["mean_iou"] for name in SPLIT_NAMES]
    medians = [summary[name]["median_iou"] for name in SPLIT_NAMES]
    if all(value is not None for value in means):
        summary["mean_iou_spread"] = float(max(means) - min(means))
        summary["median_iou_spread"] = float(max(medians) - min(medians))
    return summary


def generate_split(
    npz_path: Path = DEFAULT_NPZ_PATH,
    out_path: Path = DEFAULT_SPLIT_PATH,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    overwrite: bool = False,
    hash_source: bool = True,
) -> dict[str, Any]:
    """Generate the split from the converted dataset and write it to disk.

    Args:
        npz_path: The converted dataset produced by ``scratch/convert_data.py``.
        out_path: Where to write ``split.json``.
        seed: Seed recorded in the file and used for the assignment.
        ratios: Target patch ratios.
        overwrite: Allow replacing an existing split file. Off by default: the
            split is meant to be generated once.
        hash_source: Record the sha256 of the source ``.npz`` for provenance.

    Returns:
        The split document that was written.

    Raises:
        FileNotFoundError: If the dataset is missing.
        FileExistsError: If the split file exists and ``overwrite`` is False.
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run: python scratch/convert_data.py"
        )
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. The split is generated once on purpose; "
            "pass overwrite=True (or --overwrite) only if you intend to invalidate "
            "every result produced with the current split."
        )

    ratios = _validate_ratios(ratios)
    # npz members are decompressed on access, so 'images' is never touched here.
    with np.load(npz_path, allow_pickle=False) as handle:
        series_uid = handle["series_uid"]
        masks = handle["masks"]
        nonempty = nonempty_masks_per_patch(masks)
        shape_agreement, shape_valid = mean_pairwise_iou_nonempty(masks)
        del masks

    groups = assign_series_to_splits(series_uid, nonempty, ratios=ratios, seed=seed)
    indices = indices_for_groups(series_uid, groups)
    achieved = summarize_split(indices, groups, nonempty, ratios)
    shape_summary = summarize_shape_agreement(indices, shape_agreement, shape_valid)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Fixed train/val/test split of the preprocessed LIDC-IDRI patches, "
            "grouped by DICOM series_uid so no CT series spans two splits. "
            "Indices refer to rows of the row-aligned arrays in lidc.npz."
        ),
        "seed": int(seed),
        "grouping_key": "series_uid",
        "algorithm": (
            "Seeded shuffle of unique series, stable sort by descending patch "
            "count, then greedy assignment of each series to the split with the "
            "largest relative shortfall across the strata that series contains "
            "(weighted mean of (target - assigned) / stratum_total, weighted by "
            "the series' stratum composition); ties break toward train, then val, "
            "then test."
        ),
        "stratification": {
            "scheme": STRATIFICATION_SCHEME,
            "description": (
                "Patches are labelled by how many of their four grader masks are "
                "non-empty (0..4), i.e. ambiguity about lesion presence. The "
                "60/20/20 patch ratio is targeted within each stratum."
            ),
            "n_strata": int(nonempty.max()) + 1,
            "stratum_totals": {
                str(k): int((nonempty == k).sum()) for k in range(N_GRADERS + 1)
            },
        },
        "ratios_requested": ratios,
        "source_npz": {
            "path": str(npz_path),
            "sha256": _sha256_file(npz_path) if hash_source else None,
            "n_patches": int(series_uid.size),
        },
        "achieved": achieved,
        "secondary_diagnostic_shape_agreement": shape_summary,
        "series_uid": groups,
        "indices": {name: [int(v) for v in indices[name]] for name in SPLIT_NAMES},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2) + "\n")
    return document


def load_split(
    path: Path = DEFAULT_SPLIT_PATH,
    expected_n_patches: int | None = None,
    verify_source: Path | None = None,
) -> LoadedSplit:
    """Load the fixed split from disk, validating its integrity.

    This never regenerates and never randomizes. If the file is absent that is an
    error, not an invitation to make a new split.

    Args:
        path: Path to ``split.json``.
        expected_n_patches: If given, assert the split covers exactly this many
            patches (pass the dataset length to catch a stale split file).
        verify_source: If given, hash this ``.npz`` and assert it matches the file
            the split was generated against.

    Returns:
        The validated :class:`LoadedSplit`.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If the document is malformed, the splits overlap, they do not
            cover every patch exactly once, or a provenance check fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it once with: python -m probunet.data.splits"
        )
    document = json.loads(path.read_text())

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {document.get('schema_version')} "
            f"!= expected {SCHEMA_VERSION}"
        )
    for field in ("seed", "indices", "series_uid", "ratios_requested", "achieved"):
        if field not in document:
            raise ValueError(f"{path}: missing required field {field!r}")
    if set(document["indices"]) != set(SPLIT_NAMES):
        raise ValueError(f"{path}: indices must cover exactly {SPLIT_NAMES}")

    indices = {
        name: np.asarray(document["indices"][name], dtype=np.int64) for name in SPLIT_NAMES
    }
    series = {name: list(document["series_uid"][name]) for name in SPLIT_NAMES}

    # No series may appear in two splits.
    seen_series: dict[str, str] = {}
    for name, uids in series.items():
        for uid in uids:
            if uid in seen_series:
                raise ValueError(
                    f"{path}: series {uid} appears in both {seen_series[uid]} and {name}"
                )
            seen_series[uid] = name

    # Indices must partition 0..n-1 exactly: disjoint and complete.
    concatenated = np.concatenate([indices[name] for name in SPLIT_NAMES])
    n_patches = int(document["source_npz"]["n_patches"])
    if concatenated.size != n_patches:
        raise ValueError(
            f"{path}: splits cover {concatenated.size} patches, expected {n_patches}"
        )
    if np.unique(concatenated).size != concatenated.size:
        raise ValueError(f"{path}: duplicate indices across splits")
    if not np.array_equal(np.sort(concatenated), np.arange(n_patches)):
        raise ValueError(f"{path}: split indices are not a partition of 0..{n_patches - 1}")

    if expected_n_patches is not None and n_patches != expected_n_patches:
        raise ValueError(
            f"{path}: split covers {n_patches} patches but the dataset has "
            f"{expected_n_patches}. The split is stale; do not silently regenerate."
        )
    if verify_source is not None:
        recorded = document["source_npz"].get("sha256")
        actual = _sha256_file(Path(verify_source))
        if recorded is not None and recorded != actual:
            raise ValueError(
                f"{path}: was generated against a different {verify_source} "
                f"(recorded {recorded[:12]}..., actual {actual[:12]}...)"
            )

    return LoadedSplit(
        indices=indices,
        series_uid=series,
        seed=int(document["seed"]),
        ratios_requested=dict(document["ratios_requested"]),
        achieved=document["achieved"],
        source_npz_sha256=document["source_npz"].get("sha256"),
        n_patches=n_patches,
    )


def format_report(document: Mapping[str, Any]) -> str:
    """Render the achieved split as a human-readable table.

    Args:
        document: The split document returned by :func:`generate_split`.

    Returns:
        A multi-line report string.
    """
    achieved = document["achieved"]
    strata = document["stratification"]
    buckets = tuple(str(k) for k in range(N_GRADERS + 1))
    lines = [
        "=" * 78,
        "SPLIT REPORT",
        "=" * 78,
        f"grouping key    : {document['grouping_key']}",
        f"stratified on   : {strata['scheme']} ({strata['n_strata']} strata)",
        f"seed            : {document['seed']}",
        f"source          : {document['source_npz']['path']}",
        "",
        "TABLE 1 - empty-mask distribution over the whole dataset",
        f"  {'non-empty masks':<16} {'patches':>8} {'share':>8}",
    ]
    dataset_total = achieved["totals"]["n_patches"]
    for bucket in buckets:
        count = strata["stratum_totals"][bucket]
        lines.append(
            f"  {bucket:<16} {count:>8} {count / dataset_total:>8.4f}"
            if dataset_total
            else f"  {bucket:<16} {count:>8}"
        )
    lines += [
        f"  {'TOTAL':<16} {dataset_total:>8}",
        "",
        "TABLE 2 - per-split patch and series counts",
        f"  {'split':<6} {'series':>7} {'patches':>8} {'ratio':>8} {'target':>8} {'error':>9}",
    ]
    for name in SPLIT_NAMES:
        row = achieved[name]
        lines.append(
            f"  {name:<6} {row['n_series']:>7} {row['n_patches']:>8} "
            f"{row['patch_ratio']:>8.4f} {row['patch_ratio_requested']:>8.4f} "
            f"{row['patch_ratio_error']:>+9.5f}"
        )
    lines.append(
        f"  {'TOTAL':<6} {achieved['totals']['n_series']:>7} {dataset_total:>8}"
    )
    lines += [
        "",
        "TABLE 3 - non-empty grader masks per patch, fraction within each split",
        f"  {'split':<6} " + " ".join(f"{k:>8}" for k in buckets) + f" {'mean':>8}",
    ]
    for name in SPLIT_NAMES:
        row = achieved[name]
        fractions = row["nonempty_masks_per_patch_fraction"]
        lines.append(
            f"  {name:<6} "
            + " ".join(f"{fractions[k]:>8.4f}" for k in buckets)
            + f" {row['mean_nonempty_masks']:>8.4f}"
        )
    lines.append(
        f"  {'spread':<6} "
        + " ".join(
            f"{max(achieved[n]['nonempty_masks_per_patch_fraction'][k] for n in SPLIT_NAMES) - min(achieved[n]['nonempty_masks_per_patch_fraction'][k] for n in SPLIT_NAMES):>8.4f}"
            for k in buckets
        )
        + f" {max(achieved[n]['mean_nonempty_masks'] for n in SPLIT_NAMES) - min(achieved[n]['mean_nonempty_masks'] for n in SPLIT_NAMES):>8.4f}"
    )

    shape = document.get("secondary_diagnostic_shape_agreement")
    if shape is not None:
        lines += [
            "",
            "TABLE 4 - secondary diagnostic, NOT stratified on",
            "  mean pairwise IoU among non-empty grader masks (>= 2 non-empty required)",
            f"  {'split':<6} {'patches':>8} {'of split':>9} {'mean IoU':>9} {'median':>9}",
        ]
        for name in SPLIT_NAMES:
            row = shape[name]
            # A split with no eligible patches has no IoU to report; show it rather
            # than crashing on a None (which is how a collapsed split first showed up).
            mean_text = "      n/a" if row["mean_iou"] is None else f"{row['mean_iou']:>9.4f}"
            median_text = (
                "      n/a" if row["median_iou"] is None else f"{row['median_iou']:>9.4f}"
            )
            lines.append(
                f"  {name:<6} {row['n_patches_evaluated']:>8} "
                f"{row['fraction_of_split']:>9.4f} {mean_text} {median_text}"
            )
        if "mean_iou_spread" in shape:
            lines.append(
                f"  spread across splits: mean {shape['mean_iou_spread']:.4f}, "
                f"median {shape['median_iou_spread']:.4f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point for generating the split.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate the fixed LIDC data split.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ_PATH, help="converted dataset")
    parser.add_argument("--out", type=Path, default=DEFAULT_SPLIT_PATH, help="split file")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split seed")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[DEFAULT_RATIOS[name] for name in SPLIT_NAMES],
        help="target patch ratios (must sum to 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing split file (invalidates all previous results)",
    )
    args = parser.parse_args(argv)

    ratios = dict(zip(SPLIT_NAMES, args.ratios))
    try:
        document = generate_split(
            npz_path=args.npz,
            out_path=args.out,
            seed=args.seed,
            ratios=ratios,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}")
        return 1

    print(format_report(document))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())