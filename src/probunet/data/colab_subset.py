"""Building the Colab demo subset: selection, test exclusion, and index re-basing.

``scripts/make_colab_subset.py`` runs on the PC and writes two tracked files -- a small
``.npz`` of patches and a split file that addresses it -- so the submission notebook's
Tier 2 can train a tiny model from scratch on Colab with nothing to download. This module
is the part of that job which touches no large file and can therefore be tested on a
laptop against a synthetic stand-in dataset.

Three hazards shape everything here.

**Test contamination is the primary risk.** Patches come from the ``train`` indices of
``data/splits/split.json``, plus a small handful from ``val`` so the demo's validation loop
and its checkpoint monitor have something to run on. No test index may appear.
:func:`assert_no_test_indices` enforces that, and it is called on the **final selected
indices** rather than on the candidate pool: a pool-level check would pass while a later
top-up, a re-index or an off-by-one still pulled a test patch in. The demo's split file
carries an **empty** test list for the same reason.

**Index re-basing is the subtle one.** ``split.json`` indices address the full
15,096-patch array. The moment the subset becomes a re-indexed array of a few hundred rows,
those indices are wrong for it -- and wrong in the worst way, because they are still valid
integers that address *some* patch. So the demo gets its own split file whose train/val
lists are indices **into the subset**, and every subset row records the full-dataset index
it came from, which is what keeps the re-basing auditable after the fact. The main split
file is never reused for the demo and never rewritten.

**Format compatibility is not negotiable.** The subset must load through
:class:`~probunet.data.lidc.LidcArrays` and :class:`~probunet.data.lidc.LidcDataset` with
zero loader changes: the same key names, the same dtypes, the same array layout and the
same grader ordering as ``lidc.npz``, only fewer rows. Nothing is downcast to save space --
``np.savez_compressed`` already does the work, because the masks are almost all zeros and
compress to 0.2% of their raw size (measured, see :data:`MEASURED_BYTES_PER_PATCH`).

**Why this does not reuse ``diagnostics.stratified_indices``.** That function *silently
tops up* from other buckets when one cannot fill its slots (``diagnostics.py``, "If a small
bucket could not fill its slots, top up from whatever is left"), which is right for a
diagnostic panel that must reach a fixed size, and wrong here: a demo subset that quietly
became bucket-1-heavy would be exactly the accident the stratification exists to prevent.
:func:`draw_stratified` raises instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from probunet.data.splits import SCHEMA_VERSION, SPLIT_NAMES

DEMO_BUCKETS: tuple[int, ...] = (1, 2, 3, 4)
"""The four ambiguity buckets the demo stratifies over: non-empty grader counts 1 to 4.

Bucket 0 -- every grader empty -- is **not** a stratum and gets no quota. Measured over the
full 15,096 patches it is empty (counts are 1 -> 4963, 2 -> 2756, 3 -> 2626, 4 -> 4751), so
there is nothing to draw; the availability of every bucket is reported rather than assumed,
so if a future dataset did carry bucket-0 patches their absence from the demo would be
visible in the report instead of silent. Engineering a bucket-0 quota would be shaping the
mix beyond stratification, which is not what this is for.
"""

DEFAULT_TRAIN_PATCHES = 256
"""Train patches, 64 per bucket. See :func:`size_arithmetic` for why this number."""

DEFAULT_VAL_PATCHES = 32
"""Validation patches, 8 per bucket -- enough for the demo's validation loop and its
``val/total`` monitor to produce a real number, small enough not to dominate the budget."""

DEFAULT_SEED = 2018
"""Seed for the draw. The project's run seed; the SPLIT seed is a different, frozen 1806
and this must not be confused with it, so both are recorded in the provenance."""

MEASURED_BYTES_PER_PATCH = 30_349
"""On-disk compressed bytes per patch, **measured, not estimated**.

From the existing ``data/processed/lidc_subset.npz``: 2,154,808 bytes over 71 patches. The
breakdown is what makes this stable across a different bucket mix -- of those bytes,
2,140,598 are the images (compression ratio 0.460) and only 9,890 are the masks (ratio
0.002, because they are almost entirely zeros). Masks are 0.5% of the file, so a subset
weighted toward 4-grader patches costs essentially the same as one weighted toward
1-grader patches, and the per-patch figure can be used directly to size the draw.

Cross-check against the full file: 454,714,352 bytes over 15,096 patches = 30,121
bytes/patch, within 0.8% of the subset figure.
"""

SIZE_TARGET_BYTES = 10 * 1024**2
"""Soft target. Above this the script warns."""

SIZE_HARD_STOP_BYTES = 20 * 1024**2
"""Hard stop. Above this the script refuses to leave the file in place."""

DEMO_SPLIT_NOTE = (
    "Demo split for the Colab Tier 2 training demonstration. Its indices address "
    "data/processed/lidc_colab_demo.npz, NOT data/processed/lidc.npz -- the two are "
    "different arrays and the main split.json is invalid for this file. The test list is "
    "empty by construction: no test patch was drawn, and that was asserted on the final "
    "selected indices."
)


# ---------------------------------------------------------------------------------
# Stratified drawing
# ---------------------------------------------------------------------------------


def plan_per_bucket(total: int, buckets: tuple[int, ...] = DEMO_BUCKETS) -> dict[int, int]:
    """Split a requested patch count evenly across the ambiguity buckets.

    Args:
        total: Patches wanted overall.
        buckets: The strata, in the order the remainder is distributed over.

    Returns:
        Bucket to the number of patches wanted from it. The remainder goes to the
        lowest-numbered buckets, deterministically, so the plan is reproducible.

    Raises:
        ValueError: If ``total`` is negative or ``buckets`` is empty.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not buckets:
        raise ValueError("buckets must not be empty")
    base, remainder = divmod(total, len(buckets))
    return {
        bucket: base + (1 if position < remainder else 0)
        for position, bucket in enumerate(buckets)
    }


def draw_stratified(
    available: dict[int, np.ndarray],
    wanted: dict[int, int],
    rng: np.random.Generator,
    pool_name: str,
) -> dict[int, np.ndarray]:
    """Draw the requested number of patches from each bucket, or fail loudly.

    **A short bucket is an error, not something to make up elsewhere.** The
    ``diagnostics.stratified_indices`` path tops up from other buckets so a panel always
    reaches its requested size; that is wrong here, because a demo that silently became
    bucket-1-heavy is precisely the accident stratification exists to prevent, and it would
    be invisible in the resulting file.

    Args:
        available: Bucket to the global patch indices available in this pool.
        wanted: Bucket to the number of patches to draw.
        rng: Seeded generator, so the draw is reproducible.
        pool_name: ``"train"`` or ``"val"``, named in the error.

    Returns:
        Bucket to the drawn global indices, each sorted ascending.

    Raises:
        ValueError: If any bucket holds fewer patches than requested, naming every
            offending bucket with what was wanted and what was there.
    """
    short = [
        (bucket, count, int(available.get(bucket, np.empty(0)).size))
        for bucket, count in sorted(wanted.items())
        if int(available.get(bucket, np.empty(0)).size) < count
    ]
    if short:
        detail = "; ".join(
            f"bucket {bucket}: wanted {count}, only {have} available"
            for bucket, count, have in short
        )
        raise ValueError(
            f"the {pool_name} pool cannot fill the stratified plan -- {detail}. Reduce the "
            "requested patch count rather than letting a short bucket be topped up from "
            "another: a demo subset that quietly became bucket-1-heavy is exactly what "
            "stratifying is meant to prevent, and nothing in the resulting file would "
            "show it."
        )

    drawn: dict[int, np.ndarray] = {}
    for bucket in sorted(wanted):
        count = wanted[bucket]
        if count == 0:
            drawn[bucket] = np.empty(0, dtype=np.int64)
            continue
        members = np.sort(np.asarray(available[bucket], dtype=np.int64))
        chosen = rng.choice(members, size=count, replace=False)
        drawn[bucket] = np.sort(chosen.astype(np.int64))
    return drawn


def buckets_in_pool(
    nonempty_counts: np.ndarray, pool: np.ndarray, buckets: tuple[int, ...] = DEMO_BUCKETS
) -> dict[int, np.ndarray]:
    """Group a pool of global patch indices by ambiguity bucket.

    Args:
        nonempty_counts: Non-empty grader count per patch of the **full** dataset.
        pool: Global patch indices to group.
        buckets: The strata to report.

    Returns:
        Bucket to the global indices from ``pool`` falling in it, sorted.
    """
    pool = np.sort(np.asarray(pool, dtype=np.int64))
    counts = np.asarray(nonempty_counts)[pool]
    return {bucket: pool[counts == bucket] for bucket in buckets}


# ---------------------------------------------------------------------------------
# The test-exclusion guarantee
# ---------------------------------------------------------------------------------


def assert_no_test_indices(
    selected: np.ndarray, test_indices: np.ndarray
) -> dict[str, Any]:
    """Refuse any selection that touches the test split, and record that it was checked.

    **Called on the FINAL selected indices**, never on the candidate pool. A pool-level
    check answers "could a test patch have been drawn", which is not the question; a later
    top-up, a re-index, a concatenation in the wrong order or a plain off-by-one all leave
    the pool clean and the selection contaminated. The only check that means anything is
    the one on the rows that actually get written.

    Args:
        selected: Every global patch index about to be written to the subset.
        test_indices: The test split's global patch indices.

    Returns:
        A record of the check, for the provenance block, so the file carries the statement
        *and* the evidence rather than the statement alone.

    Raises:
        ValueError: If any selected index is a test index, naming the offenders.
    """
    selected = np.asarray(selected, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    overlap = np.intersect1d(selected, test_indices)
    if overlap.size:
        raise ValueError(
            f"TEST CONTAMINATION: {overlap.size} of the {selected.size} selected patches "
            f"are in the test split (first few: {overlap[:10].tolist()}). The Colab demo "
            "subset must never carry a test patch -- a model trained on one in a notebook "
            "a grader runs would make every reported test number meaningless. Fix the "
            "selection; do not relax this check."
        )
    return {
        "statement": (
            "No patch from the test split is present in this subset. Patches were drawn "
            "only from the train and val index lists of the source split file, and the "
            "check below was run on the FINAL selected indices, after re-basing decisions "
            "were made -- not on the candidate pool."
        ),
        "assertion": "numpy.intersect1d(selected_original_indices, test_indices).size == 0",
        "n_selected": int(selected.size),
        "n_test_indices": int(test_indices.size),
        "intersection_size": int(overlap.size),
        "verified": True,
    }


# ---------------------------------------------------------------------------------
# Re-basing
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoSelection:
    """The chosen patches, in both index spaces, plus the bucket bookkeeping.

    Attributes:
        rows: Global (full-dataset) indices of every subset row, ascending. Row ``i`` of
            the written arrays is full-dataset patch ``rows[i]``.
        train_global: Global indices assigned to the demo's train split.
        val_global: Global indices assigned to the demo's val split.
        train_rows: The same train patches as indices **into the subset**.
        val_rows: The same val patches as indices **into the subset**.
        per_bucket: ``{"train": {bucket: n}, "val": {bucket: n}}``.
        available_per_bucket: How many patches each bucket offered in each pool, so a
            report can show what was drawn from what.
    """

    rows: np.ndarray
    train_global: np.ndarray
    val_global: np.ndarray
    train_rows: np.ndarray
    val_rows: np.ndarray
    per_bucket: dict[str, dict[int, int]]
    available_per_bucket: dict[str, dict[int, int]]

    def __len__(self) -> int:
        """Number of patches in the subset."""
        return int(self.rows.size)


def rebase(train_global: np.ndarray, val_global: np.ndarray) -> DemoSelection:
    """Turn two sets of full-dataset indices into subset rows and re-based split lists.

    The subset's row order is the **ascending** full-dataset order, matching the
    convention ``scripts/export_subset.py`` already uses, so ``source_index`` is sorted and
    a reader can eyeball the mapping.

    Args:
        train_global: Full-dataset indices for the demo train split.
        val_global: Full-dataset indices for the demo val split.

    Returns:
        A :class:`DemoSelection` with ``per_bucket`` left empty for the caller to fill.

    Raises:
        ValueError: If the two sets overlap, or either carries a duplicate. Both would
            break the split file's partition requirement, and an overlap would put one
            patch in train and val at once.
    """
    train_global = np.asarray(train_global, dtype=np.int64)
    val_global = np.asarray(val_global, dtype=np.int64)

    for name, values in (("train", train_global), ("val", val_global)):
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} indices contain duplicates")
    overlap = np.intersect1d(train_global, val_global)
    if overlap.size:
        raise ValueError(
            f"train and val selections overlap on {overlap.size} patch(es) "
            f"({overlap[:5].tolist()}); a patch cannot be in both splits"
        )

    rows = np.sort(np.concatenate([train_global, val_global]))
    lookup = {int(source): position for position, source in enumerate(rows)}
    return DemoSelection(
        rows=rows,
        train_global=np.sort(train_global),
        val_global=np.sort(val_global),
        train_rows=np.array([lookup[int(i)] for i in np.sort(train_global)], dtype=np.int64),
        val_rows=np.array([lookup[int(i)] for i in np.sort(val_global)], dtype=np.int64),
        per_bucket={},
        available_per_bucket={},
    )


def select_demo_patches(
    nonempty_counts: np.ndarray,
    train_pool: np.ndarray,
    val_pool: np.ndarray,
    test_indices: np.ndarray,
    train_patches: int = DEFAULT_TRAIN_PATCHES,
    val_patches: int = DEFAULT_VAL_PATCHES,
    seed: int = DEFAULT_SEED,
    buckets: tuple[int, ...] = DEMO_BUCKETS,
) -> tuple[DemoSelection, dict[str, Any]]:
    """Draw the demo subset, assert it is test-free, and re-base its indices.

    The whole selection in one call, so the test-exclusion check cannot be skipped by a
    caller that assembles the pieces itself.

    Args:
        nonempty_counts: Non-empty grader count per patch of the full dataset.
        train_pool: Global indices of the source split's train patches.
        val_pool: Global indices of the source split's val patches.
        test_indices: Global indices of the source split's test patches.
        train_patches: Demo train patches, spread over the buckets.
        val_patches: Demo val patches, spread over the buckets.
        seed: Seed for the draw.
        buckets: The strata.

    Returns:
        ``(selection, exclusion_record)``.

    Raises:
        ValueError: If a bucket is short, if the draws overlap, or if any selected patch
            turns out to be a test patch.
    """
    rng = np.random.default_rng(seed)
    available = {
        "train": buckets_in_pool(nonempty_counts, train_pool, buckets),
        "val": buckets_in_pool(nonempty_counts, val_pool, buckets),
    }
    # Train first, then val, both off one generator: the sequence is fixed by the seed, so
    # the same seed reproduces the same subset exactly.
    drawn = {
        "train": draw_stratified(
            available["train"], plan_per_bucket(train_patches, buckets), rng, "train"
        ),
        "val": draw_stratified(
            available["val"], plan_per_bucket(val_patches, buckets), rng, "val"
        ),
    }

    selection = rebase(
        np.concatenate([drawn["train"][b] for b in sorted(drawn["train"])] or [np.empty(0)]),
        np.concatenate([drawn["val"][b] for b in sorted(drawn["val"])] or [np.empty(0)]),
    )
    # THE check, on the final rows about to be written -- see assert_no_test_indices.
    exclusion = assert_no_test_indices(selection.rows, test_indices)

    return (
        DemoSelection(
            rows=selection.rows,
            train_global=selection.train_global,
            val_global=selection.val_global,
            train_rows=selection.train_rows,
            val_rows=selection.val_rows,
            per_bucket={
                pool: {int(bucket): int(values.size) for bucket, values in sorted(by_bucket.items())}
                for pool, by_bucket in drawn.items()
            },
            available_per_bucket={
                pool: {int(bucket): int(values.size) for bucket, values in sorted(by_bucket.items())}
                for pool, by_bucket in available.items()
            },
        ),
        exclusion,
    )


# ---------------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------------


def size_arithmetic(
    train_patches: int = DEFAULT_TRAIN_PATCHES, val_patches: int = DEFAULT_VAL_PATCHES
) -> dict[str, Any]:
    """Project the on-disk size of a subset, from the measured per-patch cost.

    Reported rather than guessed: the per-patch figure is measured from an existing
    compressed export (see :data:`MEASURED_BYTES_PER_PATCH`), and the default patch count
    is chosen so the projection clears the target with headroom.

    Args:
        train_patches: Demo train patches.
        val_patches: Demo val patches.

    Returns:
        The projection and the budget it is checked against.
    """
    total = train_patches + val_patches
    projected = total * MEASURED_BYTES_PER_PATCH
    return {
        "n_patches": total,
        "bytes_per_patch_measured": MEASURED_BYTES_PER_PATCH,
        "projected_bytes": projected,
        "projected_mib": projected / 1024**2,
        "target_bytes": SIZE_TARGET_BYTES,
        "hard_stop_bytes": SIZE_HARD_STOP_BYTES,
        "max_patches_at_target": SIZE_TARGET_BYTES // MEASURED_BYTES_PER_PATCH,
        "max_patches_at_hard_stop": SIZE_HARD_STOP_BYTES // MEASURED_BYTES_PER_PATCH,
        "within_target": projected <= SIZE_TARGET_BYTES,
    }


def sha256_file(path: Path) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------
# The split document
# ---------------------------------------------------------------------------------


def build_demo_split_document(
    selection: DemoSelection,
    series_uid: np.ndarray,
    seed: int,
    subset_npz_path: Path,
    subset_npz_sha256: str,
    source_split_path: Path,
    source_split_seed: int,
    exclusion: dict[str, Any],
) -> dict[str, Any]:
    """Build the demo split document, in the schema :func:`load_split` validates.

    The indices are **subset rows**, and the test list is empty. Both facts are stated in
    the document itself, because a split file that looks like the main one but means
    something different is the trap this whole module exists around.

    Args:
        selection: The re-based selection.
        series_uid: Series UID per patch of the **full** dataset.
        seed: The seed the demo draw used.
        subset_npz_path: Path of the ``.npz`` these indices address.
        subset_npz_sha256: Its hash, so a mismatched pair is detectable.
        source_split_path: The main split file the pools came from.
        source_split_seed: That file's seed, recorded so the two are never confused.
        exclusion: The record returned by :func:`assert_no_test_indices`.

    Returns:
        A JSON-serializable document.

    Raises:
        ValueError: If the re-based indices do not partition the subset rows exactly,
            which is what ``load_split`` will refuse anyway -- better to fail here, where
            the message can say why.
    """
    n_patches = len(selection)
    covered = np.sort(np.concatenate([selection.train_rows, selection.val_rows]))
    if not np.array_equal(covered, np.arange(n_patches)):
        raise ValueError(
            f"the re-based indices do not partition 0..{n_patches - 1}: every subset row "
            "must belong to exactly one demo split, or load_split will reject the file"
        )

    series_uid = np.asarray(series_uid)
    per_split_rows = {
        "train": selection.train_rows,
        "val": selection.val_rows,
        "test": np.empty(0, dtype=np.int64),
    }
    per_split_global = {
        "train": selection.train_global,
        "val": selection.val_global,
        "test": np.empty(0, dtype=np.int64),
    }
    series = {
        name: sorted({str(uid) for uid in series_uid[per_split_global[name]]})
        for name in SPLIT_NAMES
    }
    shared = set(series["train"]) & set(series["val"])
    if shared:
        raise ValueError(
            f"{len(shared)} series appear in both demo train and demo val "
            f"({sorted(shared)[:3]}). The pools come from the series-grouped main split, "
            "so this should be impossible; load_split would refuse the file."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "grouping_key": "series_uid (inherited from the source split; not re-grouped here)",
        "note": DEMO_SPLIT_NOTE,
        "indices": {name: [int(v) for v in per_split_rows[name]] for name in SPLIT_NAMES},
        "original_indices": {
            name: [int(v) for v in per_split_global[name]] for name in SPLIT_NAMES
        },
        "series_uid": series,
        "ratios_requested": {
            name: (
                float(per_split_rows[name].size / n_patches) if n_patches else 0.0
            )
            for name in SPLIT_NAMES
        },
        "achieved": {
            name: {
                "n_patches": int(per_split_rows[name].size),
                "fraction": float(per_split_rows[name].size / n_patches) if n_patches else 0.0,
                "n_series": len(series[name]),
            }
            for name in SPLIT_NAMES
        },
        "per_bucket": {
            pool: {str(bucket): count for bucket, count in counts.items()}
            for pool, counts in selection.per_bucket.items()
        },
        "source_npz": {
            "path": str(subset_npz_path),
            "n_patches": n_patches,
            "sha256": subset_npz_sha256,
        },
        "source_split": {
            "path": str(source_split_path),
            "seed": int(source_split_seed),
            "note": (
                "The pools were taken from this file's train and val lists. Its indices "
                "address the FULL dataset and are invalid for the subset; the mapping is "
                "the 'source_index' array in the .npz and 'original_indices' above."
            ),
        },
        "test_split_exclusion": exclusion,
    }


def write_demo_split(path: Path, document: dict[str, Any]) -> Path:
    """Write the demo split document.

    Args:
        path: Destination, normally ``data/splits/colab_demo_split.json``.
        document: Output of :func:`build_demo_split_document`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------------
# The .npz
# ---------------------------------------------------------------------------------

PROVENANCE_KEY = "provenance_json"
"""Extra ``.npz`` key holding the provenance block as a JSON string.

Safe to add: :meth:`LidcArrays.load <probunet.data.lidc.LidcArrays.load>` reads named
arrays and ignores any others, so an extra key cannot change how the file loads. It is
stored as a 0-d unicode array rather than an object array so ``allow_pickle=False`` -- the
setting the loader uses -- can read it.
"""

PROVENANCE_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "generated_at",
    "git_revision",
    "source_npz_path",
    "source_npz_sha256",
    "source_n_patches",
    "source_split_path",
    "source_split_seed",
    "seed",
    "n_patches",
    "original_indices",
    "per_bucket",
    "available_per_bucket",
    "test_split_exclusion",
    "size",
    "demo_split_path",
)
"""Everything the provenance block must carry, asserted on write."""


def write_demo_subset(
    path: Path,
    images: np.ndarray,
    masks: np.ndarray,
    series_uid: np.ndarray,
    source_index: np.ndarray,
    provenance: dict[str, Any],
    keys: np.ndarray | None = None,
) -> int:
    """Write the subset ``.npz`` in exactly the full dataset's format.

    Identical key names, identical dtypes, identical layout and grader ordering -- only
    fewer rows -- so it loads through the existing dataset class with no loader change. The
    arrays are passed through untouched: **nothing is downcast**, because
    ``np.savez_compressed`` already reduces the masks to 0.2% of their raw size and a
    float16 image array would change what the demo trains on.

    Args:
        path: Destination ``.npz``.
        images: ``(n, H, W)`` float32 rows, sliced from the full array.
        masks: ``(n, 4, H, W)`` uint8 rows.
        series_uid: ``(n,)`` series UIDs.
        source_index: ``(n,)`` full-dataset row index per subset row.
        provenance: The provenance block.
        keys: Optional ``(n,)`` original pickle keys, if the source carried them.

    Returns:
        The file size in bytes.

    Raises:
        ValueError: If a dtype does not match the source format, the row counts disagree,
            or the provenance block is incomplete.
    """
    path = Path(path)
    expected = {
        "images": np.float32,
        "masks": np.uint8,
        "source_index": np.int64,
    }
    actual = {"images": images.dtype, "masks": masks.dtype, "source_index": source_index.dtype}
    wrong = {
        name: str(actual[name]) for name, want in expected.items() if actual[name] != want
    }
    if wrong:
        raise ValueError(
            f"dtype mismatch against the lidc.npz format: {wrong}. The subset must load "
            "through the existing dataset class with zero loader changes, so its dtypes "
            "have to match the full file exactly -- do not downcast to save space."
        )
    lengths = {
        "images": images.shape[0],
        "masks": masks.shape[0],
        "series_uid": series_uid.shape[0],
        "source_index": source_index.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"row counts disagree across arrays: {lengths}")

    missing = [key for key in PROVENANCE_REQUIRED_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"provenance block is missing {missing}")

    payload: dict[str, np.ndarray] = {
        "images": images,
        "masks": masks,
        "series_uid": series_uid,
        "source_index": source_index,
        PROVENANCE_KEY: np.array(json.dumps(provenance, indent=2, sort_keys=True)),
    }
    if keys is not None:
        payload["keys"] = keys

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path.stat().st_size


def read_provenance(path: Path) -> dict[str, Any]:
    """Read the provenance block back out of a subset ``.npz``.

    Args:
        path: The ``.npz``.

    Returns:
        The provenance mapping.

    Raises:
        FileNotFoundError: If the file is absent.
        KeyError: If it carries no provenance block.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"subset not found: {path}")
    with np.load(path, allow_pickle=False) as handle:
        if PROVENANCE_KEY not in handle.files:
            raise KeyError(
                f"{path} has no {PROVENANCE_KEY!r}: it was not written by "
                "write_demo_subset, so which patches it holds cannot be verified."
            )
        return json.loads(str(handle[PROVENANCE_KEY].item()))


def render_report(
    selection: DemoSelection, size_bytes: int, projection: dict[str, Any]
) -> str:
    """Render the bucket counts and the size arithmetic as a console table.

    Args:
        selection: The chosen patches.
        size_bytes: The actual on-disk size.
        projection: Output of :func:`size_arithmetic`.

    Returns:
        A plain-text report.
    """
    header = f"{'bucket':<9}{'train':>8}{'val':>8}{'total':>8}{'avail(train)':>14}{'avail(val)':>12}"
    lines = ["AMBIGUITY BUCKETS (non-empty grader masks per patch)", header, "-" * len(header)]
    for bucket in DEMO_BUCKETS:
        train = selection.per_bucket["train"].get(bucket, 0)
        val = selection.per_bucket["val"].get(bucket, 0)
        lines.append(
            f"{bucket:<9}{train:>8}{val:>8}{train + val:>8}"
            f"{selection.available_per_bucket['train'].get(bucket, 0):>14}"
            f"{selection.available_per_bucket['val'].get(bucket, 0):>12}"
        )
    lines.append(
        f"{'all':<9}{selection.train_rows.size:>8}{selection.val_rows.size:>8}"
        f"{len(selection):>8}"
    )
    lines += [
        "",
        "SIZE",
        f"  measured cost      : {projection['bytes_per_patch_measured']:,} bytes/patch "
        "(from data/processed/lidc_subset.npz: 2,154,808 B / 71 patches)",
        f"  projected          : {len(selection)} x "
        f"{projection['bytes_per_patch_measured']:,} = "
        f"{projection['projected_bytes']:,} B ({projection['projected_mib']:.2f} MiB)",
        f"  actual on disk     : {size_bytes:,} B ({size_bytes / 1024**2:.2f} MiB)",
        f"  target / hard stop : {SIZE_TARGET_BYTES / 1024**2:.0f} MiB / "
        f"{SIZE_HARD_STOP_BYTES / 1024**2:.0f} MiB",
        f"  budget allows      : {projection['max_patches_at_target']} patches at the "
        f"target, {projection['max_patches_at_hard_stop']} at the hard stop",
    ]
    return "\n".join(lines)
