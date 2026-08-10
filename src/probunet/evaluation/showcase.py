"""The pure half of the showcase export: case selection, guards, verification, I/O.

``scripts/export_showcase.py`` runs on the training PC, loads four checkpoints and writes
``data/processed/showcase.npz`` -- a tracked file the submission notebook renders with
numpy and matplotlib alone, loading no checkpoint and running no model. Everything in
*this* module is the part of that job which touches no weights: choosing which six patches
to show, deciding which patches are eligible to be chosen from, checking a recomputed
table against a published one, and writing/reading the ``.npz``.

The split exists for one reason: **the export itself can only be validated on the PC**, so
the parts that *can* be tested on a laptop with no GPU and no checkpoints are separated
out and tested there. See ``tests/test_showcase.py``, whose docstrings label that limit
explicitly.

Three things in here carry the correctness argument and are worth reading before changing:

* :func:`nearest_percentile_position` is the **pre-registered** case rule. Percentile-
  nearest, never ``argmin``/``argmax``: a single-pixel lesion makes the extremes of every
  criterion in this project unstable, so the tails would select a patch that is a metric
  artifact rather than an example of anything.
* :func:`check_published_figures` is a **value** check, not a shape check. The showcase is
  only honest if the candidates it displays are the ones that produced the published
  numbers, and the only way to demonstrate that is to recompute those numbers through the
  published code path and compare them exactly. A shape assertion cannot tell a correct
  sampling path from a divergent one.
* The guards in :func:`selection_eligibility` and :func:`ged_eligibility` are applied
  **before** percentiles are computed and their counts are **recorded**, never dropped
  silently -- an exclusion that leaves no trace in the manifest is indistinguishable from
  a bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
"""Bumped when a key is renamed or removed, so a stale notebook fails loudly."""

MANIFEST_KEY = "manifest_json"
"""The single ``.npz`` key holding the provenance manifest, as a JSON string."""

CASE_PERCENTILES: tuple[float, ...] = (5.0, 50.0, 95.0)
"""The three percentiles of the per-image criterion each set is sampled at.

Pre-registered: 5th, median, 95th. The 5th percentile of Set B is a **failure** case where
the head loses to the largest-candidate rule, and it is exported deliberately -- a
showcase that only shows wins is an advertisement, not a result.
"""

MAX_SHOWCASE_BYTES = 20 * 1024**2
"""Above this the export is warned about: the file is tracked and must survive a clone."""

EMPTY_CANDIDATE_AREA = 0
"""Foreground pixel count that marks a candidate as empty."""


# ---------------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseRecord:
    """One selected showcase case, and everything needed to justify the choice.

    Attributes:
        set_name: ``"A"`` (Phase 2 qualitative) or ``"B"`` (Phase 3 selection).
        key_prefix: Prefix of this case's ``.npz`` keys, e.g. ``"a0"``.
        percentile: The percentile targeted, in ``[0, 100]``.
        percentile_value: The criterion value at that percentile over the eligible set.
        position: Row of the patch within the per-image arrays.
        patch_index: Global patch index in the full dataset.
        bucket: Number of non-empty graders, 1 to 4.
        criterion: This patch's criterion value.
        criterion_name: Human-readable statement of what the criterion is.
    """

    set_name: str
    key_prefix: str
    percentile: float
    percentile_value: float
    position: int
    patch_index: int
    bucket: int
    criterion: float
    criterion_name: str

    def as_dict(self) -> dict[str, Any]:
        """Render as JSON-serializable plain data for the manifest.

        Returns:
            A mapping with every field, floats and ints native rather than numpy.
        """
        return {
            "set": self.set_name,
            "key_prefix": self.key_prefix,
            "percentile": float(self.percentile),
            "percentile_value": float(self.percentile_value),
            "position": int(self.position),
            "patch_index": int(self.patch_index),
            "bucket": int(self.bucket),
            "criterion": float(self.criterion),
            "criterion_name": self.criterion_name,
        }


def nearest_percentile_position(
    values: np.ndarray, eligible: np.ndarray, percentile: float
) -> tuple[int, float]:
    """Position of the eligible image whose criterion is nearest a given percentile.

    **Not ``argmin``/``argmax``, and that is the whole point of the rule.** The extremes of
    every criterion in this project are dominated by single-pixel lesions, where IoU and
    soft Dice both swing between 0 and 1 on one pixel: the most extreme patch is reliably a
    metric artifact rather than an illustrative case. A percentile target picks a patch
    that is *representative of the tail* instead of being the tail.

    The percentile is taken over the **eligible** values only, and the nearest eligible
    value to it is returned -- so an excluded patch can neither be selected nor shift the
    target.

    Ties are resolved to the lowest eligible position. That is ``np.argmin``'s existing
    behaviour rather than an added rule, and it is deterministic, which is what the case
    selection needs; it is asserted in the tests so it cannot drift.

    Args:
        values: Per-image criterion, shape ``(n,)``.
        eligible: Boolean mask of the same shape.
        percentile: Percentile to target, in ``[0, 100]``.

    Returns:
        ``(position, target)``: the row into ``values``, and the criterion value at that
        percentile over the eligible subset.

    Raises:
        ValueError: If the shapes disagree, no image is eligible, or the percentile is out
            of range.
    """
    values = np.asarray(values, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if values.shape != eligible.shape:
        raise ValueError(
            f"values {values.shape} and eligible {eligible.shape} must have one shape"
        )
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D criterion, got shape {values.shape}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if not eligible.any():
        raise ValueError(
            "no eligible image to select from. Every patch was excluded by the guards; "
            "check the guard counts before relaxing anything."
        )

    target = float(np.percentile(values[eligible], percentile))
    distance = np.abs(values - target)
    # Excluded rows cannot win, however close they sit to the target.
    distance = np.where(eligible, distance, np.inf)
    return int(np.argmin(distance)), target


def select_cases(
    values: np.ndarray,
    eligible: np.ndarray,
    patch_indices: np.ndarray,
    buckets: np.ndarray,
    set_name: str,
    criterion_name: str,
    percentiles: tuple[float, ...] = CASE_PERCENTILES,
) -> list[CaseRecord]:
    """Choose one case per percentile, mechanically and with no cherry-picking.

    Args:
        values: Per-image criterion, shape ``(n,)``.
        eligible: Boolean mask of the same shape, already carrying every guard.
        patch_indices: Global dataset index per image, shape ``(n,)``.
        buckets: Non-empty grader count per image, shape ``(n,)``.
        set_name: ``"A"`` or ``"B"``; also fixes the ``.npz`` key prefix letter.
        criterion_name: Human-readable statement of the criterion, carried into the
            manifest so a reader never has to infer what was maximized.
        percentiles: Percentiles to target.

    Returns:
        One :class:`CaseRecord` per percentile, in the order given.
    """
    letter = set_name.strip().lower()
    cases: list[CaseRecord] = []
    for ordinal, percentile in enumerate(percentiles):
        position, target = nearest_percentile_position(values, eligible, percentile)
        cases.append(
            CaseRecord(
                set_name=set_name,
                key_prefix=f"{letter}{ordinal}",
                percentile=float(percentile),
                percentile_value=target,
                position=position,
                patch_index=int(patch_indices[position]),
                bucket=int(buckets[position]),
                criterion=float(values[position]),
                criterion_name=criterion_name,
            )
        )
    return cases


def duplicate_case_positions(cases: list[CaseRecord]) -> list[int]:
    """Positions selected by more than one percentile within a set.

    Not an error -- with a heavily tied criterion two percentiles genuinely can land on one
    patch -- but it makes two panels identical, so it is surfaced rather than deduplicated.

    Args:
        cases: The cases of one set.

    Returns:
        Sorted positions that appear more than once.
    """
    seen: dict[int, int] = {}
    for case in cases:
        seen[case.position] = seen.get(case.position, 0) + 1
    return sorted(position for position, count in seen.items() if count > 1)


def render_case_table(cases: list[CaseRecord]) -> str:
    """Render the selected cases as a readable console table.

    **The bucket column is not optional.** A bucket-blind prediction was registered before
    this export ran, so which ambiguity buckets the mechanical rule happened to land on is
    itself a result and has to be visible in the console rather than buried in the file.

    Args:
        cases: All selected cases, any number of sets.

    Returns:
        A plain-text table.
    """
    header = (
        f"{'set':<4}{'key':<5}{'pct':>6}{'patch':>9}{'bucket':>8}"
        f"{'criterion':>12}{'pct value':>12}  criterion"
    )
    lines = [header, "-" * len(header)]
    for case in cases:
        lines.append(
            f"{case.set_name:<4}{case.key_prefix:<5}{case.percentile:>6.1f}"
            f"{case.patch_index:>9d}{case.bucket:>8d}"
            f"{case.criterion:>12.6f}{case.percentile_value:>12.6f}  "
            f"{case.criterion_name}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# Eligibility guards
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibilityReport:
    """Which images may be selected from, and exactly why the rest may not.

    Attributes:
        eligible: Boolean mask over images.
        counts: Guard name to the number of images it excluded. Overlapping guards each
            report their own count, so the sum can exceed ``n_excluded``.
    """

    eligible: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_eligible(self) -> int:
        """Number of images that survived every guard."""
        return int(self.eligible.sum())

    def as_dict(self) -> dict[str, Any]:
        """Render as JSON-serializable plain data for the manifest.

        Returns:
            The counts plus ``n_total``, ``n_eligible`` and ``n_excluded``.
        """
        return {
            "n_total": int(self.eligible.size),
            "n_eligible": self.n_eligible,
            "n_excluded": int(self.eligible.size - self.n_eligible),
            **{key: int(value) for key, value in self.counts.items()},
        }


def selection_eligibility(
    true_scores: np.ndarray, areas: np.ndarray
) -> EligibilityReport:
    """Guards for the selection sets (B and C), applied before any percentile.

    Two exclusions, both of them images where *no selection rule can be distinguished from
    any other*, so a showcase panel built on one would illustrate nothing:

    * **Every candidate empty** -- the ``allmt`` column of the published table. Every rule
      scores exactly 0, so the criterion is 0 by construction.
    * **All true scores tie** -- the degeneracy behind ``excl_h``/``excl_a``, where
      :func:`~probunet.evaluation.metrics.spearman_per_image` marks an image invalid
      because its candidate scores have zero variance. Head and area necessarily agree.

    The first implies the second, so their counts overlap; both are reported because they
    have different causes and the difference between them is informative (the sampler
    offered nothing, versus it offered several things that all scored alike).

    Args:
        true_scores: Soft-consensus Dice per candidate, shape ``(n_images, n_candidates)``.
        areas: Foreground pixel count per candidate, same shape.

    Returns:
        The mask and the two counts.

    Raises:
        ValueError: If the shapes disagree or are not 2-D.
    """
    true_scores = np.asarray(true_scores)
    areas = np.asarray(areas)
    if true_scores.shape != areas.shape:
        raise ValueError(
            f"true_scores {true_scores.shape} and areas {areas.shape} must have one shape"
        )
    if true_scores.ndim != 2:
        raise ValueError(f"expected (n_images, n_candidates), got {true_scores.shape}")

    all_candidates_empty = (areas <= EMPTY_CANDIDATE_AREA).all(axis=1)
    degenerate_true_tie = true_scores.max(axis=1) == true_scores.min(axis=1)
    eligible = ~(all_candidates_empty | degenerate_true_tie)
    return EligibilityReport(
        eligible=eligible,
        counts={
            "n_all_candidates_empty": int(all_candidates_empty.sum()),
            "n_degenerate_true_tie": int(degenerate_true_tie.sum()),
        },
    )


def ged_eligibility(per_variant_ged: dict[str, np.ndarray]) -> EligibilityReport:
    """Guard for Set A: exclude images whose per-image GED is undefined in either variant.

    **Only this guard applies to Set A.** The two selection guards above are defined on the
    Phase 1 head's candidate set and on soft-consensus scores -- neither of which exists
    for the Phase 2 GED comparison, whose candidates come from two different checkpoints
    entirely -- so carrying them over would mean masking one arm's data with another arm's
    degeneracy. The count of Set A images where a variant emitted only empty samples is
    *recorded* by the caller as a diagnostic instead, since it is informative without being
    a reason to exclude: two variants that both saw no lesion still have a meaningful,
    comparable GED.

    Args:
        per_variant_ged: Variant name to its per-image GED array, all the same length.

    Returns:
        The mask and one count per variant.

    Raises:
        ValueError: If the arrays differ in length or none were given.
    """
    if not per_variant_ged:
        raise ValueError("need at least one variant's GED array")
    lengths = {name: np.asarray(values).shape for name, values in per_variant_ged.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"GED arrays differ in shape: {lengths}")

    eligible = np.ones(next(iter(lengths.values())), dtype=bool)
    counts: dict[str, int] = {}
    for name, values in per_variant_ged.items():
        finite = np.isfinite(np.asarray(values, dtype=np.float64))
        counts[f"n_ged_undefined_{name}"] = int((~finite).sum())
        eligible &= finite
    return EligibilityReport(eligible=eligible, counts=counts)


# ---------------------------------------------------------------------------------
# Verification against the published results files
# ---------------------------------------------------------------------------------

SELECTION_FIGURES: tuple[tuple[str, str], ...] = (
    # (column name as printed by headroom.render_selection, dotted path into the row)
    ("random", "random.mean"),
    ("area", "area_only.mean"),
    ("head", "head.mean"),
    ("oracle", "oracle.mean"),
    ("ceil", "ceiling.mean"),
    ("edge", "head_edge_over_area"),
)
"""The published selection figures this export re-derives and checks, exactly.

Named after the columns of ``headroom.SELECTION_COLUMNS`` so a mismatch message can be
matched against the printed table without a lookup.
"""


def _dig(row: dict[str, Any], path: str) -> Any:
    """Read a dotted path out of a report row.

    Args:
        row: One bucket's row from ``headroom.per_bucket`` or from a published JSON.
        path: Dotted path, e.g. ``"random.mean"``.

    Returns:
        The value, or the string ``"<missing>"`` if any component is absent. A missing
        figure must compare unequal rather than silently match.
    """
    node: Any = row
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return "<missing>"
        node = node[part]
    return node


def _identical(left: Any, right: Any) -> bool:
    """Exact equality, with NaN equal to NaN.

    **No tolerance, deliberately.** These numbers are recomputed on the same device from
    the same weights through the same code path, so they are bit-reproducible; a tolerance
    would let a genuinely divergent sampling path pass as "close enough", which is the one
    failure this whole verification exists to catch.

    Args:
        left: Recomputed value.
        right: Published value.

    Returns:
        True if the two are the same value.
    """
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return True
    return bool(left == right)


def check_published_figures(
    recomputed: dict[str, Any],
    published: dict[str, Any],
    arm: str,
    source: str,
    figures: tuple[tuple[str, str], ...] = SELECTION_FIGURES,
) -> dict[str, Any]:
    """Assert a recomputed selection table equals the published one, figure by figure.

    **This is the central correctness check of the export.** The candidates written into
    the showcase are only the candidates behind the published numbers if recomputing those
    numbers, through the published code path, reproduces them exactly. A shape assertion
    would pass on a divergent sampling path; this cannot.

    Args:
        recomputed: ``headroom.per_bucket`` output computed in this process.
        published: The ``"buckets"`` mapping loaded from the published results JSON.
        arm: Arm label for the message, e.g. ``"phase1"``.
        source: Path of the published file, for the message.
        figures: Column name and dotted path for each figure to compare.

    Returns:
        A record of what was checked: the arm, the source, the bucket labels and the
        figure names, for the manifest.

    Raises:
        ValueError: On the first mismatch, naming the bucket, the figure, and **both**
            values; or if the two tables cover different buckets.
    """
    if set(recomputed) != set(published):
        raise ValueError(
            f"{arm}: recomputed buckets {sorted(recomputed)} do not match the buckets in "
            f"{source} ({sorted(published)}). The split or the loader differs, so no "
            "figure comparison would mean anything."
        )

    problems: list[str] = []
    for label in published:
        mine, theirs = recomputed[label], published[label]
        if int(_dig(mine, "n")) != int(_dig(theirs, "n")):
            problems.append(
                f"  bucket {label!r} n: recomputed {_dig(mine, 'n')!r} != published "
                f"{_dig(theirs, 'n')!r}"
            )
        for name, path in figures:
            left, right = _dig(mine, path), _dig(theirs, path)
            if not _identical(left, right):
                problems.append(
                    f"  bucket {label!r} {name} ({path}): recomputed {left!r} != "
                    f"published {right!r}"
                )

    if problems:
        raise ValueError(
            f"REFUSING to export: the {arm} selection table recomputed here does not match "
            f"{source} in {len(problems)} figure(s).\n"
            + "\n".join(problems)
            + "\n\nThese are computed on the same device from the same weights through the "
            "same code path, so they should be bit-identical. A difference means the "
            "candidates about to be exported are NOT the candidates behind the published "
            "table, which is the one thing this export must guarantee. Do not add a "
            "tolerance; find the divergence."
        )
    return {
        "arm": arm,
        "source": source,
        "buckets": sorted(published),
        "figures": [name for name, _ in figures],
        "matched": True,
    }


def check_published_ged(
    recomputed: dict[str, Any],
    published: dict[str, Any],
    arm: str,
    source: str,
    sample_counts: tuple[int, ...],
) -> dict[str, Any]:
    """Assert a recomputed GED report equals the published evaluation JSON, exactly.

    The Set A analogue of :func:`check_published_figures`, and it is here for the same
    reason: without it nothing would demonstrate that the samples exported for the Phase 2
    panels are the samples behind ``results/evaluation_test_*.json``.

    Args:
        recomputed: ``sampling.build_report`` output computed in this process.
        published: The published evaluation report loaded from JSON.
        arm: Arm label for the message.
        source: Path of the published file, for the message.
        sample_counts: Sample counts to compare at.

    Returns:
        A record of what was checked, for the manifest.

    Raises:
        ValueError: On the first mismatch, naming the block, the figure and both values.
    """
    problems: list[str] = []
    blocks: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "aggregate",
            recomputed["aggregate_over_all_patches"],
            published["aggregate_over_all_patches"],
        )
    ]
    mine_buckets = recomputed.get("per_bucket", {})
    their_buckets = published.get("per_bucket", {})
    if set(mine_buckets) != set(their_buckets):
        raise ValueError(
            f"{arm}: recomputed buckets {sorted(mine_buckets)} do not match {source} "
            f"({sorted(their_buckets)})"
        )
    blocks += [
        (f"bucket {label}", mine_buckets[label], their_buckets[label])
        for label in sorted(their_buckets)
    ]

    for label, mine, theirs in blocks:
        if int(_dig(mine, "n_patches")) != int(_dig(theirs, "n_patches")):
            problems.append(
                f"  {label} n_patches: recomputed {_dig(mine, 'n_patches')!r} != "
                f"published {_dig(theirs, 'n_patches')!r}"
            )
        for count in sample_counts:
            for statistic in ("mean", "median", "std"):
                path = f"ged@{count}.{statistic}"
                left, right = _dig(mine, path), _dig(theirs, path)
                if not _identical(left, right):
                    problems.append(
                        f"  {label} {path}: recomputed {left!r} != published {right!r}"
                    )

    if problems:
        raise ValueError(
            f"REFUSING to export: the {arm} GED report recomputed here does not match "
            f"{source} in {len(problems)} figure(s).\n"
            + "\n".join(problems)
            + "\n\nThe Phase 2 panels would be showing samples that did not produce the "
            "published GED table. Find the divergence; do not add a tolerance."
        )
    return {
        "arm": arm,
        "source": source,
        "sample_counts": list(sample_counts),
        "statistics": ["mean", "median", "std"],
        "matched": True,
    }


def assert_arrays_identical(
    recomputed: dict[str, np.ndarray], reference: dict[str, np.ndarray], context: str
) -> None:
    """Assert a replayed pass reproduced a first pass bit-for-bit.

    The showcase needs the *candidates themselves*, which the published measurement
    functions do not return, so they are recaptured by replaying the identical loop with a
    freshly seeded generator. That replay is only trustworthy if it lands on the same
    numbers, so every per-image column it recomputes is compared against the first pass.
    Without this, a replay that silently drifted would export candidates that never
    produced any published figure.

    Args:
        recomputed: Per-image arrays from the replay.
        reference: The same columns from the verified first pass.
        context: Label for the message.

    Raises:
        ValueError: If any column differs in shape or in a single value.
    """
    problems: list[str] = []
    for key, mine in recomputed.items():
        theirs = reference.get(key)
        if theirs is None:
            problems.append(f"  {key}: absent from the first pass")
            continue
        if mine.shape != theirs.shape:
            problems.append(f"  {key}: shape {mine.shape} != {theirs.shape}")
            continue
        differs = mine != theirs
        if np.issubdtype(mine.dtype, np.floating) and np.issubdtype(theirs.dtype, np.floating):
            # NaN != NaN, but two NaNs in the same slot are the same replay, not a drift.
            differs &= ~(np.isnan(mine) & np.isnan(theirs))
        differing = int(differs.sum())
        if differing:
            first = int(np.argmax(differs))
            problems.append(
                f"  {key}: {differing} of {mine.size} values differ; first at position "
                f"{first}, replay {mine.flat[first]!r} != first pass {theirs.flat[first]!r}"
            )
    if problems:
        raise ValueError(
            f"REFUSING to export: the {context} replay did not reproduce the verified "
            f"pass.\n" + "\n".join(problems) + "\n\nThe replay exists only to recover the "
            "candidates behind the verified numbers. If it diverges, the exported "
            "candidates are not those candidates."
        )


# ---------------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------------

MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "generated_at",
    "export_git_revision",
    "split",
    "n_patches",
    "eval_seed",
    "n_samples",
    "torch_version",
    "device",
    "variants",
    "selection_record",
    "guards",
    "assertions",
    "keys",
)
"""Every top-level manifest key the notebook is entitled to read.

Asserted by :func:`manifest_missing_keys` and by the tests, so a manifest that lost a field
fails at export time on the PC rather than at render time in a Colab session.
"""

MANIFEST_REQUIRED_VARIANT_KEYS: tuple[str, ...] = (
    "checkpoint",
    "epoch",
    "checkpoint_git_revision",
    "checkpoint_device",
    "checkpoint_torch_version",
    "latent_covariance",
    "parameter_count",
    "base_parameter_count",
    "base_parameter_sha256",
    "base_checkpoint",
)
"""Per-variant provenance. ``base_*`` is None for a plain ELBO checkpoint, which has no
frozen base -- present and null, never absent, so the notebook can print it unconditionally.
"""


def manifest_missing_keys(manifest: dict[str, Any]) -> list[str]:
    """Report manifest keys the notebook needs and the manifest lacks.

    Args:
        manifest: The manifest mapping.

    Returns:
        Sorted missing keys, top level and per variant (the latter as
        ``variants.<name>.<key>``). Empty when the manifest is complete.
    """
    missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    for name, record in (manifest.get("variants") or {}).items():
        missing += [
            f"variants.{name}.{key}"
            for key in MANIFEST_REQUIRED_VARIANT_KEYS
            if key not in record
        ]
    return sorted(missing)


def write_showcase(
    path: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> int:
    """Write the showcase ``.npz``: flat array keys plus one JSON manifest key.

    Args:
        path: Destination, normally ``data/processed/showcase.npz``.
        arrays: Flat mapping of key to array. Scalars must already be 0-d arrays of the
            intended dtype -- the dtypes are part of the format, not an accident of
            whatever numpy inferred.
        manifest: Provenance mapping, stored as a JSON string under
            :data:`MANIFEST_KEY`.

    Returns:
        The file size in bytes.

    Raises:
        ValueError: If an array key collides with the manifest key, or the manifest is
            missing a key the notebook needs.
    """
    if MANIFEST_KEY in arrays:
        raise ValueError(f"{MANIFEST_KEY!r} is reserved for the manifest")
    missing = manifest_missing_keys(manifest)
    if missing:
        raise ValueError(
            f"manifest is missing {len(missing)} key(s) the notebook reads: {missing}. "
            "The notebook prints provenance without guessing, so an absent field becomes "
            "a KeyError in a Colab session rather than here."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = dict(arrays)
    payload[MANIFEST_KEY] = np.array(json.dumps(manifest, indent=2, sort_keys=True))
    np.savez_compressed(path, **payload)
    return path.stat().st_size


def load_showcase(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a showcase export back into arrays and a manifest.

    The notebook's entry point, and the round-trip the tests exercise.

    Args:
        path: The ``.npz`` to read.

    Returns:
        ``(arrays, manifest)``. Arrays keep the dtypes they were written with; scalars come
        back as 0-d arrays.

    Raises:
        FileNotFoundError: If the file is absent.
        KeyError: If it carries no manifest.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"showcase export not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        if MANIFEST_KEY not in keys:
            raise KeyError(
                f"{path} has no {MANIFEST_KEY!r}: it was not written by write_showcase, "
                "so its provenance is unknown."
            )
        manifest = json.loads(str(data[MANIFEST_KEY].item()))
        arrays = {key: data[key] for key in keys if key != MANIFEST_KEY}
    return arrays, manifest
