"""PLUMBING-ONLY tests for the showcase export.

**These tests cannot validate the real export, and no combination of them ever could.**
The export's central claim is that the candidates it writes are the candidates that
produced ``results/consensus_selection_test.json``, and demonstrating that requires the
four checkpoints, CUDA, and the 450 MB dataset -- none of which exist on the laptop these
tests run on. What is checked here is the machinery around that claim: the percentile rule,
the eligibility guards, the exact-match verifier, the ``.npz`` round trip and the manifest
schema. Every fixture below is synthetic.

The verification that matters happens **on the PC, at export time**, inside
``scripts/export_showcase.py``: it recomputes the published tables through the published
code path and refuses to write the file if a single figure differs. Read a green run of
this module as "the plumbing is sound", never as "the export is correct".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probunet.evaluation.showcase import (
    CASE_PERCENTILES,
    SET_A_GUARD_PROVENANCE,
    SET_A_MIN_CONSENSUS_FOOTPRINT_PX,
    MANIFEST_KEY,
    MANIFEST_REQUIRED_KEYS,
    MANIFEST_REQUIRED_VARIANT_KEYS,
    SCHEMA_VERSION,
    CaseRecord,
    assert_arrays_identical,
    check_published_figures,
    check_published_ged,
    duplicate_case_positions,
    ged_eligibility,
    load_showcase,
    manifest_missing_keys,
    nearest_percentile_position,
    render_case_table,
    select_cases,
    selection_eligibility,
    set_a_eligibility,
    write_showcase,
)

# ---------------------------------------------------------------------------------
# Percentile-nearest case selection
# ---------------------------------------------------------------------------------


def test_percentile_nearest_picks_the_expected_index_on_a_hand_built_array() -> None:
    """PLUMBING-ONLY. The rule targets a percentile VALUE, not a rank.

    Values 0..10 make the median exactly 5.0 and the 90th percentile exactly 9.0, so the
    nearest-value answer is unambiguous and can be read off by hand.
    """
    values = np.arange(11, dtype=np.float64)
    eligible = np.ones(11, dtype=bool)

    assert nearest_percentile_position(values, eligible, 50.0) == (5, 5.0)
    assert nearest_percentile_position(values, eligible, 0.0) == (0, 0.0)
    assert nearest_percentile_position(values, eligible, 100.0) == (10, 10.0)
    assert nearest_percentile_position(values, eligible, 90.0) == (9, 9.0)


def test_percentile_nearest_is_not_argmin_or_argmax() -> None:
    """PLUMBING-ONLY. The reason the rule exists: an unstable extreme must not be chosen.

    One patch of a hundred sits far out at 500 -- the single-pixel-lesion artifact the rule
    is designed to avoid. ``argmax`` would take it; the 95th percentile does not, because
    one outlier among many barely moves a percentile.

    (A percentile is not immune to an outlier in a *tiny* sample -- with ten values the
    95th percentile interpolates most of the way into the outlier itself. The guarantee is
    asymptotic in the number of eligible patches, and the real export has 3019.)
    """
    values = np.concatenate([np.arange(100.0), [500.0]])
    eligible = np.ones(values.size, dtype=bool)

    position, target = nearest_percentile_position(values, eligible, 95.0)
    assert int(np.argmax(values)) == 100, "the outlier is what argmax would have taken"
    assert position == 95 and target == 95.0
    assert values[position] < 500.0


def test_percentile_nearest_resolves_a_tie_to_the_lowest_position() -> None:
    """PLUMBING-ONLY. Ties must resolve deterministically, or the cases are not repeatable.

    Every value is 2.0, so every distance to the median is 0 and the rule has nothing to
    discriminate on. It must still return one fixed answer.
    """
    values = np.full(6, 2.0)
    eligible = np.ones(6, dtype=bool)

    position, target = nearest_percentile_position(values, eligible, 50.0)
    assert (position, target) == (0, 2.0)

    # A two-way tie either side of the target resolves the same way: lowest position wins.
    values = np.array([0.0, 4.0, 6.0, 10.0])
    position, target = nearest_percentile_position(values, eligible[:4], 50.0)
    assert target == 5.0
    assert position == 1  # |4 - 5| == |6 - 5|, so the earlier row is taken


def test_percentile_nearest_on_an_even_length_array_interpolates() -> None:
    """PLUMBING-ONLY. An even-length median falls between two values, by numpy's rule.

    ``[1, 2, 3, 4]`` has median 2.5, which is not a member of the array -- so the returned
    target and the selected value differ, and the nearest member is the earlier of the two
    equidistant ones.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    eligible = np.ones(4, dtype=bool)

    position, target = nearest_percentile_position(values, eligible, 50.0)
    assert target == 2.5
    assert position == 1
    assert values[position] == 2.0


def test_percentile_is_taken_over_eligible_values_only() -> None:
    """PLUMBING-ONLY. An excluded row may neither be selected nor move the target."""
    values = np.array([100.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    eligible = np.array([False, True, True, True, True, True])

    position, target = nearest_percentile_position(values, eligible, 100.0)
    assert target == 4.0, "the excluded outlier must not set the 100th percentile"
    assert position == 5


def test_percentile_nearest_rejects_impossible_inputs() -> None:
    """PLUMBING-ONLY. Every failure is loud rather than a silently odd case."""
    values = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError, match="no eligible image"):
        nearest_percentile_position(values, np.zeros(5, dtype=bool), 50.0)
    with pytest.raises(ValueError, match=r"percentile must be in \[0, 100\]"):
        nearest_percentile_position(values, np.ones(5, dtype=bool), 101.0)
    with pytest.raises(ValueError, match="must have one shape"):
        nearest_percentile_position(values, np.ones(4, dtype=bool), 50.0)


def test_select_cases_records_percentile_patch_bucket_and_criterion() -> None:
    """PLUMBING-ONLY. Every case carries what the manifest must print about it."""
    values = np.arange(21, dtype=np.float64) - 10.0
    eligible = np.ones(21, dtype=bool)
    patch_indices = np.arange(1000, 1021, dtype=np.int64)
    buckets = np.tile(np.array([1, 2, 3, 4]), 6)[:21]

    cases = select_cases(
        values, eligible, patch_indices, buckets, "B", "head minus area"
    )

    assert [case.key_prefix for case in cases] == ["b0", "b1", "b2"]
    assert [case.percentile for case in cases] == list(CASE_PERCENTILES)
    assert cases[1].criterion == 0.0, "the median of a symmetric range"
    for case in cases:
        assert case.patch_index == int(patch_indices[case.position])
        assert case.bucket == int(buckets[case.position])
        assert case.criterion_name == "head minus area"
        assert case.as_dict()["set"] == "B"


def test_duplicate_cases_are_surfaced_not_deduplicated() -> None:
    """PLUMBING-ONLY. Two percentiles landing on one patch is reported, never hidden."""
    values = np.zeros(5)
    cases = select_cases(
        values, np.ones(5, dtype=bool), np.arange(5), np.ones(5, dtype=int), "A", "x"
    )
    assert duplicate_case_positions(cases) == [0]
    assert len(cases) == 3, "the duplicates are kept, so the caller decides what to do"


def test_render_case_table_shows_the_bucket() -> None:
    """PLUMBING-ONLY. A bucket-blind prediction was registered, so buckets must print."""
    case = CaseRecord("A", "a0", 5.0, -0.1, 7, 1234, 3, -0.0987, "ged difference")
    table = render_case_table([case])
    assert "bucket" in table
    assert "1234" in table and "3" in table and "ged difference" in table


# ---------------------------------------------------------------------------------
# Eligibility guards
# ---------------------------------------------------------------------------------


def test_selection_guards_exclude_exactly_the_intended_rows() -> None:
    """PLUMBING-ONLY. Four hand-built images, one per case the guards must separate."""
    true_scores = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # 0: every candidate empty  -> both guards
            [0.3, 0.3, 0.3, 0.3],  # 1: non-empty but all tied -> tie guard only
            [0.0, 0.2, 0.4, 0.1],  # 2: healthy                -> eligible
            [0.5, 0.5, 0.5, 0.9],  # 3: partial tie            -> eligible
        ],
        dtype=np.float32,
    )
    areas = np.array(
        [[0, 0, 0, 0], [5, 5, 5, 5], [0, 9, 20, 3], [7, 7, 7, 30]], dtype=np.int32
    )

    report = selection_eligibility(true_scores, areas)

    assert report.eligible.tolist() == [False, False, True, True]
    assert report.counts == {"n_all_candidates_empty": 1, "n_degenerate_true_tie": 2}
    assert report.as_dict() == {
        "n_total": 4,
        "n_eligible": 2,
        "n_excluded": 2,
        "n_all_candidates_empty": 1,
        "n_degenerate_true_tie": 2,
    }


def test_selection_guard_counts_overlap_and_say_so() -> None:
    """PLUMBING-ONLY. An all-empty image is also a tie, so the counts sum above the total.

    Recorded rather than collapsed: 'the sampler offered nothing' and 'it offered several
    things that scored alike' have different causes and the difference is informative.
    """
    true_scores = np.zeros((3, 4), dtype=np.float32)
    areas = np.zeros((3, 4), dtype=np.int32)

    report = selection_eligibility(true_scores, areas)

    assert report.n_eligible == 0
    assert report.counts["n_all_candidates_empty"] == 3
    assert report.counts["n_degenerate_true_tie"] == 3
    assert sum(report.counts.values()) > report.as_dict()["n_excluded"]


def test_selection_guards_reject_mismatched_inputs() -> None:
    """PLUMBING-ONLY."""
    with pytest.raises(ValueError, match="must have one shape"):
        selection_eligibility(np.zeros((2, 4)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="expected"):
        selection_eligibility(np.zeros(4), np.zeros(4))


def test_ged_guard_excludes_undefined_values_in_either_variant() -> None:
    """PLUMBING-ONLY. Exclusion is per variant, counted per variant, unioned for the mask."""
    report = ged_eligibility(
        {
            "baseline_short": np.array([0.1, np.nan, 0.3, 0.4]),
            "modernized_short": np.array([0.2, 0.2, np.inf, 0.5]),
        }
    )

    assert report.eligible.tolist() == [True, False, False, True]
    assert report.counts == {
        "n_ged_undefined_baseline_short": 1,
        "n_ged_undefined_modernized_short": 1,
    }
    assert report.as_dict()["n_excluded"] == 2


def test_ged_guard_rejects_ragged_input() -> None:
    """PLUMBING-ONLY. Arms of different lengths would pair the wrong patches."""
    with pytest.raises(ValueError, match="differ in shape"):
        ged_eligibility({"a": np.zeros(3), "b": np.zeros(4)})
    with pytest.raises(ValueError, match="at least one variant"):
        ged_eligibility({})


def test_set_a_display_guards_exclude_exactly_the_intended_rows() -> None:
    """PLUMBING-ONLY. Five hand-built images, one per case the three guards separate.

    Row 3 is the ``a1`` shape: GED defined, both arms non-empty, and a target far too small
    to see. It passes the originally specified guard and must still be excluded -- that is
    the whole reason the footprint guard exists.
    """
    report = set_a_eligibility(
        per_variant_ged={
            "baseline_short": np.array([0.1, np.nan, 0.3, 0.4, 0.5]),
            "modernized_short": np.array([0.2, 0.2, 0.3, 0.4, 0.5]),
        },
        per_variant_nonempty_samples={
            "baseline_short": np.array([6, 6, 0, 4, 6]),
            "modernized_short": np.array([6, 6, 6, 1, 6]),
        },
        consensus_footprint=np.array([100, 100, 100, 4, 100]),
        min_footprint=25,
        min_nonempty_samples=2,
    )

    assert report.eligible.tolist() == [True, False, False, False, True]
    assert report.counts["n_ged_undefined_baseline_short"] == 1
    assert report.counts["n_too_few_nonempty_samples_baseline_short"] == 1
    assert report.counts["n_too_few_nonempty_samples_modernized_short"] == 1
    assert report.counts["n_target_too_small_to_see"] == 1
    assert report.counts["min_consensus_footprint_px"] == 25
    assert report.counts["min_nonempty_samples_per_arm"] == 2
    assert report.as_dict()["n_eligible"] == 2


def test_the_originally_specified_guard_would_not_have_caught_a1() -> None:
    """PLUMBING-ONLY. The claim the disclosure rests on, asserted rather than asserted-in-prose.

    ``a1``'s real shape: both arms offered a non-empty sample, so "at least one non-empty
    sample per arm" passes it. Only the footprint threshold removes it.
    """
    a1 = dict(
        per_variant_ged={"baseline_short": np.array([0.1]),
                         "modernized_short": np.array([0.09])},
        per_variant_nonempty_samples={"baseline_short": np.array([4]),
                                      "modernized_short": np.array([1])},
        consensus_footprint=np.array([4]),
    )
    # The guard as originally specified: >= 1 non-empty per arm, no footprint threshold.
    as_specified = set_a_eligibility(**a1, min_footprint=0, min_nonempty_samples=1)
    assert as_specified.eligible.tolist() == [True], (
        "the originally specified guard passes a1, which is exactly why it was the wrong "
        "guard -- it targeted 'both arms empty', not 'the target is too small to see'"
    )
    # The footprint guard alone is what removes it.
    with_footprint = set_a_eligibility(**a1, min_footprint=25, min_nonempty_samples=1)
    assert with_footprint.eligible.tolist() == [False]
    assert with_footprint.counts["n_target_too_small_to_see"] == 1


def test_set_a_guards_are_symmetric_between_the_arms() -> None:
    """PLUMBING-ONLY. Swapping the arms cannot change who is eligible.

    The disclosure's central claim is that a guard added after the fact still cannot favour
    an arm. That is only true if the thresholds are applied identically to each, so it is
    asserted rather than argued.
    """
    ged = {"baseline_short": np.array([0.1, 0.2, 0.3]),
           "modernized_short": np.array([0.4, 0.5, 0.6])}
    samples = {"baseline_short": np.array([5, 1, 9]),
               "modernized_short": np.array([1, 7, 9])}
    footprint = np.array([100, 100, 100])

    forward = set_a_eligibility(ged, samples, footprint)
    swapped = set_a_eligibility(
        {"baseline_short": ged["modernized_short"],
         "modernized_short": ged["baseline_short"]},
        {"baseline_short": samples["modernized_short"],
         "modernized_short": samples["baseline_short"]},
        footprint,
    )
    assert forward.eligible.tolist() == swapped.eligible.tolist() == [False, False, True]


def test_set_a_guards_refuse_an_arm_missing_from_either_mapping() -> None:
    """PLUMBING-ONLY. A guard skipped for one arm is a guard that is no longer symmetric."""
    with pytest.raises(ValueError, match="not symmetric"):
        set_a_eligibility(
            per_variant_ged={"a": np.zeros(2), "b": np.zeros(2)},
            per_variant_nonempty_samples={"a": np.zeros(2)},
            consensus_footprint=np.zeros(2),
        )
    with pytest.raises(ValueError, match="consensus_footprint has shape"):
        set_a_eligibility(
            per_variant_ged={"a": np.zeros(2)},
            per_variant_nonempty_samples={"a": np.zeros(2)},
            consensus_footprint=np.zeros(3),
        )


def test_the_guard_disclosure_records_the_order_and_the_missed_failure_mode() -> None:
    """PLUMBING-ONLY. The disclosure travels with the export, so it cannot be lost."""
    assert "after case a1" in SET_A_GUARD_PROVENANCE["added"]
    assert "NOT before" in SET_A_GUARD_PROVENANCE["added"]
    assert "cherry-pick" in SET_A_GUARD_PROVENANCE["why_the_order_is_disclosed"]
    assert "wrong failure mode" in SET_A_GUARD_PROVENANCE["what_the_specified_guard_missed"]
    assert SET_A_GUARD_PROVENANCE["a1_for_the_record"]["grader_union_footprint_px"] == 4
    assert SET_A_MIN_CONSENSUS_FOOTPRINT_PX > 4, (
        "the threshold must actually exclude the case that motivated it"
    )


# ---------------------------------------------------------------------------------
# The exact-match verifier
# ---------------------------------------------------------------------------------


def _row(random: float, area: float, head: float, oracle: float, ceiling: float,
         edge: float, n: int = 10) -> dict:
    """Build one bucket row in the shape headroom.per_bucket produces."""
    return {
        "n": n,
        "random": {"mean": random},
        "area_only": {"mean": area},
        "head": {"mean": head},
        "oracle": {"mean": oracle},
        "ceiling": {"mean": ceiling},
        "head_edge_over_area": edge,
    }


def test_verifier_accepts_an_exact_match() -> None:
    """PLUMBING-ONLY."""
    table = {"1 grader": _row(0.1, 0.2, 0.3, 0.4, 0.4, 0.1), "all": _row(0.2, 0.3, 0.4, 0.5, 0.6, 0.1)}
    record = check_published_figures(table, json.loads(json.dumps(table)), "phase1", "x.json")
    assert record["matched"] is True
    assert record["figures"] == ["random", "area", "head", "oracle", "ceil", "edge"]


def test_verifier_rejects_a_difference_in_the_last_bit_and_reports_both_values() -> None:
    """PLUMBING-ONLY. No tolerance: a one-ULP difference means a divergent sampling path."""
    published = {"all": _row(0.2, 0.3, 0.4, 0.5, 0.6, 0.1)}
    drifted = {"all": _row(0.2, 0.3, np.nextafter(0.4, 1.0), 0.5, 0.6, 0.1)}

    with pytest.raises(ValueError) as raised:
        check_published_figures(drifted, published, "phase1", "results/x.json")

    message = str(raised.value)
    assert "head" in message and "0.4" in message
    assert "results/x.json" in message
    assert "Do not add a tolerance" in message


def test_verifier_rejects_a_missing_figure_rather_than_skipping_it() -> None:
    """PLUMBING-ONLY. An absent figure must compare unequal, never silently match."""
    published = {"all": _row(0.2, 0.3, 0.4, 0.5, 0.6, 0.1)}
    incomplete = {"all": {key: value for key, value in published["all"].items()
                          if key != "head_edge_over_area"}}

    with pytest.raises(ValueError, match="edge"):
        check_published_figures(incomplete, published, "phase1", "x.json")


def test_verifier_rejects_a_different_bucket_set_and_a_different_n() -> None:
    """PLUMBING-ONLY. A different split or loader invalidates every figure comparison."""
    published = {"all": _row(0.2, 0.3, 0.4, 0.5, 0.6, 0.1, n=3019)}
    with pytest.raises(ValueError, match="do not match the buckets"):
        check_published_figures({"1 grader": published["all"]}, published, "p", "x.json")
    with pytest.raises(ValueError, match=r"n: recomputed"):
        check_published_figures(
            {"all": _row(0.2, 0.3, 0.4, 0.5, 0.6, 0.1, n=3018)}, published, "p", "x.json"
        )


def _ged_block(mean: float, n_patches: int = 10) -> dict:
    """Build one GED block in the shape sampling.build_report produces."""
    return {
        "n_patches": n_patches,
        **{
            f"ged@{count}": {"mean": mean + count / 1000, "median": mean, "std": 0.1}
            for count in (1, 4, 8, 16)
        },
    }


def test_ged_verifier_accepts_a_match_and_rejects_a_drift() -> None:
    """PLUMBING-ONLY. Set A's samples must come from the published evaluation run."""
    published = {
        "aggregate_over_all_patches": _ged_block(0.3),
        "per_bucket": {"1": _ged_block(0.4, 4), "2": _ged_block(0.2, 6)},
    }
    record = check_published_ged(
        json.loads(json.dumps(published)), published, "baseline_short", "e.json", (1, 4, 8, 16)
    )
    assert record["matched"] is True

    drifted = json.loads(json.dumps(published))
    drifted["per_bucket"]["2"]["ged@16"]["mean"] = 0.201
    with pytest.raises(ValueError) as raised:
        check_published_ged(drifted, published, "baseline_short", "e.json", (1, 4, 8, 16))
    assert "bucket 2 ged@16.mean" in str(raised.value)


def test_replay_equality_check_reports_the_first_differing_position() -> None:
    """PLUMBING-ONLY. The replay is only trustworthy if it lands on the same numbers."""
    reference = {"head": np.array([0.1, 0.2, 0.3]), "index": np.array([7, 8, 9])}
    assert_arrays_identical(dict(reference), reference, "unit")

    drifted = {"head": np.array([0.1, 0.25, 0.3]), "index": np.array([7, 8, 9])}
    with pytest.raises(ValueError) as raised:
        assert_arrays_identical(drifted, reference, "unit")
    assert "first at position 1" in str(raised.value)

    with pytest.raises(ValueError, match="absent from the first pass"):
        assert_arrays_identical({"nope": np.zeros(3)}, reference, "unit")


# ---------------------------------------------------------------------------------
# File format and manifest schema
# ---------------------------------------------------------------------------------


def _synthetic_manifest() -> dict:
    """A manifest carrying every key the notebook reads, with plausible values."""
    variant = {key: "x" for key in MANIFEST_REQUIRED_VARIANT_KEYS}
    variant.update(
        {
            "checkpoint": "runs/selection-head/checkpoints/best.pt",
            "epoch": 29,
            "parameter_count": 27499098 + 150000,
            "base_parameter_count": 27499098,
            "base_parameter_sha256": "b1887d8d242f" + "0" * 52,
            "latent_covariance": "diagonal",
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-08-10T00:00:00+00:00",
        "export_git_revision": "abc1234",
        "split": "test",
        "n_patches": 3019,
        "eval_seed": 2018,
        "n_samples": 16,
        "torch_version": "2.11.0+cu128",
        "device": "cuda",
        "variants": {"selection_head": variant},
        "selection_record": [
            CaseRecord("A", "a0", 5.0, -0.1, 3, 1234, 2, -0.11, "ged difference").as_dict()
        ],
        "guards": {"set_a": {"n_eligible": 3019}, "set_b_and_c": {"n_eligible": 2654}},
        "assertions": [{"arm": "selection_head", "source": "results/x.json", "matched": True}],
        "keys": {"set_a": ["a0_image"], "set_b": [], "set_c": ["c_pred_scores"]},
    }


def test_manifest_schema_lists_every_key_the_notebook_reads() -> None:
    """PLUMBING-ONLY. A manifest that lost a field must fail on the PC, not in Colab."""
    manifest = _synthetic_manifest()
    assert manifest_missing_keys(manifest) == []

    for key in MANIFEST_REQUIRED_KEYS:
        incomplete = {name: value for name, value in manifest.items() if name != key}
        assert manifest_missing_keys(incomplete) == [key]

    stripped = json.loads(json.dumps(manifest))
    del stripped["variants"]["selection_head"]["base_parameter_sha256"]
    assert manifest_missing_keys(stripped) == [
        "variants.selection_head.base_parameter_sha256"
    ]


def test_showcase_round_trip_preserves_arrays_dtypes_and_manifest(tmp_path: Path) -> None:
    """PLUMBING-ONLY. What is written is what the notebook reads back, dtypes included.

    The dtypes are part of the format rather than an accident: image float32, masks and
    samples and candidates uint8, consensus float32, scores float32.
    """
    rng = np.random.default_rng(0)
    arrays = {
        "a0_image": rng.standard_normal((8, 8)).astype(np.float32),
        "a0_masks": (rng.random((4, 8, 8)) > 0.5).astype(np.uint8),
        "a0_consensus": (rng.integers(0, 5, (8, 8)) / 4).astype(np.float32),
        "a0_samples_baseline_short": (rng.random((16, 8, 8)) > 0.5).astype(np.uint8),
        "a0_bucket": np.asarray(3, dtype=np.int64),
        "b0_candidates": (rng.random((16, 8, 8)) > 0.5).astype(np.uint8),
        "b0_true_scores": rng.random(16).astype(np.float32),
        "b0_pred_scores": rng.random(16).astype(np.float32),
        "b0_areas": rng.integers(0, 100, 16).astype(np.int32),
        "b0_pick_head": np.asarray(4, dtype=np.int64),
        "b0_arbitrary_unselected": np.asarray(0, dtype=np.int64),
        "c_pred_scores": rng.random(64).astype(np.float32),
        "c_true_scores": rng.random(64).astype(np.float32),
        "c_areas": rng.integers(0, 100, 64).astype(np.int32),
        "c_buckets": rng.integers(1, 5, 4).astype(np.int8),
    }
    manifest = _synthetic_manifest()
    target = tmp_path / "showcase.npz"

    size = write_showcase(target, arrays, manifest)
    assert size > 0 and target.exists()

    reloaded, reloaded_manifest = load_showcase(target)

    assert set(reloaded) == set(arrays)
    for key, value in arrays.items():
        assert reloaded[key].dtype == value.dtype, key
        assert reloaded[key].shape == value.shape, key
        assert np.array_equal(reloaded[key], value), key
    assert reloaded_manifest == manifest
    assert reloaded["a0_bucket"].item() == 3
    assert reloaded["b0_arbitrary_unselected"].item() == 0


def test_write_showcase_refuses_an_incomplete_manifest(tmp_path: Path) -> None:
    """PLUMBING-ONLY. The schema is enforced at write time, on the machine that can fix it."""
    manifest = _synthetic_manifest()
    del manifest["assertions"]
    with pytest.raises(ValueError, match="missing 1 key"):
        write_showcase(tmp_path / "s.npz", {"a0_image": np.zeros((2, 2))}, manifest)


def test_write_showcase_refuses_to_shadow_the_manifest_key(tmp_path: Path) -> None:
    """PLUMBING-ONLY."""
    with pytest.raises(ValueError, match="reserved for the manifest"):
        write_showcase(
            tmp_path / "s.npz", {MANIFEST_KEY: np.zeros(1)}, _synthetic_manifest()
        )


def test_load_showcase_fails_loudly_on_a_missing_or_foreign_file(tmp_path: Path) -> None:
    """PLUMBING-ONLY. A file without provenance is refused rather than half-trusted."""
    with pytest.raises(FileNotFoundError):
        load_showcase(tmp_path / "absent.npz")

    foreign = tmp_path / "foreign.npz"
    np.savez_compressed(foreign, a0_image=np.zeros((2, 2)))
    with pytest.raises(KeyError, match=MANIFEST_KEY):
        load_showcase(foreign)
