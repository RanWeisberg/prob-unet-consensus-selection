"""Tests for GED, oracle Dice, Hungarian matching and the degenerate baselines.

The GED cases are small enough to verify by hand, and each asserts the three components
separately rather than only the total, so a sign or normalization error is localized
instead of merely detected. Two cases have entirely empty ground truth, which is the
edge the ``d = 0 for both empty`` rule exists for.

Notation: ``A`` is a non-empty mask, ``E`` an empty one, ``P`` a single pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

from probunet.evaluation.metrics import consensus_ceiling
from probunet.evaluation.headroom import (
    GRADER_COLUMNS,
    MODEL_COLUMNS,
    measure_ceilings,
    per_bucket,
    render,
)
from probunet.evaluation.metrics import (
    aggregate_over_graders,
    dice,
    distance,
    emptiest_sample_index,
    generalized_energy_distance,
    hungarian_assignment,
    hungarian_matched_iou,
    oracle_dice,
    pairwise_dice,
    pairwise_iou,
    per_grader_oracle_dice,
    random_sample_dice,
    selected_sample_dice,
    summarize,
)
from probunet.evaluation.sampling import (
    SamplingConfig,
    build_report,
    collect_per_patch_metrics,
    draw_prior_samples,
)
from probunet.variants import ProbUNetVariant

SIZE = 8


def block_mask(rows: int = 2, columns: int = 2, offset: int = 0) -> torch.Tensor:
    """A rectangular non-empty mask of shape (SIZE, SIZE)."""
    mask = torch.zeros(SIZE, SIZE, dtype=torch.uint8)
    mask[offset : offset + rows, offset : offset + columns] = 1
    return mask


EMPTY = torch.zeros(SIZE, SIZE, dtype=torch.uint8)
A = block_mask()
B_MASK = block_mask(rows=3, columns=1, offset=4)


def pixel_mask(row: int = 0, column: int = 0) -> torch.Tensor:
    """A single-pixel mask."""
    mask = torch.zeros(SIZE, SIZE, dtype=torch.uint8)
    mask[row, column] = 1
    return mask


def stack(masks: list[torch.Tensor]) -> torch.Tensor:
    """Stack 2-D masks into a (1, k, H, W) batch of one patch."""
    return torch.stack(masks).unsqueeze(0)


# --------------------------------------------------------------------------- #
# distance and pairwise helpers
# --------------------------------------------------------------------------- #
def test_distance_is_one_minus_iou() -> None:
    """The GED distance is 0 for both-empty and 1 for empty-vs-anything."""
    assert distance(EMPTY, EMPTY).item() == 0.0
    assert distance(A, A).item() == 0.0
    assert distance(EMPTY, A).item() == 1.0


def test_pairwise_iou_shape_and_values() -> None:
    """Pairwise IoU produces an (B, n, m) matrix consistent with binary_iou."""
    samples = stack([A, EMPTY])
    graders = stack([A, A, EMPTY, B_MASK])
    matrix = pairwise_iou(samples, graders)
    assert matrix.shape == (1, 2, 4)
    assert matrix[0, 0, 0].item() == 1.0          # A vs A
    assert matrix[0, 0, 2].item() == 0.0          # A vs empty
    assert matrix[0, 1, 2].item() == 1.0          # empty vs empty
    assert matrix[0, 0, 3].item() == 0.0          # A vs disjoint B


def test_pairwise_dice_matches_dice() -> None:
    """Pairwise Dice agrees with the scalar primitive."""
    samples = stack([A])
    graders = stack([A, EMPTY])
    matrix = pairwise_dice(samples, graders)
    assert matrix[0, 0, 0].item() == pytest.approx(dice(A, A).item())
    assert matrix[0, 0, 1].item() == pytest.approx(dice(A, EMPTY).item())


def test_pairwise_validates_shapes() -> None:
    """Rank and batch/spatial mismatches are rejected."""
    with pytest.raises(ValueError, match="expected"):
        pairwise_iou(A.unsqueeze(0), stack([A]))
    with pytest.raises(ValueError, match="incompatible"):
        pairwise_iou(stack([A]), torch.zeros(2, 1, SIZE, SIZE, dtype=torch.uint8))


# --------------------------------------------------------------------------- #
# GED on hand-verifiable cases
# --------------------------------------------------------------------------- #
def test_ged_perfect_single_sample() -> None:
    """Case 1: sample equals all four identical graders. Every term is 0."""
    result = generalized_energy_distance(stack([A]), stack([A, A, A, A]))
    assert result["d_ys"].item() == pytest.approx(0.0)
    assert result["d_ss"].item() == pytest.approx(0.0)
    assert result["d_yy"].item() == pytest.approx(0.0)
    assert result["d_squared"].item() == pytest.approx(0.0)


def test_ged_all_empty_everywhere() -> None:
    """Case 2: empty ground truth and an empty sample. d = 0 throughout, so GED = 0.

    Without the both-empty convention this would be 0/0 and the metric would punish a
    correct prediction of lesion absence.
    """
    result = generalized_energy_distance(stack([EMPTY]), stack([EMPTY] * 4))
    assert result["d_ys"].item() == pytest.approx(0.0)
    assert result["d_yy"].item() == pytest.approx(0.0)
    assert result["d_squared"].item() == pytest.approx(0.0)


def test_ged_empty_ground_truth_nonempty_sample() -> None:
    """Case 3: empty ground truth, a one-pixel sample. GED = 2 * 1 - 0 - 0 = 2."""
    result = generalized_energy_distance(stack([pixel_mask()]), stack([EMPTY] * 4))
    assert result["d_ys"].item() == pytest.approx(1.0)
    assert result["d_ss"].item() == pytest.approx(0.0)
    assert result["d_yy"].item() == pytest.approx(0.0)
    assert result["d_squared"].item() == pytest.approx(2.0)


def test_ged_split_graders_single_sample() -> None:
    """Case 4: graders [E,E,A,A], sample [A].

    cross  = (1 + 1 + 0 + 0) / 4       = 0.5
    sample = d(A,A) / 1                = 0.0
    grader = 8 / 16                    = 0.5
    GED    = 2*0.5 - 0.0 - 0.5         = 0.5
    """
    result = generalized_energy_distance(stack([A]), stack([EMPTY, EMPTY, A, A]))
    assert result["d_ys"].item() == pytest.approx(0.5)
    assert result["d_ss"].item() == pytest.approx(0.0)
    assert result["d_yy"].item() == pytest.approx(0.5)
    assert result["d_squared"].item() == pytest.approx(0.5)


def test_ged_matched_distribution_is_zero() -> None:
    """Case 5: graders [E,E,A,A], samples [A,E] -- the sample distribution matches.

    cross  = (1 + 1 + 0 + 0 + 0 + 0 + 1 + 1) / 8 = 0.5
    sample = (0 + 1 + 1 + 0) / 4                  = 0.5
    grader = 8 / 16                               = 0.5
    GED    = 2*0.5 - 0.5 - 0.5                    = 0.0

    A distribution that reproduces the graders exactly scores 0, which is the property
    that makes GED a distribution metric rather than an accuracy.
    """
    result = generalized_energy_distance(stack([A, EMPTY]), stack([EMPTY, EMPTY, A, A]))
    assert result["d_ys"].item() == pytest.approx(0.5)
    assert result["d_ss"].item() == pytest.approx(0.5)
    assert result["d_yy"].item() == pytest.approx(0.5)
    assert result["d_squared"].item() == pytest.approx(0.0)


def test_ged_self_distance_includes_the_diagonal() -> None:
    """The n^2 normalization keeps i == j, whose distance is 0.

    Two identical samples give d_ss = 0; two maximally different ones give 0.5, not the
    1.0 an off-diagonal-only mean would produce.
    """
    same = generalized_energy_distance(stack([A, A]), stack([A] * 4))
    assert same["d_ss"].item() == pytest.approx(0.0)

    different = generalized_energy_distance(stack([A, B_MASK]), stack([A] * 4))
    assert different["d_ss"].item() == pytest.approx(0.5)


def test_ged_equals_squared_distribution_difference_for_disjoint_modes() -> None:
    """With disjoint modes the estimator reduces to ``sum_i (p_i - q_i)^2``.

    For pairwise-disjoint masks the distance behaves like the discrete metric
    ``d(x,y) = [x != y]``, and the estimator collapses to the squared difference of the
    two empirical mode distributions. Graders ``[A,A,B,B]`` give ``p = (0.5, 0.5)`` and
    samples ``[A,A,A,B]`` give ``q = (0.75, 0.25)``, so::

        d_ys = 8/16 = 0.5,  d_ss = 6/16 = 0.375,  d_yy = 8/16 = 0.5
        GED  = 2(0.5) - 0.375 - 0.5 = 0.125 = (0.25)^2 + (0.25)^2

    This identity is also why the estimator cannot go negative in this regime.
    """
    result = generalized_energy_distance(
        stack([A, A, A, B_MASK]), stack([A, A, B_MASK, B_MASK])
    )
    assert result["d_ys"].item() == pytest.approx(0.5)
    assert result["d_ss"].item() == pytest.approx(0.375)
    assert result["d_yy"].item() == pytest.approx(0.5)
    assert result["d_squared"].item() == pytest.approx(0.125)
    assert result["d_squared"].item() == pytest.approx(0.25**2 + 0.25**2)


def test_ged_is_not_clamped_and_stays_nonnegative_here() -> None:
    """The estimator is returned raw; the negative counter is a safety net.

    Because the self-distance sums keep their zero diagonals, this estimator is the
    energy distance between the two *empirical* distributions rather than an unbiased
    population estimate, and no negative case could be constructed for it. Values are
    still reported unclamped, and ``summarize`` counts negatives, so if a real run ever
    produces one it is visible rather than hidden.
    """
    torch.manual_seed(0)
    for _ in range(20):
        samples = (torch.rand(4, 6, SIZE, SIZE) > 0.8).to(torch.uint8)
        graders = (torch.rand(4, 4, SIZE, SIZE) > 0.8).to(torch.uint8)
        values = generalized_energy_distance(samples, graders)["d_squared"]
        assert torch.all(values >= -1e-6), values


def test_ged_is_per_patch() -> None:
    """Each patch in a batch is scored independently."""
    samples = torch.stack([stack([A])[0], stack([pixel_mask()])[0]])
    graders = torch.stack([stack([A] * 4)[0], stack([EMPTY] * 4)[0]])
    result = generalized_energy_distance(samples, graders)
    assert result["d_squared"].shape == (2,)
    assert result["d_squared"][0].item() == pytest.approx(0.0)
    assert result["d_squared"][1].item() == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Single-sample quality
# --------------------------------------------------------------------------- #
def test_oracle_dice_picks_the_best_sample() -> None:
    """With one perfect sample among two, oracle Dice is 1.0 by inspection."""
    samples = stack([A, B_MASK])
    graders = stack([A, A, A, A])
    assert oracle_dice(samples, graders).item() == pytest.approx(1.0)

    # A and B are disjoint, so the second sample scores 0 against every grader.
    expected_random = (1.0 + 0.0) / 2
    assert random_sample_dice(samples, graders).item() == pytest.approx(expected_random)
    # Oracle is the ceiling, random the floor.
    assert oracle_dice(samples, graders) > random_sample_dice(samples, graders)


def test_oracle_dice_reflects_the_grader_aggregation() -> None:
    """A sample matching only some graders scores differently under mean vs min."""
    samples = stack([A])
    graders = stack([A, A, EMPTY, EMPTY])
    assert oracle_dice(samples, graders, "mean").item() == pytest.approx(0.5)
    assert oracle_dice(samples, graders, "min").item() == pytest.approx(0.0)
    assert oracle_dice(samples, graders, "max").item() == pytest.approx(1.0)


def test_per_grader_oracle_is_at_least_the_single_best_sample() -> None:
    """Letting every grader pick its own sample can only help."""
    samples = stack([A, B_MASK])
    graders = stack([A, A, B_MASK, B_MASK])
    single = oracle_dice(samples, graders).item()
    per_grader = per_grader_oracle_dice(samples, graders).item()
    assert per_grader == pytest.approx(1.0)
    assert per_grader >= single
    assert single == pytest.approx(0.5)


def test_selected_sample_dice_scores_the_chosen_index() -> None:
    """Selection is explicit, so any rule can be scored the same way."""
    samples = stack([A, B_MASK])
    graders = stack([A, A, A, A])
    assert selected_sample_dice(samples, graders, torch.tensor([0])).item() == pytest.approx(1.0)
    assert selected_sample_dice(samples, graders, torch.tensor([1])).item() == pytest.approx(0.0)


def test_aggregate_over_graders_options() -> None:
    """Every documented aggregation works and an unknown one is rejected."""
    scores = torch.tensor([[[0.0, 0.5, 1.0, 1.0]]])
    assert aggregate_over_graders(scores, "mean").item() == pytest.approx(0.625)
    assert aggregate_over_graders(scores, "median").item() == pytest.approx(0.75)
    assert aggregate_over_graders(scores, "min").item() == pytest.approx(0.0)
    assert aggregate_over_graders(scores, "max").item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="aggregate must be"):
        aggregate_over_graders(scores, "geometric")


# --------------------------------------------------------------------------- #
# Hungarian matching
# --------------------------------------------------------------------------- #
def test_hungarian_recovers_a_known_permutation() -> None:
    """Samples that are a shuffle of the graders match perfectly, one for one."""
    masks = [block_mask(offset=index) for index in range(4)]
    permutation = [2, 0, 3, 1]
    samples = stack([masks[index] for index in permutation])
    graders = stack(masks)

    assert hungarian_matched_iou(samples, graders).item() == pytest.approx(1.0)
    rows, columns = hungarian_assignment(pairwise_iou(samples, graders))[0]
    recovered = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    for sample_index, grader_index in enumerate(permutation):
        assert recovered[sample_index] == grader_index


def test_hungarian_forces_one_to_one() -> None:
    """One good sample cannot be matched to every grader.

    Two graders, two samples, but only one sample is any good: matching must pair the
    other sample with the remaining grader, so the mean is 0.5 rather than 1.0.
    """
    samples = stack([A, EMPTY])
    graders = stack([A, B_MASK])
    assert hungarian_matched_iou(samples, graders).item() == pytest.approx(0.5)


def test_hungarian_with_fewer_samples_than_graders() -> None:
    """n < m matches only min(n, m) pairs, so n = 1 is the best single grader."""
    samples = stack([A])
    graders = stack([A, B_MASK, EMPTY, EMPTY])
    assert hungarian_matched_iou(samples, graders).item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Degenerate baselines
# --------------------------------------------------------------------------- #
def test_empty_predictor_scores_well_when_most_graders_are_empty() -> None:
    """The trap this baseline documents.

    On a bucket-1 patch -- three of four graders empty, which is 33% of the dataset --
    an all-empty prediction gets Dice 0.75. Any single-sample number must be read
    against this, not against zero.
    """
    graders = stack([A, EMPTY, EMPTY, EMPTY])
    empty_sample = stack([EMPTY])
    assert random_sample_dice(empty_sample, graders).item() == pytest.approx(0.75)
    assert oracle_dice(empty_sample, graders).item() == pytest.approx(0.75)


def test_empty_predictor_ged_is_sample_count_independent() -> None:
    """More empty samples change nothing: every component is unchanged."""
    graders = stack([A, EMPTY, EMPTY, EMPTY])
    one = generalized_energy_distance(stack([EMPTY]), graders)
    eight = generalized_energy_distance(stack([EMPTY] * 8), graders)
    for key in ("d_ys", "d_ss", "d_yy", "d_squared"):
        assert one[key].item() == pytest.approx(eight[key].item()), key


def test_emptiest_sample_index() -> None:
    """The trivial selection rule picks the smallest-area sample, ties to lowest index."""
    samples = stack([block_mask(rows=3, columns=3), pixel_mask(), EMPTY])
    assert emptiest_sample_index(samples).item() == 2
    assert emptiest_sample_index(stack([EMPTY, EMPTY])).item() == 0


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def test_summarize_reports_spread_and_negatives() -> None:
    """The paper's LIDC figure is a distribution, so spread is reported, not just a mean."""
    stats = summarize(np.array([-0.5, 0.0, 0.5, 1.0, 1.5]))
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["median"] == pytest.approx(0.5)
    assert stats["q25"] == pytest.approx(0.0)
    assert stats["q75"] == pytest.approx(1.0)
    assert stats["iqr"] == pytest.approx(1.0)
    assert stats["n_negative"] == 1


def test_summarize_handles_empty_and_nan() -> None:
    """An empty bucket yields n = 0 rather than a crash."""
    assert summarize(np.array([]))["n"] == 0
    assert summarize(np.array([np.nan, np.nan]))["n"] == 0
    assert summarize(np.array([np.nan, 1.0]))["n"] == 1


# --------------------------------------------------------------------------- #
# Sampling and the report, end to end on a tiny model
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_model_and_loader(tmp_path):
    """A tiny trained-shaped model plus a deterministic eval loader over synthetic data."""
    from probunet.data.lidc import DataConfig, build_data
    from probunet.data.splits import generate_split
    from probunet.model.prob_unet import ProbUNet, ProbUNetConfig

    npz = tmp_path / "lidc.npz"
    rng = np.random.default_rng(0)
    n_patches = 24
    series = np.array([f"s{i // 4:03d}" for i in range(n_patches)], dtype=np.str_)
    images = rng.random((n_patches, 16, 16), dtype=np.float32)
    masks = np.zeros((n_patches, 4, 16, 16), dtype=np.uint8)
    for row in range(n_patches):
        for slot in range((row % 4) + 1):
            masks[row, slot, : 2 + slot, : 2 + (row % 3)] = 1
    np.savez_compressed(npz, images=images, masks=masks, series_uid=series)
    split = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=split)

    torch.manual_seed(0)
    model = ProbUNet(ProbUNetConfig(latent_dim=2, base_channels=4, num_downs=2, convs_per_scale=1))
    data = build_data(DataConfig(npz_path=npz, split_path=split, batch_size=4))
    return model, data


def test_draw_prior_samples_shapes_and_determinism(tiny_model_and_loader) -> None:
    """Sampling is reproducible under a seeded generator."""
    model, data = tiny_model_and_loader
    batch = next(iter(data.loaders["val"]))
    first = draw_prior_samples(model, batch["image"], 5, torch.Generator().manual_seed(1))
    second = draw_prior_samples(model, batch["image"], 5, torch.Generator().manual_seed(1))
    third = draw_prior_samples(model, batch["image"], 5, torch.Generator().manual_seed(2))

    assert first.shape == (batch["image"].shape[0], 5, 16, 16)
    assert first.dtype == torch.uint8
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_collect_and_report_end_to_end(tiny_model_and_loader) -> None:
    """The whole evaluation path runs and produces the documented keys."""
    model, data = tiny_model_and_loader
    config = SamplingConfig(sample_counts=(1, 2, 4), seed=7)
    variant = ProbUNetVariant(model, generator=torch.Generator().manual_seed(config.seed))
    per_patch = collect_per_patch_metrics(
        variant, data.loaders["val"], config, torch.device("cpu")
    )

    expected = len(data.datasets["val"])
    for key, values in per_patch.items():
        assert values.shape == (expected,), key
    for count in config.sample_counts:
        assert f"ged@{count}" in per_patch
        assert f"oracle_dice@{count}" in per_patch
        assert f"hungarian_iou@{count}" in per_patch
        assert f"emptiest_sample_dice@{count}" in per_patch
    assert "empty_ged" in per_patch

    report = build_report(per_patch, config)
    aggregate = report["aggregate_over_all_patches"]
    assert aggregate["n_patches"] == expected
    assert aggregate["ged@4"]["n"] == expected
    assert report["per_bucket"], "no buckets were populated"
    for block in report["per_bucket"].values():
        assert block["n_patches"] > 0
        assert block["lesion_area_median_px"] is not None


def test_evaluation_is_reproducible(tiny_model_and_loader) -> None:
    """Same seed gives identical metrics; a different seed does not."""
    model, data = tiny_model_and_loader
    device = torch.device("cpu")

    def run(seed: int) -> dict:
        """Evaluate with a variant seeded by ``seed``."""
        config = SamplingConfig(sample_counts=(4,), seed=seed)
        variant = ProbUNetVariant(model, generator=torch.Generator().manual_seed(seed))
        return collect_per_patch_metrics(variant, data.loaders["val"], config, device)

    first, second, other = run(11), run(11), run(12)

    assert np.array_equal(first["ged@4"], second["ged@4"])
    assert not np.array_equal(first["ged@4"], other["ged@4"])


def test_oracle_is_never_below_random_in_practice(tiny_model_and_loader) -> None:
    """Oracle selection is by construction at least as good as an unselected sample."""
    model, data = tiny_model_and_loader
    config = SamplingConfig(sample_counts=(4,), seed=3)
    variant = ProbUNetVariant(model, generator=torch.Generator().manual_seed(config.seed))
    per_patch = collect_per_patch_metrics(
        variant, data.loaders["val"], config, torch.device("cpu")
    )
    assert np.all(per_patch["oracle_dice@4"] >= per_patch["random_sample_dice@4"] - 1e-6)


def test_sampling_config_validation() -> None:
    """Sample counts must be positive and strictly increasing."""
    with pytest.raises(ValueError, match="must not be empty"):
        SamplingConfig(sample_counts=())
    with pytest.raises(ValueError, match="positive"):
        SamplingConfig(sample_counts=(0, 4))
    with pytest.raises(ValueError, match="increasing"):
        SamplingConfig(sample_counts=(8, 4))


def test_collect_rejects_a_raw_model(tiny_model_and_loader) -> None:
    """Passing a model instead of a variant fails with an actionable message.

    ``ProbUNet`` also has a ``sample`` method, but its signature is
    ``sample(encoded, n_samples)``. Without this guard the mistake surfaced as
    "'Tensor' object has no attribute 'prior'" from deep inside the model.
    """
    model, data = tiny_model_and_loader
    with pytest.raises(TypeError, match="ProbUNetVariant"):
        collect_per_patch_metrics(
            model, data.loaders["val"], SamplingConfig(sample_counts=(1,)), torch.device("cpu")
        )


# --------------------------------------------------------------------------- #
# Consensus headroom: the Phase 3 pre-registration pass
# --------------------------------------------------------------------------- #
def headroom_arrays(**overrides: np.ndarray) -> dict[str, np.ndarray]:
    """Build per-patch arrays for :func:`per_bucket`, one patch per bucket.

    Args:
        **overrides: Columns to replace.

    Returns:
        A mapping matching ``measure_split``'s output contract.
    """
    base = {
        "random": np.array([0.10, 0.20, 0.30, 0.40]),
        "oracle": np.array([0.25, 0.40, 0.55, 0.70]),
        "all_empty": np.zeros(4),
        "emptiest": np.zeros(4),
        "ceiling": np.array([0.40, 0.667, 0.857, 1.0]),
        "nonempty_frac": np.ones(4),
        "n_nonempty": np.array([1, 2, 3, 4]),
        "index": np.arange(4),
    }
    base.update(overrides)
    return base


def test_headroom_reports_ok_when_the_pathology_is_gone() -> None:
    """oracle above all-empty on every bucket is the result Phase 3 needs."""
    report = per_bucket(headroom_arrays())
    assert set(report) == {"1 grader", "2 graders", "3 graders", "4 graders", "all"}
    for label, row in report.items():
        assert row["verdict"] == "ok", label
        assert row["oracle_beats_all_empty"] is True
    assert report["1 grader"]["headroom_oracle_minus_random"] == pytest.approx(0.15)
    # Reported as a fraction of what any mask could achieve, never against 1.0.
    assert report["1 grader"]["oracle_fraction_of_ceiling"] == pytest.approx(0.25 / 0.40)


def test_headroom_flags_a_genuine_all_empty_win() -> None:
    """If all-empty still beats oracle, the target is not fixed and the run must stop."""
    report = per_bucket(
        headroom_arrays(all_empty=np.array([0.75, 0.0, 0.0, 0.0]))
    )
    assert report["1 grader"]["verdict"] == "all_empty_wins"
    assert report["1 grader"]["oracle_beats_all_empty"] is False
    assert report["4 graders"]["verdict"] == "ok"


def test_headroom_separates_a_degenerate_tie_from_a_failure() -> None:
    """A model emitting only empty candidates ties at 0 -- that is NOT all-empty winning.

    Both score exactly 0 because there is nothing to choose between, so the pass says
    nothing about the target. Reporting it as a refutation of soft consensus would blame
    the target for an undertrained checkpoint. Observed for real on a 2-step smoke
    checkpoint, which is what prompted the three-way verdict.
    """
    report = per_bucket(
        headroom_arrays(
            random=np.zeros(4),
            oracle=np.zeros(4),
            emptiest=np.zeros(4),
            nonempty_frac=np.zeros(4),
        )
    )
    for label, row in report.items():
        assert row["verdict"] == "degenerate_tie", label
        assert row["oracle_beats_all_empty"] is False
        # The column that explains why.
        assert row["nonempty_frac"]["mean"] == 0.0


def test_headroom_table_renders_every_bucket() -> None:
    """The rendered table carries one row per bucket plus the aggregate."""
    text = render(per_bucket(headroom_arrays()))
    for label in ("1 grader", "2 graders", "3 graders", "4 graders", "all"):
        assert label in text
    assert "nonempty" in text and "ceiling" in text and "headroom" in text


def test_grader_and_model_columns_are_disjoint() -> None:
    """The split between what needs weights and what does not is explicit.

    ``ceiling`` and ``all_empty`` are functions of the grader masks alone, so they are
    properties of the dataset and the split -- final, and identical across every arm and
    every checkpoint. Nothing that varies with a model may leak into that set.
    """
    assert set(GRADER_COLUMNS).isdisjoint(MODEL_COLUMNS)
    assert set(GRADER_COLUMNS) == {"ceiling", "all_empty"}


def test_measure_ceilings_needs_no_model() -> None:
    """The ceiling table is computable from grader masks alone, on CPU, with no weights.

    This is what lets the report notebook build the table without downloading a
    checkpoint, and it is why the four ceilings cannot move between runs.
    """
    generator = torch.Generator().manual_seed(5)
    graders = torch.stack(
        [
            torch.stack(
                [
                    (torch.rand(8, 8, generator=generator) > 0.6).to(torch.uint8)
                    if grader < count
                    else torch.zeros(8, 8, dtype=torch.uint8)
                    for grader in range(4)
                ]
            )
            for count in (1, 2, 3, 4)
        ]
    )
    batches = [{"masks": graders, "index": torch.arange(4)}]

    results = measure_ceilings(batches)
    assert set(results) == {"ceiling", "all_empty", "n_nonempty", "index"}
    assert results["n_nonempty"].tolist() == [1, 2, 3, 4]
    # Identical to calling the primitive directly -- one definition, not two.
    assert np.allclose(results["ceiling"], consensus_ceiling(graders).numpy())
    # Every patch here has a non-empty grader, so all-empty is mechanically 0.
    assert np.allclose(results["all_empty"], 0.0)

    # per_bucket handles the model-free columns without inventing a verdict.
    report = per_bucket(results)
    assert "ceiling" in report["1 grader"]
    assert "verdict" not in report["1 grader"]
    assert "ceiling" in render(report)


def test_every_documented_column_is_actually_produced() -> None:
    """A legend must never describe output the table does not print.

    That mismatch happened here once -- ``orc|off`` stayed in the legend after it was
    dropped from the header -- and it is the same class as a config documenting an
    invocation that could not work. The legend is now generated from the same tuples that
    generate the columns, so the two cannot drift; this asserts the other half, that every
    documented column resolves to a key ``per_bucket`` really emits.
    """
    from probunet.evaluation.headroom import (
        RANKING_COLUMNS,
        SELECTION_COLUMNS,
        per_bucket,
        render_selection,
    )

    results = {
        "head": np.array([0.4, 0.6]),
        "area_only": np.array([0.3, 0.5]),
        "random": np.array([0.2, 0.3]),
        "oracle": np.array([0.5, 0.7]),
        "emptiest": np.zeros(2),
        "all_empty": np.zeros(2),
        "ceiling": np.array([0.4, 1.0]),
        "nonempty_frac": np.array([0.5, 1.0]),
        "area_picks_largest": np.ones(2),
        "head_spearman": np.array([0.9, 0.8]),
        "area_spearman": np.array([0.7, 0.6]),
        "pred_mean": np.array([0.1, 0.2]),
        "pred_std_within_image": np.array([0.01, 0.02]),
        "pred_spread_within_image": np.array([0.03, 0.04]),
        "pred_of_chosen": np.array([0.15, 0.25]),
        "n_nonempty": np.array([1, 4]),
        "index": np.arange(2),
    }
    report = per_bucket(results)
    row = report["all"]

    for name, key, _, _ in (*SELECTION_COLUMNS, *RANKING_COLUMNS):
        assert key in row, f"column {name!r} documents {key!r}, which per_bucket never emits"

    text = render_selection(report)
    for name, _, _, legend in (*SELECTION_COLUMNS, *RANKING_COLUMNS):
        assert name in text, f"{name!r} is documented but not printed"
        if legend:
            assert legend.split(".")[0][:30] in text


def test_the_area_control_reduced_to_the_largest_area_rule_on_the_measured_run() -> None:
    """On the recorded run's candidate set, the area control picked the largest candidate.

    **Scoped deliberately to the measurement, not to the architecture.**
    ``AreaOnlyScorer`` is a ReLU MLP on ``log1p(area)`` and is *not* guaranteed monotone --
    measured over five seeds it is monotone at initialization for only one, and 9-47 of 800
    training steps show ``argmax != largest-area``. So this asserts what was observed on
    this checkpoint and split, which is what licenses describing the control as the
    deterministic rule **for this run**, and nothing stronger.

    Skips when the tracked results file is absent, so it runs wherever the record is.
    """
    record = REPO_ROOT / "results" / "consensus_selection_val.json"
    if not record.exists():
        pytest.skip(f"no recorded selection table at {record}")

    buckets = json.loads(record.read_text())["buckets"]
    if "area_picks_largest" not in buckets["all"]:
        pytest.skip(
            f"{record.name} predates the area_picks_largest column; re-run "
            "scripts/consensus_headroom.py --head-checkpoint to record it"
        )
    for label, row in buckets.items():
        assert row["area_picks_largest"]["mean"] == pytest.approx(1.0, abs=1e-9), (
            f"{label}: the area control no longer reduces to the largest-area rule "
            f"({row['area_picks_largest']['mean']}). The report's wording depends on this, "
            "so re-check FINDINGS 4.5 before changing the number."
        )
