"""Tests for GED, oracle Dice, Hungarian matching and the degenerate baselines.

The GED cases are small enough to verify by hand, and each asserts the three components
separately rather than only the total, so a sign or normalization error is localized
instead of merely detected. Two cases have entirely empty ground truth, which is the
edge the ``d = 0 for both empty`` rule exists for.

Notation: ``A`` is a non-empty mask, ``E`` an empty one, ``P`` a single pixel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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
