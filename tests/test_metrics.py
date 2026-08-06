"""Tests for the shared overlap primitives.

The empty-mask convention is the thing that must not drift: two empty masks score 1.0,
so the GED distance ``1 - IoU`` is 0 and the metric rewards correct agreement on lesion
absence. 38% of the grader masks in this dataset are empty, so this is a common path,
not an edge case.
"""

from __future__ import annotations

import pytest
import torch

from probunet.evaluation.metrics import (
    EMPTY_VS_EMPTY_SCORE,
    N_GRADERS,
    aggregate_over_graders,
    all_empty_consensus_dice,
    binary_iou,
    consensus,
    consensus_ceiling,
    consensus_oracle,
    consensus_random,
    consensus_scores,
    consensus_selected,
    dice,
    pairwise_dice,
    soft_dice,
)


def mask(rows: list[list[int]]) -> torch.Tensor:
    """Build a small uint8 mask from nested lists."""
    return torch.tensor(rows, dtype=torch.uint8)


# --------------------------------------------------------------------------- #
# The empty-mask convention
# --------------------------------------------------------------------------- #
def test_both_empty_scores_one() -> None:
    """Two empty masks agree perfectly: IoU and Dice are 1.0, so GED distance is 0."""
    empty = torch.zeros(4, 4, dtype=torch.uint8)
    assert binary_iou(empty, empty).item() == EMPTY_VS_EMPTY_SCORE == 1.0
    assert dice(empty, empty).item() == 1.0
    # This is the property CLAUDE.md requires of the GED distance.
    assert (1.0 - binary_iou(empty, empty)).item() == 0.0


def test_empty_against_nonempty_scores_zero() -> None:
    """An empty mask against a non-empty one has no overlap."""
    empty = torch.zeros(2, 2, dtype=torch.uint8)
    full = torch.ones(2, 2, dtype=torch.uint8)
    assert binary_iou(empty, full).item() == 0.0
    assert dice(empty, full).item() == 0.0
    assert binary_iou(full, empty).item() == 0.0


# --------------------------------------------------------------------------- #
# Hand-computed overlaps
# --------------------------------------------------------------------------- #
def test_identical_nonempty_masks() -> None:
    """A mask against itself scores 1.0."""
    a = mask([[1, 0], [0, 1]])
    assert binary_iou(a, a).item() == 1.0
    assert dice(a, a).item() == 1.0


def test_half_overlap() -> None:
    """One shared pixel out of two each: IoU 1/3, Dice 2/4 = 1/2."""
    a = mask([[1, 1], [0, 0]])
    b = mask([[1, 0], [1, 0]])
    assert binary_iou(a, b).item() == pytest.approx(1.0 / 3.0)
    assert dice(a, b).item() == pytest.approx(0.5)


def test_subset() -> None:
    """One pixel inside four: IoU 1/4, Dice 2/5."""
    a = mask([[1, 1], [1, 1]])
    b = mask([[1, 0], [0, 0]])
    assert binary_iou(a, b).item() == pytest.approx(0.25)
    assert dice(a, b).item() == pytest.approx(2.0 / 5.0)


def test_disjoint() -> None:
    """No shared pixels: both scores 0."""
    a = mask([[1, 0], [0, 0]])
    b = mask([[0, 0], [0, 1]])
    assert binary_iou(a, b).item() == 0.0
    assert dice(a, b).item() == 0.0


def test_dice_is_at_least_iou() -> None:
    """Dice >= IoU always, with equality only at 0 and 1."""
    torch.manual_seed(0)
    a = (torch.rand(20, 8, 8) > 0.7).to(torch.uint8)
    b = (torch.rand(20, 8, 8) > 0.7).to(torch.uint8)
    assert torch.all(dice(a, b) >= binary_iou(a, b) - 1e-6)


# --------------------------------------------------------------------------- #
# Batching and dtypes
# --------------------------------------------------------------------------- #
def test_reduces_over_last_two_dims_only() -> None:
    """Leading dimensions are preserved, so (B, S, H, W) gives (B, S)."""
    a = torch.zeros(3, 5, 4, 4, dtype=torch.uint8)
    b = torch.zeros(3, 5, 4, 4, dtype=torch.uint8)
    assert binary_iou(a, b).shape == (3, 5)
    assert dice(a, b).shape == (3, 5)
    assert binary_iou(a[0, 0], b[0, 0]).shape == ()


def test_batch_entries_are_independent() -> None:
    """Each batch element is scored on its own."""
    a = torch.stack([torch.ones(2, 2), torch.zeros(2, 2)]).to(torch.uint8)
    b = torch.stack([torch.ones(2, 2), torch.zeros(2, 2)]).to(torch.uint8)
    scores = binary_iou(a, b)
    assert scores.tolist() == [1.0, 1.0]  # second pair is empty-vs-empty

    c = torch.stack([torch.ones(2, 2), torch.ones(2, 2)]).to(torch.uint8)
    assert binary_iou(a, c).tolist() == [1.0, 0.0]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool, torch.float32])
def test_accepts_common_dtypes(dtype: torch.Tensor) -> None:
    """Masks may arrive as uint8, int64, bool or a binary float tensor."""
    a = (torch.rand(4, 4) > 0.5).to(dtype)
    b = (torch.rand(4, 4) > 0.5).to(dtype)
    assert binary_iou(a, b).dtype == torch.float32
    assert dice(a, b).dtype == torch.float32


def test_output_is_float32() -> None:
    """MPS has no float64, so overlap scores stay float32."""
    a = torch.ones(4, 4, dtype=torch.uint8)
    assert binary_iou(a, a).dtype == torch.float32
    assert dice(a, a).dtype == torch.float32


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_soft_masks_are_rejected() -> None:
    """A probability map is not a mask; silently thresholding it would be wrong."""
    soft = torch.full((4, 4), 0.6)
    hard = torch.ones(4, 4)
    with pytest.raises(ValueError, match="outside"):
        binary_iou(soft, hard)
    with pytest.raises(ValueError, match="outside"):
        dice(soft, hard)


def test_shape_mismatch_rejected() -> None:
    """Comparing different shapes is an error, not a broadcast."""
    with pytest.raises(ValueError, match="shape mismatch"):
        binary_iou(torch.zeros(4, 4), torch.zeros(4, 5))


def test_rank_too_low_rejected() -> None:
    """A 1-D input has no spatial dims to reduce."""
    with pytest.raises(ValueError, match="at least 2 dims"):
        binary_iou(torch.zeros(4), torch.zeros(4))


# --------------------------------------------------------------------------- #
# One-pixel masks: the small-lesion tail
# --------------------------------------------------------------------------- #
def test_single_pixel_masks() -> None:
    """The smallest non-empty mask in the dataset is one pixel.

    IoU there is brutally sensitive: two adjacent single pixels score 0, and a single
    pixel inside a 150-pixel lesion scores 1/150. Any per-bucket number from the
    small-lesion tail has to be read with this in mind.
    """
    one = torch.zeros(16, 16, dtype=torch.uint8)
    one[0, 0] = 1
    neighbour = torch.zeros(16, 16, dtype=torch.uint8)
    neighbour[0, 1] = 1

    assert binary_iou(one, one).item() == 1.0
    assert binary_iou(one, neighbour).item() == 0.0
    assert dice(one, neighbour).item() == 0.0

    lesion = torch.zeros(16, 16, dtype=torch.uint8)
    lesion[:10, :15] = 1  # 150 pixels, containing the single pixel
    assert int(lesion.sum()) == 150
    assert binary_iou(one, lesion).item() == pytest.approx(1.0 / 150.0)
    assert dice(one, lesion).item() == pytest.approx(2.0 / 151.0)


def test_one_pixel_disagreement_dominates_iou() -> None:
    """Adding one pixel to a one-pixel mask halves its IoU."""
    one = torch.zeros(8, 8, dtype=torch.uint8)
    one[0, 0] = 1
    two = torch.zeros(8, 8, dtype=torch.uint8)
    two[0, 0] = 1
    two[0, 1] = 1
    assert binary_iou(one, two).item() == pytest.approx(0.5)
    # The same absolute error on a large mask barely registers.
    big = torch.zeros(8, 8, dtype=torch.uint8)
    big[:5, :8] = 1
    bigger = big.clone()
    bigger[5, 0] = 1
    assert binary_iou(big, bigger).item() == pytest.approx(40.0 / 41.0)


# --------------------------------------------------------------------------- #
# Soft consensus: the Phase 3 selection target
# --------------------------------------------------------------------------- #
def graders_with(n_nonempty: int, area: int = 8, size: int = 8) -> torch.Tensor:
    """Build ``(1, 4, size, size)`` graders where ``n_nonempty`` share one identical mask.

    The idealized bucket geometry the per-bucket ceilings are derived from: the non-empty
    graders agree exactly, so the ceiling is analytic.

    Args:
        n_nonempty: How many of the four graders see a lesion.
        area: Foreground pixels in each non-empty grader.
        size: Tile side length.

    Returns:
        A uint8 tensor of shape ``(1, 4, size, size)``.
    """
    masks = torch.zeros(1, N_GRADERS, size, size, dtype=torch.uint8)
    flat = masks.view(1, N_GRADERS, -1)
    for grader in range(n_nonempty):
        flat[0, grader, :area] = 1
    return masks


def test_consensus_is_the_grader_fraction() -> None:
    """Each pixel of c is the fraction of graders that included it."""
    graders = graders_with(3, area=4)
    soft = consensus(graders)
    assert soft.shape == (1, 8, 8)
    assert soft.dtype == torch.float32
    flat = soft.view(-1)
    assert torch.allclose(flat[:4], torch.full((4,), 0.75))
    assert torch.allclose(flat[4:], torch.zeros(60))
    # Every attainable value lies on the quarter grid.
    for count in range(N_GRADERS + 1):
        values = torch.unique(consensus(graders_with(count, area=4)))
        assert torch.isin(
            values, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        ).all(), f"{count} graders produced off-grid values {values}"


def test_consensus_refuses_to_average_an_average() -> None:
    """A consensus map fed back in must fail loudly, not silently halve.

    The values would still look plausible, so nothing downstream would catch it.
    """
    soft = consensus(graders_with(2)).unsqueeze(1).expand(-1, N_GRADERS, -1, -1)
    with pytest.raises(ValueError, match="outside"):
        consensus(soft)


def test_consensus_rejects_the_wrong_rank() -> None:
    """A (B, H, W) map is not a grader stack."""
    with pytest.raises(ValueError, match=r"\(B, m, H, W\)"):
        consensus(torch.zeros(1, 8, 8, dtype=torch.uint8))


@pytest.mark.parametrize(
    ("n_nonempty", "expected"),
    [(1, 0.4), (2, 2 / 3), (3, 6 / 7), (4, 1.0)],
)
def test_perfect_mask_hits_the_bucket_ceiling(n_nonempty: int, expected: float) -> None:
    """THE bounded-score fact: a perfect mask scores 0.40 in bucket 1, not 1.0.

    ``2(kA/4) / (A + kA/4)`` for ``k`` agreeing graders, i.e. 0.40 / 0.667 / 0.857 / 1.00.
    Absolute values are low BY CONSTRUCTION and must never be read against 1.0 -- this
    test is where that ladder is pinned.
    """
    graders = graders_with(n_nonempty, area=8)
    perfect = graders[:, :1, :, :]  # exactly grader 0's mask
    score = consensus_scores(perfect, graders)
    assert score.shape == (1, 1)
    assert float(score) == pytest.approx(expected, abs=1e-6)
    # And it is genuinely the best any binary mask could do here.
    assert float(consensus_ceiling(graders)) == pytest.approx(expected, abs=1e-6)


def test_empty_scores_zero_wherever_a_grader_saw_something() -> None:
    """The pathology is gone: empty is worthless on every real patch.

    Under per-grader MEAN Dice an empty mask scored 0.75 on a 3-empty image against 0.25
    for a correct one, so a head trained on it learned to prefer empty. Both halves are
    asserted here so the inversion cannot quietly return.
    """
    for n_nonempty in (1, 2, 3, 4):
        graders = graders_with(n_nonempty, area=8)
        assert float(all_empty_consensus_dice(graders)) == 0.0, n_nonempty

    # The old target, for contrast: empty beats correct in bucket 1.
    graders = graders_with(1, area=8)
    empty = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    perfect = graders[:, :1, :, :]
    old_empty = float(aggregate_over_graders(pairwise_dice(empty, graders), "mean"))
    old_perfect = float(aggregate_over_graders(pairwise_dice(perfect, graders), "mean"))
    assert old_empty == pytest.approx(0.75) and old_perfect == pytest.approx(0.25)
    assert old_empty > old_perfect, "the documented inversion is not reproduced"

    # Under soft consensus the ordering is the right way round.
    new_empty = float(consensus_scores(empty, graders))
    new_perfect = float(consensus_scores(perfect, graders))
    assert new_perfect > new_empty == 0.0


def test_all_empty_consensus_uses_the_shared_convention() -> None:
    """Synthetic only: no patch in this dataset has zero non-empty graders.

    Measured 2026-08-06 over all 15,096 patches (counts 1->4963, 2->2756, 3->2626,
    4->4751, and 0 in every split), so this branch is DEFENSIVE. It still must not invent
    a second convention.
    """
    graders = graders_with(0)
    assert float(consensus(graders).sum()) == 0.0
    assert float(all_empty_consensus_dice(graders)) == EMPTY_VS_EMPTY_SCORE == 1.0
    # A non-empty sample against an all-zero consensus is pure false positive: 0.
    nonempty = torch.ones(1, 1, 8, 8, dtype=torch.uint8)
    assert float(consensus_scores(nonempty, graders)) == 0.0


def test_soft_dice_keeps_the_hard_mask_hard() -> None:
    """Soft target, binary sample -- a soft sample is rejected, not silently thresholded."""
    graders = graders_with(2, area=8)
    soft = consensus(graders)
    with pytest.raises(ValueError, match="outside"):
        soft_dice(soft, soft)  # sample must be binary
    with pytest.raises(ValueError, match="floating point"):
        soft_dice(graders[:, 0], graders[:, 0])  # target must be soft
    # x3, not x2: consensus here holds {0, 0.5}, and doubling that lands on {0, 1.0},
    # which is legitimately in range.
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        soft_dice(graders[:, 0], soft * 3.0)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        soft_dice(graders[:, 0], soft - 1.0)


def test_consensus_baselines_reduce_the_same_scores() -> None:
    """oracle / random / selected are three reductions of one score matrix."""
    torch.manual_seed(0)
    graders = torch.zeros(2, N_GRADERS, 8, 8, dtype=torch.uint8)
    graders[:, :3, :4, :4] = 1
    samples = (torch.rand(2, 5, 8, 8) > 0.5).to(torch.uint8)

    scores = consensus_scores(samples, graders)
    assert scores.shape == (2, 5)
    assert torch.allclose(consensus_oracle(samples, graders), scores.amax(dim=1))
    assert torch.allclose(consensus_random(samples, graders), scores.mean(dim=1))
    choice = torch.tensor([2, 4])
    assert torch.allclose(
        consensus_selected(samples, graders, choice),
        torch.stack([scores[0, 2], scores[1, 4]]),
    )
    # The oracle is bounded by what any mask could achieve.
    assert (consensus_oracle(samples, graders) <= consensus_ceiling(graders) + 1e-6).all()


def test_consensus_scores_validate_shapes() -> None:
    """Mismatched batch or spatial dims fail rather than broadcast."""
    graders = graders_with(2)
    with pytest.raises(ValueError, match="incompatible"):
        consensus_scores(torch.zeros(2, 3, 8, 8, dtype=torch.uint8), graders)
    with pytest.raises(ValueError, match=r"\(B, k, H, W\)"):
        consensus_scores(torch.zeros(1, 8, 8, dtype=torch.uint8), graders)
