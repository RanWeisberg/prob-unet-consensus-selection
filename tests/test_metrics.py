"""Tests for the shared overlap primitives.

The empty-mask convention is the thing that must not drift: two empty masks score 1.0,
so the GED distance ``1 - IoU`` is 0 and the metric rewards correct agreement on lesion
absence. 38% of the grader masks in this dataset are empty, so this is a common path,
not an edge case.
"""

from __future__ import annotations

import pytest
import torch

from probunet.evaluation.metrics import EMPTY_VS_EMPTY_SCORE, binary_iou, dice


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
