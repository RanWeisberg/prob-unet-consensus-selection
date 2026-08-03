"""Overlap metrics: the shared primitives every evaluation number is built from.

Only two functions live here, deliberately: :func:`binary_iou` and :func:`dice`. GED,
oracle Dice and Hungarian-matched IoU are all defined in terms of these, so there is
exactly one place where the edge cases are decided.

**The empty-mask convention.** Two empty masks have IoU **1.0** and Dice **1.0**: they
agree perfectly that there is no lesion. This makes the GED distance
``d = 1 - IoU`` equal **0** for two empty masks, which is what CLAUDE.md requires --
the metric must reward correct agreement on lesion absence. Getting this wrong
silently skews every reported result, and it is not a rare case: 38% of the grader
masks in this dataset are empty and 33% of patches have only one non-empty grader.

An empty mask against a non-empty one has IoU 0.0, which needs no special handling
(the intersection is empty and the union is not).
"""

from __future__ import annotations

import torch
from torch import Tensor

EMPTY_VS_EMPTY_SCORE = 1.0
"""Score for comparing two empty masks: perfect agreement on lesion absence."""


def _as_bool(mask: Tensor) -> Tensor:
    """Interpret a mask as boolean.

    Args:
        mask: Integer, boolean or floating tensor holding binary values.

    Returns:
        A boolean tensor.

    Raises:
        ValueError: If a floating tensor holds values other than 0 and 1, which would
            mean soft labels or probabilities were passed where a hard mask is
            required -- silently thresholding them at 0 would be wrong.
    """
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point():
        unique = torch.unique(mask)
        if not torch.isin(unique, torch.tensor([0.0, 1.0], device=mask.device)).all():
            raise ValueError(
                "floating mask holds values outside {0, 1}; threshold or argmax it "
                f"before computing overlap (found up to {unique.numel()} distinct "
                "values). Soft masks are not supported."
            )
    return mask != 0


def _check_shapes(a: Tensor, b: Tensor) -> None:
    """Check that two masks are comparable.

    Args:
        a: First mask.
        b: Second mask.

    Raises:
        ValueError: If the shapes differ or are lower than 2-D.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dim() < 2:
        raise ValueError(f"masks must have at least 2 dims (H, W), got {tuple(a.shape)}")


def binary_iou(a: Tensor, b: Tensor) -> Tensor:
    """Intersection over union of two binary masks, over the last two dimensions.

    Args:
        a: Mask of shape ``(..., H, W)``.
        b: Mask of the same shape.

    Returns:
        Float32 IoU of shape ``a.shape[:-2]`` (a scalar for 2-D inputs). Two empty
        masks score :data:`EMPTY_VS_EMPTY_SCORE`.

    Raises:
        ValueError: If the shapes differ, or a floating mask is not binary.
    """
    _check_shapes(a, b)
    left, right = _as_bool(a), _as_bool(b)
    dims = (-2, -1)
    intersection = (left & right).sum(dim=dims).to(torch.float32)
    union = (left | right).sum(dim=dims).to(torch.float32)
    # union == 0 means both masks are empty: perfect agreement, not a division by zero.
    return torch.where(
        union > 0,
        intersection / union.clamp(min=1.0),
        torch.full_like(union, EMPTY_VS_EMPTY_SCORE),
    )


def dice(a: Tensor, b: Tensor) -> Tensor:
    """Dice coefficient of two binary masks, over the last two dimensions.

    Args:
        a: Mask of shape ``(..., H, W)``.
        b: Mask of the same shape.

    Returns:
        Float32 Dice of shape ``a.shape[:-2]``. Two empty masks score
        :data:`EMPTY_VS_EMPTY_SCORE`, consistent with :func:`binary_iou`.

    Raises:
        ValueError: If the shapes differ, or a floating mask is not binary.
    """
    _check_shapes(a, b)
    left, right = _as_bool(a), _as_bool(b)
    dims = (-2, -1)
    intersection = (left & right).sum(dim=dims).to(torch.float32)
    total = left.sum(dim=dims).to(torch.float32) + right.sum(dim=dims).to(torch.float32)
    return torch.where(
        total > 0,
        2.0 * intersection / total.clamp(min=1.0),
        torch.full_like(total, EMPTY_VS_EMPTY_SCORE),
    )