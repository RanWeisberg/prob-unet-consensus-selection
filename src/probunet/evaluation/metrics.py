"""Evaluation metrics, all built on one definition of overlap.

:func:`binary_iou` and :func:`dice` are the only place overlap is computed. Everything
else here -- the generalized energy distance, oracle Dice, Hungarian-matched IoU, the
degenerate baselines -- is defined in terms of them, so there is exactly one place where
the edge cases are decided. In particular :func:`pairwise_iou` loops over the sample axis
calling :func:`binary_iou` rather than deriving overlap from
``intersection = S @ Y.T``: the fast route would be a second definition of IoU with its
own empty-mask branch, and the loop costs about 1.5 s over the whole validation split.

**The empty-mask convention.** Two empty masks have IoU **1.0** and Dice **1.0**: they
agree perfectly that there is no lesion. The GED distance ``d = 1 - IoU`` is therefore
**0** for two empty masks, which is what the paper requires so the metric rewards
correct agreement on lesion absence. This is not a rare case: 38% of the grader masks in
this dataset are empty and 33% of patches have only one non-empty grader.

**IoU is foreground-only**, not averaged over classes. The reference's Cityscapes
evaluation averages per-class IoU with ``nanmean``, which for binary data would mix in a
background IoU of ~0.99 and produce a far more forgiving metric -- ``d = 0.505`` instead
of ``1.000`` for an empty prediction against a 150-pixel lesion. The paper's explicit
"if both masks are empty, d = 0" rule is itself evidence for foreground-only: under
class averaging that case falls out of ``nanmean([1.0, NaN]) = 1.0`` automatically and
would need no rule at all. It is only necessary when a single foreground IoU has to
resolve 0/0. The paper also speaks of "masks of the lesion", and follow-up work on this
same preprocessed LIDC data reports foreground IoU. See DEVIATIONS.md entry 7.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

AGGREGATIONS = ("mean", "median", "min", "max")
"""Ways to reduce a sample's per-grader scores into one number.

The consensus-selection extension's target is still open (CLAUDE.md), so the choice is
exposed rather than hard-coded. ``mean`` is the default everywhere.
"""

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


def distance(a: Tensor, b: Tensor) -> Tensor:
    """The GED distance ``d(x, y) = 1 - IoU(x, y)``.

    Two empty masks give exactly 0, because :func:`binary_iou` scores them 1.0.

    Args:
        a: Mask of shape ``(..., H, W)``.
        b: Mask of the same shape.

    Returns:
        Float32 distances of shape ``a.shape[:-2]``.
    """
    return 1.0 - binary_iou(a, b)


def pairwise_iou(a: Tensor, b: Tensor) -> Tensor:
    """IoU between every mask in ``a`` and every mask in ``b``.

    Implemented as a loop over ``a``'s mask axis calling :func:`binary_iou`, so overlap
    has a single definition. See the module docstring for why the vectorized formulation
    is deliberately avoided.

    Args:
        a: Masks of shape ``(B, n, H, W)``.
        b: Masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 IoU matrix of shape ``(B, n, m)``.

    Raises:
        ValueError: If either input is not 4-D, or their batch or spatial dims differ.
    """
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError(
            f"expected (B, k, H, W) for both, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape[0] != b.shape[0] or a.shape[-2:] != b.shape[-2:]:
        raise ValueError(
            f"incompatible shapes {tuple(a.shape)} and {tuple(b.shape)}"
        )
    columns = b.shape[1]
    return torch.stack(
        [binary_iou(a[:, i : i + 1].expand(-1, columns, -1, -1), b) for i in range(a.shape[1])],
        dim=1,
    )


def pairwise_dice(a: Tensor, b: Tensor) -> Tensor:
    """Dice between every mask in ``a`` and every mask in ``b``.

    Args:
        a: Masks of shape ``(B, n, H, W)``.
        b: Masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 Dice matrix of shape ``(B, n, m)``.

    Raises:
        ValueError: If the shapes are incompatible.
    """
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError(
            f"expected (B, k, H, W) for both, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape[0] != b.shape[0] or a.shape[-2:] != b.shape[-2:]:
        raise ValueError(f"incompatible shapes {tuple(a.shape)} and {tuple(b.shape)}")
    columns = b.shape[1]
    return torch.stack(
        [dice(a[:, i : i + 1].expand(-1, columns, -1, -1), b) for i in range(a.shape[1])],
        dim=1,
    )


def generalized_energy_distance(samples: Tensor, graders: Tensor) -> dict[str, Tensor]:
    """The IoU-based generalized energy distance, per patch.

    Implements the paper's estimator (Appendix B) for ``n`` model samples and ``m``
    ground-truth masks::

        D^2 = (2 / (n*m)) * sum_ij d(S_i, Y_j)
              - (1 / n^2)  * sum_ij d(S_i, S_j)
              - (1 / m^2)  * sum_ij d(Y_i, Y_j)

    The two self-distance sums run over **all** pairs including ``i == j``, whose
    distance is 0. This matches the authors' ``calc_energy_distances``, which takes a
    plain mean over both axes of the full square matrices.

    Keeping those zero diagonals makes this the energy distance between the two
    *empirical* distributions rather than an unbiased population estimate. For
    pairwise-disjoint masks it reduces exactly to ``sum_i (p_i - q_i)^2`` over mode
    frequencies, so it is non-negative in that regime and no negative case could be
    constructed for it. Values are nonetheless returned **unclamped**, and
    :func:`summarize` counts negatives, so anything unexpected in a real run is visible
    rather than quietly floored at zero.

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.
        graders: Ground-truth masks of shape ``(B, m, H, W)``.

    Returns:
        A dict of per-patch tensors of shape ``(B,)``: ``d_squared`` plus the three
        components ``d_ys``, ``d_ss`` and ``d_yy``, so a sign or normalization error can
        be localized rather than merely observed.
    """
    d_ys = (1.0 - pairwise_iou(samples, graders)).mean(dim=(1, 2))
    d_ss = (1.0 - pairwise_iou(samples, samples)).mean(dim=(1, 2))
    d_yy = (1.0 - pairwise_iou(graders, graders)).mean(dim=(1, 2))
    return {
        "d_squared": 2.0 * d_ys - d_ss - d_yy,
        "d_ys": d_ys,
        "d_ss": d_ss,
        "d_yy": d_yy,
    }


def aggregate_over_graders(scores: Tensor, aggregate: str = "mean") -> Tensor:
    """Reduce a ``(B, n, m)`` score matrix over the grader axis.

    Args:
        scores: Per-(sample, grader) scores of shape ``(B, n, m)``.
        aggregate: One of :data:`AGGREGATIONS`. ``median`` interpolates, so for an even
            number of graders it is the true midpoint rather than the lower value.

    Returns:
        Per-sample scores of shape ``(B, n)``.

    Raises:
        ValueError: If ``aggregate`` is unknown.
    """
    if aggregate == "mean":
        return scores.mean(dim=-1)
    if aggregate == "median":
        return torch.quantile(scores, 0.5, dim=-1)
    if aggregate == "min":
        return scores.amin(dim=-1)
    if aggregate == "max":
        return scores.amax(dim=-1)
    raise ValueError(f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}")


def oracle_dice(samples: Tensor, graders: Tensor, aggregate: str = "mean") -> Tensor:
    """Dice of the best single sample: ``max_i [ agg_j Dice(S_i, Y_j) ]``.

    This is the ceiling the consensus-selection head targets, because it selects **one**
    sample -- exactly what the head must do without ground truth. Contrast
    :func:`per_grader_oracle_dice`, which lets every grader pick its own best sample and
    is therefore more permissive.

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.
        graders: Ground-truth masks of shape ``(B, m, H, W)``.
        aggregate: How to combine a sample's per-grader Dice scores.

    Returns:
        Per-patch Dice of shape ``(B,)``.
    """
    return aggregate_over_graders(pairwise_dice(samples, graders), aggregate).amax(dim=1)


def per_grader_oracle_dice(samples: Tensor, graders: Tensor) -> Tensor:
    """``mean_j [ max_i Dice(S_i, Y_j) ]``: each grader takes its best-matching sample.

    Reported because parts of the literature call this "oracle". It is never the
    extension's target, since it does not correspond to selecting a single output.

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.
        graders: Ground-truth masks of shape ``(B, m, H, W)``.

    Returns:
        Per-patch Dice of shape ``(B,)``.
    """
    return pairwise_dice(samples, graders).amax(dim=1).mean(dim=1)


def random_sample_dice(samples: Tensor, graders: Tensor, aggregate: str = "mean") -> Tensor:
    """Expected Dice of an unselected sample: ``mean_i [ agg_j Dice(S_i, Y_j) ]``.

    The floor the extension must beat. Computed as the mean over all ``n`` draws rather
    than by literally drawing one, which has the same expectation with much less
    variance.

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.
        graders: Ground-truth masks of shape ``(B, m, H, W)``.
        aggregate: How to combine a sample's per-grader Dice scores.

    Returns:
        Per-patch Dice of shape ``(B,)``.
    """
    return aggregate_over_graders(pairwise_dice(samples, graders), aggregate).mean(dim=1)


def selected_sample_dice(
    samples: Tensor, graders: Tensor, selection: Tensor, aggregate: str = "mean"
) -> Tensor:
    """Dice of one chosen sample per patch.

    The generic scoring path for any selection rule: the emptiest-sample baseline here,
    and the consensus-selection head later.

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.
        graders: Ground-truth masks of shape ``(B, m, H, W)``.
        selection: Index of the chosen sample per patch, shape ``(B,)``.
        aggregate: How to combine the chosen sample's per-grader Dice scores.

    Returns:
        Per-patch Dice of shape ``(B,)``.
    """
    scores = aggregate_over_graders(pairwise_dice(samples, graders), aggregate)
    return scores.gather(1, selection.to(torch.int64).unsqueeze(1)).squeeze(1)


def emptiest_sample_index(samples: Tensor) -> Tensor:
    """Index of the sample with the fewest foreground pixels, per patch.

    A deliberately trivial selection rule. With 33% of patches having three of four
    graders empty, "always predict nothing" scores well, so any learned selector must be
    shown to beat this and not merely to beat random selection.

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.

    Returns:
        Chosen indices of shape ``(B,)``. Ties go to the lowest index.
    """
    areas = (samples != 0).flatten(start_dim=2).sum(dim=2)
    return areas.argmin(dim=1)


def hungarian_assignment(iou: Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Optimal one-to-one matching between samples and graders, per patch.

    Maximizes the total IoU of the matched pairs. With ``n != m`` only ``min(n, m)``
    pairs are matched, so ``n = 1`` degenerates to "the single sample against its
    best-matching grader".

    Args:
        iou: IoU matrix of shape ``(B, n, m)``.

    Returns:
        One ``(sample_indices, grader_indices)`` pair per patch.
    """
    matrix = iou.detach().to(torch.float32).cpu().numpy()
    # linear_sum_assignment minimizes, so negate to maximize overlap.
    return [linear_sum_assignment(-matrix[b]) for b in range(matrix.shape[0])]


def hungarian_matched_iou(samples: Tensor, graders: Tensor) -> Tensor:
    """Mean IoU over an optimal one-to-one matching of samples to graders.

    GED can reward a diverse set of samples that individually match nothing. Matching
    each grader to a distinct sample asks a different question: does the sample *set*
    cover the grader set, one for one?

    Args:
        samples: Model samples of shape ``(B, n, H, W)``.
        graders: Ground-truth masks of shape ``(B, m, H, W)``.

    Returns:
        Per-patch mean matched IoU of shape ``(B,)``.
    """
    iou = pairwise_iou(samples, graders)
    matched = [
        iou[batch, rows, columns].mean()
        for batch, (rows, columns) in enumerate(hungarian_assignment(iou))
    ]
    return torch.stack(matched)


def summarize(values: np.ndarray | Tensor) -> dict[str, float | int]:
    """Summarize a per-patch metric as a distribution, not just a mean.

    The paper's LIDC figure is a scatter plot, so the spread matters as much as the
    centre. ``n_negative`` is reported because the GED estimator is biased and may go
    below zero for small sample counts; that is visible here rather than clamped away.

    Args:
        values: Per-patch values.

    Returns:
        Count, mean, std, median, quartiles, IQR, min, max and negative count. All-NaN
        or empty input yields ``n = 0`` and None elsewhere.
    """
    array = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None,
                "q75": None, "iqr": None, "min": None, "max": None, "n_negative": 0}
    q25, q75 = (float(x) for x in np.percentile(array, [25, 75]))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "min": float(array.min()),
        "max": float(array.max()),
        "n_negative": int((array < 0).sum()),
    }
