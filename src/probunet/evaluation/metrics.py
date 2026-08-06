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

Retained for the **distribution** metrics, where the four graders stay separate. It is
**not** the selection target: Phase 3 selects on soft consensus (see :func:`consensus`),
because every one of these aggregations is broken by the empty-grader structure of this
data -- ``mean`` scores an empty mask 0.75 against 0.25 for a correct one on the 33% of
patches with three empty graders, ``median`` scores it 1.0, and ``min`` is degenerate
across three of four buckets. ``mean`` remains the default for reporting.
"""

EMPTY_VS_EMPTY_SCORE = 1.0
"""Score for comparing two empty masks: perfect agreement on lesion absence.

**The single empty-mask convention.** Used by :func:`binary_iou`, :func:`dice` and
:func:`soft_dice` alike; there must never be a second one.
"""

N_GRADERS = 4
"""Independent expert annotations per patch in LIDC-IDRI."""


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


def consensus(graders: Tensor) -> Tensor:
    """Soft consensus map: the equal-weighted average of the grader masks.

    ``c = (1/4) * sum_k m_k``, so each pixel lands in ``{0, 0.25, 0.5, 0.75, 1.0}`` and
    reads as **the fraction of graders who included this pixel**. This is the Phase 3
    selection target (FINDINGS 4.4).

    **Equal weighting is an explicit assumption, not an inferred one.** The four LIDC
    annotators are anonymous and carry no identity across patches, so per-grader
    reliability cannot be estimated from this data -- which is also why STAPLE has nothing
    to fit here and is cited as considered-and-rejected rather than used.

    **This is the SELECTION target only.** GED, sample diversity and every other
    distribution metric keep the four masks separate; collapsing them into one average for
    those would discard exactly the grader spread Phase 1 exists to measure.

    Args:
        graders: Binary grader masks of shape ``(B, m, H, W)``, any integer, boolean or
            binary floating dtype.

    Returns:
        Float32 consensus of shape ``(B, H, W)``.

    Raises:
        ValueError: If the input is not 4-D, or is not binary. The binarity check is what
            stops an **already-averaged** map from being averaged a second time -- the
            values would still look plausible, so nothing downstream would catch it.
    """
    if graders.dim() != 4:
        raise ValueError(
            f"expected grader masks of shape (B, m, H, W), got {tuple(graders.shape)}"
        )
    # _as_bool raises on a floating tensor holding anything outside {0, 1}, which is the
    # double-averaging guard: a consensus map arriving here fails loudly.
    return _as_bool(graders).to(torch.float32).mean(dim=1)


def soft_dice(samples: Tensor, target: Tensor) -> Tensor:
    """Dice between **binary** samples and a **soft** target, over the last two dims.

    ``2 * sum(s * c) / (sum(s) + sum(c))``. The sample stays hard -- it is the artifact
    that would actually be delivered -- while the target carries the graders' partial
    agreement.

    Deliberately a separate function rather than a relaxation of :func:`dice`: ``dice``
    routes through :func:`_as_bool`, which **raises** on a floating mask holding values
    outside ``{0, 1}``, and that guard is correct and stays. What is shared is the thing
    that must be shared -- :data:`EMPTY_VS_EMPTY_SCORE`, so there is still exactly one
    empty-mask convention.

    **Scores are bounded well below 1 by construction.** Against a consensus built from
    one non-empty grader of area ``A``, a *perfect* mask scores
    ``2(0.25A) / (A + 0.25A) = 0.40``. The per-bucket ceilings are 0.40 / 0.667 / 0.857 /
    1.00 for 1/2/3/4 non-empty graders. Report against those, never against 1.0.

    Args:
        samples: Binary masks of shape ``(..., H, W)``.
        target: Soft target of the same shape, values in ``[0, 1]``.

    Returns:
        Float32 scores of shape ``samples.shape[:-2]``.

    Raises:
        ValueError: If the shapes differ, the samples are not binary, or the target falls
            outside ``[0, 1]``.
    """
    _check_shapes(samples, target)
    if not target.is_floating_point():
        raise ValueError(
            f"soft target must be floating point, got {target.dtype}. For a binary "
            "target use dice()."
        )
    if bool((target < 0).any() or (target > 1).any()):
        raise ValueError("soft target holds values outside [0, 1]")

    hard = _as_bool(samples).to(torch.float32)
    soft = target.to(torch.float32)
    dims = (-2, -1)
    intersection = (hard * soft).sum(dim=dims)
    total = hard.sum(dim=dims) + soft.sum(dim=dims)
    # total == 0 means an empty sample against an all-zero consensus, i.e. every grader
    # saw no lesion and the model agreed. Same convention as binary_iou and dice.
    #
    # MEASURED 2026-08-06: no patch in this dataset has zero non-empty graders (counts are
    # 1 -> 4963, 2 -> 2756, 3 -> 2626, 4 -> 4751 over all 15,096; and 0 in each of train,
    # val and test). So this branch is DEFENSIVE -- it is tested synthetically and plays
    # no part in any reported number. An empty sample against a non-empty consensus scores
    # 0, which is the whole point of the target.
    return torch.where(
        total > 0,
        2.0 * intersection / total.clamp(min=1e-12),
        torch.full_like(total, EMPTY_VS_EMPTY_SCORE),
    )


def consensus_ceiling(graders: Tensor) -> Tensor:
    """The best soft-consensus Dice any binary mask could achieve on this image.

    The maximum of ``2*sum(s*c) / (sum(s) + sum(c))`` over binary ``s`` is attained by
    thresholding ``c`` at its most favourable level, so it is computed exactly by trying
    each distinct non-zero value of ``c`` as a threshold. For ``k`` identical non-empty
    graders this reduces to the familiar ladder 0.40 / 0.667 / 0.857 / 1.00.

    Reported alongside every soft-consensus number, because absolute values are low by
    design and are meaningless without the ceiling beside them.

    Args:
        graders: Binary grader masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 ceiling of shape ``(B,)``.
    """
    soft = consensus(graders)
    levels = torch.arange(1, N_GRADERS + 1, device=soft.device, dtype=torch.float32)
    levels = levels / N_GRADERS
    # (B, L, H, W): one candidate mask per threshold level.
    candidates = (soft.unsqueeze(1) >= levels.view(1, -1, 1, 1)).to(torch.uint8)
    scores = soft_dice(candidates, soft.unsqueeze(1).expand_as(candidates))
    return scores.amax(dim=1)


def consensus_scores(samples: Tensor, graders: Tensor) -> Tensor:
    """Soft-consensus Dice of every sample against its image's consensus.

    The single scoring primitive Phase 3 is built on: the head's regression target, and
    the quantity every consensus baseline below reduces.

    Args:
        samples: Binary masks of shape ``(B, n, H, W)``.
        graders: Binary grader masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 scores of shape ``(B, n)``.

    Raises:
        ValueError: If either input is not 4-D or their batch/spatial dims differ.
    """
    if samples.dim() != 4 or graders.dim() != 4:
        raise ValueError(
            f"expected (B, k, H, W) for both, got {tuple(samples.shape)} and "
            f"{tuple(graders.shape)}"
        )
    if samples.shape[0] != graders.shape[0] or samples.shape[-2:] != graders.shape[-2:]:
        raise ValueError(
            f"incompatible shapes {tuple(samples.shape)} and {tuple(graders.shape)}"
        )
    soft = consensus(graders).unsqueeze(1).expand(-1, samples.shape[1], -1, -1)
    return soft_dice(samples, soft)


def consensus_oracle(samples: Tensor, graders: Tensor) -> Tensor:
    """Best achievable soft-consensus Dice over the sample set: the ceiling the head chases.

    Args:
        samples: Binary masks of shape ``(B, n, H, W)``.
        graders: Binary grader masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 per-patch scores of shape ``(B,)``.
    """
    return consensus_scores(samples, graders).amax(dim=1)


def consensus_random(samples: Tensor, graders: Tensor) -> Tensor:
    """Expected soft-consensus Dice of an unselected sample: the floor the head must beat.

    The mean over all ``n`` draws rather than one literal draw -- same expectation, far
    less variance.

    Args:
        samples: Binary masks of shape ``(B, n, H, W)``.
        graders: Binary grader masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 per-patch scores of shape ``(B,)``.
    """
    return consensus_scores(samples, graders).mean(dim=1)


def consensus_selected(samples: Tensor, graders: Tensor, selection: Tensor) -> Tensor:
    """Soft-consensus Dice of one chosen sample per patch.

    The generic path for any selection rule -- the emptiest-sample baseline now, the
    trained head later.

    Args:
        samples: Binary masks of shape ``(B, n, H, W)``.
        graders: Binary grader masks of shape ``(B, m, H, W)``.
        selection: Chosen sample index per patch, shape ``(B,)``.

    Returns:
        Float32 per-patch scores of shape ``(B,)``.
    """
    scores = consensus_scores(samples, graders)
    return scores.gather(1, selection.to(torch.int64).unsqueeze(1)).squeeze(1)


def all_empty_consensus_dice(graders: Tensor) -> Tensor:
    """Soft-consensus Dice of the degenerate all-empty predictor.

    **0.0 on every patch that has at least one non-empty grader**, which in this dataset
    is every patch. Under the old per-grader *mean* Dice the same predictor scored 0.75 on
    bucket 1 and beat best-of-16; that inversion is what soft consensus removes, and this
    function is how the pre-registration pass demonstrates it rather than asserting it.

    Args:
        graders: Binary grader masks of shape ``(B, m, H, W)``.

    Returns:
        Float32 per-patch scores of shape ``(B,)``.
    """
    soft = consensus(graders)
    empty = torch.zeros_like(soft, dtype=torch.uint8)
    return soft_dice(empty, soft)


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

    The per-grader-aggregate oracle, reported for continuity with the Phase 1 table.
    **It is no longer the quantity the head targets** -- Phase 3 selects on soft consensus,
    so :func:`consensus_oracle` is the ceiling the head chases. Under the default
    ``aggregate="mean"`` this one is in fact unreachable in bucket 1, where it measures
    0.7458 against an all-empty mask's 0.7500 (FINDINGS 4.4); that is the arithmetic that
    ruled the per-grader aggregates out. Contrast :func:`per_grader_oracle_dice`, which
    lets every grader pick its own best sample and is therefore more permissive still.

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


def spearman_per_image(
    predicted: Tensor, target: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-image Spearman correlation between predicted and true candidate scores.

    Measures whether the head **ranks** candidates correctly, which is the only thing
    selection actually needs. It is reported beside the regression loss because the two
    can disagree loudly: on bucket-1 images almost every candidate scores near 0, so a head
    that predicts a constant ~0.1 everywhere achieves a small Huber loss while ranking
    nothing. That is the image-only shortcut FINDINGS 4.4 pre-registers a fallback for, and
    a healthy loss beside a near-zero Spearman is exactly its signature.

    **Degenerate images are excluded and counted, never silently dropped.** If all of an
    image's candidates share one score, Spearman is undefined -- zero variance in the
    denominator. This is not rare here: on bucket 1 empty candidates all score exactly
    0.000, so ties are common and whole images can be constant. The excluded *fraction* is
    itself a finding, because it measures how often the sampler offers no real choice, so
    the caller receives the validity mask rather than an average that quietly rests on a
    subset.

    Ties within an otherwise varying image are handled by average ranking, the standard
    convention, so a partial tie degrades the correlation rather than invalidating it.

    Args:
        predicted: Predicted scores of shape ``(B, n)``.
        target: True scores of the same shape.

    Returns:
        ``(rho, valid)``, both on the **CPU**: ``rho`` is ``(B,)`` float32 with NaN where
        undefined, and ``valid`` is a ``(B,)`` bool mask, True where both sides had
        non-degenerate variance.

    Raises:
        ValueError: If the shapes differ or there are fewer than two candidates.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"shape mismatch: {tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    if predicted.dim() != 2:
        raise ValueError(f"expected (B, n), got {tuple(predicted.shape)}")
    if predicted.shape[1] < 2:
        raise ValueError("need at least two candidates to correlate")

    # CPU before float64: MPS has no float64 at all, and ranks want the precision more
    # than they want the device. These are (B, n) with n = 16, so the transfer is free.
    left = _average_ranks(predicted.detach().cpu().to(torch.float64))
    right = _average_ranks(target.detach().cpu().to(torch.float64))
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)

    numerator = (left * right).sum(dim=1)
    denominator = left.pow(2).sum(dim=1).sqrt() * right.pow(2).sum(dim=1).sqrt()
    # A constant side gives a zero-length rank vector: the correlation is undefined, not
    # zero. Zero would read as "the head ranked randomly", which is a different claim.
    valid = denominator > 1e-12
    rho = torch.where(valid, numerator / denominator.clamp_min(1e-12), torch.nan)
    return rho.to(torch.float32), valid


def _average_ranks(values: Tensor) -> Tensor:
    """Rank each row, averaging the ranks of tied values.

    Args:
        values: Scores of shape ``(B, n)``.

    Returns:
        Ranks of shape ``(B, n)``, float64.
    """
    order = values.argsort(dim=1)
    ranks = torch.empty_like(values)
    positions = torch.arange(
        values.shape[1], dtype=values.dtype, device=values.device
    ).expand_as(values)
    ranks.scatter_(1, order, positions)

    # Average the ranks within each group of equal values, per row.
    for row in range(values.shape[0]):
        unique, inverse = torch.unique(values[row], return_inverse=True)
        if unique.numel() == values.shape[1]:
            continue  # no ties in this row
        totals = torch.zeros(unique.numel(), dtype=values.dtype, device=values.device)
        counts = torch.zeros_like(totals)
        totals.index_add_(0, inverse, ranks[row])
        counts.index_add_(0, inverse, torch.ones_like(ranks[row]))
        ranks[row] = (totals / counts)[inverse]
    return ranks
