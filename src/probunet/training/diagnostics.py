"""Latent-space diagnostics: what the loss curve cannot show you.

Under the paper's reduction convention the KL term is about 0.0006% of the objective at
initialization, so the near-term risk is **not** posterior collapse but the opposite: an
unconstrained latent whose *prior* never learns to cover the grader variants. Training
loss falls happily while that happens, because training draws ``z`` from the posterior.
The failure only appears at inference, when ``z`` comes from the prior.

Read the four signatures together, never one alone:

=====================================  ==================================================
observation                            interpretation
=====================================  ==================================================
``prior_posterior_ce_ratio`` >> 1      Prior is not covering the variants. Posterior-z
                                       reconstructs well, prior-z does not.
``sample_diversity_iou`` -> 1.0 AND    Prior has genuinely **collapsed** to a point.
``nonempty_sample_fraction`` > 0
``sample_diversity_iou`` -> 1.0 AND    Model has not learned foreground **yet**. All
``nonempty_sample_fraction`` ~ 0       samples are empty, and two empty masks have IoU
                                       1.0 by the convention that makes GED reward
                                       agreement on lesion absence. Expected early:
                                       the background-to-foreground ratio is 176:1.
                                       **Not** a collapse alarm.
``prior_sigma_mean`` -> 0              Collapse again, seen from the sigma side.
``kl`` -> 0 while ``ce`` falls         Posterior collapse: the opposite failure.
=====================================  ==================================================

That third row is why ``nonempty_sample_fraction`` exists. Diversity alone cannot
distinguish "hasn't learned anything yet" from "collapsed", because both read 1.0.

**Latent-geometry diagnostics (Phase 2).** Three quantities measure how many directions
the posterior actually uses, and they answer three different questions:

============================  ====================================================
metric                        question
============================  ====================================================
``kl_dim_{i}``                Which **coordinate axis** carries information? The
                              Phase 1 series, kept for continuity. Under a full
                              covariance it is a marginal and does not sum to the
                              total -- see :func:`per_dim_kl`.
``kl_whitened_{i}``           How much does each **informative direction** carry,
                              regardless of axis? Sums exactly to
                              ``kl_snapshot_total``. Sorted descending; the index
                              carries no identity across epochs.
``effrank_snapshot`` /        How **many** directions are informative? The headline
``effrank_val_mean``          number for the FINDINGS 3.2 hypothesis.
============================  ====================================================

The ``_snapshot`` and ``_val`` suffixes are load-bearing and must survive into any report
table: ``_snapshot`` is 32 images at grader 0, computed every epoch; ``_val`` is every
validation image at all four graders, computed at the diagnostics cadence. Only the
second can support a claim about a small shift in effective rank.

The fixed diagnostic sets are **stratified over the ambiguity buckets** rather than
taken as the first N by index, so the panel always shows single-grader cases -- 33% of
the data and the hard case for the consensus-selection extension. The chosen indices are
recorded to disk so panels are identical across runs and comparable between phases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from probunet.data.lidc import LidcDataset
from probunet.evaluation.metrics import binary_iou
from probunet.model.encoder import LatentDistribution, LatentStats
from probunet.model.prob_unet import Encoded, ProbUNet

LOGGER = logging.getLogger(__name__)

N_GRADERS = 4
PANEL_PAD = 2
PANEL_SEPARATOR_VALUE = 0.35


@dataclass(frozen=True)
class DiagnosticSets:
    """Fixed image sets used by the diagnostics.

    Attributes:
        diversity: Global patch indices for the diversity measure.
        panel: Global patch indices shown in the qualitative panel.
        buckets: Ambiguity bucket of each panel index, for labelling.
    """

    diversity: np.ndarray
    panel: np.ndarray
    buckets: dict[str, list[int]]

    def to_dict(self) -> dict[str, object]:
        """Render as a JSON-serializable mapping."""
        return {
            "diversity": [int(i) for i in self.diversity],
            "panel": [int(i) for i in self.panel],
            "panel_buckets": self.buckets,
        }


def stratified_indices(
    dataset: LidcDataset, count: int, seed: int
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Pick ``count`` patch indices spread evenly over the ambiguity buckets.

    Args:
        dataset: The split to draw from.
        count: How many indices to pick.
        seed: Seed making the choice deterministic.

    Returns:
        A ``(indices, per_bucket)`` pair: the chosen global patch indices, sorted, and
        a mapping from bucket label to the indices chosen from it.

    Raises:
        ValueError: If the dataset has no patches.
    """
    if len(dataset) == 0:
        raise ValueError("cannot draw diagnostic indices from an empty dataset")

    buckets = {
        bucket: members
        for bucket, members in dataset.buckets().items()
        if members.size > 0
    }
    rng = np.random.default_rng(seed)
    ordered = sorted(buckets)
    # Spread requested slots over the populated buckets, remainder to the first ones.
    base, remainder = divmod(count, len(ordered))
    chosen: dict[str, list[int]] = {}
    picked: list[int] = []
    for position, bucket in enumerate(ordered):
        want = base + (1 if position < remainder else 0)
        members = np.sort(buckets[bucket])
        take = min(want, members.size)
        if take == 0:
            continue
        selection = rng.choice(members, size=take, replace=False)
        chosen[str(bucket)] = sorted(int(i) for i in selection)
        picked.extend(int(i) for i in selection)

    # If a small bucket could not fill its slots, top up from whatever is left.
    if len(picked) < count:
        remaining = np.setdiff1d(np.sort(dataset.indices), np.array(picked, dtype=np.int64))
        if remaining.size:
            extra = rng.choice(
                remaining, size=min(count - len(picked), remaining.size), replace=False
            )
            picked.extend(int(i) for i in extra)
    return np.array(sorted(picked), dtype=np.int64), chosen


def build_diagnostic_sets(
    dataset: LidcDataset, diversity_images: int, panel_images: int, seed: int
) -> DiagnosticSets:
    """Choose the fixed diversity and panel sets, both ambiguity-stratified.

    The panel is drawn as a subset of the diversity set where possible, so the two
    views describe the same images.

    Args:
        dataset: The validation split.
        diversity_images: Size of the diversity set.
        panel_images: Size of the panel.
        seed: Seed making the choice deterministic.

    Returns:
        The chosen :class:`DiagnosticSets`.
    """
    diversity, _ = stratified_indices(dataset, diversity_images, seed)
    panel, panel_buckets = stratified_indices(dataset, panel_images, seed + 1)
    return DiagnosticSets(diversity=diversity, panel=panel, buckets=panel_buckets)


def save_diagnostic_sets(sets: DiagnosticSets, path: Path) -> None:
    """Record the chosen indices so panels are comparable across runs.

    Args:
        sets: The chosen sets.
        path: Destination JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sets.to_dict(), indent=2) + "\n")


def reparameterize(
    distribution: LatentDistribution, generator: torch.Generator | None = None
) -> Tensor:
    """Draw a latent sample with an optionally seeded generator.

    ``Distribution.rsample`` accepts no generator, so the reparameterization is written
    out. The diagnostics need a fixed noise sequence: with only 8 to 64 images, sampling
    noise would otherwise make panels incomparable between epochs.

    Both latent parameterizations are supported, and both are the *same* draw torch
    would make -- ``z = mu + L @ eps`` with ``L`` diagonal is elementwise
    ``mu + sigma * eps``, so the diagonal branch below is that formula specialized rather
    than a second algorithm. The branches stay separate anyway, because the diagonal one
    is pinned byte-for-byte by ``tests/fingerprints/phase1_latent.json``.

    Args:
        distribution: An ``Independent(Normal(...), 1)`` (diagonal) or a
            ``MultivariateNormal(loc, scale_tril=L)`` (full covariance).
        generator: Optional CPU generator supplying the noise. When None, the
            distribution's own ``rsample`` is used, which draws from the global RNG.

    Returns:
        A sample of shape ``(batch, latent_dim)``.
    """
    if generator is None:
        return distribution.rsample()

    if isinstance(distribution, MultivariateNormal):
        loc = distribution.loc
        factor = distribution.scale_tril
        # Noise is drawn on the CPU generator and then moved, exactly as the diagonal
        # branch does, so a seeded diagnostic gives the same z on every backend.
        noise = torch.randn(loc.shape, generator=generator, dtype=loc.dtype).to(loc.device)
        # z = mu + L eps. matmul over the last axis; unsqueeze/squeeze rather than einsum
        # so the shape contract is visible.
        return loc + torch.matmul(factor, noise.unsqueeze(-1)).squeeze(-1)

    base: Normal = distribution.base_dist  # type: ignore[assignment]
    noise = torch.randn(
        base.loc.shape, generator=generator, dtype=base.loc.dtype
    ).to(base.loc.device)
    return base.loc + base.scale * noise


def sigma_stats(stats: LatentStats, prefix: str) -> dict[str, float]:
    """Summarize a latent distribution's predicted standard deviations.

    Args:
        stats: The parameters the encoder produced.
        prefix: Metric name prefix, e.g. ``"prior"``.

    Returns:
        Mean and standard deviation of sigma, and the mean absolute mu.
    """
    # .detach() before the float() conversions: these are logging scalars, and reading a
    # tensor that still requires grad both warns and needlessly keeps the graph alive.
    # Numerically identical -- the Phase 1 fingerprint asserts that.
    mu = stats.mu.detach()
    sigma = stats.sigma.detach()
    return {
        f"{prefix}_sigma_mean": float(sigma.mean()),
        f"{prefix}_sigma_std": float(sigma.std()),
        f"{prefix}_sigma_min": float(sigma.min()),
        f"{prefix}_mu_abs_mean": float(mu.abs().mean()),
    }


def per_dim_kl(posterior: LatentDistribution, prior: LatentDistribution) -> Tensor:
    """Axis-aligned KL per latent **coordinate**, averaged over the batch.

    A dimension whose KL sits at zero is inactive along that coordinate axis. This is the
    series behind the Phase 1 finding that one of six dimensions carried 98.8% of the KL
    (FINDINGS 2.3), and it is kept for **both** parameterizations so that the Phase 1 and
    Phase 2 tables are the same measurement rather than two different ones.

    **For a full covariance this is a marginal, not a decomposition -- do not sum it.**
    The diagonal case is genuinely separable, so the values do add up to the total KL.
    The full case reports the KL between the two *marginals* along each coordinate, which
    ignores every cross-coordinate term; the values will not sum to ``kl_divergence`` and
    presenting them as a breakdown of it would be wrong. The quantity that does decompose
    the total exactly is :func:`whitened_kl_decomposition`, and it is also the one that
    does not assume the informative directions are coordinate axes -- which, once the
    covariance is full, they have no reason to be.

    Args:
        posterior: ``Q(z | X, Y)``.
        prior: ``P(z | X)``.

    Returns:
        A tensor of shape ``(latent_dim,)``.
    """
    if isinstance(posterior, MultivariateNormal) or isinstance(prior, MultivariateNormal):
        # Marginals of a multivariate Gaussian: mean i and the i-th diagonal entry of
        # Sigma = L L^T, whose square root is the marginal standard deviation.
        marginals = [
            Normal(
                loc=distribution.loc,
                scale=torch.sqrt(
                    (distribution.scale_tril**2).sum(dim=-1).clamp_min(0.0)
                ),
            )
            for distribution in (posterior, prior)
        ]
        return kl_divergence(marginals[0], marginals[1]).mean(dim=0)

    # The wrapped Independent sums over latent dims, so the base distributions are used
    # here to see which dimensions carry information. Unchanged from Phase 1 and pinned
    # by tests/fingerprints/phase1_latent.json.
    per_element = kl_divergence(posterior.base_dist, prior.base_dist)
    return per_element.mean(dim=0)


def effective_rank(weights: Tensor) -> Tensor:
    """Entropy-based effective rank ``exp(H)`` of a non-negative weight vector.

    The weights are normalized to sum to one and their Shannon entropy is exponentiated,
    giving the *effective number of contributing components*: ``1`` when one component
    carries everything, ``N`` when all ``N`` contribute equally, and a continuous value
    between. Continuity is the reason for using this rather than counting entries above a
    threshold -- a threshold would need a magic number, and CLAUDE.md forbids those.

    **What is fed in matters more than the formula, and the naive choice inverts the
    answer.** Applied to the raw eigenvalues of ``Sigma_q`` this measures how isotropic
    the posterior is, which is very nearly the opposite of how much information it
    carries: a posterior that has collapsed onto the prior in five of six directions is
    *almost isotropic*, so the raw spectrum reports a high rank precisely when the latent
    is most degenerate. Applied to the prior-whitened per-direction KL from
    :func:`whitened_kl_decomposition` it measures the number of directions that actually
    transmit information, which is the quantity FINDINGS 3.2 hypothesizes should rise
    under a full covariance. On Phase 1's measured geometry the two readings are 5.50 and
    1.00 respectively; ``test_effective_rank_inverts_on_the_raw_spectrum`` pins that gap
    so nobody can quietly swap the input back.

    Args:
        weights: Non-negative weights of shape ``(..., N)``. Negative entries are clamped
            to zero: the whitened KL terms are non-negative analytically, and any
            negative value is float round-off at the collapse floor.

    Returns:
        A tensor of shape ``(...)``, in ``[1, N]``.

    Note:
        If the weights sum to zero -- total KL underflowing, i.e. the posterior exactly
        equalling the prior in every direction -- the normalized vector is all zeros,
        ``H`` is 0 and this returns ``1.0``. That is the reading which argues *against*
        the Phase 2 hypothesis, so the degenerate case cannot manufacture support for it.
    """
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=-1, keepdim=True)
    # clamp_min rather than a branch: at total == 0 every p is 0, xlogy(0, 0) is 0, and
    # the result falls out as exp(0) == 1. See the note above.
    probabilities = weights / total.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
    return torch.exp(entropy)


def _loc_and_scale_tril(distribution: LatentDistribution) -> tuple[Tensor, Tensor]:
    """Return ``(loc, L)`` for either latent parameterization.

    Args:
        distribution: An ``Independent(Normal(...), 1)`` or a ``MultivariateNormal``.

    Returns:
        The mean of shape ``(B, N)`` and the lower-triangular factor of shape
        ``(B, N, N)``. For the diagonal family the factor is ``diag(sigma)``, which is
        the same covariance written in the form the whitening needs.
    """
    if isinstance(distribution, MultivariateNormal):
        return distribution.loc.detach(), distribution.scale_tril.detach()
    base: Normal = distribution.base_dist  # type: ignore[assignment]
    return base.loc.detach(), torch.diag_embed(base.scale.detach())


@dataclass(frozen=True)
class WhitenedKL:
    """The prior-whitened KL decomposition, per image.

    Attributes:
        per_direction: ``(B, N)`` non-negative per-direction KL contributions, sorted
            **descending** within each image. They sum to :attr:`total` exactly.
        eigenvalues: ``(B, N)`` eigenvalues of the prior-whitened posterior covariance,
            sorted descending. Dimensionless: ``1.0`` means the posterior matches the
            prior's spread in that direction.
        total: ``(B,)`` total KL per image, equal to ``kl_divergence(Q, P)``.
    """

    per_direction: Tensor
    eigenvalues: Tensor
    total: Tensor

    @property
    def effective_rank(self) -> Tensor:
        """``(B,)`` effective number of informative directions; see :func:`effective_rank`."""
        return effective_rank(self.per_direction)

    @property
    def min_eigenvalue_gap(self) -> Tensor:
        """``(B,)`` smallest gap between consecutive (descending) eigenvalues.

        The degeneracy companion. The eigendecomposition fixes individual directions only
        up to rotation *within* an eigenspace, so when two eigenvalues coincide the split
        of the mean-shift term between them is arbitrary and the individual
        :attr:`per_direction` values stop being meaningful -- while :attr:`total` and the
        sorted eigenvalue spectrum both remain exact. A gap near zero is the signal to
        read the total and the spectrum shape and not the per-direction values.

        Readable without a scale reference because the eigenvalues are ratios to the
        prior's variance, so ``1.0`` is a natural unit rather than an arbitrary one.
        """
        return (self.eigenvalues[..., :-1] - self.eigenvalues[..., 1:]).min(dim=-1).values


def whitened_kl_decomposition(
    posterior: LatentDistribution, prior: LatentDistribution
) -> WhitenedKL:
    r"""Decompose ``KL(Q || P)`` into exact, non-negative per-direction contributions.

    Whiten by the prior's Cholesky factor and then diagonalize what is left::

        M     = Lp^-1 Lq                 so   Sigma~ = M M^T = Lp^-1 Sigma_q Lp^-T
        Delta = Lp^-1 (mu_q - mu_p)
        Sigma~ = U diag(lambda) U^T      and  d = U^T Delta

        kl_i = 1/2 ( lambda_i + d_i^2 - 1 - ln lambda_i )     with   sum_i kl_i = KL(Q||P)

    Every term is non-negative (``1/2(x - 1 - ln x) >= 0`` for ``x > 0``), so this is a
    genuine decomposition rather than a signed rearrangement, and it is **invariant to
    rotations of the latent coordinates** -- which is the whole point once the covariance
    is full and the informative direction has no reason to lie along a coordinate axis.
    Unlike :func:`per_dim_kl` under a full covariance, these values really do sum to the
    total, and ``Trainer`` logs that total alongside them so the identity is checkable
    from the logs rather than taken on trust.

    **Ordering is by descending ``kl_i`` and there is deliberately no per-direction
    tracking across epochs.** A persistent per-direction identity does not merely take
    effort to compute -- under a full covariance it does not exist: the eigenbasis is free
    to rotate between epochs, so "direction 2" at epoch 100 and at epoch 125 need not
    describe the same subspace at all. What is comparable across epochs is the *shape* of
    the sorted spectrum and the effective rank derived from it. ``kl_dim_*`` remains the
    coordinate-indexed series for continuity with Phase 1.

    Three numerical choices, all load-bearing:

    * **The SVD of ``M``, not an eigendecomposition of ``M M^T``.** Forming the product
      squares the condition number, and ``eigh`` of a near-singular product can return a
      small *negative* eigenvalue, whose logarithm is NaN. With the SVD,
      ``lambda = s**2`` is non-negative by construction. This is the same argument that
      makes the model parameterize ``L`` rather than ``Sigma`` (Phase 2 spec,
      constraint 5), applied to the measurement instead of the model.
    * **``u - log1p(u)`` with ``u = lambda - 1``, never ``lambda - 1 - log(lambda)``.**
      **Do not "simplify" this back.** ``lambda ~ 1`` means the posterior matches the
      prior in that direction -- exactly the dead direction this measurement exists to
      detect -- and there ``lambda - 1`` and ``ln lambda`` are two nearly equal numbers
      whose difference is ``u**2/2``. Subtracting them independently rounded loses the
      answer to cancellation; ``log1p`` computes the small difference directly.
    * **The linear algebra runs on the CPU.** ``torch.linalg.eigh`` has no MPS kernel at
      all and ``linalg_svd`` only reaches MPS through a CPU fallback that warns on every
      call, so a device-resident decomposition would either crash or emit noise every
      validation epoch on the development machine. The matrices are ``N x N`` with
      ``N = 6``, so the transfer costs nothing and the choice is explicit rather than
      left to a fallback that ``PYTORCH_ENABLE_MPS_FALLBACK`` can switch off.

    Args:
        posterior: ``Q(z | X, Y)``, either latent family.
        prior: ``P(z | X)``, either latent family. The two need not match, though in
            practice both come from the same model and therefore do.

    Returns:
        The :class:`WhitenedKL` for the batch, on the CPU, in the input dtype.
    """
    mu_q, chol_q = _loc_and_scale_tril(posterior)
    mu_p, chol_p = _loc_and_scale_tril(prior)
    mu_q, chol_q, mu_p, chol_p = (
        tensor.cpu() for tensor in (mu_q, chol_q, mu_p, chol_p)
    )

    # Lp^-1 Lq and Lp^-1 (mu_q - mu_p) by triangular solve: never form an inverse.
    whitened_factor = torch.linalg.solve_triangular(chol_p, chol_q, upper=False)
    shift = torch.linalg.solve_triangular(
        chol_p, (mu_q - mu_p).unsqueeze(-1), upper=False
    ).squeeze(-1)

    # Left singular vectors of M diagonalize M M^T; the singular values are already
    # descending, so lambda comes out descending too.
    directions, singular_values, _ = torch.linalg.svd(whitened_factor)
    eigenvalues = singular_values * singular_values
    projected = torch.matmul(
        directions.transpose(-1, -2), shift.unsqueeze(-1)
    ).squeeze(-1)

    offset = eigenvalues - 1.0
    per_direction = 0.5 * (offset - torch.log1p(offset) + projected * projected)
    total = per_direction.sum(dim=-1)
    # Sorting after the sum, so the total is unaffected by the ordering.
    per_direction = per_direction.sort(dim=-1, descending=True).values
    return WhitenedKL(
        per_direction=per_direction, eigenvalues=eigenvalues, total=total
    )


def prior_spectrum(prior: LatentDistribution) -> dict[str, float]:
    """Eigenvalue spectrum and condition number of the prior's own covariance.

    **Why this is needed to read the whitened numbers correctly.** The whitened
    decomposition is asymmetric by construction: it divides out ``Sigma_p`` and asks how
    ``Q`` deviates from ``P``. That is the right frame for "how much information does the
    posterior transmit", and it stays right whatever shape the prior takes -- but it means
    ``lambda ~ 1`` says **"matches the prior in that direction"**, which is *not* the same
    claim as "is isotropic". If the prior itself becomes strongly anisotropic, a posterior
    sitting on it still reads ``lambda ~ 1`` while being far from spherical. Nothing about
    the collapse measurement breaks; the *English sentence* a reader attaches to it does.
    This family supplies the missing half, so the whitened spectrum can be reported
    against the frame it was measured in rather than against an assumed unit ball.

    It is also a secondary finding in its own right. Phase 1's prior is diagonal and can
    only be axis-aligned; a Phase 2 prior is free to correlate, and **"the prior itself
    became correlated"** would be a real result -- the prior is what inference actually
    samples from, so its geometry is closer to the reported metrics than the posterior's
    is.

    Computed from ``Sigma_p = Lp Lp^T`` via the singular values of ``Lp``, so the
    eigenvalues are ``s**2`` and non-negative by construction -- the same reasoning as in
    :func:`whitened_kl_decomposition`, and on the CPU for the same MPS-kernel reason.

    Args:
        prior: ``P(z | X)``, either latent family. Under the diagonal parameterization the
            eigenvalues are just the sorted variances, which is the correct Phase 1
            baseline for this series rather than a degenerate case to skip.

    Returns:
        ``prior_eigval_{i}`` for each direction (descending, batch-mean), the batch-mean
        ``prior_condition_number``, and ``prior_condition_number_max`` -- the worst image
        in the batch, since an average condition number hides the one prior that has gone
        singular.
    """
    _, factor = _loc_and_scale_tril(prior)
    eigenvalues = torch.linalg.svdvals(factor.cpu()) ** 2
    # svdvals returns descending, so index 0 is the largest.
    condition = eigenvalues[..., 0] / eigenvalues[..., -1].clamp_min(
        torch.finfo(eigenvalues.dtype).tiny
    )
    metrics = {
        f"prior_eigval_{index}": float(value)
        for index, value in enumerate(eigenvalues.mean(dim=0))
    }
    metrics["prior_condition_number"] = float(condition.mean())
    metrics["prior_condition_number_max"] = float(condition.max())
    return metrics


def whitened_kl_snapshot(decomposition: WhitenedKL) -> dict[str, float]:
    """Flatten a decomposition into the per-epoch snapshot scalars.

    These share the one-batch, grader-0 sampling of :func:`per_dim_kl` as used by
    ``Trainer._latent_stats``, so they carry the same caveat FINDINGS 2.3 records: they
    are a *diagnostic snapshot*, not a population estimate. The full-validation-set
    counterpart is :class:`EffectiveRankAccumulator`, and the two families are named
    ``*_snapshot`` and ``*_val`` so a report table can never confuse them.

    Args:
        decomposition: The batch's :class:`WhitenedKL`.

    Returns:
        ``kl_whitened_{i}`` for each direction (descending), the batch-mean total as
        ``kl_snapshot_total``, the batch-worst ``whitened_eigengap_min``, and the
        batch-mean ``effrank_snapshot``.
    """
    per_direction = decomposition.per_direction.mean(dim=0)
    metrics = {
        f"kl_whitened_{index}": float(value)
        for index, value in enumerate(per_direction)
    }
    metrics["kl_snapshot_total"] = float(decomposition.total.mean())
    # Minimum over directions AND over images: one degenerate image is enough to make the
    # batch-mean per-direction values unreliable, so the warning is deliberately the
    # worst case rather than an average of warnings.
    metrics["whitened_eigengap_min"] = float(decomposition.min_eigenvalue_gap.min())
    metrics["effrank_snapshot"] = float(decomposition.effective_rank.mean())
    return metrics


class EffectiveRankAccumulator:
    """Collects per-image effective rank across a full validation pass.

    **Why this exists rather than reusing the snapshot.** The snapshot is 32 images at one
    grader. That was adequate for Phase 1's finding, where one dimension carried 98.8% of
    the KL and the signal dwarfed any sampling error. It is *not* adequate for Phase 2's
    claim, which is that the effective rank **rose**: a move from 1.0 to 1.3 is exactly
    the size of thing a 32-image sample cannot distinguish from noise, and reporting it
    from a snapshot would be the kind of result this project's guardrails exist to
    prevent. So the headline number comes from every validation image at all four graders,
    with a dispersion estimate attached.

    The cost is genuinely marginal: ``Trainer.validate`` already computes the prior and
    all four posteriors for every validation batch, so this adds one batched ``6 x 6`` SVD
    and a small CPU transfer per posterior over work already done. It nonetheless runs at
    the **diagnostics cadence** rather than every epoch, to keep those transfers off the
    per-epoch path on a multi-day run.

    One image contributes four samples, one per grader. They are **pooled for the mean and
    clustered by image for the spread**, because those two jobs have different right
    answers: the grader-to-grader variation in how much information the posterior carries
    is real signal that belongs in the mean, but the four samples from one image are
    **not independent** -- they share an image, a prior, and a U-Net encoding. A standard
    error formed as ``pooled_std / sqrt(4 * images)`` would silently claim twice the
    precision the data supports. So the spread is also reported at the image level: average
    the four graders' effective rank within each image first, then take the standard
    deviation across the image-level values. ``effrank_val_image_std`` is the one to divide
    by ``sqrt(effrank_val_image_count)``; ``effrank_val_std`` is descriptive only.
    """

    def __init__(self, n_graders: int = 4) -> None:
        """Start an empty accumulator.

        Args:
            n_graders: Number of graders contributing per image. Used only to check that
                every grader saw the same images before clustering by image.
        """
        self._n_graders = n_graders
        self._ranks: dict[int, list[Tensor]] = {}
        self._totals: list[Tensor] = []
        self._per_direction: list[Tensor] = []
        self._prior_eigenvalues: list[Tensor] = []

    def update(
        self, posterior: LatentDistribution, prior: LatentDistribution, grader: int = 0
    ) -> None:
        """Accumulate one batch, at one grader.

        Args:
            posterior: ``Q(z | X, Y)`` for this batch and grader.
            prior: ``P(z | X)`` for this batch.
            grader: Which grader this posterior came from. Batches must arrive in the same
                image order for every grader, which is what lets the image-level clustering
                line the four values of an image up with each other -- true of
                ``Trainer.validate``, whose grader loop sits *inside* the batch loop.
        """
        decomposition = whitened_kl_decomposition(posterior, prior)
        self._ranks.setdefault(grader, []).append(decomposition.effective_rank)
        self._totals.append(decomposition.total)
        self._per_direction.append(decomposition.per_direction)
        if grader == 0:
            # The prior depends on the image alone, so it is identical across the four
            # graders. Accumulated from grader 0 only: taking it every time would count
            # each image four times and quietly turn the image count into a pair count.
            _, factor = _loc_and_scale_tril(prior)
            self._prior_eigenvalues.append(torch.linalg.svdvals(factor.cpu()) ** 2)

    def metrics(self) -> dict[str, float]:
        """Aggregate into report-ready scalars.

        Returns:
            The pooled mean and its descriptive spread and quartiles, the **image-clustered**
            standard deviation and the image count behind it, the pooled sample count, and
            the mean total KL. Empty if nothing was accumulated, so a caller can merge
            unconditionally. The image-clustered keys are omitted -- rather than faked --
            when the graders did not all cover the same images.

        Note:
            The mean of a per-image effective rank is **not** the effective rank of the
            mean covariance, and per-image is the deliberate choice. Averaging the
            covariances first would let images whose informative directions point
            different ways add up to an isotropic average, reporting a high rank for a
            population in which every individual posterior is rank-1. The question is how
            many directions *each* posterior uses, so each posterior is measured and the
            measurements are then summarized.
        """
        if not self._ranks:
            return {}
        per_grader = {
            grader: torch.cat(chunks).to(torch.float64)
            for grader, chunks in sorted(self._ranks.items())
        }
        ranks = torch.cat(list(per_grader.values()))
        totals = torch.cat(self._totals).to(torch.float64)
        quartiles = torch.quantile(ranks, torch.tensor([0.25, 0.75], dtype=torch.float64))
        metrics = {
            "effrank_val_mean": float(ranks.mean()),
            # Descriptive only -- these samples are not independent. Do NOT form a standard
            # error from this; use effrank_val_image_std instead.
            "effrank_val_std": float(ranks.std(unbiased=True)) if ranks.numel() > 1 else 0.0,
            "effrank_val_p25": float(quartiles[0]),
            "effrank_val_p75": float(quartiles[1]),
            "effrank_val_count": float(ranks.numel()),
            "kl_val_total_mean": float(totals.mean()),
        }

        # The full-validation counterpart of the kl_whitened_* snapshot: the spectrum
        # shape at a population size that can actually carry a before/after comparison.
        # Still sums to kl_val_total_mean, since every image's parts sum to its own total.
        spectrum = torch.cat(self._per_direction).to(torch.float64).mean(dim=0)
        metrics.update(
            {f"kl_whitened_val_{index}": float(value) for index, value in enumerate(spectrum)}
        )
        if self._prior_eigenvalues:
            prior_eigenvalues = torch.cat(self._prior_eigenvalues).to(torch.float64)
            condition = prior_eigenvalues[:, 0] / prior_eigenvalues[:, -1].clamp_min(
                torch.finfo(torch.float64).tiny
            )
            metrics.update(
                {
                    f"prior_eigval_val_{index}": float(value)
                    for index, value in enumerate(prior_eigenvalues.mean(dim=0))
                }
            )
            metrics["prior_condition_number_val"] = float(condition.mean())
            metrics["prior_condition_number_val_max"] = float(condition.max())

        sizes = {value.numel() for value in per_grader.values()}
        if len(per_grader) == self._n_graders and len(sizes) == 1:
            # (images, graders) -> mean within image -> spread across images.
            by_image = torch.stack(list(per_grader.values()), dim=1).mean(dim=1)
            metrics["effrank_val_image_mean"] = float(by_image.mean())
            metrics["effrank_val_image_std"] = (
                float(by_image.std(unbiased=True)) if by_image.numel() > 1 else 0.0
            )
            metrics["effrank_val_image_count"] = float(by_image.numel())
        return metrics


def mean_pairwise_iou(samples: Tensor) -> Tensor:
    """Mean pairwise IoU among sampled masks, averaged over images.

    Args:
        samples: Boolean or integer masks of shape ``(B, S, H, W)`` with ``S >= 2``.

    Returns:
        A scalar tensor in [0, 1]. Values near 1 mean the samples agree with each
        other -- which is either collapse or an all-empty model; read it together with
        :func:`nonempty_sample_fraction`.

    Raises:
        ValueError: If fewer than two samples are supplied.
    """
    if samples.dim() != 4:
        raise ValueError(f"expected (B, S, H, W), got {tuple(samples.shape)}")
    n_samples = samples.shape[1]
    if n_samples < 2:
        raise ValueError(f"need at least 2 samples to form a pair, got {n_samples}")
    scores = [
        binary_iou(samples[:, i], samples[:, j])
        for i in range(n_samples)
        for j in range(i + 1, n_samples)
    ]
    return torch.stack(scores, dim=0).mean()


def nonempty_sample_fraction(samples: Tensor) -> Tensor:
    """Fraction of sampled masks that contain at least one foreground pixel.

    Disambiguates a diversity score of 1.0: near zero means the model has not learned
    foreground yet, above zero means the prior really has collapsed.

    Args:
        samples: Masks of shape ``(B, S, H, W)``.

    Returns:
        A scalar tensor in [0, 1].
    """
    flat = samples.reshape(samples.shape[0], samples.shape[1], -1)
    return (flat != 0).any(dim=2).to(torch.float32).mean()


def make_panel(
    images: Tensor, grader_masks: Tensor, samples: Tensor
) -> Tensor:
    """Tile images, grader masks and prior samples into one greyscale grid.

    One row per image: the image, then its four grader masks, then the prior samples.
    Tiled by hand rather than with ``torchvision.utils.make_grid`` to avoid adding a
    dependency for fifteen lines of indexing.

    Args:
        images: Images of shape ``(B, 1, H, W)``, values in [0, 1].
        grader_masks: Grader masks of shape ``(B, 4, H, W)``.
        samples: Prior sample masks of shape ``(B, S, H, W)``.

    Returns:
        A tensor of shape ``(1, rows, cols)`` suitable for ``add_image``.
    """
    batch, _, height, width = images.shape
    columns = 1 + grader_masks.shape[1] + samples.shape[1]
    cell_h, cell_w = height + PANEL_PAD, width + PANEL_PAD
    panel = torch.full(
        (1, batch * cell_h, columns * cell_w),
        PANEL_SEPARATOR_VALUE,
        dtype=torch.float32,
    )

    def place(row: int, column: int, tile: Tensor) -> None:
        top = row * cell_h + PANEL_PAD // 2
        left = column * cell_w + PANEL_PAD // 2
        panel[0, top : top + height, left : left + width] = tile

    for row in range(batch):
        place(row, 0, images[row, 0].detach().to(torch.float32).cpu())
        for grader in range(grader_masks.shape[1]):
            place(row, 1 + grader, grader_masks[row, grader].detach().to(torch.float32).cpu())
        for sample in range(samples.shape[1]):
            place(
                row,
                1 + grader_masks.shape[1] + sample,
                samples[row, sample].detach().to(torch.float32).cpu(),
            )
    return panel.clamp(0.0, 1.0)


def logits_to_mask(logits: Tensor) -> Tensor:
    """Convert class logits to a binary foreground mask.

    Args:
        logits: Logits of shape ``(..., C, H, W)``.

    Returns:
        A uint8 mask of shape ``(..., H, W)``, 1 where the foreground class wins.
    """
    return logits.argmax(dim=-3).to(torch.uint8)


def prior_samples_for_images(
    model: ProbUNet,
    encoded: Encoded,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw prior samples as hard masks, re-running only ``f_comb``.

    Args:
        model: The model.
        encoded: State from ``model.encode``, whose U-Net pass is reused.
        n_samples: Samples per image.
        generator: Optional generator for reproducible noise.

    Returns:
        A uint8 mask tensor of shape ``(B, n_samples, H, W)``.
    """
    masks = []
    for _ in range(n_samples):
        z = reparameterize(encoded.prior, generator)
        masks.append(logits_to_mask(model.reconstruct(encoded, z)))
    return torch.stack(masks, dim=1)
