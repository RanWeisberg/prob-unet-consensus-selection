"""Tests for the negative-ELBO objective.

The reduction convention is the thing that fails silently, so it is pinned from both
directions: the KL is checked against hand-computed values for known diagonal
Gaussians (including the identical case, where it must be exactly zero), and the
cross-entropy against hand-computed values for constant logits.

Analytic KL for diagonal Gaussians, per dimension::

    KL(q||p) = log(sigma_p / sigma_q)
             + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2)
             - 1/2
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Independent, MultivariateNormal, Normal, kl_divergence

from probunet.losses import ElboConfig, ElboLoss, elbo_from_output, elbo_loss, kl_term
from probunet.model import ProbUNet, ProbUNetConfig, ProbUNetOutput

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_NPZ = REPO_ROOT / "data" / "processed" / "lidc.npz"
REAL_SPLIT = REPO_ROOT / "data" / "splits" / "split.json"

LN2 = math.log(2.0)


def diagonal(mu: float, sigma: float, dims: int, batch: int = 1) -> Independent:
    """Build an ``Independent(Normal, 1)`` with constant mean and scale.

    Args:
        mu: Mean for every dimension.
        sigma: Standard deviation for every dimension.
        dims: Number of latent dimensions.
        batch: Batch size.

    Returns:
        The distribution, with batch shape ``(batch,)`` and event shape ``(dims,)``.
    """
    return Independent(
        Normal(torch.full((batch, dims), mu), torch.full((batch, dims), sigma)), 1
    )


# --------------------------------------------------------------------------- #
# KL against hand-computed values
# --------------------------------------------------------------------------- #
def test_kl_is_exactly_zero_for_identical_distributions() -> None:
    """KL(q||q) must be exactly 0, not merely small."""
    for dims in (1, 6):
        distribution = diagonal(0.0, 1.0, dims)
        assert kl_term(distribution, distribution).item() == 0.0

    shifted = diagonal(2.5, 0.7, 6)
    assert kl_term(shifted, shifted).item() == 0.0


def test_kl_shifted_mean() -> None:
    """mu_q=1, mu_p=0, sigma=1: 0.5 per dim, so 3.0 over 6 dims."""
    posterior = diagonal(1.0, 1.0, 6)
    prior = diagonal(0.0, 1.0, 6)
    assert kl_term(posterior, prior).item() == pytest.approx(3.0, abs=1e-6)


def test_kl_wider_posterior() -> None:
    """sigma_q=2, sigma_p=1, equal means: log(1/2) + 4/2 - 1/2 = 0.8068528."""
    expected = math.log(1.0 / 2.0) + (2.0**2) / (2.0 * 1.0**2) - 0.5
    assert expected == pytest.approx(0.8068528, abs=1e-6)
    value = kl_term(diagonal(0.0, 2.0, 1), diagonal(0.0, 1.0, 1))
    assert value.item() == pytest.approx(expected, abs=1e-6)


def test_kl_narrower_posterior() -> None:
    """sigma_q=1, sigma_p=2, equal means: log(2) + 1/8 - 1/2 = 0.3181472."""
    expected = math.log(2.0) + (1.0**2) / (2.0 * 2.0**2) - 0.5
    assert expected == pytest.approx(0.3181472, abs=1e-6)
    value = kl_term(diagonal(0.0, 1.0, 1), diagonal(0.0, 2.0, 1))
    assert value.item() == pytest.approx(expected, abs=1e-6)


def test_kl_combined_shift_and_scale() -> None:
    """A case with both a mean shift and a scale change, over several dims."""
    mu_q, sigma_q, mu_p, sigma_p, dims = 0.5, 1.5, -0.25, 0.8, 4
    per_dim = (
        math.log(sigma_p / sigma_q)
        + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2 * sigma_p**2)
        - 0.5
    )
    value = kl_term(diagonal(mu_q, sigma_q, dims), diagonal(mu_p, sigma_p, dims))
    assert value.item() == pytest.approx(dims * per_dim, rel=1e-6)


def test_kl_sums_over_latent_dims_not_averages() -> None:
    """Doubling the latent dimension doubles the KL: it is summed, not averaged."""
    six = kl_term(diagonal(1.0, 1.0, 6), diagonal(0.0, 1.0, 6))
    twelve = kl_term(diagonal(1.0, 1.0, 12), diagonal(0.0, 1.0, 12))
    assert twelve.item() == pytest.approx(2.0 * six.item(), rel=1e-6)


def test_kl_averages_over_batch() -> None:
    """Batch size does not change the KL magnitude: it is averaged."""
    one = kl_term(diagonal(1.0, 1.0, 6, batch=1), diagonal(0.0, 1.0, 6, batch=1))
    eight = kl_term(diagonal(1.0, 1.0, 6, batch=8), diagonal(0.0, 1.0, 6, batch=8))
    assert eight.item() == pytest.approx(one.item(), rel=1e-6)


def test_plain_normal_is_rejected() -> None:
    """A bare Normal would average over latent dims instead of summing.

    Still rejected after Phase 2 widened the guard to admit MultivariateNormal. This is
    the case the guard exists for: a plain Normal's kl_divergence is
    per-latent-dimension, so the batch mean would silently divide the KL by latent_dim
    and redefine beta.
    """
    posterior = Normal(torch.zeros(2, 6), torch.ones(2, 6))
    prior = Normal(torch.zeros(2, 6), torch.ones(2, 6))
    with pytest.raises(TypeError, match="expected one of"):
        kl_term(posterior, prior)


def test_plain_normal_is_rejected_when_its_shape_would_slip_through() -> None:
    """The type check is why this fails; the shape check alone would pass it.

    A Normal with ``(B,)``-shaped parameters gives a ``(B,)``-shaped kl_divergence, which
    satisfies "one value per batch element" while still meaning "a single latent dimension,
    then averaged". The shape guard cannot tell that apart from a correctly wrapped
    distribution, so the type guard is what catches it.
    """
    posterior = Normal(torch.zeros(4), torch.ones(4))
    prior = Normal(torch.zeros(4), torch.ones(4))
    assert kl_divergence(posterior, prior).shape == (4,)  # a shape-only guard would pass
    with pytest.raises(TypeError, match="expected one of"):
        kl_term(posterior, prior)


def test_wrongly_wrapped_independent_is_rejected() -> None:
    """The shape check is why this fails; the type check alone would pass it.

    An Independent with reinterpreted_batch_ndims=2 folds the batch axis into the event
    shape too, so kl_divergence returns a scalar and the batch mean silently disappears.
    """
    posterior = Independent(Normal(torch.zeros(2, 6), torch.ones(2, 6)), 2)
    prior = Independent(Normal(torch.zeros(2, 6), torch.ones(2, 6)), 2)
    with pytest.raises(ValueError, match="one value per batch element"):
        kl_term(posterior, prior)


# --------------------------------------------------------------------------- #
# Phase 2: the KL path accepts a full covariance
# --------------------------------------------------------------------------- #
def full(mu: float, sigma: float, dims: int, batch: int = 1, correlation: float = 0.0):
    """Build a MultivariateNormal with a constant diagonal and one off-diagonal entry.

    Args:
        mu: Constant mean.
        sigma: Constant diagonal of the Cholesky factor.
        dims: Latent dimensionality.
        batch: Batch size.
        correlation: Value placed in the strict lower triangle.

    Returns:
        The distribution.
    """
    factor = torch.diag_embed(torch.full((batch, dims), sigma))
    if correlation:
        mask = torch.ones(dims, dims, dtype=torch.bool).tril(-1)
        factor = factor.clone()
        factor[:, mask] = correlation
    return MultivariateNormal(torch.full((batch, dims), mu), scale_tril=factor)


def test_multivariate_normal_is_accepted() -> None:
    """The Phase 2 latent passes the guard and returns a scalar."""
    value = kl_term(full(1.0, 1.0, 6, batch=4, correlation=0.3), full(0.0, 1.0, 6, batch=4))
    assert value.dim() == 0
    assert torch.isfinite(value)
    assert value > 0


def test_full_kl_is_zero_for_identical_distributions() -> None:
    """The same sanity check the diagonal path gets, on the full path."""
    same = full(0.5, 1.2, 6, batch=3, correlation=0.4)
    assert kl_term(same, same).item() == pytest.approx(0.0, abs=1e-6)


def test_full_kl_reduction_matches_the_diagonal_convention() -> None:
    """Sum over latent dims, mean over batch -- unchanged by the flag.

    Asserted the same two ways the diagonal path is: the KL scales with latent_dim (so it
    is summed, not averaged), and it does not scale with batch size (so it is meaned).
    """
    two = kl_term(full(1.0, 1.0, 2, batch=1), full(0.0, 1.0, 2, batch=1))
    six = kl_term(full(1.0, 1.0, 6, batch=1), full(0.0, 1.0, 6, batch=1))
    assert six.item() == pytest.approx(3.0 * two.item(), rel=1e-5)

    one = kl_term(full(1.0, 1.0, 6, batch=1), full(0.0, 1.0, 6, batch=1))
    eight = kl_term(full(1.0, 1.0, 6, batch=8), full(0.0, 1.0, 6, batch=8))
    assert eight.item() == pytest.approx(one.item(), rel=1e-5)


def test_mixing_the_two_families_fails_loudly() -> None:
    """A full posterior against a diagonal prior raises rather than misreducing.

    torch registers no KL between MultivariateNormal and Independent(Normal), so the pair
    raises NotImplementedError from torch itself. That is the safe outcome and worth
    pinning: this project never mixes the families -- both latent nets are built from the
    same ``model.latent_covariance`` flag, and
    ``LatentEncoder.distribution_from_stats`` refuses a mismatch between its own
    configuration and its parameters -- but if a future change ever produced the pair, it
    would stop rather than quietly compute a wrong objective.
    """
    with pytest.raises(NotImplementedError, match="No KL"):
        kl_term(
            full(1.0, 1.0, 6, batch=4, correlation=0.2), diagonal(0.0, 1.0, 6, batch=4)
        )


# --------------------------------------------------------------------------- #
# Cross-entropy reduction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("batch", "size"), [(1, 2), (2, 2), (4, 8)])
def test_ce_is_summed_over_pixels_and_meaned_over_batch(batch: int, size: int) -> None:
    """Constant zero logits give ln(2) per pixel, so ce = pixels * ln(2)."""
    logits = torch.zeros(batch, 2, size, size)
    target = torch.zeros(batch, size, size, dtype=torch.int64)
    posterior = prior = diagonal(0.0, 1.0, 6, batch=batch)

    terms = elbo_loss(logits, target, posterior, prior)
    pixels = size * size
    assert terms["ce_per_pixel"].item() == pytest.approx(LN2, abs=1e-6)
    assert terms["ce"].item() == pytest.approx(pixels * LN2, rel=1e-6)
    # KL is zero here, so the total is the CE.
    assert terms["kl"].item() == 0.0
    assert terms["total"].item() == pytest.approx(terms["ce"].item(), rel=1e-6)


def test_ce_matches_hand_computed_asymmetric_logits() -> None:
    """logits [2, 0] with target 0 gives log(1 + e^-2) per pixel."""
    batch, size = 2, 2
    logits = torch.zeros(batch, 2, size, size)
    logits[:, 0] = 2.0
    target = torch.zeros(batch, size, size, dtype=torch.int64)
    expected_per_pixel = math.log(1.0 + math.exp(-2.0))
    assert expected_per_pixel == pytest.approx(0.126928, abs=1e-6)

    terms = elbo_loss(logits, target, diagonal(0.0, 1.0, 6, batch), diagonal(0.0, 1.0, 6, batch))
    assert terms["ce_per_pixel"].item() == pytest.approx(expected_per_pixel, abs=1e-6)
    assert terms["ce"].item() == pytest.approx(size * size * expected_per_pixel, rel=1e-6)


def test_ce_equals_torch_sum_reduction_over_batch() -> None:
    """The documented identity with F.cross_entropy(reduction='sum')."""
    torch.manual_seed(0)
    batch, size = 3, 4
    logits = torch.randn(batch, 2, size, size)
    target = (torch.rand(batch, size, size) > 0.5).to(torch.int64)
    terms = elbo_loss(logits, target, diagonal(0.0, 1.0, 6, batch), diagonal(0.0, 1.0, 6, batch))

    expected = F.cross_entropy(logits, target, reduction="sum") / batch
    assert terms["ce"].item() == pytest.approx(expected.item(), rel=1e-6)
    assert terms["ce"].item() == pytest.approx(
        terms["ce_per_pixel"].item() * size * size, rel=1e-6
    )


def test_ce_does_not_scale_with_batch_size() -> None:
    """Averaging over the batch means a repeated batch has the same CE."""
    logits = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.int64)
    single = elbo_loss(logits, target, diagonal(0.0, 1.0, 6), diagonal(0.0, 1.0, 6))

    repeated = elbo_loss(
        logits.repeat(5, 1, 1, 1),
        target.repeat(5, 1, 1),
        diagonal(0.0, 1.0, 6, batch=5),
        diagonal(0.0, 1.0, 6, batch=5),
    )
    assert repeated["ce"].item() == pytest.approx(single["ce"].item(), rel=1e-6)


def test_ce_scales_with_pixel_count() -> None:
    """Doubling each spatial dimension quadruples the CE: it is summed over pixels."""
    small = elbo_loss(
        torch.zeros(1, 2, 4, 4),
        torch.zeros(1, 4, 4, dtype=torch.int64),
        diagonal(0.0, 1.0, 6),
        diagonal(0.0, 1.0, 6),
    )
    large = elbo_loss(
        torch.zeros(1, 2, 8, 8),
        torch.zeros(1, 8, 8, dtype=torch.int64),
        diagonal(0.0, 1.0, 6),
        diagonal(0.0, 1.0, 6),
    )
    assert large["ce"].item() == pytest.approx(4.0 * small["ce"].item(), rel=1e-6)
    assert large["ce_per_pixel"].item() == pytest.approx(
        small["ce_per_pixel"].item(), rel=1e-6
    )


def test_batch_loss_is_the_mean_of_per_item_losses() -> None:
    """Both terms are batch means, so the whole is the mean of the parts."""
    torch.manual_seed(1)
    batch, size = 4, 4
    logits = torch.randn(batch, 2, size, size)
    target = (torch.rand(batch, size, size) > 0.5).to(torch.int64)
    mus = torch.randn(batch, 6)
    posterior = Independent(Normal(mus, torch.ones(batch, 6)), 1)
    prior = Independent(Normal(torch.zeros(batch, 6), torch.ones(batch, 6)), 1)

    whole = elbo_loss(logits, target, posterior, prior)
    parts = [
        elbo_loss(
            logits[i : i + 1],
            target[i : i + 1],
            Independent(Normal(mus[i : i + 1], torch.ones(1, 6)), 1),
            Independent(Normal(torch.zeros(1, 6), torch.ones(1, 6)), 1),
        )
        for i in range(batch)
    ]
    for key in ("total", "ce", "kl"):
        expected = sum(part[key].item() for part in parts) / batch
        assert whole[key].item() == pytest.approx(expected, rel=1e-5), key


# --------------------------------------------------------------------------- #
# beta
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0, 10.0])
def test_beta_scales_only_the_kl_term(beta: float) -> None:
    """total = ce + beta * kl, exactly."""
    torch.manual_seed(2)
    logits = torch.randn(2, 2, 4, 4)
    target = (torch.rand(2, 4, 4) > 0.5).to(torch.int64)
    posterior = diagonal(1.0, 1.0, 6, batch=2)
    prior = diagonal(0.0, 1.0, 6, batch=2)

    terms = elbo_loss(logits, target, posterior, prior, beta=beta)
    assert terms["total"].item() == pytest.approx(
        terms["ce"].item() + beta * terms["kl"].item(), rel=1e-6
    )


def test_beta_zero_removes_the_kl() -> None:
    """beta=0 leaves the reconstruction term alone."""
    terms = elbo_loss(
        torch.zeros(2, 2, 4, 4),
        torch.zeros(2, 4, 4, dtype=torch.int64),
        diagonal(1.0, 1.0, 6, batch=2),
        diagonal(0.0, 1.0, 6, batch=2),
        beta=0.0,
    )
    assert terms["kl"].item() > 0
    assert terms["total"].item() == pytest.approx(terms["ce"].item(), rel=1e-9)


def test_negative_beta_rejected() -> None:
    """A negative beta would reward divergence from the prior."""
    with pytest.raises(ValueError, match="beta"):
        ElboConfig(beta=-0.1)
    with pytest.raises(ValueError, match="beta"):
        elbo_loss(
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 2, 2, dtype=torch.int64),
            diagonal(0.0, 1.0, 6),
            diagonal(0.0, 1.0, 6),
            beta=-1.0,
        )


def test_module_wrapper_matches_function() -> None:
    """ElboLoss is a thin wrapper around elbo_loss."""
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 4, 4)
    target = (torch.rand(2, 4, 4) > 0.5).to(torch.int64)
    posterior = diagonal(0.5, 1.2, 6, batch=2)
    prior = diagonal(0.0, 1.0, 6, batch=2)

    loss = ElboLoss(ElboConfig(beta=2.0))
    assert loss.beta == 2.0
    from_module = loss(logits, target, posterior, prior)
    from_function = elbo_loss(logits, target, posterior, prior, beta=2.0)
    for key in from_function:
        assert from_module[key].item() == pytest.approx(from_function[key].item())


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_input_validation() -> None:
    """Rank, dtype and shape violations are named clearly."""
    posterior = prior = diagonal(0.0, 1.0, 6, batch=2)
    target = torch.zeros(2, 4, 4, dtype=torch.int64)

    with pytest.raises(ValueError, match=r"logits must be"):
        elbo_loss(torch.zeros(2, 4, 4), target, posterior, prior)
    with pytest.raises(ValueError, match=r"target must be \(B, H, W\)"):
        elbo_loss(torch.zeros(2, 2, 4, 4), torch.zeros(2, 1, 4, 4, dtype=torch.int64), posterior, prior)
    with pytest.raises(ValueError, match="int64"):
        elbo_loss(torch.zeros(2, 2, 4, 4), target.float(), posterior, prior)
    with pytest.raises(ValueError, match="batch mismatch"):
        elbo_loss(torch.zeros(3, 2, 4, 4), target, posterior, prior)
    with pytest.raises(ValueError, match="spatial mismatch"):
        elbo_loss(torch.zeros(2, 2, 8, 8), target, posterior, prior)


# --------------------------------------------------------------------------- #
# Numerics and dtype
# --------------------------------------------------------------------------- #
def test_all_terms_are_float32() -> None:
    """MPS has no float64: nothing may silently upcast."""
    import numpy as np

    terms = elbo_loss(
        torch.zeros(2, 2, 4, 4),
        torch.zeros(2, 4, 4, dtype=torch.int64),
        diagonal(0.5, 1.0, 6, batch=2),
        diagonal(0.0, 1.0, 6, batch=2),
        # A numpy float64 beta must not drag the objective to float64.
        beta=np.float64(1.0),
    )
    for key, value in terms.items():
        assert value.dtype == torch.float32, key
        assert value.dim() == 0, key


def test_extreme_logits_stay_finite() -> None:
    """Cross-entropy is computed in a log-sum-exp-stable way."""
    for magnitude in (50.0, -50.0):
        logits = torch.full((2, 2, 4, 4), magnitude)
        logits[:, 0] = -magnitude
        terms = elbo_loss(
            logits,
            torch.zeros(2, 4, 4, dtype=torch.int64),
            diagonal(0.0, 1.0, 6, batch=2),
            diagonal(0.0, 1.0, 6, batch=2),
        )
        assert torch.isfinite(terms["total"]), magnitude


def test_all_empty_target_is_valid() -> None:
    """An all-background target is legitimate data, not an error."""
    torch.manual_seed(4)
    logits = torch.randn(2, 2, 8, 8)
    target = torch.zeros(2, 8, 8, dtype=torch.int64)
    terms = elbo_loss(logits, target, diagonal(0.0, 1.0, 6, batch=2), diagonal(0.0, 1.0, 6, batch=2))
    assert torch.isfinite(terms["total"])
    assert terms["ce"].item() > 0


# --------------------------------------------------------------------------- #
# Integration with the model
# --------------------------------------------------------------------------- #
def test_gradients_reach_every_component() -> None:
    """Unlike the logits alone, the full objective reaches the prior too.

    In sub-stage 1 ``logits.sum().backward()`` left ``prior_net.grad`` as None,
    because z comes from the posterior. The KL term is the only path to the prior, so
    this is the test that the objective actually trains it.
    """
    torch.manual_seed(5)
    model = ProbUNet(ProbUNetConfig())
    image = torch.rand(2, 1, 32, 32)
    target = (torch.rand(2, 32, 32) > 0.7).to(torch.int64)

    output = model(image, target)
    terms = elbo_from_output(output, target)
    terms["total"].backward()

    for name, module in (
        ("unet", model.unet),
        ("prior_net", model.prior_net),
        ("posterior_net", model.posterior_net),
        ("fcomb", model.fcomb),
    ):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} received no gradient from the objective"
        assert all(torch.isfinite(g).all() for g in grads), f"{name} has non-finite grads"


def test_elbo_from_output_requires_a_posterior() -> None:
    """Without a posterior, z did not come from Q and the ELBO is ill-defined."""
    torch.manual_seed(6)
    model = ProbUNet(ProbUNetConfig())
    image = torch.rand(1, 1, 32, 32)
    target = torch.zeros(1, 32, 32, dtype=torch.int64)

    encoded = model.encode(image)  # no mask -> no posterior
    assert encoded.posterior is None
    z = encoded.prior.rsample()
    output = ProbUNetOutput(logits=model.reconstruct(encoded, z), z=z, encoded=encoded)
    with pytest.raises(ValueError, match="no posterior"):
        elbo_from_output(output, target)


def test_deterministic_for_fixed_inputs() -> None:
    """The objective is a pure function of its arguments."""
    torch.manual_seed(7)
    logits = torch.randn(2, 2, 4, 4)
    target = (torch.rand(2, 4, 4) > 0.5).to(torch.int64)
    posterior = diagonal(0.3, 0.9, 6, batch=2)
    prior = diagonal(0.0, 1.0, 6, batch=2)

    first = elbo_loss(logits, target, posterior, prior)
    second = elbo_loss(logits, target, posterior, prior)
    for key in first:
        assert first[key].item() == second[key].item(), key


# --------------------------------------------------------------------------- #
# Magnitude regression on real data
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (REAL_NPZ.exists() and REAL_SPLIT.exists()), reason="converted dataset absent"
)
def test_initial_magnitudes_on_real_data() -> None:
    """Pin the measured balance at initialization.

    Under the sum-over-pixels convention CE is ~1.9e4 per image and KL is well under
    1, so the KL is a negligible fraction of the objective at the start. If a future
    change switches the CE to a pixel mean, ce drops by 16384x and this fails --
    which is the point, because that change would silently redefine beta.
    """
    from probunet.data.lidc import DataConfig, build_data

    torch.manual_seed(0)
    model = ProbUNet(ProbUNetConfig())
    data = build_data(DataConfig(batch_size=32))
    data.set_epoch(0)
    batch = next(iter(data.loaders["train"]))

    with torch.no_grad():
        output = model(batch["image"], batch["mask"])
        terms = elbo_from_output(output, batch["mask"])

    pixels = 128 * 128
    assert 15_000 < terms["ce"].item() < 25_000, terms["ce"].item()
    assert 0.0 < terms["kl"].item() < 1.0, terms["kl"].item()
    assert terms["ce"].item() == pytest.approx(
        terms["ce_per_pixel"].item() * pixels, rel=1e-5
    )
    # KL is a negligible share of the objective at initialization.
    assert terms["kl"].item() / terms["ce"].item() < 1e-4
