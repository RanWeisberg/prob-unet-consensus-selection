"""The Probabilistic U-Net training objective (negative ELBO).

``L = CE(prediction, Y) + beta * KL(Q || P)``, minimized.

**Reduction convention -- read this before changing anything here.**

* Cross-entropy is **summed over pixels and averaged over the batch**.
* KL is **summed over latent dimensions and averaged over the batch**. The sum over
  latent dims is not written by hand: the distributions are
  ``Independent(Normal(...), 1)``, so ``kl_divergence`` returns one value per batch
  element and the sum is part of the type. :func:`elbo_loss` rejects distributions
  that are not wrapped that way, because a plain ``Normal`` would silently give a
  per-latent-dimension KL that then gets averaged instead of summed.

``beta`` is only meaningful relative to this convention, and both failure modes are
silent:

* Using **pixel-mean** cross-entropy instead of the pixel sum multiplies the relative
  weight of the KL term by the number of pixels -- 16384 at 128x128. Measured on real
  batches at initialization, KL is 0.0006% of the objective under the sum convention
  and 10.5% under the mean convention, and that share grows as CE falls, which drives
  the posterior to collapse onto the prior.
* Weighting the KL too weakly leaves the latent space unconstrained, so the prior
  never learns to cover the grader variants. The training loss cannot reveal this;
  only comparing posterior-z against prior-z reconstructions can.

The convention above is the one the authors' TF1 code uses:
``utils/training_utils.py`` returns ``ce_sum = reduce_sum(ce_per_pixel) / batch_size``,
``model/probabilistic_unet.py`` takes ``rec_loss['sum']`` for the objective and
``reduce_mean`` over an already latent-summed ``MultivariateNormalDiag`` KL, and
``training/prob_unet_config.py`` sets ``beta = 1.0``. The learning rate of 1e-4 is
calibrated to this gradient scale.

There is deliberately **no reduction option**. A configurable reduction is precisely
how ``beta`` comes to mean something other than the paper's ``beta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution, kl_divergence

if TYPE_CHECKING:  # import only for type checking: losses must not import model at runtime
    from probunet.model.prob_unet import ProbUNetOutput

DEFAULT_BETA: float = 1.0
"""The paper's value (``prob_unet_config.py``), valid under the reduction above."""


@dataclass(frozen=True)
class ElboConfig:
    """Configuration of the training objective.

    Attributes:
        beta: Weight of the KL term. 1.0 is the paper's value and is only meaningful
            under this module's reduction convention.

    Raises:
        ValueError: If ``beta`` is negative.
    """

    beta: float = DEFAULT_BETA

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")


def _validate(logits: Tensor, target: Tensor) -> None:
    """Check the prediction/target contract.

    Args:
        logits: Class logits.
        target: Class-index target.

    Raises:
        ValueError: If ranks, dtypes or shapes are wrong.
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B, C, H, W), got {tuple(logits.shape)}")
    if target.dim() != 3:
        raise ValueError(
            f"target must be (B, H, W) class indices, got {tuple(target.shape)}"
        )
    if target.dtype != torch.int64:
        raise ValueError(
            f"target must be int64 class indices for cross-entropy, got {target.dtype}"
        )
    if logits.shape[0] != target.shape[0]:
        raise ValueError(
            f"batch mismatch: logits {logits.shape[0]} vs target {target.shape[0]}"
        )
    if logits.shape[2:] != target.shape[1:]:
        raise ValueError(
            f"spatial mismatch: logits {tuple(logits.shape[2:])} vs target "
            f"{tuple(target.shape[1:])}"
        )


def kl_term(posterior: Distribution, prior: Distribution) -> Tensor:
    """KL(Q || P), summed over latent dims and averaged over the batch.

    Args:
        posterior: ``Q(z | X, Y)`` as ``Independent(Normal(...), 1)``.
        prior: ``P(z | X)`` as ``Independent(Normal(...), 1)``.

    Returns:
        A scalar tensor.

    Raises:
        ValueError: If ``kl_divergence`` does not return one value per batch element.
            That happens when the distributions are plain ``Normal`` rather than
            ``Independent(..., 1)``, in which case the latent dimensions would be
            averaged rather than summed -- a silent factor of ``latent_dim``.
    """
    per_item = kl_divergence(posterior, prior)
    if per_item.dim() != 1:
        raise ValueError(
            "kl_divergence returned shape "
            f"{tuple(per_item.shape)}; expected one value per batch element. Wrap the "
            "latent distributions in Independent(Normal(...), 1) so the sum over "
            "latent dimensions is explicit."
        )
    return per_item.mean()


def elbo_loss(
    logits: Tensor,
    target: Tensor,
    posterior: Distribution,
    prior: Distribution,
    beta: float = DEFAULT_BETA,
) -> dict[str, Tensor]:
    """Compute the negative ELBO and its parts.

    Args:
        logits: Class logits of shape ``(B, C, H, W)``.
        target: Class indices of shape ``(B, H, W)``, dtype int64.
        posterior: ``Q(z | X, Y)``, an ``Independent(Normal(...), 1)``.
        prior: ``P(z | X)``, an ``Independent(Normal(...), 1)``.
        beta: Weight of the KL term.

    Returns:
        A dict of scalar tensors:

        * ``total`` -- ``ce + beta * kl``, the quantity to minimize. This is the
          negative ELBO; the reference returns ``elbo = -(rec + beta*kl)`` and
          minimizes ``-elbo``, so the sign works out the same.
        * ``ce`` -- cross-entropy, summed over pixels, averaged over the batch.
        * ``kl`` -- KL, summed over latent dims, averaged over the batch.
        * ``ce_per_pixel`` -- ``ce`` divided by the pixel count, for logging only.
          The reference logs this quantity while optimizing ``ce``.

    Raises:
        ValueError: If the inputs violate the contract, or ``beta`` is negative.
    """
    _validate(logits, target)
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

    batch_size = logits.shape[0]
    pixels = logits.shape[-2] * logits.shape[-1]

    # Sum over pixels, mean over batch. Equivalent to the reference's
    # reduce_sum(ce_per_pixel) / batch_size.
    ce = F.cross_entropy(logits, target, reduction="sum") / batch_size
    kl = kl_term(posterior, prior)

    # as_tensor with ce's dtype keeps everything float32 even if beta arrives as a
    # numpy float64 scalar; MPS has no float64.
    beta_tensor = torch.as_tensor(beta, dtype=ce.dtype, device=ce.device)
    return {
        "total": ce + beta_tensor * kl,
        "ce": ce,
        "kl": kl,
        "ce_per_pixel": ce / pixels,
    }


def elbo_from_output(
    output: ProbUNetOutput,
    target: Tensor,
    beta: float = DEFAULT_BETA,
) -> dict[str, Tensor]:
    """Compute the objective directly from a :class:`ProbUNetOutput`.

    Args:
        output: The result of ``ProbUNet.forward``, whose ``z`` was drawn from the
            posterior.
        target: Class indices of shape ``(B, H, W)``, dtype int64.
        beta: Weight of the KL term.

    Returns:
        The same dict as :func:`elbo_loss`.

    Raises:
        ValueError: If the output carries no posterior, which means ``z`` did not come
            from ``Q(z | X, Y)`` and the objective would be ill-defined.
    """
    if output.posterior is None:
        raise ValueError(
            "output has no posterior: the ELBO requires z drawn from Q(z | X, Y). "
            "Call ProbUNet.forward(image, mask) rather than encode() without a mask."
        )
    return elbo_loss(output.logits, target, output.posterior, output.prior, beta=beta)


class ElboLoss(nn.Module):
    """Module wrapper holding ``beta``, for the training loop to own."""

    def __init__(self, config: ElboConfig | None = None) -> None:
        """Build the loss.

        Args:
            config: Objective configuration; defaults to :class:`ElboConfig`.
        """
        super().__init__()
        self.config = config or ElboConfig()

    @property
    def beta(self) -> float:
        """Weight of the KL term."""
        return self.config.beta

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        posterior: Distribution,
        prior: Distribution,
    ) -> dict[str, Tensor]:
        """Compute the objective.

        Args:
            logits: Class logits of shape ``(B, C, H, W)``.
            target: Class indices of shape ``(B, H, W)``, dtype int64.
            posterior: ``Q(z | X, Y)``.
            prior: ``P(z | X)``.

        Returns:
            The same dict as :func:`elbo_loss`.
        """
        return elbo_loss(logits, target, posterior, prior, beta=self.config.beta)
