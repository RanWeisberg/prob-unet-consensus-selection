"""Training objectives.

The baseline has exactly one: the negative ELBO of :mod:`probunet.losses.elbo`.
Alternative losses (Dice, focal, Tversky) belong to the flag-gated modernization
phase and are deliberately absent here.
"""

from probunet.losses.elbo import (
    DEFAULT_BETA,
    ElboConfig,
    ElboLoss,
    elbo_from_output,
    elbo_loss,
    kl_term,
)

__all__ = [
    "DEFAULT_BETA",
    "ElboConfig",
    "ElboLoss",
    "elbo_from_output",
    "elbo_loss",
    "kl_term",
]