"""Cross-cutting helpers: run environment, device selection, provenance."""

from probunet.utils.runtime import (
    describe_device,
    git_revision,
    rng_state,
    seed_everything,
    select_device,
    set_rng_state,
)

__all__ = [
    "describe_device",
    "git_revision",
    "rng_state",
    "seed_everything",
    "select_device",
    "set_rng_state",
]
