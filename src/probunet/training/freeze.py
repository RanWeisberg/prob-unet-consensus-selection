"""Freezing a base model, with the assertion made explicit and logged.

The extension is defined as a head trained on top of a frozen Probabilistic U-Net: it
must not alter the generative model, the prior/posterior dynamics or the GED behaviour.
The claim the report makes -- *distribution metrics unchanged, single-sample quality
improved* -- is only true if the base really is frozen, and a base that quietly keeps
training would produce a result that looks like the extension working.

So freezing is not a one-line ``requires_grad = False`` buried in a constructor. It
returns a record that the training loop logs, and it raises if anything is still
trainable.
"""

from __future__ import annotations

import logging

from torch import nn

LOGGER = logging.getLogger(__name__)


def freeze_module(module: nn.Module, name: str = "base model") -> dict[str, object]:
    """Freeze every parameter of a module and assert that it worked.

    Args:
        module: The module to freeze.
        name: Label used in the log line and error message.

    Returns:
        A record with the frozen and trainable parameter counts and the training flag,
        suitable for logging and for storing in a checkpoint.

    Raises:
        RuntimeError: If any parameter remains trainable, or the module is still in
            training mode, after freezing. Both would silently invalidate the
            extension's central claim.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()

    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    record = {
        "name": name,
        "frozen_parameters": frozen,
        "trainable_parameters": trainable,
        "training_mode": module.training,
    }
    if trainable != 0:
        raise RuntimeError(
            f"{name} still has {trainable} trainable parameters after freezing"
        )
    if module.training:
        raise RuntimeError(f"{name} is still in training mode after freezing")

    LOGGER.info(
        "froze %s: %d parameters frozen, %d trainable, training=%s",
        name,
        frozen,
        trainable,
        module.training,
    )
    return record


def assert_frozen(module: nn.Module, name: str = "base model") -> None:
    """Check that a module is frozen, without changing it.

    Args:
        module: The module to check.
        name: Label used in the error message.

    Raises:
        RuntimeError: If any parameter is trainable or the module is in training mode.
    """
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"{name} is not frozen: {trainable} trainable parameters")
    if module.training:
        raise RuntimeError(f"{name} is not frozen: still in training mode")
