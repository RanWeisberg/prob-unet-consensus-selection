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

import hashlib
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


def parameter_fingerprint(module: nn.Module) -> str:
    """Hash a module's parameter values, for before/after comparison across an epoch.

    **This is the check that catches an optimizer accidentally handed
    ``model.parameters()``.** ``requires_grad = False`` and ``eval()`` are both *intent*:
    they are what a correct run sets, and asserting them proves only that nobody undid
    them. Neither notices an optimizer that was built over the wrong parameter set before
    they were applied, or a stray in-place update. Comparing the actual values before and
    after a full epoch notices, because it measures the outcome rather than the
    configuration.

    Hashes values only, not gradients or optimizer state, so it answers exactly one
    question: did any number in this module move?

    Args:
        module: The module to fingerprint. Pass the **base only** -- fingerprinting a
            wrapper that also contains the trainable head would change every epoch by
            design and the check would have to be thrown away.

    Returns:
        A hex digest over every parameter, in sorted name order so it is independent of
        registration order.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode())
        # Detach and move to CPU before hashing: the bytes must not depend on device or
        # on whether the tensor happens to carry grad.
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_unchanged(
    module: nn.Module, fingerprint: str, name: str = "base model", context: str = ""
) -> None:
    """Assert a module's parameters still hash to ``fingerprint``.

    Args:
        module: The module to check -- the **base only**.
        fingerprint: The digest from :func:`parameter_fingerprint` taken earlier.
        name: Label used in the error message.
        context: Optional description of the interval covered, e.g. ``"epoch 3"``.

    Raises:
        RuntimeError: If any parameter value changed. The most likely cause is by far the
            most damaging one, so it is named in the message.
    """
    current = parameter_fingerprint(module)
    if current != fingerprint:
        where = f" during {context}" if context else ""
        raise RuntimeError(
            f"{name} parameters CHANGED{where}: the base is not actually frozen. The "
            "usual cause is an optimizer constructed over model.parameters() instead of "
            "the head's parameters alone. Every Phase 3 number depends on this not "
            "happening -- 'distribution metrics unchanged' is false if the base moved."
        )


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
