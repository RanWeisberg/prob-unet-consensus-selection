"""What must match between two head runs for their comparison to mean anything.

The headline Phase 3 ablation is **head-on-Phase1 versus head-on-Phase2**, and it is only
a measurement of the base if everything *except* the base is held fixed. A head trained
with a different candidate count, a different loss, a different budget or a different
evaluation seed is a different experiment wearing the same name.

**This module refuses rather than warns.** A warning gets read past -- especially at the
end of a multi-day run, in a log nobody reads until the number looks odd. A refusal cannot
be. The failure mode it exists to prevent is not a crash; it is a plausible-looking table
in a report that compares two things which were never comparable.

The one field that is *expected* to differ is ``model.latent_covariance`` -- that is the
variable under test -- so it is deliberately absent from the critical set and is reported
alongside as the difference the comparison is *about*.
"""

from __future__ import annotations

from typing import Any

ABLATION_CRITICAL_FIELDS: tuple[tuple[str, str], ...] = (
    # (dotted path into the config dict, why it must match)
    ("head.train_samples", "candidates per training step changes what the head saw"),
    ("head.eval_samples", "candidates per image at evaluation changes the score itself"),
    (
        "head.eval_seed",
        "the arms must draw COMPARABLE candidate sets, or the comparison mixes the "
        "head's contribution with sampling noise",
    ),
    ("head.huber_delta", "the regression loss is part of what was optimized"),
    ("head.scorer_channels", "head architecture and width"),
    ("head.mean_centered_targets", "a different regression target is a different task"),
    ("model.base_channels", "sets the head's input width, so it is head architecture too"),
    ("optim.name", "optimizer settings"),
    ("optim.lr", "optimizer settings"),
    ("optim.weight_decay", "optimizer settings"),
    ("optim.betas", "optimizer settings"),
    ("optim.eps", "optimizer settings"),
    ("schedule.name", "the learning-rate trajectory is part of the budget"),
    ("train.epochs", "budget"),
    ("train.iterations", "budget"),
    ("train.mode", "both arms must be selection_head runs"),
    ("data.batch_size", "changes the number of steps and the gradient noise scale"),
    ("data.split_path", "a different split is a different dataset"),
    ("run.seed", "head initialization and the training candidate draw"),
)
"""Config fields that must be identical across the arms of a head ablation.

``model.latent_covariance`` is **deliberately excluded**: it is the variable under test.
"""

VARYING_FIELD = "model.latent_covariance"
"""The one field the ablation is about, reported rather than enforced."""


def _lookup(config: dict[str, Any], path: str) -> Any:
    """Read a dotted path out of a config dict.

    Args:
        config: A resolved config mapping.
        path: Dotted path, e.g. ``"head.train_samples"``.

    Returns:
        The value, or the string ``"<missing>"`` if any component is absent -- a checkpoint
        written before a field existed must compare unequal to one that has it, not
        silently match.
    """
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return "<missing>"
        node = node[part]
    # Lists and tuples both spell the same config value depending on whether it came from
    # YAML or a dataclass, so normalize before comparing.
    return tuple(node) if isinstance(node, (list, tuple)) else node


def ablation_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields that must match across a head ablation.

    Args:
        config: A resolved config mapping, e.g. ``CheckpointState.config``.

    Returns:
        Mapping from dotted path to value, for every critical field.
    """
    return {path: _lookup(config, path) for path, _ in ABLATION_CRITICAL_FIELDS}


def assert_comparable(signatures: dict[str, dict[str, Any]]) -> None:
    """Refuse to compare head runs that differ in anything but the base.

    Args:
        signatures: Arm name to its :func:`ablation_signature`. Fewer than two arms is
            trivially comparable.

    Raises:
        ValueError: If any critical field differs between arms. The message names every
            offending field, its value in each arm, and why that field matters -- so the
            fix is obvious without reading this module.
    """
    if len(signatures) < 2:
        return

    reasons = dict(ABLATION_CRITICAL_FIELDS)
    problems: list[str] = []
    for path in reasons:
        values = {arm: signature.get(path) for arm, signature in signatures.items()}
        if len({repr(value) for value in values.values()}) > 1:
            rendered = ", ".join(f"{arm}={value!r}" for arm, value in sorted(values.items()))
            problems.append(f"  {path}: {rendered}\n      why it matters: {reasons[path]}")

    if problems:
        raise ValueError(
            "REFUSING to compare these head runs: they differ in "
            f"{len(problems)} field(s) that would invalidate the ablation.\n"
            + "\n".join(problems)
            + "\n\nThe head-on-Phase1 vs head-on-Phase2 comparison is only a measurement "
            "of the BASE if everything else is held fixed. Retrain the arms with matching "
            "settings, or compare something else. This is a refusal rather than a warning "
            "because a warning at the end of a multi-day run gets read past, and the "
            "result would be a plausible-looking table comparing two things that were "
            "never comparable."
        )