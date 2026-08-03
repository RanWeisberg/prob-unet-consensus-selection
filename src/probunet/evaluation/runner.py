"""Turning a checkpoint into a report.

One code path, used by both ``scripts/evaluate.py`` (a single checkpoint) and
``scripts/compare.py`` (several at once), so the two cannot drift apart. Everything here
runs on whatever device :func:`probunet.utils.runtime.select_device` picks.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch

from probunet.data.lidc import build_data
from probunet.evaluation.sampling import SamplingConfig, build_report, collect_per_patch_metrics
from probunet.model.prob_unet import ProbUNet
from probunet.training.checkpoint import CheckpointState, load_checkpoint
from probunet.training.config import ExperimentConfig
from probunet.utils.runtime import git_revision
from probunet.variants import ProbUNetVariant, SegmentationVariant

LOGGER = logging.getLogger(__name__)


def load_variant(
    checkpoint: Path,
    device: torch.device,
    name: str | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> tuple[SegmentationVariant, ExperimentConfig, CheckpointState]:
    """Load a checkpoint into an evaluatable variant.

    The checkpoint is read twice: once to recover the configuration the model was built
    with, and once to fill the weights of the model that configuration describes.

    Args:
        checkpoint: Path to a full checkpoint or a weights-only export.
        device: Device to place the model on.
        name: Label for reports; defaults to the run name in the checkpoint's config.
        batch_size: Override the config's batch size.
        seed: Seed for sampling noise; defaults to the run seed.

    Returns:
        A ``(variant, config, state)`` triple.
    """
    state = load_checkpoint(checkpoint, restore_rng=False)
    config = ExperimentConfig.from_dict(state.config)
    if batch_size is not None:
        config = dataclasses.replace(
            config, data=dataclasses.replace(config.data, batch_size=batch_size)
        )

    model = ProbUNet(config.model).to(device)
    load_checkpoint(checkpoint, model=model, map_location=device, restore_rng=False)
    model.eval()

    generator = torch.Generator().manual_seed(
        config.run.seed if seed is None else seed
    )
    variant = ProbUNetVariant(
        model, name=name or config.run.name, generator=generator
    )
    return variant, config, state


def evaluate_variant(
    variant: SegmentationVariant,
    config: ExperimentConfig,
    split: str,
    sampling: SamplingConfig,
    device: torch.device,
    state: CheckpointState | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one variant on one split and build its report.

    Args:
        variant: The variant to evaluate.
        config: Configuration supplying the dataset and split paths.
        split: ``"val"`` or ``"test"``.
        sampling: Sampling configuration.
        device: Device to run on.
        state: Checkpoint provenance, if the variant came from one.
        checkpoint: Checkpoint path, recorded in the report.

    Returns:
        The report, with a ``provenance`` block so every number is traceable.
    """
    data = build_data(config.data)
    per_patch = collect_per_patch_metrics(variant, data.loaders[split], sampling, device)
    report = build_report(per_patch, sampling)
    report["variant"] = getattr(variant, "name", "unknown")
    report["selects_a_sample"] = any(key.startswith("selected_dice@") for key in per_patch)
    report["provenance"] = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "split": split,
        "checkpoint_epoch": state.epoch if state else None,
        "checkpoint_git_revision": state.git_revision if state else None,
        "checkpoint_device": state.device if state else None,
        "checkpoint_monitor": state.monitor if state else None,
        "checkpoint_best_metric": state.best_metric if state else None,
        "evaluation_git_revision": git_revision(),
        "evaluation_device": str(device),
        "torch_version": torch.__version__,
        "seed": sampling.seed,
        "aggregate": sampling.aggregate,
    }
    return report


def evaluate_checkpoint(
    checkpoint: Path,
    split: str,
    sampling: SamplingConfig,
    device: torch.device,
    name: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it in one call.

    Args:
        checkpoint: Path to the checkpoint.
        split: ``"val"`` or ``"test"``.
        sampling: Sampling configuration.
        device: Device to run on.
        name: Label for reports.
        batch_size: Override the config's batch size.

    Returns:
        The report for that checkpoint.
    """
    variant, config, state = load_variant(
        checkpoint, device, name=name, batch_size=batch_size, seed=sampling.seed
    )
    LOGGER.info(
        "%s: epoch %d, trained on %s, git %s",
        getattr(variant, "name", "variant"),
        state.epoch,
        state.device,
        state.git_revision,
    )
    return evaluate_variant(
        variant, config, split, sampling, device, state=state, checkpoint=checkpoint
    )