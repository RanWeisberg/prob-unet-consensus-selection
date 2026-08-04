"""Checkpoint save and resume.

A checkpoint carries everything needed to reproduce or continue a run: the model and
optimizer state, the scheduler, the epoch and step, the resolved configuration, the
seed, the git revision, and the RNG state of every generator the run depends on -- so a
resumed run continues the exact sequence rather than a statistically similar one.

**On unpickling.** These files are written by ``torch.save`` and loaded with
``weights_only=False``, because they carry the config and RNG state alongside tensors.
That is safe here only because they are produced locally by this project. Never load a
checkpoint from an untrusted source; the pickle caveat that applies to the source
dataset applies equally to checkpoints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from probunet.utils.runtime import git_revision, rng_state, set_rng_state

LOGGER = logging.getLogger(__name__)

BEST_NAME = "best.pt"
LAST_NAME = "last.pt"


@dataclass
class CheckpointState:
    """What a loaded checkpoint tells us about the run it came from.

    Attributes:
        epoch: The epoch that had just finished.
        global_step: Optimizer steps taken.
        best_metric: Best monitored value so far, or None.
        monitor: Name of the monitored metric.
        config: The resolved configuration as a plain dict.
        seed: The run seed.
        git_revision: Revision the run was launched from.
        device: Device string the run used, for the cross-backend caveat.
        metrics: Metrics recorded at save time.
        history: Per-epoch metric records from the start of the run.
    """

    epoch: int
    global_step: int
    best_metric: float | None
    monitor: str
    config: dict[str, Any]
    seed: int
    git_revision: str
    device: str
    metrics: dict[str, float]
    history: list[dict[str, float]]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    seed: int,
    device: str,
    monitor: str,
    best_metric: float | None,
    metrics: dict[str, float],
    loader_generator_state: torch.Tensor | None = None,
    history: list[dict[str, float]] | None = None,
) -> None:
    """Write a checkpoint atomically.

    The file is written to a temporary path and then renamed, so an interrupted save
    cannot leave a truncated checkpoint where a valid one used to be.

    Args:
        path: Destination file.
        model: Model whose state to save.
        optimizer: Optimizer whose state to save.
        scheduler: Scheduler whose state to save, if any.
        epoch: Epoch that just finished.
        global_step: Optimizer steps taken.
        config: Resolved configuration as a plain dict.
        seed: Run seed.
        device: Device string, recorded because seeds do not reproduce across backends.
        monitor: Name of the monitored metric.
        best_metric: Best monitored value so far.
        metrics: Metrics at save time.
        loader_generator_state: State of the training DataLoader's generator, so a
            resume replays the same batch order.
        history: Per-epoch metrics so far. Carried in the checkpoint because a run
            spanning days may be resumed several times, and without it ``summary.json``
            would hold only the epochs since the last resume -- a loss curve silently
            truncated to its own tail.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "monitor": monitor,
        # Stored as JSON text so the settings can be read without unpickling.
        "config_json": json.dumps(config, indent=2, default=str),
        "seed": seed,
        "device": device,
        "git_revision": git_revision(),
        "metrics": metrics,
        "history": list(history or []),
        "rng": rng_state(),
        "loader_generator": loader_generator_state,
        "torch_version": torch.__version__,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    LOGGER.info("wrote checkpoint %s (epoch %d, step %d)", path, epoch, global_step)


def load_checkpoint(
    path: Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> CheckpointState:
    """Load a checkpoint, optionally restoring model, optimizer and scheduler in place.

    Args:
        path: Checkpoint file.
        model: If given, load the model state into it.
        optimizer: If given, load the optimizer state into it.
        scheduler: If given, load the scheduler state into it.
        map_location: Where to map tensors while loading.
        restore_rng: Restore RNG state, which is what makes a resume replay the exact
            sequence. Turn it off to evaluate a checkpoint without perturbing the
            current process's generators.

    Returns:
        The :class:`CheckpointState` describing the run.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only=False: the payload carries the config and RNG state, not only
    # tensors. Safe for locally produced files only -- see the module docstring.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model"])
    # A weights-only export has no optimizer, scheduler or RNG state. Loading one for
    # evaluation is fine; resuming from one is not, and the absent keys say so.
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and "rng" in payload:
        set_rng_state(payload["rng"])

    return CheckpointState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_metric=payload.get("best_metric"),
        monitor=str(payload.get("monitor", "val/total")),
        config=json.loads(payload["config_json"]),
        seed=int(payload["seed"]),
        git_revision=str(payload.get("git_revision", "unknown")),
        device=str(payload.get("device", "unknown")),
        metrics=dict(payload.get("metrics", {})),
        history=list(payload.get("history", [])),
    )


def loader_generator_state(path: Path) -> torch.Tensor | None:
    """Read just the DataLoader generator state from a checkpoint.

    Args:
        path: Checkpoint file.

    Returns:
        The saved generator state, or None if the checkpoint has none.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload.get("loader_generator")


def is_improvement(candidate: float, best: float | None, mode: str) -> bool:
    """Decide whether a monitored value improves on the best seen.

    Args:
        candidate: The new value.
        best: The best value so far, or None if there is none yet.
        mode: ``"min"`` or ``"max"``.

    Returns:
        True if ``candidate`` is better.
    """
    if best is None:
        return True
    return candidate < best if mode == "min" else candidate > best


WEIGHTS_ONLY_FORMAT = "weights_only"


def export_weights(source: Path, destination: Path) -> dict[str, Any]:
    """Write a weights-only copy of a checkpoint.

    A full checkpoint carries the optimizer state, and Adam keeps two moment buffers per
    parameter, so it is roughly three times the size of the weights alone -- about 330 MB
    against 110 MB for this model. That matters for the artifact a teammate or a Colab
    session downloads, and none of it is needed to evaluate.

    The export keeps the configuration, epoch and git revision, so it is still traceable
    to the run that produced it. What it drops is everything only a *resume* needs:
    optimizer, scheduler and RNG state. Full checkpoints remain the authoritative
    resumable artifact.

    Args:
        source: Full checkpoint to read.
        destination: Where to write the export.

    Returns:
        A summary with both file sizes and the ratio.

    Raises:
        FileNotFoundError: If the source is missing.
    """
    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)

    export = {
        "format": WEIGHTS_ONLY_FORMAT,
        "model": payload["model"],
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "best_metric": payload.get("best_metric"),
        "monitor": payload.get("monitor", "val/total"),
        "config_json": payload["config_json"],
        "seed": payload["seed"],
        "device": payload.get("device", "unknown"),
        "git_revision": payload.get("git_revision", "unknown"),
        "torch_version": payload.get("torch_version", torch.__version__),
        "exported_from": str(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(export, temporary)
    temporary.replace(destination)

    source_bytes = source.stat().st_size
    destination_bytes = destination.stat().st_size
    summary = {
        "source": str(source),
        "destination": str(destination),
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "ratio": source_bytes / max(destination_bytes, 1),
    }
    LOGGER.info(
        "exported weights: %.1f MiB -> %.1f MiB (%.2fx smaller)",
        source_bytes / 1024**2,
        destination_bytes / 1024**2,
        summary["ratio"],
    )
    return summary


def is_weights_only(path: Path) -> bool:
    """Whether a file is a weights-only export rather than a full checkpoint.

    Args:
        path: Checkpoint file.

    Returns:
        True if the payload is a weights-only export.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload.get("format") == WEIGHTS_ONLY_FORMAT
