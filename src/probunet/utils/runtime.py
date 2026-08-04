"""Run environment: device selection, seeding and provenance.

Kept in one place because these three decisions are what make a run reproducible and
attributable, and all three are logged at startup.

Note on reproducibility: seeds do **not** reproduce across backends. A run seeded
identically on CUDA and on MPS will diverge. Every result that is compared against
another in the report must therefore come from the same device, which is why
:func:`select_device` logs its choice loudly and the device is recorded in every
checkpoint.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

DEVICE_PREFERENCE = ("cuda", "mps", "cpu")
"""Automatic selection order, per CLAUDE.md."""


def select_device(requested: str = "auto") -> torch.device:
    """Choose a compute device, preferring cuda, then mps, then cpu.

    Args:
        requested: ``"auto"`` for automatic selection, or an explicit device string
            such as ``"cpu"``, ``"cuda"`` or ``"mps"``.

    Returns:
        The selected device.

    Raises:
        ValueError: If an explicitly requested device is unavailable. Falling back
            silently would mean a run labelled "cuda" quietly executing on cpu.
    """
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device 'cuda' requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("device 'mps' requested but MPS is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Return a human-readable description of a device.

    Args:
        device: The device to describe.

    Returns:
        A one-line description including the accelerator name where available.
    """
    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"cuda:{index} ({name}, {total:.1f} GiB)"
    if device.type == "mps":
        return "mps (Apple Silicon unified memory; float64 unsupported)"
    return f"cpu ({os.cpu_count()} cores)"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, and optionally request deterministic kernels.

    Args:
        seed: The seed to apply.
        deterministic: Ask cuDNN for deterministic algorithms and disable its
            autotuner. This does not make MPS deterministic -- no such switch
            exists there -- so cross-device reproducibility is never guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_revision(repo: Path | None = None) -> str:
    """Return the current git revision, marked ``-dirty`` if the tree is modified.

    Args:
        repo: Repository root; defaults to the package's repository.

    Returns:
        The short SHA, optionally suffixed with ``-dirty``, or ``"unknown"`` when git
        is unavailable or this is not a repository. Provenance that silently reads
        "unknown" is better than a crash, but it is worth noticing in the log.
    """
    root = repo or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def rng_state() -> dict[str, object]:
    """Capture the RNG state of every generator a run depends on.

    ``torch.get_rng_state()`` is the **CPU** generator only. A tensor operation on an
    accelerator draws from that device's own generator, so a checkpoint that saves the CPU
    state alone does not let an accelerated run replay its sequence -- and this model
    samples ``z`` on the training device every step. Both accelerator generators are
    therefore captured as well:

    * **CUDA** -- where the long 240k-iteration run happens, and the reason this matters:
      a ~33-hour job will very likely be resumed at least once.
    * **MPS** -- where development happens. Verified directly: restoring only the CPU
      state does not reproduce a subsequent ``torch.randn(..., device="mps")``.

    Returns:
        A dict suitable for storing in a checkpoint and passing to
        :func:`set_rng_state`.
    """
    state: dict[str, object] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, object]) -> None:
    """Restore RNG state captured by :func:`rng_state`.

    Missing entries are skipped, so a checkpoint taken on one backend can still be
    resumed on another -- with the caveat that the sequence will not match. That caveat is
    why every checkpoint records the device it was written on.

    Args:
        state: The captured state.
    """
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    if "python" in state:
        random.setstate(state["python"])  # type: ignore[arg-type]
    if "cuda" in state and torch.cuda.is_available():
        saved = state["cuda"]
        # set_rng_state_all requires one state per visible device, so a resume on a
        # machine with a different GPU count would raise deep inside torch. Skip loudly
        # instead: the run continues, it just does not replay the exact sequence.
        if isinstance(saved, (list, tuple)) and len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([_as_byte_tensor(item) for item in saved])
        else:
            LOGGER.warning(
                "checkpoint holds %s CUDA RNG state(s) but this machine has %d device(s); "
                "skipping CUDA RNG restore, so the sampling sequence will not match",
                len(saved) if isinstance(saved, (list, tuple)) else "unreadable",
                torch.cuda.device_count(),
            )
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(_as_byte_tensor(state["mps"]))


def _as_byte_tensor(value: object) -> torch.Tensor:
    """Coerce a saved RNG state back into the CPU ``ByteTensor`` torch demands.

    Necessary because checkpoints are loaded with ``map_location=<training device>``, and
    that moves **every** storage in the payload -- including the RNG state tensors -- onto
    the accelerator. ``torch.set_rng_state`` then rejects them:
    ``TypeError: RNG state must be a torch.ByteTensor``. So resuming any run on MPS or
    CUDA raised, while resuming on CPU worked; a CPU-only test cannot see the difference.

    Args:
        value: A saved RNG state, possibly living on an accelerator.

    Returns:
        The same state as a contiguous CPU ``uint8`` tensor.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
