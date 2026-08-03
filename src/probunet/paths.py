"""Canonical artifact locations, and which of them git tracks.

The distinction that matters: **training artifacts are ignored, results are tracked.**

A checkpoint is hundreds of megabytes, tied to one machine, and nobody needs it to read
a number. A results summary is a few kilobytes of JSON and is the only thing the
notebook and the report actually consume. Putting them in the same directory means
either committing gigabytes or shipping a repo whose notebook has nothing to read --
which is exactly the trap this module exists to prevent.

==========================  ==========  =========================================
path                        git         contents
==========================  ==========  =========================================
``runs/``                   ignored     checkpoints, TensorBoard events, per-run logs
``experiments/``            ignored     same, alternative name
``data/raw/``               ignored     the source pickle
``data/processed/*``        ignored     the full converted ``.npz`` (~450 MB)
``data/processed/lidc.json``       TRACKED  conversion provenance (a few KB)
``data/processed/lidc_subset.npz`` TRACKED  panel/diversity patches only (a few MB)
``data/splits/``            TRACKED     the frozen split, plus its notes
``results/``                TRACKED     evaluation and comparison JSON/CSV
``configs/``                TRACKED     the three variant configs
==========================  ==========  =========================================

``tests/test_paths.py`` asserts this table against ``git check-ignore``, so a stray
``.gitignore`` edit cannot silently make ``results/`` unreachable from Colab.
"""

from __future__ import annotations

from pathlib import Path

# --- ignored: large, machine-specific training artifacts ------------------------
RUNS_DIR = Path("runs")
EXPERIMENTS_DIR = Path("experiments")
DATA_RAW_DIR = Path("data/raw")

# --- data ----------------------------------------------------------------------
DATA_PROCESSED_DIR = Path("data/processed")
FULL_NPZ = DATA_PROCESSED_DIR / "lidc.npz"
"""The full converted dataset. Ignored: ~450 MB compressed."""

SUBSET_NPZ = DATA_PROCESSED_DIR / "lidc_subset.npz"
"""The stratified panel/diversity patches only. Tracked, so the notebook can draw
qualitative samples on Colab without the full dataset."""

CONVERSION_SIDECAR = DATA_PROCESSED_DIR / "lidc.json"

# --- tracked: small, portable, the things a reader needs ------------------------
SPLITS_DIR = Path("data/splits")
SPLIT_PATH = SPLITS_DIR / "split.json"
RESULTS_DIR = Path("results")
CONFIGS_DIR = Path("configs")

COMPARISON_JSON = RESULTS_DIR / "comparison.json"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

TRACKED_PATHS: tuple[Path, ...] = (
    SPLIT_PATH,
    CONVERSION_SIDECAR,
    SUBSET_NPZ,
    RESULTS_DIR / "comparison.json",
    CONFIGS_DIR / "baseline.yaml",
)
"""Paths git must NOT ignore. Asserted in tests."""

IGNORED_PATHS: tuple[Path, ...] = (
    RUNS_DIR / "any" / "checkpoints" / "best.pt",
    EXPERIMENTS_DIR / "any" / "checkpoints" / "best.pt",
    DATA_RAW_DIR / "data_lidc.pickle",
    FULL_NPZ,
)
"""Paths git MUST ignore. Asserted in tests."""


def results_path(name: str, results_dir: Path | None = None) -> Path:
    """Build a path inside the tracked results directory, creating it if needed.

    Args:
        name: File name, e.g. ``"evaluation_val.json"``.
        results_dir: Override for the results directory.

    Returns:
        The full path. The parent directory is created.
    """
    directory = results_dir or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
