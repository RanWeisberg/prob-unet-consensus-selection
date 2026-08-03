"""Tests for the tracked-versus-ignored artifact split.

The failure this guards against is silent and only shows up for someone else: if
``results/`` were gitignored, ``compare.py`` would keep working locally while a fresh
clone -- a teammate's, a grader's, or a Colab session's -- would find nothing for the
notebook to read. Same for the subset ``.npz`` the qualitative panels need.

These assertions run ``git check-ignore``, so they test the real ignore rules rather than
a copy of them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from probunet import paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def is_ignored(path: Path) -> bool:
    """Ask git whether a path is ignored.

    Args:
        path: Repository-relative path. It need not exist.

    Returns:
        True if git would ignore it.

    Raises:
        pytest.skip.Exception: If this is not a git repository.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        pytest.skip("not a git repository, or git unavailable")
    return result.returncode == 0


@pytest.mark.parametrize("path", paths.TRACKED_PATHS, ids=lambda p: str(p))
def test_tracked_paths_are_not_ignored(path: Path) -> None:
    """Results, the split, the subset export and the configs must ship with the repo."""
    assert not is_ignored(path), (
        f"{path} is gitignored, so it will not reach a fresh clone. The notebook and the "
        "report read these files."
    )


@pytest.mark.parametrize("path", paths.IGNORED_PATHS, ids=lambda p: str(p))
def test_ignored_paths_are_ignored(path: Path) -> None:
    """Checkpoints, TensorBoard events and the full dataset must never be committed."""
    assert is_ignored(path), f"{path} is NOT gitignored and could be committed"


def test_results_directory_is_tracked_and_documented() -> None:
    """results/ exists with a README, so the tracked/ignored split is discoverable."""
    readme = REPO_ROOT / paths.RESULTS_DIR / "README.md"
    assert readme.exists(), "results/README.md is missing"
    assert not is_ignored(paths.RESULTS_DIR / "README.md")
    text = readme.read_text()
    assert "ignored" in text and "tracked" in text


def test_runs_and_results_are_different_trees() -> None:
    """The whole point: training artifacts and results do not share a directory."""
    assert paths.RUNS_DIR != paths.RESULTS_DIR
    assert is_ignored(paths.RUNS_DIR / "x" / "checkpoints" / "best.pt")
    assert not is_ignored(paths.RESULTS_DIR / "comparison.json")


def test_no_large_artifact_is_tracked() -> None:
    """No checkpoint, dataset or TensorBoard event is in the index."""
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if listing.returncode != 0:
        pytest.skip("not a git repository")
    tracked = listing.stdout.split()
    offenders = [
        name
        for name in tracked
        if name.endswith((".pt", ".pth", ".pickle", ".pkl"))
        or "tfevents" in name
        or (name.endswith(".npz") and "subset" not in name)
    ]
    assert not offenders, f"large artifacts are tracked: {offenders}"


def test_results_path_creates_the_directory(tmp_path: Path) -> None:
    """results_path() makes the directory, so a script never fails on a fresh clone."""
    target = tmp_path / "fresh_results"
    assert not target.exists()
    produced = paths.results_path("evaluation_val.json", results_dir=target)
    assert target.is_dir()
    assert produced == target / "evaluation_val.json"