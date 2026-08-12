"""The submission archive: what it contains, and what it refuses to build.

The refusals are the point of these tests. A missing ``showcase.npz`` or an unexecuted
notebook does not break anything at build time -- the archive is written, it is the right
size, and it opens. The failure only appears when someone else unzips it, which is far too
late. So each refusal is asserted here, on a synthetic repository with exactly that defect.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from probunet import submission
from probunet.submission import (
    INCLUDED_DIRS,
    INCLUDED_FILES,
    SubmissionError,
    archive_name,
    build_archive,
    ensure_dist_gitignored,
    is_excluded,
    iter_members,
    notebook_has_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS = ("123456789", "987654321")


def notebook_source(*, with_outputs: bool) -> str:
    """A minimal notebook, with or without a saved output.

    Args:
        with_outputs: Whether the single code cell carries an output.

    Returns:
        The notebook as JSON text.
    """
    outputs = [{"output_type": "stream", "name": "stdout", "text": ["ok\n"]}] if with_outputs else []
    return json.dumps({
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# title"]},
            {"cell_type": "code", "execution_count": 1, "metadata": {},
             "outputs": outputs, "source": ["print('ok')"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    })


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A repository-shaped tree carrying one example of everything the rules mention.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        The root of the synthetic repository.
    """
    root = tmp_path / "repo"
    contents = {
        # allowlisted
        "README.md": "# readme",
        "pyproject.toml": "[project]",
        "DEVIATIONS.md": "# deviations",
        "notebooks/submission.ipynb": notebook_source(with_outputs=True),
        "data/processed/showcase.npz": "arrays",
        "data/processed/lidc_colab_demo.npz": "patches",
        "src/probunet/__init__.py": "",
        "src/probunet/model/prob_unet.py": "x = 1",
        "tests/test_example.py": "",
        "tests/fingerprints/phase1_latent.json": "{}",
        "configs/baseline.yaml": "run: {}",
        "scripts/train.py": "",
        "results/comparison_test.json": "{}",
        "data/splits/split.json": "{}",
        # excluded
        "FINDINGS.md": "the working record",
        "CLAUDE.md": "the project spec",
        "data/processed/lidc.npz": "450 MB in real life",
        "data/processed/lidc_subset.npz": "2 MB in real life",
        "runs/baseline/checkpoints/best.pt": "weights",
        "src/probunet/__pycache__/prob_unet.pyc": "bytecode",
        "src/probunet_consensus_selection.egg-info/PKG-INFO": "metadata",
        "notebooks/scratch_notes.ipynb": "{}",
        ".git/config": "",
        ".pytest_cache/v/cache/lastfailed": "{}",
        "dist/previous.zip": "",
        "scripts/__pycache__/train.pyc": "bytecode",
        "src/.DS_Store": "desktop clutter",
    }
    for relative, text in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / ".gitignore").write_text("runs/\n")
    return root


def archived_names(root: Path) -> set[str]:
    """The archive names :func:`iter_members` would write for a tree.

    Args:
        root: Repository root.

    Returns:
        The set of archive names.
    """
    return {arcname for _, arcname in iter_members(root)}


# --------------------------------------------------------------------------- #
# The allowlist and the exclusions
# --------------------------------------------------------------------------- #
def test_included_paths_are_archived(fake_repo: Path) -> None:
    """Every allowlisted file and directory reaches the archive."""
    names = archived_names(fake_repo)
    for expected in (
        "README.md",
        "pyproject.toml",
        "DEVIATIONS.md",
        "notebooks/submission.ipynb",
        "data/processed/showcase.npz",
        "data/processed/lidc_colab_demo.npz",
        "src/probunet/__init__.py",
        "src/probunet/model/prob_unet.py",
        "tests/test_example.py",
        "tests/fingerprints/phase1_latent.json",
        "configs/baseline.yaml",
        "scripts/train.py",
        "results/comparison_test.json",
        "data/splits/split.json",
    ):
        assert expected in names, f"{expected} is missing from the archive"


def test_excluded_paths_are_not_archived(fake_repo: Path) -> None:
    """Checkpoints, the full dataset, caches and private files stay out.

    Each of these is excluded twice over -- it is outside the allowlist *and* named in an
    exclusion rule -- because one guard is not enough for a 450 MB dataset or a file that
    is deliberately untracked.
    """
    names = archived_names(fake_repo)
    for forbidden in (
        "FINDINGS.md",
        "CLAUDE.md",
        "data/processed/lidc.npz",
        "data/processed/lidc_subset.npz",
        "runs/baseline/checkpoints/best.pt",
        "src/probunet/__pycache__/prob_unet.pyc",
        "src/probunet_consensus_selection.egg-info/PKG-INFO",
        "notebooks/scratch_notes.ipynb",
        ".git/config",
        ".pytest_cache/v/cache/lastfailed",
        "dist/previous.zip",
        "scripts/__pycache__/train.pyc",
        "src/.DS_Store",
    ):
        assert forbidden not in names, f"{forbidden} must not be archived"

    assert not any(name.endswith((".pt", ".pyc", ".zip")) for name in names)
    assert not any(part in name.split("/") for name in names
                   for part in ("runs", "__pycache__", ".git", "dist"))


def test_exclusions_hold_inside_an_allowlisted_directory(fake_repo: Path) -> None:
    """A checkpoint dropped into an included directory is still excluded."""
    stray = fake_repo / "results" / "best.pt"
    stray.write_text("weights")
    assert is_excluded(Path("results/best.pt"))
    assert "results/best.pt" not in archived_names(fake_repo)


def test_the_real_repository_matches_the_allowlist() -> None:
    """The allowlist names paths that exist, so a rename cannot silently empty the archive."""
    for relative in INCLUDED_FILES:
        assert (REPO_ROOT / relative).is_file(), f"{relative} is on the allowlist but absent"
    for relative in INCLUDED_DIRS:
        assert (REPO_ROOT / relative).is_dir(), f"{relative} is on the allowlist but absent"


# --------------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------------- #
def test_build_refuses_a_notebook_with_no_saved_outputs(fake_repo: Path) -> None:
    """An unexecuted notebook opens blank for anyone without the training hardware."""
    (fake_repo / "notebooks" / "submission.ipynb").write_text(
        notebook_source(with_outputs=False)
    )
    assert not notebook_has_outputs(fake_repo / "notebooks" / "submission.ipynb")
    with pytest.raises(SubmissionError, match="no saved outputs"):
        build_archive(fake_repo, IDS)


@pytest.mark.parametrize("artifact", ["showcase.npz", "lidc_colab_demo.npz"])
def test_build_refuses_a_missing_notebook_artifact(fake_repo: Path, artifact: str) -> None:
    """Without these two exports the notebook has nothing to render or train on."""
    (fake_repo / "data" / "processed" / artifact).unlink()
    with pytest.raises(SubmissionError, match=artifact):
        build_archive(fake_repo, IDS)


def test_build_refuses_a_report_path_that_does_not_exist(fake_repo: Path) -> None:
    """A typo in --report must fail, not silently drop the report."""
    with pytest.raises(SubmissionError, match="report not found"):
        build_archive(fake_repo, IDS, report=fake_repo / "nope.pdf")


def test_archive_name_requires_two_numeric_ids() -> None:
    """The archive is named after the two IDs, so they are validated rather than trusted."""
    assert archive_name(IDS) == "123456789_987654321.zip"
    with pytest.raises(SubmissionError, match="two student IDs"):
        archive_name(("123456789",))
    with pytest.raises(SubmissionError, match="student ID"):
        archive_name(("123456789", "report.pdf"))


# --------------------------------------------------------------------------- #
# The archive itself
# --------------------------------------------------------------------------- #
def test_archive_extracts_to_the_expected_tree(fake_repo: Path, tmp_path: Path) -> None:
    """The zip opens, and what comes out carries the marker files a reader needs."""
    archive = build_archive(fake_repo, IDS)
    assert archive.path.name == "123456789_987654321.zip"
    assert archive.path.parent == fake_repo / "dist"

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive.path) as bundle:
        assert bundle.testzip() is None
        bundle.extractall(extracted)

    for marker in (
        "README.md",
        "notebooks/submission.ipynb",
        "data/processed/showcase.npz",
        "data/processed/lidc_colab_demo.npz",
        "data/splits/split.json",
        "configs/baseline.yaml",
        "src/probunet/__init__.py",
        "results/comparison_test.json",
    ):
        assert (extracted / marker).is_file(), f"{marker} missing after extraction"
    assert not (extracted / "runs").exists()
    assert not (extracted / "FINDINGS.md").exists()

    assert archive.file_count == len(archived_names(fake_repo))
    assert archive.counts["src"] == 2
    assert archive.counts["(root)"] == 3          # README, pyproject, DEVIATIONS


def test_a_report_is_placed_at_the_archive_root(fake_repo: Path) -> None:
    """With --report the PDF ships at the root; without it, the build warns loudly."""
    report = fake_repo / "report.pdf"
    report.write_text("%PDF-1.4")

    with_report = build_archive(fake_repo, IDS, report=report)
    with zipfile.ZipFile(with_report.path) as bundle:
        assert "report.pdf" in bundle.namelist()
    assert not with_report.warnings

    without_report = build_archive(fake_repo, IDS)
    with zipfile.ZipFile(without_report.path) as bundle:
        assert "report.pdf" not in bundle.namelist()
    assert any("REPORT" in warning.upper() for warning in without_report.warnings)


def test_a_large_archive_warns_but_still_builds(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size is suspicious, not wrong: it warns rather than refusing."""
    monkeypatch.setattr(submission, "SIZE_WARNING_BYTES", 1)
    archive = build_archive(fake_repo, IDS)
    assert archive.path.is_file()
    assert any("above the" in warning for warning in archive.warnings)


def test_dist_is_gitignored(tmp_path: Path) -> None:
    """The rule is added when absent and left alone when present."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("runs/\n")

    assert ensure_dist_gitignored(root) is True
    assert "dist/" in (root / ".gitignore").read_text()
    assert ensure_dist_gitignored(root) is False

    assert ensure_dist_gitignored(REPO_ROOT) is False, (
        "this repository already ignores dist/; the packaging script must not rewrite it"
    )
