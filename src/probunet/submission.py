"""Collect the repository into a submission archive.

The repository stays the single source of truth: this module reads it in place and writes
one ``.zip``. Nothing is copied into a parallel tree, because a parallel tree drifts and a
drifted copy is exactly the kind of difference nobody notices until after the archive has
been sent.

What goes in is an **allowlist** (:data:`INCLUDED_FILES`, :data:`INCLUDED_DIRS`), not
everything-minus-a-few-things: a new untracked directory in the working tree then cannot
end up in the archive by default. The exclusions in :data:`EXCLUDED_PATHS` and friends are
a second, redundant filter over that allowlist — checkpoints, the 450 MB dataset and the
caches are large or private enough that one guard is not enough.

Three conditions **refuse the build** rather than warn, because each of them produces an
archive that looks complete and is not:

* the notebook carries no saved outputs — it would open blank for anyone without the
  training hardware, and its figures are most of what it has to say;
* ``showcase.npz`` is missing — every figure in sections 0 to 6 reads it;
* ``lidc_colab_demo.npz`` is missing — section 7 has nothing to train on.

A missing report PDF only warns: the archive is still worth building without it, and the
warning is loud.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from probunet.paths import COLAB_DEMO_NPZ, FULL_NPZ, SHOWCASE_NPZ, SUBSET_NPZ

NOTEBOOK = Path("notebooks/submission.ipynb")
"""The one notebook that ships. Others, if any, stay in the repository."""

CONVERSION_SCRIPTS: tuple[Path, ...] = (
    Path("scratch/inspect_data.py"),
    Path("scratch/convert_data.py"),
)
"""The two scripts the README's dataset section tells the reader to run.

They are the only part of ``scratch/`` that ships. The rest of that directory is one-off
analysis. ``scratch/`` is gitignored, so these two files can be absent from a fresh clone --
:func:`build_archive` warns in that case rather than refusing, because an archive without
them is still worth building and the warning says exactly what it lacks."""

INCLUDED_FILES: tuple[Path, ...] = (
    Path("README.md"),
    Path("pyproject.toml"),
    Path("DEVIATIONS.md"),
    NOTEBOOK,
    SHOWCASE_NPZ,
    COLAB_DEMO_NPZ,
    *CONVERSION_SCRIPTS,
)
"""Individual files to archive, each at its repository-relative path."""

INCLUDED_DIRS: tuple[Path, ...] = (
    Path("src"),
    Path("tests"),
    Path("configs"),
    Path("scripts"),
    Path("results"),
    Path("data/splits"),
)
"""Directories to archive recursively, subject to the exclusions below."""

EXCLUDED_PATHS: frozenset[Path] = frozenset({
    FULL_NPZ,
    SUBSET_NPZ,
    Path("FINDINGS.md"),
    Path("CLAUDE.md"),
})
"""Files that must never be archived even if an allowlisted directory contains them."""

EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "experiments",
    "runs",
})
"""Directory names pruned wherever they appear."""

EXCLUDED_SUFFIXES: frozenset[str] = frozenset({
    ".ckpt", ".pt", ".pth",          # checkpoints: hundreds of MB, and never needed to read a number
    ".pyc", ".pyo",
    ".zip",
})
"""File suffixes pruned wherever they appear."""

EXCLUDED_FILE_NAMES: frozenset[str] = frozenset({".DS_Store"})
"""File names pruned wherever they appear. Desktop clutter, invisible until it is archived."""

REQUIRED_ARTIFACTS: tuple[Path, ...] = (SHOWCASE_NPZ, COLAB_DEMO_NPZ)
"""Tracked exports without which the notebook cannot render or train."""

DIST_DIR = Path("dist")
"""Where archives are written. Gitignored — see :func:`ensure_dist_gitignored`."""

SIZE_WARNING_BYTES = 50 * 1024**2
"""Archive size above which the build warns. Not a refusal; large is suspicious, not wrong."""

STUDENT_ID = re.compile(r"^\d{4,12}$")
"""Accepted shape of a student ID, so a swapped argument order fails loudly."""


class SubmissionError(RuntimeError):
    """A condition that would produce an archive that looks complete and is not."""


@dataclass
class SubmissionArchive:
    """The result of a build.

    Attributes:
        path: The written ``.zip``.
        size_bytes: Its size on disk.
        counts: File count per top-level entry of the archive.
        warnings: Non-fatal problems, in the order they were found.
    """

    path: Path
    size_bytes: int
    counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        """Total number of files in the archive."""
        return sum(self.counts.values())


def archive_name(student_ids: Sequence[str]) -> str:
    """Build the archive file name from two student IDs.

    Args:
        student_ids: Exactly two ID strings.

    Returns:
        ``"<id1>_<id2>.zip"``.

    Raises:
        SubmissionError: If there are not exactly two IDs, or one is not a number.
    """
    if len(student_ids) != 2:
        raise SubmissionError(f"expected two student IDs, got {len(student_ids)}")
    for identifier in student_ids:
        if not STUDENT_ID.match(identifier):
            raise SubmissionError(
                f"{identifier!r} does not look like a student ID (4 to 12 digits)"
            )
    return f"{student_ids[0]}_{student_ids[1]}.zip"


def is_excluded(relative: Path) -> bool:
    """Whether a repository-relative path is filtered out of the archive.

    Args:
        relative: Path relative to the repository root.

    Returns:
        True if the path is excluded.
    """
    if relative in EXCLUDED_PATHS:
        return True
    if relative.suffix in EXCLUDED_SUFFIXES or relative.name in EXCLUDED_FILE_NAMES:
        return True
    return any(
        part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info") for part in relative.parts
    )


def notebook_has_outputs(path: Path) -> bool:
    """Whether a notebook has at least one code cell with a saved output.

    Args:
        path: The ``.ipynb``.

    Returns:
        True if any code cell carries a non-empty ``outputs`` list.
    """
    if not path.is_file():
        return False
    notebook = json.loads(path.read_text())
    return any(
        cell.get("cell_type") == "code" and cell.get("outputs")
        for cell in notebook.get("cells", [])
    )


def preflight(root: Path) -> list[str]:
    """Find the conditions that must refuse a build.

    Args:
        root: Repository root.

    Returns:
        One message per problem; empty when the repository is ready to archive.
    """
    problems: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (root / artifact).is_file():
            problems.append(
                f"{artifact.as_posix()} is missing; the notebook cannot run without it"
            )
    if not (root / NOTEBOOK).is_file():
        problems.append(f"{NOTEBOOK.as_posix()} is missing")
    elif not notebook_has_outputs(root / NOTEBOOK):
        problems.append(
            f"{NOTEBOOK.as_posix()} has no saved outputs; execute it and save before "
            f"building"
        )
    return problems


def iter_members(root: Path) -> Iterator[tuple[Path, str]]:
    """Walk the allowlist and yield the files to archive.

    Args:
        root: Repository root.

    Yields:
        ``(absolute path, archive name)`` pairs, archive names being
        repository-relative POSIX paths.
    """
    for relative in INCLUDED_FILES:
        path = root / relative
        if path.is_file() and not is_excluded(relative):
            yield path, relative.as_posix()

    for directory in INCLUDED_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if is_excluded(relative):
                continue
            yield path, relative.as_posix()


def ensure_dist_gitignored(root: Path) -> bool:
    """Make sure the archive directory is gitignored, appending the rule if absent.

    Args:
        root: Repository root.

    Returns:
        True if the rule was added, False if it was already there.
    """
    gitignore = root / ".gitignore"
    existing = gitignore.read_text().splitlines() if gitignore.is_file() else []
    if any(line.strip().rstrip("/") == DIST_DIR.name for line in existing):
        return False
    block = ["", "# submission archives, built from the repository by scripts/make_submission.py",
             f"{DIST_DIR.name}/"]
    gitignore.write_text("\n".join(existing + block).lstrip("\n") + "\n")
    return True


def build_archive(
    root: Path,
    student_ids: Sequence[str],
    report: Path | None = None,
    out_dir: Path = DIST_DIR,
) -> SubmissionArchive:
    """Write the submission archive.

    Args:
        root: Repository root.
        student_ids: The two student IDs, which name the archive.
        report: Optional PDF to place at the archive root. Its absence warns.
        out_dir: Directory for the archive, relative to ``root``.

    Returns:
        The build record, including any warnings.

    Raises:
        SubmissionError: If a preflight condition fails, or a named report is missing.
    """
    name = archive_name(student_ids)
    problems = preflight(root)
    if problems:
        raise SubmissionError(
            "refusing to build; each of these produces an archive that looks complete "
            "and is not:\n  - " + "\n  - ".join(problems)
        )

    warnings: list[str] = []
    absent = [s.as_posix() for s in CONVERSION_SCRIPTS if not (root / s).is_file()]
    if absent:
        warnings.append(
            "the dataset instructions in README.md point at " + ", ".join(absent)
            + ", which this working tree does not have (scratch/ is gitignored), so the "
              "archive will not contain them"
        )
    if report is None:
        warnings.append(
            "NO REPORT PDF. The archive is missing the required report; pass --report PATH."
        )
    elif not Path(report).is_file():
        raise SubmissionError(f"report not found: {report}")

    target = root / out_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path, arcname in iter_members(root):
            bundle.write(path, arcname)
            head = arcname.split("/")[0]
            counts[head if "/" in arcname else "(root)"] += 1
        if report is not None:
            bundle.write(report, Path(report).name)
            counts["(root)"] += 1

    size = target.stat().st_size
    if size > SIZE_WARNING_BYTES:
        warnings.append(
            f"archive is {size / 1024**2:.1f} MiB, above the {SIZE_WARNING_BYTES / 1024**2:.0f} "
            f"MiB mark; check that no checkpoint or dataset slipped in"
        )
    return SubmissionArchive(path=target, size_bytes=size, counts=dict(counts),
                             warnings=warnings)
