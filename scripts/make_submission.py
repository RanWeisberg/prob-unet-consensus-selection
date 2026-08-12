"""Build the submission archive from this repository.

The repository is read in place and one ``dist/<id1>_<id2>.zip`` is written; nothing is
copied into a parallel tree first. What is included, what is excluded, and the three
conditions that refuse a build are all defined in :mod:`probunet.submission`.

Usage::

    python scripts/make_submission.py 123456789 987654321 --report report.pdf
    python scripts/make_submission.py 123456789 987654321      # warns: no report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probunet.submission import (  # noqa: E402 -- after the sys.path line, by design
    SIZE_WARNING_BYTES,
    SubmissionError,
    build_archive,
    ensure_dist_gitignored,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the archive and print what went into it.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("student_ids", nargs=2, metavar="ID",
                        help="the two student ID numbers; they name the archive")
    parser.add_argument("--report", type=Path, default=None,
                        help="the report PDF, placed at the archive root")
    parser.add_argument("--out-dir", type=Path, default=Path("dist"),
                        help="where to write the archive (default: dist/)")
    arguments = parser.parse_args(argv)

    if ensure_dist_gitignored(REPO_ROOT):
        print(f"added {arguments.out_dir}/ to .gitignore")

    try:
        archive = build_archive(
            REPO_ROOT, arguments.student_ids, arguments.report, arguments.out_dir
        )
    except SubmissionError as error:
        print(f"\nERROR: {error}\n", file=sys.stderr)
        return 1

    print(f"archive : {archive.path.relative_to(REPO_ROOT)}")
    print(f"size    : {archive.size_bytes / 1024**2:.2f} MiB "
          f"(warns above {SIZE_WARNING_BYTES / 1024**2:.0f} MiB)")
    print(f"files   : {archive.file_count}")
    for name, count in sorted(archive.counts.items()):
        print(f"  {name:<12} {count:>5}")

    for warning in archive.warnings:
        print(f"\n!!! WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
