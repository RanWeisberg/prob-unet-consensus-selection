"""Write a weights-only copy of a checkpoint.

A full checkpoint carries the Adam state -- two moment buffers per parameter -- so it is
roughly three times the size of the weights alone (~330 MB against ~110 MB here). The
export keeps the config, epoch and git revision, so it stays traceable to its run, and
drops only what a *resume* needs. Full checkpoints remain the authoritative resumable
artifact; exports are what a teammate or a Colab session downloads.

Usage::

    python scripts/export_weights.py runs/baseline/checkpoints/best.pt
    python scripts/export_weights.py runs/*/checkpoints/best.pt --out-dir exports
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.training.checkpoint import export_weights  # noqa: E402

LOGGER = logging.getLogger("probunet.export_weights")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="checkpoints to export")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="destination directory (default: alongside each checkpoint)",
    )
    parser.add_argument(
        "--suffix", default="_weights", help="filename suffix (default: %(default)s)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(message)s",
    )

    for source in args.checkpoints:
        # Name the export after its run, not just "best_weights.pt": three runs would
        # otherwise produce three indistinguishable files in one download folder.
        run = source.parent.parent.name if source.parent.name == "checkpoints" else ""
        stem = f"{run}_{source.stem}{args.suffix}" if run else f"{source.stem}{args.suffix}"
        destination = (args.out_dir or source.parent) / f"{stem}.pt"
        summary = export_weights(source, destination)
        print(
            f"{summary['source']} -> {summary['destination']}  "
            f"{summary['source_bytes'] / 1024**2:.1f} MiB -> "
            f"{summary['destination_bytes'] / 1024**2:.1f} MiB "
            f"({summary['ratio']:.2f}x smaller)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
