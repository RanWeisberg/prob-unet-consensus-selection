"""Evaluate several checkpoints and write one comparison file.

Produces ``results/comparison.json`` and ``results/comparison.csv`` holding every metric
-- aggregate and per ambiguity bucket -- for every variant plus the degenerate baselines.
These files are small and **tracked**, and they are what the notebook and the report
read. Checkpoints themselves stay in the ignored ``runs/`` tree.

``--split`` is required and has no default, for the same reason as in ``evaluate.py``:
``test`` is evaluated once, at the end.

Usage::

    python scripts/compare.py --split val \\
        --checkpoint baseline=runs/baseline/checkpoints/best.pt \\
        --checkpoint modernized=runs/modernized/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.evaluation.runner import evaluate_checkpoint  # noqa: E402
from probunet.evaluation.sampling import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    DEFAULT_SAMPLE_COUNTS,
    SamplingConfig,
)
from probunet.paths import COMPARISON_CSV, COMPARISON_JSON  # noqa: E402
from probunet.utils.runtime import describe_device, git_revision, select_device  # noqa: E402

LOGGER = logging.getLogger("probunet.compare")

CSV_COLUMNS = (
    "variant",
    "split",
    "scope",
    "metric",
    "n_samples",
    "n_patches",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "iqr",
    "min",
    "max",
    "n_negative",
)


def parse_checkpoint(argument: str) -> tuple[str, Path]:
    """Parse a ``name=path`` checkpoint argument.

    Args:
        argument: The raw ``name=path`` string.

    Returns:
        A ``(name, path)`` pair.

    Raises:
        argparse.ArgumentTypeError: If the argument is malformed or the file is missing.
    """
    if "=" not in argument:
        raise argparse.ArgumentTypeError(
            f"expected name=path, got {argument!r} (e.g. baseline=runs/baseline/checkpoints/best.pt)"
        )
    name, _, raw = argument.partition("=")
    path = Path(raw)
    if not name:
        raise argparse.ArgumentTypeError(f"empty variant name in {argument!r}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"checkpoint not found: {path}")
    return name, path


def flatten(reports: dict[str, dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Flatten nested reports into CSV rows.

    Args:
        reports: Variant name to report.
        split: Split the reports came from.

    Returns:
        One row per (variant, scope, metric). ``scope`` is ``"all"`` or ``"bucket_k"``.
    """
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        scopes = {"all": report["aggregate_over_all_patches"]}
        scopes.update({f"bucket_{k}": v for k, v in report["per_bucket"].items()})
        for scope, block in scopes.items():
            for metric, stats in block.items():
                if not isinstance(stats, dict):
                    continue
                name, _, count = metric.partition("@")
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "scope": scope,
                        "metric": name,
                        "n_samples": int(count) if count else "",
                        "n_patches": block["n_patches"],
                        **{key: stats.get(key) for key in CSV_COLUMNS[6:]},
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a named checkpoint; repeat for each variant",
    )
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--samples", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=COMPARISON_JSON)
    parser.add_argument("--csv", type=Path, default=COMPARISON_CSV)
    parser.add_argument(
        "--note",
        default=None,
        help=(
            "free-text provenance note recorded in the JSON, e.g. to mark a file as a "
            "placeholder from a smoke run rather than a reportable result"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.split == "test":
        LOGGER.warning(
            "evaluating on TEST. This is for final reported numbers only; use val while "
            "still iterating."
        )

    device = select_device(args.device or "auto")
    LOGGER.info("device: %s", describe_device(device))
    sampling = SamplingConfig(
        sample_counts=tuple(args.samples), seed=args.seed, aggregate=args.aggregate
    )

    reports: dict[str, Any] = {}
    for name, path in args.checkpoint:
        LOGGER.info("evaluating %s from %s", name, path)
        reports[name] = evaluate_checkpoint(
            path, args.split, sampling, device, name=name, batch_size=args.batch_size
        )

    document = {
        "note": args.note,
        "split": args.split,
        "sample_counts": list(args.samples),
        "seed": args.seed,
        "aggregate": args.aggregate,
        "device": str(device),
        "git_revision": git_revision(),
        "variants": reports,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2, default=float) + "\n")

    rows = flatten(reports, args.split)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("wrote %s (%d variants)", args.json, len(reports))
    LOGGER.info("wrote %s (%d rows)", args.csv, len(rows))

    # A compact console summary of the headline numbers.
    largest = max(args.samples)
    print(f"\n{'variant':<16} {'GED@' + str(largest):>10} {'random':>9} {'oracle':>9} {'empty':>9}")
    for name, report in reports.items():
        block = report["aggregate_over_all_patches"]
        selected = block.get(f"selected_dice@{largest}")
        line = (
            f"{name:<16} {block[f'ged@{largest}']['mean']:>10.4f} "
            f"{block[f'random_sample_dice@{largest}']['mean']:>9.4f} "
            f"{block[f'oracle_dice@{largest}']['mean']:>9.4f} "
            f"{block['empty_dice']['mean']:>9.4f}"
        )
        if selected:
            line += f"   selected={selected['mean']:.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())