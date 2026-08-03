"""Evaluate a trained Probabilistic U-Net checkpoint.

Reports the generalized energy distance at several sample counts, single-sample quality
(oracle / random / Hungarian), and two degenerate baselines, aggregate and per ambiguity
bucket.

``--split`` is **required and has no default**. Development and every iteration belong on
``val``; ``test`` is evaluated once, at the end, for the final report numbers. Choosing
``test`` prints a prominent warning and is recorded in the results file.

Usage::

    python scripts/evaluate.py --checkpoint runs/baseline/checkpoints/best.pt --split val
    python scripts/evaluate.py --checkpoint runs/baseline/checkpoints/best.pt --split test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

# Allow running from a bare checkout without an editable install.
SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.evaluation.runner import evaluate_variant, load_variant  # noqa: E402
from probunet.evaluation.sampling import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    DEFAULT_SAMPLE_COUNTS,
    SamplingConfig,
)
from probunet.paths import results_path  # noqa: E402
from probunet.utils.runtime import describe_device, select_device  # noqa: E402

LOGGER = logging.getLogger("probunet.evaluate")

TEST_WARNING = """
================================================================================
EVALUATING ON THE TEST SPLIT
The test split is meant to be touched ONCE, for the final reported numbers. If you
are still iterating on the model, the loss, the schedule or the extension, stop and
use --split val instead. Every number produced here is recorded with the git
revision and checkpoint path so it can be audited later.
================================================================================
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint to evaluate")
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
        help="which split to evaluate; REQUIRED, no default. test is for final numbers only",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_COUNTS),
        help="sample counts to report (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED, help="sampling seed")
    parser.add_argument("--aggregate", default="mean", help="grader aggregation for Dice")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--device", default=None, help="override device")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results JSON path (default: results/evaluation_<split>.json, which is TRACKED)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def format_report(report: dict, split: str) -> str:
    """Render the report as plain tables.

    Args:
        report: Output of :func:`build_report`.
        split: Split name, for the header.

    Returns:
        A multi-line string.
    """
    lines: list[str] = ["=" * 96, f"EVALUATION on {split.upper()}", "=" * 96]
    counts = report["sample_counts"]
    aggregate = report["aggregate_over_all_patches"]

    lines += [
        f"patches evaluated      : {aggregate['n_patches']}",
        f"median lesion area     : {aggregate['lesion_area_median_px']:.1f} px",
        "",
        "TABLE 1 - generalized energy distance by sample count (lower is better)",
        f"  {'n':>3} {'mean':>9} {'median':>9} {'q25':>9} {'q75':>9} {'IQR':>9} {'<0':>6}",
    ]
    for count in counts:
        stats = aggregate[f"ged@{count}"]
        lines.append(
            f"  {count:>3} {stats['mean']:>9.4f} {stats['median']:>9.4f} "
            f"{stats['q25']:>9.4f} {stats['q75']:>9.4f} {stats['iqr']:>9.4f} "
            f"{stats['n_negative']:>6}"
        )
    lines.append("  components at the largest n:")
    largest = counts[-1]
    for component, label in (("ged_ys", "2*E[d(S,Y)]"), ("ged_ss", "E[d(S,S')]"), ("ged_yy", "E[d(Y,Y')]")):
        value = aggregate[f"{component}@{largest}"]["mean"]
        shown = 2 * value if component == "ged_ys" else value
        lines.append(f"    {label:<14} {shown:>9.4f}")

    lines += [
        "",
        "TABLE 2 - single-sample quality (higher is better)",
        f"  {'n':>3} {'random':>9} {'emptiest':>9} {'oracle':>9} {'per-grader':>11} {'hungIoU':>9}",
    ]
    for count in counts:
        lines.append(
            f"  {count:>3} "
            f"{aggregate[f'random_sample_dice@{count}']['mean']:>9.4f} "
            f"{aggregate[f'emptiest_sample_dice@{count}']['mean']:>9.4f} "
            f"{aggregate[f'oracle_dice@{count}']['mean']:>9.4f} "
            f"{aggregate[f'oracle_dice_per_grader@{count}']['mean']:>11.4f} "
            f"{aggregate[f'hungarian_iou@{count}']['mean']:>9.4f}"
        )

    lines += [
        "  hungIoU matches min(n, 4) pairs one-to-one, so values at n < 4 are NOT",
        "  comparable with n >= 4: at n=1 a single sample is scored against its best",
        "  grader, while at n >= 4 every grader must be covered by a distinct sample.",
        "",
        "TABLE 3 - degenerate all-empty predictor (n-independent)",
        f"  {'GED':>9} {'Dice':>9} {'oracleDice':>11} {'hungIoU':>9}",
        f"  {aggregate['empty_ged']['mean']:>9.4f} {aggregate['empty_dice']['mean']:>9.4f} "
        f"{aggregate['empty_oracle_dice']['mean']:>11.4f} "
        f"{aggregate['empty_hungarian_iou']['mean']:>9.4f}",
        "  An all-empty predictor scores this well because 33% of patches have three of",
        "  four graders empty. Any single-sample number must be read against it.",
    ]

    lines += [
        "",
        "TABLE 4 - per ambiguity bucket (non-empty grader count)",
        f"  {'bucket':>6} {'patches':>8} {'medArea':>8} "
        + " ".join(f"{'GED@' + str(c):>9}" for c in counts)
        + f" {'random':>8} {'oracle':>8} {'empty':>8}",
    ]
    for bucket, block in report["per_bucket"].items():
        row = (
            f"  {bucket:>6} {block['n_patches']:>8} "
            f"{block['lesion_area_median_px']:>8.1f} "
            + " ".join(f"{block[f'ged@{c}']['mean']:>9.4f}" for c in counts)
            + f" {block[f'random_sample_dice@{largest}']['mean']:>8.4f}"
            + f" {block[f'oracle_dice@{largest}']['mean']:>8.4f}"
            + f" {block['empty_dice']['mean']:>8.4f}"
        )
        lines.append(row)
    lines += [
        "",
        "  'random', 'oracle' and 'empty' are Dice at the largest sample count.",
        "  The smallest non-empty mask in this dataset is 1 pixel, so IoU in the",
        "  small-lesion tail is extremely sensitive: read a poor bucket number against",
        "  its median lesion area before concluding the model is worse there.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.split == "test":
        LOGGER.warning(TEST_WARNING)

    device = select_device(args.device or "auto")
    variant, config, state = load_variant(
        args.checkpoint, device, batch_size=args.batch_size, seed=args.seed
    )

    LOGGER.info("checkpoint    : %s", args.checkpoint)
    LOGGER.info(
        "trained on    : %s (epoch %d, git %s)", state.device, state.epoch, state.git_revision
    )
    LOGGER.info("monitor       : %s = %s", state.monitor, state.best_metric)
    LOGGER.info("device        : %s", describe_device(device))
    LOGGER.info("split         : %s", args.split)

    sampling = SamplingConfig(
        sample_counts=tuple(args.samples), seed=args.seed, aggregate=args.aggregate
    )
    report = evaluate_variant(
        variant, config, args.split, sampling, device, state=state, checkpoint=args.checkpoint
    )

    print(format_report(report, args.split))
    # Results go to the TRACKED results/ tree, not the ignored run directory: the
    # notebook and the report read these files, so they have to ship with the repo.
    out = args.out or results_path(f"evaluation_{args.split}_{variant.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    # Guard required on Windows and macOS, where DataLoader workers are spawned.
    raise SystemExit(main())
