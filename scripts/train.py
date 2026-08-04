"""Train the baseline Probabilistic U-Net.

Usage::

    python scripts/train.py --config configs/smoke.yaml       # verify the loop, <1 min
    python scripts/train.py --config configs/baseline.yaml    # the real run
    python scripts/train.py --config configs/baseline.yaml --resume runs/baseline/checkpoints/last.pt

The resolved configuration, seed, device and git revision are logged at startup and the
configuration is written to ``<run_dir>/config.resolved.yaml``, so a result can always be
traced back to what produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from a bare checkout without an editable install.
SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.training.config import ExperimentConfig  # noqa: E402
from probunet.training.trainer import Trainer  # noqa: E402
from probunet.utils.runtime import git_revision  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--device", default=None, help="override run.device")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--epochs", type=int, default=None, help="override the budget with an epoch count"
    )
    budget.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="override the budget with an optimizer-step count (paper: 240000)",
    )
    parser.add_argument("--name", default=None, help="override run.name")
    parser.add_argument("--out-dir", type=Path, default=None, help="override run.out_dir")
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level (default: %(default)s)"
    )
    return parser


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply command-line overrides to a loaded configuration.

    Args:
        config: The configuration loaded from YAML.
        args: Parsed command-line arguments.

    Returns:
        The configuration with overrides applied.
    """
    import dataclasses

    run_changes = {}
    if args.device is not None:
        run_changes["device"] = args.device
    if args.name is not None:
        run_changes["name"] = args.name
    if args.out_dir is not None:
        run_changes["out_dir"] = args.out_dir
    if run_changes:
        config = dataclasses.replace(config, run=dataclasses.replace(config.run, **run_changes))
    # The budget is exactly one of iterations/epochs, so an override must clear the other
    # rather than set both -- TrainConfig rejects having both, and the point of a CLI
    # override is to replace the configured budget, not to conflict with it.
    if args.epochs is not None:
        config = dataclasses.replace(
            config,
            train=dataclasses.replace(config.train, epochs=args.epochs, iterations=None),
        )
    elif args.iterations is not None:
        config = dataclasses.replace(
            config,
            train=dataclasses.replace(config.train, iterations=args.iterations, epochs=None),
        )
    return config


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
    logger = logging.getLogger("probunet.train")

    config = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    logger.info("config        : %s", args.config)
    logger.info("git revision  : %s", git_revision())

    trainer = Trainer(config)
    if args.resume is not None:
        trainer.resume(args.resume)

    summary = trainer.train()
    (config.run.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    logger.info(
        "finished: best %s=%s after %d epochs (%d steps)",
        summary["monitor"],
        summary["best_metric"],
        summary["epochs"],
        summary["global_step"],
    )
    return 0


if __name__ == "__main__":
    # Required on Windows (the RTX 3070 machine) and macOS, where DataLoader workers are
    # spawned rather than forked: without this guard each worker would re-execute the
    # module and start its own training run.
    raise SystemExit(main())
