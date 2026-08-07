"""Pre-registration pass: is the empty-mask pathology actually gone, and how much
headroom is there for the selection head?

**Run this BEFORE building the head.** It answers two questions that decide whether Phase 3
is worth building at all, and it answers them from the frozen Phase 1 checkpoint, with no
head in existence:

1. **Is the pathology gone?** Under the old per-grader *mean* Dice, an all-empty mask
   scored 0.75 on bucket 1 and beat best-of-16 (oracle@16 = 0.7458 < 0.7500), so no
   selector over those samples could have won. Soft consensus should send all-empty to
   **exactly 0** wherever any grader saw a lesion. This pass demonstrates that on real
   data rather than asserting it from arithmetic.
2. **How much is there to win?** ``oracle@16 - random`` is the gap the head is trying to
   capture. If that gap is small, the head cannot help however good it is.

**Predicted direction, registered before running** (see FINDINGS 4.4):

===========================  ==================================================
row                          prediction
===========================  ==================================================
all-empty, every bucket      exactly 0.000 -- mechanical, not empirical
oracle@16 > all-empty        all four buckets, decisively
oracle@16 vs ceiling         below it everywhere; bucket 1 furthest below
random < oracle@16           everywhere; headroom largest in buckets 3-4
emptiest-sample              near 0, tracking all-empty
===========================  ==================================================

**If all-empty beats oracle@16 on ANY bucket, stop and report.** The target is not fixed
and the framework should not be built on it.

Usage::

    python scripts/consensus_headroom.py \
        --checkpoint runs/baseline/checkpoints/best.pt \
        --split val --out results/consensus_headroom_baseline.json

The measurement logic lives in ``probunet.evaluation.headroom``; this file is the CLI.

Lives in ``scripts/`` rather than ``scratch/`` for two reasons: ``scratch/`` is gitignored
and so would not reach the training machine through git, and this pass produces a
**pre-registered result that belongs in the report**, so the code behind it is tracked.

Run it on the machine holding the checkpoint, and **not while a training run is live** --
a full validation pass alongside a running job risks exhausting an 8 GiB card.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from probunet.data.lidc import build_data  # noqa: E402
from probunet.evaluation.headroom import (  # noqa: E402
    EVAL_SAMPLES,
    measure_ceilings,
    measure_selection,
    measure_split,
    per_bucket,
    render,
    render_selection,
)
from probunet.extension.ablation import assert_not_a_smoke_run  # noqa: E402
from probunet.extension.head import SelectionHead  # noqa: E402
from probunet.evaluation.sampling import DEFAULT_EVAL_SEED  # noqa: E402
from probunet.model.prob_unet import ProbUNet  # noqa: E402
from probunet.training.checkpoint import load_checkpoint  # noqa: E402
from probunet.training.config import ExperimentConfig  # noqa: E402
from probunet.training.freeze import freeze_module  # noqa: E402
from probunet.utils.runtime import git_revision, seed_everything, select_device  # noqa: E402

LOGGER = logging.getLogger("probunet.consensus_headroom")

def main() -> None:
    """Parse arguments, run the pass, print and optionally save the table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="checkpoint to sample from. Omit for CEILING-ONLY mode: the ceiling and the "
             "all-empty score depend on the grader masks alone, so they need no weights.",
    )
    parser.add_argument(
        "--head-checkpoint", type=Path, default=None,
        help="a trained selection-head checkpoint. Switches to the Stage 5 SELECTION "
             "table: head, its size-prior control, random, oracle and the fraction of the "
             "oracle-minus-random gap each captured, per ambiguity bucket.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="config supplying the data pipeline. Required in ceiling-only mode; "
             "otherwise the checkpoint's own config is used.",
    )
    parser.add_argument(
        "--split", required=True, choices=("val", "test"),
        help="no default: development happens on val, test is touched once at the end",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help=f"candidates per image; defaults to head.eval_samples, else {EVAL_SAMPLES}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    if (
        arguments.checkpoint is None
        and arguments.config is None
        and arguments.head_checkpoint is None
    ):
        parser.error(
            "give --head-checkpoint for the selection table, --checkpoint for the "
            "headroom pass, or --config for ceiling-only mode"
        )
    if arguments.checkpoint is not None and not arguments.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {arguments.checkpoint}. Run this on the machine that "
            "trained it."
        )

    state = None
    if arguments.head_checkpoint is not None:
        state = load_checkpoint(arguments.head_checkpoint)
        # A smoke/gate checkpoint must never reach a results table.
        assert_not_a_smoke_run(state.config, arguments.head_checkpoint)
        config = ExperimentConfig.from_dict(state.config)
    elif arguments.config is not None:
        config = ExperimentConfig.from_yaml(arguments.config)
    else:
        state = load_checkpoint(arguments.checkpoint)
        config = ExperimentConfig.from_dict(state.config)
    device = select_device(arguments.device or config.run.device)
    seed_everything(config.run.seed, deterministic=config.run.deterministic)

    data = build_data(config.data)
    loader = data.loaders[arguments.split]

    freeze_record = None
    if arguments.head_checkpoint is not None:
        base = ProbUNet(config.model).to(device)
        head = SelectionHead(base, channels=config.head.scorer_channels).to(device)
        load_checkpoint(arguments.head_checkpoint, model=head, map_location=device)
        head.eval()
        head.assert_base_frozen()
        freeze_record = dict(head.freeze_record)
        samples = arguments.samples or config.head.eval_samples
        seed = config.head.eval_seed
        LOGGER.info(
            "SELECTION table over %d patches from %s: %d shared candidates, seed %d",
            len(data.datasets[arguments.split]), arguments.split, samples, seed,
        )
        results = measure_selection(head, loader, device, samples, seed)
    elif arguments.checkpoint is None:
        # Ceiling-only: grader masks are the whole input, so there is nothing to load and
        # nothing that could vary between arms.
        LOGGER.info(
            "CEILING-ONLY over %d patches from %s: no model, no weights",
            len(data.datasets[arguments.split]), arguments.split,
        )
        results = measure_ceilings(loader, device)
    else:
        model = ProbUNet(config.model).to(device)
        load_checkpoint(arguments.checkpoint, model=model)
        # No head here, but the base must be frozen and in eval mode for the same reason
        # it will be in Phase 3: a base still in training mode would give other samples.
        freeze_record = freeze_module(model, "base model")
        LOGGER.info(
            "scoring %d patches from %s with %d shared candidates per image (seed %d)",
            len(data.datasets[arguments.split]), arguments.split,
            arguments.samples or EVAL_SAMPLES, arguments.seed,
        )
        results = measure_split(
            model, loader, device, arguments.samples or EVAL_SAMPLES, arguments.seed
        )
    report = per_bucket(results)

    record = {
        "mode": (
            "selection" if arguments.head_checkpoint
            else "ceiling_only" if state is None else "full"
        ),
        "checkpoint": None if state is None else str(arguments.checkpoint),
        "checkpoint_epoch": None if state is None else state.epoch,
        "checkpoint_device": None if state is None else state.device,
        "checkpoint_git_revision": None if state is None else state.git_revision,
        "checkpoint_torch_version": None if state is None else state.torch_version,
        "latent_covariance": config.model.latent_covariance,
        "split": arguments.split,
        "n_samples": arguments.samples,
        "sampling_seed": arguments.seed,
        "freeze_record": freeze_record,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_version": torch.version.cuda,
            "git_revision": git_revision(),
        },
        "buckets": report,
    }
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(record, indent=2) + "\n")
        LOGGER.info("wrote %s", arguments.out)

    print()
    print(render_selection(report) if arguments.head_checkpoint else render(report))
    print()
    if arguments.head_checkpoint is not None:
        return
    if state is None:
        print(
            "Ceiling-only: these depend on the grader masks alone, so they are FINAL -- "
            "they cannot move when the pass is rerun on a checkpoint."
        )
        return
    failed = [k for k, row in report.items() if row["verdict"] == "all_empty_wins"]
    tied = [k for k, row in report.items() if row["verdict"] == "degenerate_tie"]
    if failed:
        print(
            "STOP: all-empty beats oracle on " + ", ".join(failed) + ".\n"
            "The soft-consensus target has NOT removed the empty-mask pathology on these "
            "buckets. Report this before building the head -- do not proceed to Stage 3."
        )
    elif tied:
        print(
            "INCONCLUSIVE on " + ", ".join(tied) + ": oracle ties all-empty.\n"
            "Check the 'nonempty' column. If it is ~0 the model emitted only empty "
            "candidates, so every selection rule scores 0 and this pass says nothing "
            "about the target -- that is a checkpoint problem, not a target problem."
        )
    else:
        print(
            "Pathology cleared: oracle beats all-empty on every bucket. The 'headroom' "
            "column is what the head is trying to capture."
        )


if __name__ == "__main__":
    main()