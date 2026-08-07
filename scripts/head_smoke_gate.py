"""Pre-run gate for the selection head. Runs the smoke config and PRINTS A VERDICT.

**Run this on the TRAINING machine before the real head run.** Passing on macOS/MPS says
nothing about CUDA: the head's paths differ by backend, and two MPS-only bugs in the
Spearman path were found by running it rather than by the CPU test suite.

It exists as a script rather than a log to read because a log gets skimmed. Every check
below prints ``PASS`` or ``FAIL`` and the process exits non-zero if any fails.

**The check that matters most is Spearman, not the loss.** A collapsing training loss is
consistent with the head learning, and equally consistent with it predicting each image's
mean target -- which gives a near-zero Huber loss and *zero ranking ability*. That is the
image-only shortcut (FINDINGS 4.4), and the loss curve cannot distinguish the two. Spearman
and the area-control column can, which is why both are reported here.

Usage::

    python scripts/head_smoke_gate.py --base-checkpoint runs/baseline/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from probunet.training.config import ExperimentConfig  # noqa: E402
from probunet.training.freeze import parameter_fingerprint  # noqa: E402
from probunet.training.trainer import Trainer  # noqa: E402

LOGGER = logging.getLogger("probunet.head_gate")
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "smoke_head.yaml"


class Verdict:
    """Collects PASS/FAIL lines and reports whether everything passed."""

    def __init__(self) -> None:
        """Start with no checks recorded."""
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str) -> None:
        """Record one check.

        Args:
            ok: Whether it passed.
            name: Short label.
            detail: The measured values behind the verdict.
        """
        self.rows.append((bool(ok), name, detail))

    @property
    def passed(self) -> bool:
        """Whether every recorded check passed."""
        return all(ok for ok, _, _ in self.rows)

    def render(self) -> str:
        """Render the verdict block.

        Returns:
            One ``PASS``/``FAIL`` line per check, then the overall result.
        """
        width = max(len(name) for _, name, _ in self.rows)
        lines = [
            f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}"
            for ok, name, detail in self.rows
        ]
        lines += ["", "GATE PASSED" if self.passed else "GATE FAILED -- do not start the real run"]
        return "\n".join(lines)


def run_gate(base_checkpoint: Path, device: str | None) -> Verdict:
    """Train the smoke head and evaluate every pass criterion.

    Args:
        base_checkpoint: The frozen base to fit on.
        device: Optional device override.

    Returns:
        The completed :class:`Verdict`.
    """
    config = ExperimentConfig.from_yaml(CONFIG)
    verdict = Verdict()

    with tempfile.TemporaryDirectory() as temporary:
        config = dataclasses.replace(
            config,
            run=dataclasses.replace(
                config.run,
                out_dir=Path(temporary),
                device=device or config.run.device,
            ),
        )
        trainer = Trainer(config, base_checkpoint=base_checkpoint)
        before = parameter_fingerprint(trainer.head.base)
        counts = trainer.head.parameter_counts()

        summary = trainer.train()
        verdict.check(True, "1 completes", f"{summary['epochs']} epochs, {summary['global_step']} steps")

        # 2. The freeze held, measured rather than asserted.
        after = parameter_fingerprint(trainer.head.base)
        verdict.check(
            after == before and counts["base"] > 0,
            "2 base frozen",
            f"{counts['base']:,} frozen, sha256 {before[:12]} unchanged={after == before}",
        )

        history = summary["history"]
        losses = [record["train/total"] for record in history]
        verdict.check(
            losses[-1] < losses[0],
            "3 loss decreases",
            " -> ".join(f"{value:.5f}" for value in losses),
        )

        last: dict[str, Any] = history[-1]
        selected = last["val/selected_consensus_dice"]
        oracle = last["val/oracle_consensus_dice"]
        ceiling = last["val/ceiling"]
        ordered = np.isfinite([selected, oracle, ceiling]).all() and (
            selected <= oracle + 1e-6 <= ceiling + 1e-6
        )
        verdict.check(
            bool(ordered),
            "4 scores ordered",
            f"selected {selected:.5f} <= oracle {oracle:.5f} <= ceiling {ceiling:.5f}",
        )

        # 5. THE discriminating check. A near-zero loss with a near-zero Spearman is the
        # image-only shortcut, not a trained head, and the loss curve cannot tell them
        # apart.
        rho = last["val/spearman"]
        excluded = last["val/spearman_excluded_fraction"]
        verdict.check(
            bool(np.isfinite(rho)) and excluded < 1.0,
            "5 spearman ran",
            f"rho {rho:+.4f} over {last['val/spearman_images']:.0f} images, "
            f"{excluded:.1%} excluded as degenerate",
        )

        # Not pass/fail -- too few steps to demand a margin -- but the numbers that decide
        # whether the real run is learning overlap or area, printed so they are seen.
        head_gap = last.get("val/headroom_captured", float("nan"))
        area_gap = last.get("val/headroom_captured_area_only", float("nan"))
        LOGGER.info(
            "head parameters: %s scorer + %s area control (disjoint)",
            f"{counts['scorer']:,}",
            f"{counts['area_baseline']:,}",
        )
        LOGGER.info(
            "headroom captured: head %.1f%% vs area-only control %.1f%% "
            "(informational at 32 steps; on the REAL run head must be well above area)",
            100 * head_gap,
            100 * area_gap,
        )
        LOGGER.info(
            "head huber %.6f vs area huber %.6f",
            last.get("train/head_huber", float("nan")),
            last.get("train/area_huber", float("nan")),
        )
    return verdict


def main() -> int:
    """Parse arguments, run the gate, print the verdict.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    verdict = run_gate(arguments.base_checkpoint, arguments.device)
    print()
    print(verdict.render())
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())