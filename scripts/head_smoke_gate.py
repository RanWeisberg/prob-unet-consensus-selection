"""Pre-run gate for the selection head. Runs the smoke config and PRINTS A VERDICT.

**The behavioural gate exists only on the training machine.** ``runs/`` is gitignored and
the Mac never trains, so a trained base will never be present there. A Mac run is a
**plumbing check permanently**, not just on one occasion, and this script says so in its own
output rather than leaving a passing plumbing check to be mistaken for a passing real one.

Two regimes, decided by measurement rather than by assumption:

* **TRAINED BASE** -- the full gate. Criteria 4 and 5 are checked against **pre-registered
  numeric bands** read from ``results/consensus_headroom_baseline.json``, so they can
  actually fail.
* **STAND-IN BASE** -- an untrained or barely-trained checkpoint. Criteria 4 and 5 are
  **SKIPPED, not passed.** Ordering three numbers that are all approximately zero is an
  assertion that passes on noise, and a near-tied target vector can yield a near-perfect
  rank correlation that means nothing.

A missing base checkpoint is a **refusal**, not a fallback: a gate that invented its own
base could not detect the thing it exists to detect.

Usage::

    # training machine, full behavioural gate:
    python scripts/head_smoke_gate.py --base-checkpoint runs/baseline/checkpoints/best.pt \\
        --require-behavioural

    # code-only machine, plumbing check:
    python scripts/head_smoke_gate.py --base-checkpoint <any real checkpoint>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from probunet.evaluation.metrics import consensus_scores  # noqa: E402
from probunet.training.config import ExperimentConfig  # noqa: E402
from probunet.training.freeze import parameter_fingerprint  # noqa: E402
from probunet.training.trainer import Trainer  # noqa: E402

LOGGER = logging.getLogger("probunet.head_gate")
ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "smoke_head.yaml"
REFERENCE = ROOT / "results" / "consensus_headroom_baseline.json"

STAND_IN_ORACLE_RATIO = 0.10
"""Below this ``oracle / ceiling``, the base is treated as a stand-in.

Measured on both sides of the line, so it is not a guessed threshold: the pre-registered
Phase 1 base gives ``0.5244 / 0.6446 = 0.813``, and a 2-epoch smoke checkpoint gives
``0.0035 / 0.5831 = 0.006``. Two orders of magnitude apart, so anything in between is
already anomalous and is worth landing on the conservative side of.
"""

ORACLE_TOLERANCE = 0.10
"""Absolute band around the pre-registered oracle.

Wide on purpose: the gate evaluates a few hundred images rather than the full 3021, so
sampling variation is real. It is still far tighter than the two-orders-of-magnitude gap
between a trained and a stand-in base, which is what this has to catch.
"""

DEGENERATE_TARGET_TIE_FRACTION = 0.95
"""Above this tie fraction, the rank correlation is reported as uninformative.

With a stand-in base almost every candidate is empty, and soft-consensus Dice against a
never-empty consensus is 0 for all of them. A near-tied target vector can then produce a
near-perfect rho that means nothing at all.
"""


class Verdict:
    """Collects PASS / FAIL / SKIP lines and reports the overall result."""

    def __init__(self) -> None:
        """Start with no checks recorded."""
        self.rows: list[tuple[str, str, str]] = []

    def check(self, ok: bool, name: str, detail: str) -> None:
        """Record a pass/fail check.

        Args:
            ok: Whether it passed.
            name: Short label.
            detail: The measured values behind the verdict.
        """
        self.rows.append(("PASS" if ok else "FAIL", name, detail))

    def skip(self, name: str, detail: str) -> None:
        """Record a check that was deliberately **not** run.

        Args:
            name: Short label.
            detail: Why it was skipped.
        """
        self.rows.append(("SKIP", name, detail))

    @property
    def failed(self) -> bool:
        """Whether any check failed."""
        return any(status == "FAIL" for status, _, _ in self.rows)

    @property
    def skipped(self) -> bool:
        """Whether any check was skipped."""
        return any(status == "SKIP" for status, _, _ in self.rows)

    def render(self, behavioural: bool) -> str:
        """Render the verdict block.

        Args:
            behavioural: Whether the behavioural criteria ran.

        Returns:
            One line per check, then the overall result.
        """
        width = max(len(name) for _, name, _ in self.rows)
        lines = [f"{status}  {name:<{width}}  {detail}" for status, name, detail in self.rows]
        lines.append("")
        if self.failed:
            lines.append("GATE FAILED -- do not start the real run")
        elif not behavioural:
            lines += [
                "PLUMBING ONLY - BEHAVIOURAL CRITERIA SKIPPED",
                "",
                "The plumbing works. Nothing here says the head LEARNS anything: that",
                "question needs a trained base, which lives only on the training machine.",
                "Re-run there with --require-behavioural before the real run.",
            ]
        else:
            lines.append("GATE PASSED -- behavioural criteria checked against pre-registered bands")
        return "\n".join(lines)


def load_reference() -> dict[str, Any] | None:
    """Read the pre-registered headroom bands.

    Returns:
        The aggregate row and the candidate count it was measured at, or None if the
        tracked reference file is absent.
    """
    if not REFERENCE.exists():
        return None
    record = json.loads(REFERENCE.read_text())
    aggregate = record["buckets"]["all"]
    return {
        "n_samples": record.get("n_samples"),
        "random": aggregate["random"]["mean"],
        "oracle": aggregate["oracle"]["mean"],
        "ceiling": aggregate["ceiling"]["mean"],
    }


@torch.no_grad()
def target_distribution(trainer: Trainer) -> dict[str, float]:
    """Describe the regression targets the head is being fitted to.

    **Printed beside rho because rho alone cannot be trusted.** If almost every candidate
    scores identically -- which is what a stand-in base produces, since empty candidates all
    score exactly 0 against a never-empty consensus -- then a near-perfect rank correlation
    is an artifact of ties, not evidence of ranking ability.

    Args:
        trainer: A trainer whose head and validation loader are built.

    Returns:
        Count of distinct values, fraction exactly zero, min/median/max, and the tie
        fraction feeding the rank computation.
    """
    batch = next(iter(trainer.data.loaders["val"]))
    image = batch["image"].to(trainer.device)
    graders = batch["masks"].to(trainer.device)
    _, candidates = trainer.head.sample_candidates(
        image, trainer.config.head.eval_samples, torch.Generator().manual_seed(
            trainer.config.head.eval_seed
        )
    )
    targets = consensus_scores(candidates, graders).cpu().numpy()

    # Tie fraction per image, averaged: 1 - (distinct values / candidates).
    per_image_distinct = np.array([len(np.unique(row)) for row in targets])
    tie_fraction = float(1.0 - (per_image_distinct / targets.shape[1]).mean())
    return {
        "distinct_values": float(len(np.unique(targets))),
        "fraction_exactly_zero": float((targets == 0.0).mean()),
        "min": float(targets.min()),
        "median": float(np.median(targets)),
        "max": float(targets.max()),
        "tie_fraction": tie_fraction,
    }


def run_gate(
    base_checkpoint: Path, device: str | None, expect_base_sha: str | None
) -> tuple[Verdict, bool]:
    """Train the smoke head and evaluate every criterion appropriate to the base regime.

    Args:
        base_checkpoint: The frozen base to fit on.
        device: Optional device override.
        expect_base_sha: Optional expected base parameter sha256 prefix.

    Returns:
        ``(verdict, behavioural)`` -- the completed verdict and whether the behavioural
        criteria actually ran.

    Raises:
        FileNotFoundError: If the base checkpoint is absent. **A refusal, not a fallback**:
            a gate that quietly substituted its own base could not detect anything.
    """
    base_checkpoint = Path(base_checkpoint)
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"base checkpoint not found: {base_checkpoint}\n"
            "REFUSING to run: the gate does not substitute a stand-in base. Note that "
            "runs/ is gitignored, so a trained checkpoint is present only on the machine "
            "that produced it."
        )

    config = ExperimentConfig.from_yaml(CONFIG)
    verdict = Verdict()
    reference = load_reference()

    with tempfile.TemporaryDirectory() as temporary:
        # A temp run dir: the gate must leave no checkpoint behind that could later be
        # mistaken for a real head result.
        config = dataclasses.replace(
            config,
            run=dataclasses.replace(
                config.run, out_dir=Path(temporary), device=device or config.run.device
            ),
        )
        trainer = Trainer(config, base_checkpoint=base_checkpoint)
        before = parameter_fingerprint(trainer.head.base)
        counts = trainer.head.parameter_counts()

        summary = trainer.train()
        verdict.check(
            True, "1 completes", f"{summary['epochs']} epochs, {summary['global_step']} steps"
        )

        after = parameter_fingerprint(trainer.head.base)
        identity = f"{counts['base']:,} frozen, base sha256 {before[:12]}"
        if expect_base_sha:
            identity += f" (expected {expect_base_sha[:12]})"
        verdict.check(
            after == before
            and counts["base"] > 0
            and (not expect_base_sha or before.startswith(expect_base_sha)),
            "2 base frozen",
            identity + f", unchanged={after == before}",
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
        random_baseline = last["val/random_consensus_dice"]
        oracle = last["val/oracle_consensus_dice"]
        ceiling = last["val/ceiling"]
        ratio = oracle / max(ceiling, 1e-12)

        # ---- regime, decided by measurement -------------------------------------------
        behavioural = ratio >= STAND_IN_ORACLE_RATIO
        LOGGER.info(
            "base regime: %s (oracle/ceiling = %.4f; stand-in below %.2f)",
            "TRAINED" if behavioural else "STAND-IN",
            ratio,
            STAND_IN_ORACLE_RATIO,
        )

        targets = target_distribution(trainer)
        LOGGER.info(
            "target distribution: %d distinct, %.1f%% exactly zero, min %.4f median %.4f "
            "max %.4f, tie fraction %.3f",
            int(targets["distinct_values"]),
            100 * targets["fraction_exactly_zero"],
            targets["min"],
            targets["median"],
            targets["max"],
            targets["tie_fraction"],
        )
        LOGGER.info(
            "head parameters: %s scorer + %s area control (disjoint); frozen base %s",
            f"{counts['scorer']:,}",
            f"{counts['area_baseline']:,}",
            f"{counts['base']:,}",
        )
        LOGGER.info(
            "headroom captured: head %.1f%% vs area-only control %.1f%%; "
            "head huber %.6f vs area huber %.6f",
            100 * last.get("val/headroom_captured", float("nan")),
            100 * last.get("val/headroom_captured_area_only", float("nan")),
            last.get("train/head_huber", float("nan")),
            last.get("train/area_huber", float("nan")),
        )

        if not behavioural:
            reason = (
                f"stand-in base: oracle {oracle:.5f} is {100 * ratio:.1f}% of ceiling "
                f"{ceiling:.4f}; ordering near-zero numbers passes on noise"
            )
            verdict.skip("4 scores in band", reason)
            verdict.skip(
                "5 spearman informative",
                f"targets {targets['tie_fraction']:.1%} tied, "
                f"{100 * targets['fraction_exactly_zero']:.0f}% exactly zero -- rho is an "
                "artifact of ties here",
            )
            return verdict, False

        # ---- behavioural criteria, against pre-registered bands ------------------------
        if reference is None:
            verdict.skip("4 scores in band", f"no pre-registered reference at {REFERENCE}")
        elif reference["n_samples"] != config.head.eval_samples:
            # No interpolation invented. Oracle grows with the candidate count as a
            # max-order statistic, and deriving its value at another K needs the per-patch
            # oracle-vs-K curve, which the summary file does not carry.
            verdict.skip(
                "4 scores in band",
                f"reference is at {reference['n_samples']} candidates but the gate uses "
                f"{config.head.eval_samples}; no pre-registered band for that count",
            )
        else:
            in_band = abs(oracle - reference["oracle"]) <= ORACLE_TOLERANCE
            beats_random = selected >= random_baseline
            verdict.check(
                bool(in_band and beats_random and selected <= oracle + 1e-6),
                "4 scores in band",
                f"oracle {oracle:.4f} vs pre-registered {reference['oracle']:.4f} "
                f"+-{ORACLE_TOLERANCE}; selected {selected:.4f} vs random "
                f"{random_baseline:.4f}",
            )

        rho = last["val/spearman"]
        excluded = last["val/spearman_excluded_fraction"]
        informative = targets["tie_fraction"] < DEGENERATE_TARGET_TIE_FRACTION
        if not informative:
            verdict.skip(
                "5 spearman informative",
                f"targets {targets['tie_fraction']:.1%} tied -- rho {rho:+.4f} is "
                "uninformative, not a pass",
            )
        else:
            verdict.check(
                bool(np.isfinite(rho)) and excluded < 1.0,
                "5 spearman informative",
                f"rho {rho:+.4f} over {last['val/spearman_images']:.0f} images, "
                f"{excluded:.1%} excluded, targets {targets['tie_fraction']:.1%} tied",
            )

    return verdict, True


def main() -> int:
    """Parse arguments, run the gate, print the verdict.

    Returns:
        0 on success, 1 on any failure, 2 when ``--require-behavioural`` was asked for but
        the behavioural criteria did not run.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--expect-base-sha",
        default=None,
        help="expected base parameter sha256 prefix, so a gate run against the wrong "
             "checkpoint fails instead of quietly measuring something else",
    )
    parser.add_argument(
        "--require-behavioural",
        action="store_true",
        help="exit non-zero if the behavioural criteria were skipped. Use this on the "
             "training machine, where a plumbing-only result means the gate did not run.",
    )
    arguments = parser.parse_args()

    try:
        verdict, behavioural = run_gate(
            arguments.base_checkpoint, arguments.device, arguments.expect_base_sha
        )
    except FileNotFoundError as error:
        print(f"\nFAIL  0 base present  {error}")
        print("\nGATE FAILED -- do not start the real run")
        return 1

    print()
    print(verdict.render(behavioural))
    if verdict.failed:
        return 1
    if arguments.require_behavioural and not behavioural:
        print("\n--require-behavioural was set and the behavioural criteria did not run.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
