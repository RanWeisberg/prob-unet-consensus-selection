"""Pre-registration measurement: has the empty-mask pathology gone, and what headroom
remains for the selection head?

Under the old per-grader **mean** Dice an all-empty mask scored 0.75 on bucket 1 and beat
best-of-16 (oracle@16 = 0.7458 < 0.7500), so no selector over those samples could have
won -- the target itself was inverted (FINDINGS 4.4). Soft consensus should send all-empty
to **exactly 0** wherever any grader saw a lesion. This module measures that on real data
rather than asserting it from arithmetic, and reports ``oracle - random`` as the gap the
head is trying to capture.

The logic lives here rather than in ``scripts/consensus_headroom.py`` so it can be tested
and so the report's notebook can import it; the script is a thin CLI over it.

**The verdict is three-way, and the third value matters.** ``oracle == all_empty`` is a
*tie*, not a win for all-empty, and it has an entirely different cause: it is what a model
emitting only empty candidates produces, in which case every selection rule scores 0 and
the pass says nothing about the target at all. Collapsing that into "all-empty wins" would
report a broken or undertrained checkpoint as a refutation of soft consensus. The
``nonempty_frac`` column is what tells the two apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from probunet.evaluation.metrics import (
    all_empty_consensus_dice,
    consensus_ceiling,
    consensus_oracle,
    consensus_random,
    consensus_selected,
    emptiest_sample_index,
    summarize,
)
from probunet.evaluation.sampling import draw_prior_samples
from probunet.model.prob_unet import ProbUNet

EVAL_SAMPLES = 16
"""Candidates per image. 16 so the numbers sit alongside the existing Phase 1 table."""

BUCKET_LABELS = {1: "1 grader", 2: "2 graders", 3: "3 graders", 4: "4 graders"}

VERDICT_TOLERANCE = 1e-9
"""Below this, oracle and all-empty are a tie rather than a winner and a loser."""


@torch.no_grad()
def measure_split(
    model: ProbUNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Score every patch in a split under the soft-consensus metric.

    **One shared candidate set per image.** The 16 candidates are drawn once and random,
    oracle, emptiest-sample and the ceiling are all computed from that same set. Scoring
    them from independent draws would confound a selection rule's contribution with
    sampling noise -- the difference between two rules has to come from the *choice*, not
    from two different sets of things to choose between.

    Args:
        model: The frozen base model.
        loader: An eval-mode loader yielding ``image`` and ``masks``.
        device: Device to run on.
        n_samples: Candidates per image.
        seed: Sampling seed; :data:`DEFAULT_EVAL_SEED` keeps head-on-Phase1 and
            head-on-Phase2 comparable.

    Returns:
        Per-patch arrays: ``random``, ``oracle``, ``all_empty``, ``emptiest``,
        ``ceiling``, ``nonempty_frac`` (fraction of this image's candidates carrying any
        foreground), ``n_nonempty`` (the ambiguity bucket) and ``index``.
    """
    # A CPU generator, matching every other seeded sampling path in the project, so the
    # draw is reproducible across backends.
    generator = torch.Generator().manual_seed(seed)
    columns: dict[str, list[np.ndarray]] = {
        key: [] for key in
        ("random", "oracle", "all_empty", "emptiest", "ceiling", "nonempty_frac",
         "n_nonempty", "index")
    }

    for batch in loader:
        image = batch["image"].to(device)
        graders = batch["masks"].to(device)

        samples = draw_prior_samples(model, image, n_samples, generator)

        emptiest = emptiest_sample_index(samples)
        values = {
            "random": consensus_random(samples, graders),
            "oracle": consensus_oracle(samples, graders),
            "all_empty": all_empty_consensus_dice(graders),
            "emptiest": consensus_selected(samples, graders, emptiest),
            "ceiling": consensus_ceiling(graders),
            # Per image, not batch-averaged: a model emitting only empty candidates makes
            # every selection rule tie at 0, and this column is what tells that apart from
            # a failure of the target itself.
            "nonempty_frac": (samples.flatten(start_dim=2) != 0).any(dim=2)
            .to(torch.float32).mean(dim=1),
            "n_nonempty": (graders.flatten(start_dim=2).sum(dim=2) > 0).sum(dim=1),
            "index": batch["index"],
        }
        for key, value in values.items():
            columns[key].append(value.detach().cpu().numpy())

    return {key: np.concatenate(parts) for key, parts in columns.items()}


def per_bucket(results: dict[str, np.ndarray]) -> dict[str, Any]:
    """Break the per-patch arrays down by ambiguity bucket, plus an aggregate row.

    Args:
        results: Output of :func:`measure_split`.

    Returns:
        A mapping from bucket label (and ``"all"``) to that group's summary.
    """
    report: dict[str, Any] = {}
    buckets = [(label, results["n_nonempty"] == count)
               for count, label in sorted(BUCKET_LABELS.items())]
    buckets.append(("all", np.ones_like(results["n_nonempty"], dtype=bool)))

    for label, selector in buckets:
        if not selector.any():
            continue
        group = {key: results[key][selector]
                 for key in ("random", "oracle", "all_empty", "emptiest", "ceiling",
                             "nonempty_frac")}
        oracle, all_empty = group["oracle"].mean(), group["all_empty"].mean()
        # Three-way, not two. A TIE is not a win for all-empty, and it has a completely
        # different cause: it is what a model that emits only empty candidates produces,
        # in which case every selection rule scores 0 and the pass says nothing about the
        # target at all. Collapsing tie into "all-empty wins" would report a broken or
        # untrained checkpoint as a refutation of soft consensus.
        if oracle > all_empty + VERDICT_TOLERANCE:
            verdict = "ok"
        elif oracle < all_empty - VERDICT_TOLERANCE:
            verdict = "all_empty_wins"
        else:
            verdict = "degenerate_tie"
        report[label] = {
            "n": int(selector.sum()),
            **{key: summarize(value) for key, value in group.items()},
            "headroom_oracle_minus_random": float(
                oracle - group["random"].mean()
            ),
            # THE question this pass exists to answer.
            "verdict": verdict,
            "oracle_beats_all_empty": verdict == "ok",
            "oracle_fraction_of_ceiling": float(
                oracle / max(group["ceiling"].mean(), 1e-12)
            ),
        }
    return report


def render(report: dict[str, Any]) -> str:
    """Render the pre-registration table as plain text.

    Args:
        report: Output of :func:`per_bucket`.

    Returns:
        A block ready to paste into FINDINGS.
    """
    labels = {
        "ok": "OK",
        "all_empty_wins": "*** ALL-EMPTY WINS ***",
        "degenerate_tie": "tie -- see nonempty",
    }
    header = (
        f"{'bucket':<12}{'n':>6}{'random':>9}{'oracle':>9}{'all-empty':>10}"
        f"{'emptiest':>9}{'ceiling':>9}{'headroom':>9}{'nonempty':>9}  verdict"
    )
    lines = [header, "-" * len(header)]
    for label, row in report.items():
        lines.append(
            f"{label:<12}{row['n']:>6}"
            f"{row['random']['mean']:>9.4f}{row['oracle']['mean']:>9.4f}"
            f"{row['all_empty']['mean']:>10.4f}{row['emptiest']['mean']:>9.4f}"
            f"{row['ceiling']['mean']:>9.4f}"
            f"{row['headroom_oracle_minus_random']:>9.4f}"
            f"{row['nonempty_frac']['mean']:>9.4f}  {labels[row['verdict']]}"
        )
    return "\n".join(lines)
