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

from typing import TYPE_CHECKING, Any

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
from probunet.evaluation.metrics import consensus_scores
from probunet.evaluation.sampling import draw_prior_samples
from probunet.model.prob_unet import ProbUNet

if TYPE_CHECKING:  # avoid a runtime cycle: extension.head imports training.diagnostics
    from probunet.extension.head import SelectionHead

EVAL_SAMPLES = 16
"""Candidates per image. 16 so the numbers sit alongside the existing Phase 1 table."""

BUCKET_LABELS = {1: "1 grader", 2: "2 graders", 3: "3 graders", 4: "4 graders"}

VERDICT_TOLERANCE = 1e-9
"""Below this, oracle and all-empty are a tie rather than a winner and a loser."""

GRADER_COLUMNS = ("ceiling", "all_empty")
"""Columns that depend on the grader masks ALONE -- no model, no checkpoint, no weights.

These are properties of the dataset and the split, so they are **final**: they do not move
when the pass is rerun on a different checkpoint, and comparing them across arms is
meaningless because they cannot differ. :func:`measure_ceilings` computes exactly these,
which is what lets the report notebook import the ceiling table without loading weights.
"""

MODEL_COLUMNS = ("random", "oracle", "emptiest", "nonempty_frac")
"""Columns that require sampling from a model, and therefore a checkpoint."""

BOOKKEEPING_COLUMNS = ("n_nonempty", "index")
"""Per-patch bookkeeping, grouped on rather than summarized."""


@torch.no_grad()
def measure_ceilings(
    loader: torch.utils.data.DataLoader, device: torch.device | None = None
) -> dict[str, np.ndarray]:
    """Compute the grader-only columns: the achievable ceiling and the all-empty score.

    **No model, no checkpoint, no weights.** Both quantities are functions of the four
    grader masks alone, so this runs anywhere the data is -- which is what lets the report
    notebook produce the ceiling table on CPU without downloading a checkpoint, and what
    makes the ceilings final rather than per-run.

    Args:
        loader: An eval-mode loader yielding ``masks``.
        device: Optional device for the arithmetic; CPU is fine and is the default.

    Returns:
        Per-patch ``ceiling``, ``all_empty``, ``n_nonempty`` and ``index``.
    """
    device = device or torch.device("cpu")
    columns: dict[str, list[np.ndarray]] = {
        key: [] for key in ("ceiling", "all_empty", "n_nonempty", "index")
    }
    for batch in loader:
        graders = batch["masks"].to(device)
        values = {
            "ceiling": consensus_ceiling(graders),
            "all_empty": all_empty_consensus_dice(graders),
            "n_nonempty": (graders.flatten(start_dim=2).sum(dim=2) > 0).sum(dim=1),
            "index": batch["index"],
        }
        for key, value in values.items():
            columns[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in columns.items()}


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


@torch.no_grad()
def measure_selection(
    head: "SelectionHead",
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Score a trained head against every baseline, on one shared candidate set.

    The Stage 5 results table. Every rule -- head, size-prior control, random, oracle,
    emptiest -- chooses from the **same** candidates for a given image, so the differences
    between them come from the choice and not from being handed different things to choose
    between.

    Args:
        head: The trained selection head, with its frozen base.
        loader: An eval-mode loader yielding ``image`` and ``masks``.
        device: Device to run on.
        n_samples: Candidates per image.
        seed: Sampling seed, so the arms see comparable draws.

    Returns:
        Per-patch arrays, one column per selection rule plus ``ceiling``,
        ``nonempty_frac``, ``n_nonempty`` and ``index``.
    """
    generator = torch.Generator().manual_seed(seed)
    columns: dict[str, list[np.ndarray]] = {}

    for batch in loader:
        image = batch["image"].to(device)
        graders = batch["masks"].to(device)

        features, candidates = head.sample_candidates(image, n_samples, generator)
        scores = consensus_scores(candidates, graders)
        chosen = head.select(features, candidates)
        by_area = head.select_by_area(candidates)

        values = {
            "head": consensus_selected(candidates, graders, chosen),
            "area_only": consensus_selected(candidates, graders, by_area),
            "random": scores.mean(dim=1),
            "oracle": scores.amax(dim=1),
            "emptiest": consensus_selected(
                candidates, graders, emptiest_sample_index(candidates)
            ),
            "all_empty": all_empty_consensus_dice(graders),
            "ceiling": consensus_ceiling(graders),
            "nonempty_frac": (candidates.flatten(start_dim=2) != 0)
            .any(dim=2)
            .to(torch.float32)
            .mean(dim=1),
            "n_nonempty": (graders.flatten(start_dim=2).sum(dim=2) > 0).sum(dim=1),
            "index": batch["index"],
        }
        for key, value in values.items():
            columns.setdefault(key, []).append(value.detach().cpu().numpy())

    return {key: np.concatenate(parts) for key, parts in columns.items()}


def render_selection(report: dict[str, Any]) -> str:
    """Render the Stage 5 selection table.

    The ``gap`` column -- the fraction of the oracle-minus-random headroom captured -- is
    the one to lead with: it is scale-free across buckets whose ceilings run 0.40 to 0.89,
    so it is the only column comparable between rows. ``area`` sits immediately beside
    ``head`` on purpose: if the head barely beats its size-prior control, it learned area
    rather than spatial agreement, and adjacency makes that impossible to miss.

    Args:
        report: Output of :func:`per_bucket` over :func:`measure_selection`.

    Returns:
        A plain-text table.
    """
    header = (
        f"{'bucket':<11}{'n':>5}{'random':>8}{'head':>8}{'area':>8}{'oracle':>8}"
        f"{'ceil':>7}{'gap%':>7}{'area%':>7}{'allmt':>7}{'orc|off':>8}"
    )
    lines = [header, "-" * len(header)]
    for label, row in report.items():
        lines.append(
            f"{label:<11}{row['n']:>5}"
            f"{row['random']['mean']:>8.4f}{row['head']['mean']:>8.4f}"
            f"{row['area_only']['mean']:>8.4f}{row['oracle']['mean']:>8.4f}"
            f"{row['ceiling']['mean']:>7.4f}"
            f"{100 * row['gap_captured_head']:>7.1f}"
            f"{100 * row['gap_captured_area_only']:>7.1f}"
            f"{row['all_candidates_empty_fraction']:>7.3f}"
            f"{row.get('oracle_where_offered', float('nan')):>8.4f}"
        )
    lines += [
        "",
        "gap%    = fraction of the oracle-minus-random headroom the HEAD captured "
        "(lead with this: scale-free across buckets)",
        "area%   = the same for the size-prior CONTROL. If gap% is not well above it, the "
        "head learned area, not overlap.",
        "allmt   = fraction of images where EVERY candidate was empty; those score 0 under "
        "every rule and cap any selector",
        "orc|off = oracle over only the images where the sampler offered a non-empty "
        "candidate -- separates 'offered nothing' from 'offered something poorly located'",
    ]
    return "\n".join(lines)


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
        group = {
            key: value[selector]
            for key, value in results.items()
            if key not in BOOKKEEPING_COLUMNS
        }
        if "oracle" not in group:
            # Ceiling-only mode: no model, so no verdict to reach.
            report[label] = {
                "n": int(selector.sum()),
                **{key: summarize(value) for key, value in group.items()},
            }
            continue
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
        random_mean = group["random"].mean()
        gap = max(float(oracle - random_mean), 1e-12)
        report[label] = {
            "n": int(selector.sum()),
            **{key: summarize(value) for key, value in group.items()},
            "headroom_oracle_minus_random": float(oracle - random_mean),
            # THE number to lead with. Scale-free across buckets whose ceilings run 0.40 to
            # 0.89, so it is the only column comparable BETWEEN buckets. From the means,
            # not per image: the per-image denominator is zero whenever every candidate
            # scores alike, which on bucket 1 is common.
            **{
                f"gap_captured_{key}": float((group[key].mean() - random_mean) / gap)
                for key in ("head", "area_only", "emptiest")
                if key in group
            },
            # THE question this pass exists to answer.
            "verdict": verdict,
            "oracle_beats_all_empty": verdict == "ok",
            "oracle_fraction_of_ceiling": float(
                oracle / max(group["ceiling"].mean(), 1e-12)
            ),
        }
        if "nonempty_frac" in group:
            # Images where EVERY candidate is empty. They score exactly 0 under every
            # rule, so they cap what any selector could achieve and belong in the table
            # rather than hidden inside an average. This separates "the sampler offered
            # nothing" from "the sampler offered something poorly localized".
            offered = group["nonempty_frac"] > 0
            report[label]["all_candidates_empty_fraction"] = float((~offered).mean())
            if "oracle" in group and offered.any():
                report[label]["oracle_where_offered"] = float(group["oracle"][offered].mean())
                report[label]["n_offered"] = int(offered.sum())
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
    if report and "verdict" not in next(iter(report.values())):
        header = f"{'bucket':<12}{'n':>6}{'ceiling':>10}{'all-empty':>11}"
        lines = [header, "-" * len(header)]
        for label, row in report.items():
            lines.append(
                f"{label:<12}{row['n']:>6}"
                f"{row['ceiling']['mean']:>10.4f}{row['all_empty']['mean']:>11.4f}"
            )
        return "\n".join(lines)

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
