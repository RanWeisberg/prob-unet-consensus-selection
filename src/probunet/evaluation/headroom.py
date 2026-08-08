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
from probunet.evaluation.metrics import consensus_scores, spearman_per_image
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
        predicted = head.score_candidates(features, candidates)
        area_scores = head.score_by_area(candidates)
        chosen = predicted.argmax(dim=1)
        by_area = area_scores.argmax(dim=1)

        # Does the area control reduce to "pick the largest-area candidate"? Measured per
        # image rather than assumed: the scorer is a ReLU MLP on log1p(area), which is NOT
        # guaranteed monotone -- measured inversions occur on synthetic data -- so whether
        # it collapses to that deterministic rule is a property of the data and has to be
        # reported per run.
        areas = (candidates != 0).flatten(start_dim=2).sum(dim=2)
        picks_largest = (by_area == areas.argmax(dim=1)).to(torch.float32)

        head_rho, head_valid = spearman_per_image(predicted, scores)
        area_rho, area_valid = spearman_per_image(area_scores, scores)

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
            "area_picks_largest": picks_largest,
            # The head's predicted-score distribution. A head emitting near-constant scores
            # that merely rank non-empty above empty is a different finding from one that
            # discriminates WITHIN the non-empty candidates, and the within-image spread is
            # what separates them.
            "pred_spread_within_image": predicted.amax(dim=1) - predicted.amin(dim=1),
            "pred_std_within_image": predicted.std(dim=1),
            "pred_mean": predicted.mean(dim=1),
            "pred_of_chosen": predicted.gather(1, chosen.unsqueeze(1)).squeeze(1),
        }
        for key, value in values.items():
            columns.setdefault(key, []).append(value.detach().cpu().numpy())
        # Rank correlations come back on the CPU already (float64 ranks; see
        # metrics.spearman_per_image), and carry a validity mask rather than a NaN average.
        for key, (rho, valid) in (
            ("head_spearman", (head_rho, head_valid)),
            ("area_spearman", (area_rho, area_valid)),
        ):
            masked = rho.numpy().astype(np.float64).copy()
            masked[~valid.numpy()] = np.nan
            columns.setdefault(key, []).append(masked)

    return {key: np.concatenate(parts) for key, parts in columns.items()}


def _stat(row: dict[str, Any], key: str, field: str = "mean") -> float:
    """Read a summary statistic, tolerating an absent column or an all-degenerate one.

    ``summarize`` returns ``None`` for every statistic when no finite value survived --
    which happens for a Spearman column whose images were all degenerate -- and ``None``
    cannot be formatted as a float.

    Args:
        row: One bucket's report row.
        key: Column name.
        field: Statistic within that column's summary.

    Returns:
        The value, or NaN when absent or undefined.
    """
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get(field)
    return float("nan") if value is None else float(value)


SELECTION_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    # (header, report key, format, legend -- empty legend means self-explanatory)
    ("random", "random", "8.4f", ""),
    ("area", "area_only", "8.4f", "the size-prior control: sees candidate area, never the image"),
    ("head", "head", "8.4f", ""),
    ("oracle", "oracle", "8.4f", ""),
    ("ceil", "ceiling", "7.4f", "best any binary mask could score; NEVER report against 1.0"),
    ("edge", "head_edge_over_area", "+8.4f",
     "head - area. THE HEADLINE: the head's contribution beyond a size prior"),
    ("e/tot", "head_edge_frac_of_total_headroom", "7.1%",
     "edge as a fraction of TOTAL headroom (oracle - random)"),
    ("e/left", "head_edge_frac_of_headroom_area_left", "7.1%",
     "edge as a fraction of the headroom the AREA CONTROL LEFT (oracle - area). e/tot and "
     "e/left trend in OPPOSITE directions across buckets; report both, claim neither"),
    ("allmt", "all_candidates_empty_fraction", "7.3f",
     "fraction of images where EVERY candidate was empty. Those score 0 under every rule "
     "and are a hard cap on any selector"),
    ("orc|off", "oracle_where_offered", "8.4f",
     "oracle over ONLY the images where the sampler offered a non-empty candidate -- "
     "separates 'offered nothing' from 'offered something poorly located'"),
    ("lrgst", "area_picks_largest", "7.3f",
     "fraction of images where the area control picked the LARGEST-area candidate. 1.000 "
     "means it reduced to that deterministic rule ON THIS CANDIDATE SET"),
)
"""Columns of the main selection table, with the legend text each one owns."""

RANKING_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("rho_h", "head_spearman", "8.3f", "head rank correlation with the true scores"),
    ("rho_a", "area_spearman", "8.3f",
     "the same for the area control. rho_h - rho_a is the head's incremental RANKING "
     "ability; if rho_a is high, most of the ranking signal in this task is size"),
    ("excl_h", "head_spearman_excluded_fraction", "8.3f",
     "fraction of images excluded as degenerate (all candidates tied), counted not dropped"),
    ("excl_a", "area_spearman_excluded_fraction", "8.3f", ""),
    ("pr_mean", "pred_mean", "9.4f", "mean predicted score"),
    ("pr_std", "pred_std_within_image", "8.4f",
     "WITHIN-IMAGE spread of the head's predicted scores. Near zero means the scores are "
     "nearly constant and candidate ORDER is doing the selection rather than the head"),
    ("pr_rng", "pred_spread_within_image", "8.4f", "within-image max - min"),
    ("pr_chos", "pred_of_chosen", "9.4f", "predicted score of the candidate actually chosen"),
)
"""Columns of the ranking / score-distribution table."""


def _stat(row: dict[str, Any], key: str, field: str = "mean") -> float:
    """Read a summary statistic, tolerating an absent column or an all-degenerate one.

    ``summarize`` returns ``None`` for every statistic when no finite value survived --
    which happens for a Spearman column whose images were all degenerate -- and ``None``
    cannot be formatted as a float.

    Args:
        row: One bucket's report row.
        key: Column name.
        field: Statistic within that column's summary.

    Returns:
        The value, or NaN when absent or undefined.
    """
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get(field)
    return float("nan") if value is None else float(value)


def _render_block(
    report: dict[str, Any], columns: tuple[tuple[str, str, str, str], ...], title: str
) -> list[str]:
    """Render one table block plus the legend for the columns it actually prints.

    The legend is generated **from the same tuple that generates the columns**, so a
    legend can never describe output that is not produced. That mismatch happened once
    here -- ``orc|off`` was documented after it had been dropped from the header -- and it
    is the same class as a config documenting an invocation that could not work.

    Args:
        report: Output of :func:`per_bucket`.
        columns: Column specifications.
        title: Block heading.

    Returns:
        The block's lines.
    """
    header = f"{title:<11}{'n':>5}" + "".join(
        f"{name:>{spec.lstrip('+').split('.')[0]}}" for name, _, spec, _ in columns
    )
    lines = [header, "-" * len(header)]
    for label, row in report.items():
        cells = []
        for _, key, spec, _ in columns:
            value = _stat(row, key)
            if spec.endswith("%"):
                width, precision = spec[:-1].split(".")
                cells.append(f"{100 * value:>{width}.{precision}f}")
            else:
                cells.append(f"{value:>{spec}}")
        lines.append(f"{label:<11}{row['n']:>5}" + "".join(cells))
    legend = [f"  {name:<8}= {text}" for name, _, _, text in columns if text]
    return lines + ([""] + legend if legend else [])


def render_selection(report: dict[str, Any]) -> str:
    """Render the Stage 5 selection table as two blocks.

    Split in two because thirteen columns on one line is unreadable, and the two blocks
    answer different questions: the first is *how well does each rule select*, the second
    is *is the head actually ranking, or are its scores nearly constant*.

    Args:
        report: Output of :func:`per_bucket` over :func:`measure_selection`.

    Returns:
        A plain-text report.
    """
    lines = _render_block(report, SELECTION_COLUMNS, "bucket")
    lines += ["", ""] + _render_block(report, RANKING_COLUMNS, "bucket")
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
        for key in ("head_spearman", "area_spearman"):
            if key in group:
                finite = np.isfinite(group[key])
                report[label][f"{key}_excluded_fraction"] = float((~finite).mean())
        if "head" in group and "area_only" in group:
            # The denominator question (FINDINGS 4.5). Reported BOTH ways, because the
            # bucket trend reverses depending on which is used and neither direction is
            # the finding on its own.
            head_mean, area_mean = group["head"].mean(), group["area_only"].mean()
            report[label]["head_edge_over_area"] = float(head_mean - area_mean)
            report[label]["head_edge_frac_of_total_headroom"] = float(
                (head_mean - area_mean) / gap
            )
            remaining = max(float(oracle - area_mean), 1e-12)
            report[label]["head_edge_frac_of_headroom_area_left"] = float(
                (head_mean - area_mean) / remaining
            )
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
