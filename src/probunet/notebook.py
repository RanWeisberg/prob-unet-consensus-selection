"""Display helpers for ``notebooks/submission.ipynb``.

The notebook is a **thin narrative layer**: markdown, a call, a figure. Everything it
needs beyond that lives here, because CLAUDE.md's rule for the notebook is that logic
belongs in the package and cells carry explanation. ``tests/test_variants.py`` asserts
that no cell defines a function or a class, so this module is what makes that possible
rather than a matter of restraint.

Nothing here computes a result. These are formatting and rendering helpers only: they load
tracked JSON, align text into tables, crop an image to its lesion, and draw mask overlays.
Every *number* the notebook shows is read from ``results/*.json`` or
``data/processed/showcase.npz`` and formatted here -- never recomputed, and never typed
into a cell.

**numpy and matplotlib only.** The notebook must run in a fresh Colab with no
``pip install``, so nothing here may reach for pandas, seaborn, or anything else whose
version could differ between machines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

PALETTE: dict[str, str] = {
    "random": "#9aa0a6",
    "area": "#e8a33d",
    "head": "#2f6fb3",
    "oracle": "#3f9e5a",
    "ceiling": "#6b4fa0",
    "baseline": "#2f6fb3",
    "modernized": "#c0504d",
    "grader": "#c0504d",
    "other": "#8899aa",
}
"""One colour per role, so a colour means the same thing in every figure."""

RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
"""Figure defaults, applied once by the notebook's library cell."""

FIGSIZE: dict[str, tuple[float, float]] = {
    "tile_row": (12.2, 2.5),
    "tile_grid": (12.2, 3.4),
    "chart": (7.8, 3.2),
    "chart_pair": (11.4, 3.4),
    "case": (12.6, 4.2),
    "arms": (12.2, 3.2),
    "demo_row": (13.0, 1.8),
}
"""Figure sizes by role, so every section's figures line up at the same widths."""


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------


def load_results(results_dir: Path, name: str) -> dict[str, Any]:
    """Load one tracked results JSON.

    Args:
        results_dir: The repository's ``results/`` directory.
        name: File name, e.g. ``"evaluation_test_baseline.json"``.

    Returns:
        The parsed record.
    """
    return json.loads((Path(results_dir) / name).read_text())


def load_manifest(showcase: Mapping[str, Any]) -> dict[str, Any]:
    """Read the provenance manifest out of a loaded ``showcase.npz``.

    Args:
        showcase: The loaded ``.npz``.

    Returns:
        The manifest mapping.
    """
    return json.loads(str(showcase["manifest_json"].item()))


def stat(block: Mapping[str, Any], key: str, field: str = "mean") -> float:
    """Read one statistic out of a results block, tolerating a bare scalar.

    Results files store most quantities as a summary dict and a few (``head_edge_over_area``
    and friends) as plain floats. This reads either without the caller having to know which.

    Args:
        block: One bucket's row, or an ``aggregate_over_all_patches`` block.
        key: Column name.
        field: Statistic within that column's summary.

    Returns:
        The value as a float.
    """
    node = block[key]
    return float(node[field]) if isinstance(node, dict) else float(node)


def case_arrays(showcase: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    """Collect one showcase case's arrays, keyed without their prefix.

    Args:
        showcase: The loaded ``showcase.npz``.
        prefix: Case key prefix, e.g. ``"b2"``.

    Returns:
        Mapping from the un-prefixed key to its array.
    """
    return {
        key[len(prefix) + 1:]: showcase[key]
        for key in showcase.files
        if key.startswith(prefix + "_")
    }


# ---------------------------------------------------------------------------------
# Text tables
# ---------------------------------------------------------------------------------


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    title: str | None = None,
    note: str | None = None,
) -> None:
    """Print a plain aligned text table.

    Deliberately not pandas: a DataFrame's repr is a version surface, and the notebook has
    to render identically on a grader's Colab and on the machine that produced it.

    Args:
        headers: Column headings.
        rows: Row values; anything is accepted and stringified.
        title: Optional heading printed above the table.
        note: Optional footnote printed below it.
    """
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    if title:
        print(title)
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(cells[0])))
    print("  ".join("-" * w for w in widths))
    for row in cells[1:]:
        print("  ".join(
            cell.rjust(widths[i]) if i else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        ))
    if note:
        print(note)


def describe_monotonicity(values: Sequence[float], label: str) -> str:
    """State whether a per-bucket series is monotone, measured rather than asserted.

    The validation split produced two cleanly monotone (and opposed) per-bucket trends for
    the head's edge. That property did **not** survive the test split, and a sentence
    claiming it would now be false. So the notebook computes the verdict from the numbers
    it is about to print instead of carrying a claim that can go stale.

    Args:
        values: The series, in bucket order.
        label: Name of the series, for the sentence.

    Returns:
        A one-line verdict.
    """
    deltas = [b - a for a, b in zip(values, values[1:])]
    rising = all(d > 0 for d in deltas)
    falling = all(d < 0 for d in deltas)
    shape = ("monotonically rising" if rising else
             "monotonically falling" if falling else "NOT monotone")
    return (f"{label}: {values[0]:.1f}% -> {values[-1]:.1f}% across buckets, {shape} "
            f"(steps {', '.join(f'{d:+.1f}' for d in deltas)})")


# ---------------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------------


def crop_box(
    masks: Iterable[np.ndarray], margin: int = 14, minimum: int = 40
) -> tuple[int, int, int, int]:
    """A square crop around everything non-zero across several mask stacks.

    LIDC lesions here run from 1 pixel to a few hundred in a 128x128 frame, so an uncropped
    grid of 16 samples shows dots. This is a **display transform only**: every panel of a
    figure uses one box, and the notebook prints the box in the caption so the crop is
    never silent.

    Args:
        masks: Arrays of shape ``(H, W)`` or ``(n, H, W)``.
        margin: Pixels of context around the union of the foreground.
        minimum: Smallest box side, so a 1-pixel lesion still gets a readable frame.

    Returns:
        ``(row_start, row_stop, col_start, col_stop)``.
    """
    stacks = [np.asarray(m) for m in masks]
    height, width = stacks[0].shape[-2:]
    union = np.zeros((height, width), dtype=bool)
    for arr in stacks:
        union |= (arr.reshape(-1, height, width) != 0).any(axis=0)
    if not union.any():
        return 0, height, 0, width

    rows, cols = np.where(union)
    r0, r1 = int(rows.min()) - margin, int(rows.max()) + 1 + margin
    c0, c1 = int(cols.min()) - margin, int(cols.max()) + 1 + margin
    size = max(r1 - r0, c1 - c0, minimum)
    row_centre, col_centre = (r0 + r1) // 2, (c0 + c1) // 2
    r0 = max(0, row_centre - size // 2)
    c0 = max(0, col_centre - size // 2)
    return r0, min(height, r0 + size), c0, min(width, c0 + size)


def panel(
    ax: Any,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    box: tuple[int, int, int, int] | None = None,
    colour: str = PALETTE["grader"],
    title: str | None = None,
    alpha: float = 0.55,
) -> None:
    """Draw one tile: a greyscale CT patch with an optional binary mask over it.

    Args:
        ax: Target axes.
        image: ``(H, W)`` float image in ``[0, 1]``.
        mask: Optional ``(H, W)`` binary mask.
        box: Optional crop from :func:`crop_box`.
        colour: Overlay colour.
        title: Optional tile title.
        alpha: Overlay opacity; the contour is drawn opaque regardless.
    """
    r0, r1, c0, c1 = box or (0, image.shape[0], 0, image.shape[1])
    ax.imshow(image[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if mask is not None:
        cropped = np.asarray(mask)[r0:r1, c0:c1].astype(float)
        rgba = np.zeros((*cropped.shape, 4))
        rgba[..., :3] = plt.matplotlib.colors.to_rgb(colour)
        rgba[..., 3] = cropped * alpha
        ax.imshow(rgba, interpolation="nearest")
        if cropped.any():
            ax.contour(cropped, levels=[0.5], colors=[colour], linewidths=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=7.5, pad=2)


def consensus_panel(ax: Any, consensus: np.ndarray,
                    box: tuple[int, int, int, int] | None = None) -> Any:
    """Draw the soft consensus map on a fixed 0-1 scale.

    Fixed ``vmin``/``vmax`` rather than autoscaled, so the five possible levels always map
    to the same colours and two figures can be compared by eye.

    Args:
        ax: Target axes.
        consensus: ``(H, W)`` float map with values in ``{0, .25, .5, .75, 1}``.
        box: Optional crop.

    Returns:
        The image handle, for a colourbar.
    """
    r0, r1, c0, c1 = box or (0, consensus.shape[0], 0, consensus.shape[1])
    handle = ax.imshow(consensus[r0:r1, c0:c1], cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("soft consensus", fontsize=7.5, pad=2)
    return handle


def mask_areas(stack: np.ndarray) -> np.ndarray:
    """Foreground pixel count per mask in a stack.

    Args:
        stack: ``(n, H, W)`` binary masks.

    Returns:
        ``(n,)`` integer areas.
    """
    arr = np.asarray(stack)
    return (arr.reshape(arr.shape[0], -1) != 0).sum(axis=1)


# ---------------------------------------------------------------------------------
# The two composite figures
# ---------------------------------------------------------------------------------


def phase2_case_figure(showcase: Mapping[str, Any], prefix: str) -> dict[str, int]:
    """One Set A case: the same patch's samples under both Phase 2 arms.

    Args:
        showcase: The loaded ``showcase.npz``.
        prefix: Case key prefix, ``"a0"``, ``"a1"`` or ``"a2"``.

    Returns:
        Non-empty sample counts per arm, so the caller can report legibility.
    """
    c = case_arrays(showcase, prefix)
    base, modern = c["samples_baseline_short"], c["samples_modernized_short"]
    box = crop_box([c["masks"], base, modern], margin=10)
    areas = {"baseline-short": mask_areas(base), "modernized-short": mask_areas(modern)}

    fig = plt.figure(figsize=FIGSIZE["arms"])
    grid = fig.add_gridspec(
        2, 11, width_ratios=[1.3, 0.3] + [1] * 4 + [0.3] + [1] * 4,
        hspace=0.30, wspace=0.06, left=0.03, right=0.99, top=0.83, bottom=0.06,
    )
    panel(fig.add_subplot(grid[0, 0]), c["image"], box=box, title="CT patch")
    consensus_panel(fig.add_subplot(grid[1, 0]), c["consensus"], box)

    for row, (name, arr) in enumerate((("baseline-short", base),
                                       ("modernized-short", modern))):
        colour = PALETTE["baseline"] if row == 0 else PALETTE["modernized"]
        for k in range(8):
            ax = fig.add_subplot(grid[row, 2 + k + (1 if k >= 4 else 0)])
            panel(ax, c["image"], arr[k], box=box, colour=colour,
                  title=f"{int(areas[name][k])}px")
            if k == 0:
                ax.set_ylabel(name, fontsize=7, color=colour)

    nonempty = {name: int((a > 0).sum()) for name, a in areas.items()}
    r0, r1, c0, c1 = box
    fig.suptitle(
        f"TEST — patch {int(c['patch_index'])}, bucket {int(c['bucket'])}   |   "
        f"GED {float(c['ged_baseline_short']):.3f} -> "
        f"{float(c['ged_modernized_short']):.3f}  (criterion "
        f"{float(c['criterion']):+.4f})   |   first 8 of 16 samples per arm, non-empty "
        + " vs ".join(f"{v}/16" for v in nonempty.values())
        + f"   |   crop {r0}:{r1}, {c0}:{c1}", fontsize=8.5)
    plt.show()
    return nonempty


def selection_case_figure(
    showcase: Mapping[str, Any], prefix: str, headline: str
) -> dict[str, int]:
    """One Set B case: all 16 candidates with each selection rule's pick outlined.

    Args:
        showcase: The loaded ``showcase.npz``.
        prefix: Case key prefix, ``"b0"``, ``"b1"`` or ``"b2"``.
        headline: Sentence appended to the figure title.

    Returns:
        The index each selection rule picked.

    Note:
        The export also carries ``<prefix>_arbitrary_unselected``, candidate index 0, which
        earlier versions drew as a contrast tile. It is no longer rendered: on case ``b2``
        that index is also the head's pick and the oracle, so a tile labelled as an
        unselected draw was in fact the best candidate in the set. The published random
        quantity is the mean over all 16, printed as the last row of the table below.
    """
    c = case_arrays(showcase, prefix)
    candidates, true_scores = c["candidates"], c["true_scores"]
    picks = {
        "head": int(c["pick_head"]),
        "area": int(c["pick_area"]),
        "oracle": int(c["pick_oracle"]),
    }
    box = crop_box([c["masks"], candidates], margin=10)
    areas = mask_areas(candidates)

    # Taller than it looks like it needs: the top row's tiles carry an xlabel naming the
    # rule that picked them, and it collides with the second row's titles otherwise.
    fig, axes = plt.subplots(2, 9, figsize=FIGSIZE["case"])
    panel(axes[0, 0], c["image"], box=box, title="CT patch")
    consensus_panel(axes[1, 0], c["consensus"], box)

    for i in range(len(candidates)):
        ax = axes[i // 8, 1 + i % 8]
        tags = [name for name, index in picks.items() if index == i]
        colour = (PALETTE["head"] if "head" in tags else
                  PALETTE["area"] if "area" in tags else
                  PALETTE["oracle"] if "oracle" in tags else PALETTE["other"])
        panel(ax, c["image"], candidates[i], box=box, colour=colour,
              title=f"#{i}   {true_scores[i]:.3f}")
        if tags:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.4)
                spine.set_color(colour)
            ax.set_xlabel(" / ".join(tags), fontsize=6.8, color=colour, labelpad=1)

    r0, r1, c0, c1 = box
    fig.suptitle(
        f"TEST — patch {int(c['patch_index'])}, bucket {int(c['bucket'])}. Tile titles are "
        f"TRUE soft-consensus Dice. {headline}   (crop {r0}:{r1}, {c0}:{c1})", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.2)
    plt.show()

    table(["rule", "picked", "true score", "candidate area"],
          [[name, f"#{index}", f"{true_scores[index]:.4f}", f"{int(areas[index])} px"]
           for name, index in picks.items()]
          + [["ceiling", "—", f"{float(c['ceiling']):.4f}", "—"],
             ["mean over all 16", "—", f"{float(c['mean_score']):.4f}",
              "the published 'random' column"]])
    return picks


# ---------------------------------------------------------------------------------
# Provenance printers
# ---------------------------------------------------------------------------------


def print_export_verification(manifest: Mapping[str, Any]) -> None:
    """Print one confirmation line per results file the export re-derived.

    ``scripts/export_showcase.py`` recomputes the published aggregate tables through the
    same code path that wrote them and records, per file, whether every figure matched
    exactly. This renders that record.

    Args:
        manifest: The showcase manifest.
    """
    for a in manifest["assertions"]:
        what = (f"{len(a['figures'])} figures x {len(a['buckets'])} buckets"
                if "figures" in a else
                "GED @ " + ", ".join(str(n) for n in a["sample_counts"])
                + f" ({', '.join(a['statistics'])})")
        name = Path(a["source"].replace("\\", "/")).name
        print(f"  {'MATCHED' if a['matched'] else 'FAILED':7s}  {name:46s}  {what}")
    print(f"  export {manifest['export_git_revision']}, "
          f"{manifest['generated_at'][:19]} UTC, no tolerance")


def set_a_guard_line(manifest: Mapping[str, Any]) -> str:
    """One line describing the Set A legibility guard and what it retained.

    Args:
        manifest: The showcase manifest.

    Returns:
        A single sentence, or a note if the export predates the guard.
    """
    guard = manifest.get("set_a_display_guard")
    if guard is None:
        return "This export predates the Set A legibility guard."
    per_bucket = manifest["guards"]["set_a"].get("per_bucket_eligible", {})
    retained = ", ".join(f"{k}:{v}" for k, v in per_bucket.items())
    return (f"Set A legibility guard (set after the first export was inspected): grader-union "
            f"footprint >= {guard['min_consensus_footprint_px']} px and "
            f">= {guard['min_nonempty_samples_per_arm']} non-empty samples per arm, applied "
            f"symmetrically to both arms. Eligible {guard['n_eligible_after']:,} of "
            f"{guard['n_total']:,} test patches "
            f"({100 * guard['n_eligible_after'] / guard['n_total']:.1f}%)"
            + (f"; per bucket {retained}." if retained else "."))


# ---------------------------------------------------------------------------------
# Figures built from tracked arrays
# ---------------------------------------------------------------------------------


def graders_figure(showcase: Mapping[str, Any], prefix: str) -> None:
    """One patch: the CT crop, its four grader masks, and the soft consensus.

    Args:
        showcase: The loaded ``showcase.npz``.
        prefix: Case key prefix, e.g. ``"b2"``.
    """
    c = case_arrays(showcase, prefix)
    box = crop_box([c["masks"]], margin=16)
    fig, axes = plt.subplots(1, 6, figsize=FIGSIZE["tile_row"])
    panel(axes[0], c["image"], box=box, title="CT patch (128x128)")
    for i in range(4):
        panel(axes[i + 1], c["image"], c["masks"][i], box=box, colour=PALETTE["grader"],
              title=f"grader {i + 1} — {int(c['masks'][i].sum())} px")
    bar = fig.colorbar(consensus_panel(axes[5], c["consensus"], box), ax=axes[5],
                       fraction=0.046, ticks=[0, .25, .5, .75, 1])
    bar.ax.tick_params(labelsize=6)
    fig.suptitle(f"TEST — patch {int(c['patch_index'])}, ambiguity bucket "
                 f"{int(c['bucket'])} of 4, achievable ceiling {float(c['ceiling']):.3f}",
                 fontsize=9)
    fig.tight_layout()
    plt.show()


def samples_figure(showcase: Mapping[str, Any], prefix: str, title: str,
                   key: str = "candidates", colour: str = PALETTE["head"]) -> None:
    """A 2x8 grid of one patch's prior samples, areas in the tile titles.

    Args:
        showcase: The loaded ``showcase.npz``.
        prefix: Case key prefix.
        title: Leading text of the figure title.
        key: Which stack to draw, un-prefixed.
        colour: Overlay colour.
    """
    c = case_arrays(showcase, prefix)
    stack = c[key]
    box = crop_box([c["masks"], stack], margin=10)
    areas = mask_areas(stack)
    fig, axes = plt.subplots(2, 8, figsize=FIGSIZE["tile_grid"])
    for i, ax in enumerate(axes.ravel()):
        panel(ax, c["image"], stack[i], box=box, colour=colour,
              title=f"#{i}   {int(areas[i])} px")
    fig.suptitle(
        f"{title}   |   patch {int(c['patch_index'])}, bucket {int(c['bucket'])}, "
        f"{int((areas > 0).sum())} of {len(areas)} samples non-empty   |   grader areas "
        f"{[int(m.sum()) for m in c['masks']]} px", fontsize=8.5)
    fig.tight_layout()
    plt.show()


def bucket_ged_figure(
    buckets: Sequence[str],
    base: Sequence[float],
    modern: Sequence[float],
    n_patches: Sequence[int],
    labels: Sequence[str],
    sample_count: int,
) -> list[float]:
    """Per-bucket GED for two arms, beside the percentage change per bucket.

    Args:
        buckets: Bucket names in order.
        base: First arm's GED per bucket.
        modern: Second arm's GED per bucket.
        n_patches: Patch count per bucket.
        labels: The two arm names.
        sample_count: Sample count the GED was measured at.

    Returns:
        The percentage change per bucket, for the caller's table.
    """
    pct = [100.0 * (y - x) / x for x, y in zip(base, modern)]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["chart_pair"])
    x = np.arange(len(buckets))

    axes[0].bar(x - 0.2, base, 0.4, label=labels[0], color=PALETTE["baseline"])
    axes[0].bar(x + 0.2, modern, 0.4, label=labels[1], color=PALETTE["modernized"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{k}\n(n={n:,})" for k, n in zip(buckets, n_patches)])
    axes[0].set_ylabel(f"GED^2 @{sample_count}")
    axes[0].set_title("TEST — GED per ambiguity bucket")
    axes[0].legend(fontsize=7.5)

    axes[1].bar(x, pct, 0.55,
                color=[PALETTE["modernized"] if p > 0 else PALETTE["oracle"] for p in pct])
    axes[1].axhline(0, color="k", lw=0.8)
    for xi, p in zip(x, pct):
        axes[1].annotate(f"{p:+.1f}%", (xi, p), ha="center",
                         va="bottom" if p > 0 else "top", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buckets)
    axes[1].set_ylabel("change in GED^2 (%), negative is better")
    axes[1].set_title("worse where existence is in doubt, better where graders agree")

    fig.supxlabel("non-empty grader masks", fontsize=8.5)
    fig.tight_layout()
    plt.show()
    return pct


def score_bars(labels: Sequence[str], values: Sequence[float], title: str) -> None:
    """One bar per selection rule, annotated with its score.

    Args:
        labels: Rule names; each must have an entry in :data:`PALETTE`.
        values: Soft-consensus Dice per rule.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE["chart"])
    bars = ax.bar(list(labels), list(values), color=[PALETTE[k] for k in labels], width=0.62)
    for rect, value in zip(bars, values):
        ax.annotate(f"{value:.4f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("soft-consensus Dice")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def denominator_figure(buckets: Sequence[str], e_tot: Sequence[float],
                       e_left: Sequence[float]) -> None:
    """The head's edge as a fraction of each of the two correct denominators.

    Args:
        buckets: Bucket names in order.
        e_tot: Edge as a percentage of total headroom.
        e_left: Edge as a percentage of the headroom the area control leaves.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE["chart"])
    x = np.arange(len(buckets))
    ax.plot(x, e_tot, "o-", color=PALETTE["area"], label="of total headroom (oracle - random)")
    ax.plot(x, e_left, "s-", color=PALETTE["head"], label="of headroom the area rule leaves")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("non-empty grader masks")
    ax.set_ylabel("head's edge (%)")
    ax.set_title("TEST — the two denominators trend in opposite directions")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    plt.show()


def score_agreement_figure(showcase: Mapping[str, Any]) -> float:
    """Predicted against true score for every candidate, and the outcome per rule.

    Args:
        showcase: The loaded ``showcase.npz``, carrying the Set C arrays.

    Returns:
        The Pearson correlation between predicted and true candidate scores.
    """
    pred, true = showcase["c_pred_scores"], showcase["c_true_scores"]
    per_candidate_bucket = np.repeat(showcase["c_buckets"], int(showcase["c_n_candidates"]))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["chart_pair"])
    for bucket in np.unique(showcase["c_buckets"]):
        keep = per_candidate_bucket == bucket
        axes[0].scatter(true[keep], pred[keep], s=2.5, alpha=0.18, label=f"bucket {bucket}")
    limits = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    axes[0].plot(limits, limits, "k--", lw=0.9, label="perfect prediction")
    axes[0].set_xlabel("true soft-consensus Dice")
    axes[0].set_ylabel("predicted score")
    axes[0].set_title(f"TEST — {len(pred):,} candidates over "
                      f"{len(showcase['c_buckets']):,} patches")
    legend = axes[0].legend(fontsize=7, markerscale=4)
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    axes[1].hist([showcase["c_random_scores"], showcase["c_area_scores"],
                  showcase["c_head_scores"], showcase["c_oracle_scores"]],
                 bins=28, histtype="step", lw=1.4,
                 color=[PALETTE[k] for k in ("random", "area", "head", "oracle")],
                 label=["random", "area", "head", "oracle"])
    axes[1].set_xlabel("soft-consensus Dice of the selected sample")
    axes[1].set_ylabel("patches")
    axes[1].set_title("TEST — per-patch outcome by rule")
    axes[1].legend(fontsize=7.5)

    fig.tight_layout()
    plt.show()
    return float(np.corrcoef(true, pred)[0, 1])


def composition_figure(labels: Sequence[str], first: Sequence[float],
                       second: Sequence[float], arms: Sequence[str], title: str) -> None:
    """The same four selection rules under two frozen bases, deltas annotated.

    Args:
        labels: Rule names for the x axis.
        first: Scores on the first base.
        second: Scores on the second base.
        arms: The two base names, for the legend.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE["chart"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, first, 0.4, label=arms[0], color=PALETTE["baseline"])
    ax.bar(x + 0.2, second, 0.4, label=arms[1], color=PALETTE["modernized"])
    for xi, (a, b) in enumerate(zip(first, second)):
        ax.annotate(f"{b - a:+.4f}", (xi, max(a, b)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("soft-consensus Dice")
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    plt.show()


def demo_loss_figure(history: Sequence[Mapping[str, Any]]) -> None:
    """The demo run's ELBO curve on a log axis, so the first drop and the tail both read.

    Args:
        history: The per-epoch records from the demo run's ``summary.json``.
    """
    epochs = [h["epoch"] + 1 for h in history]
    fig, ax = plt.subplots(figsize=FIGSIZE["chart"])
    ax.plot(epochs, [h["train/total"] for h in history], "o-", color=PALETTE["baseline"],
            label="train")
    ax.plot(epochs, [h.get("val/total") for h in history], "s--", color=PALETTE["area"],
            label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO total (log scale)")
    ax.set_title("demo run — not a reported result")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    plt.show()


def demo_samples_figure(image: np.ndarray, masks: np.ndarray, samples: np.ndarray,
                        title: str) -> None:
    """The demo model's prior samples beside the four graders of one patch.

    Args:
        image: ``(H, W)`` CT patch.
        masks: ``(4, H, W)`` grader masks.
        samples: ``(n, H, W)`` prior samples from the demo model.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 1 + len(masks) + len(samples), figsize=FIGSIZE["demo_row"])
    panel(axes[0], image, title="CT patch")
    for i, mask in enumerate(masks):
        panel(axes[1 + i], image, mask, colour=PALETTE["grader"], title=f"grader {i + 1}")
    for i, sample in enumerate(samples):
        panel(axes[1 + len(masks) + i], image, sample, colour=PALETTE["head"],
              title=f"sample {i + 1}")
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    plt.show()
