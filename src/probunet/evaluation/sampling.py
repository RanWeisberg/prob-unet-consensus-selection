"""Drawing prior samples and turning them into per-patch metrics.

Sampling draws ``max(sample_counts)`` masks per image **once** and evaluates every
requested sample count as a **prefix** of that single set. Two reasons: it costs one pass
instead of four, and it makes "GED decreases as more samples are drawn" a comparison
within one set of draws rather than across independent ones -- which is the qualitative
claim the paper's Figure 4a supports and the one worth checking.

Everything here runs on whatever device :func:`probunet.utils.runtime.select_device`
picks. Nothing is CUDA-specific.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from probunet.evaluation.metrics import (
    emptiest_sample_index,
    generalized_energy_distance,
    hungarian_matched_iou,
    oracle_dice,
    per_grader_oracle_dice,
    random_sample_dice,
    selected_sample_dice,
    summarize,
)
from probunet.model.prob_unet import ProbUNet
from probunet.training.diagnostics import logits_to_mask, reparameterize
from probunet.variants import SegmentationVariant

LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLE_COUNTS: tuple[int, ...] = (1, 4, 8, 16)
"""Sample counts reported by the paper and the follow-up literature."""

DEFAULT_EVAL_SEED = 2018


@dataclass(frozen=True)
class SamplingConfig:
    """How to draw and score samples.

    Attributes:
        sample_counts: Sample counts to report, evaluated as prefixes of one draw.
        seed: Seed for the sampling noise, so a re-run reproduces exactly.
        aggregate: How to reduce a sample's per-grader scores; see
            :data:`probunet.evaluation.metrics.AGGREGATIONS`.

    Raises:
        ValueError: If the counts are empty, non-positive or unsorted.
    """

    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS
    seed: int = DEFAULT_EVAL_SEED
    aggregate: str = "mean"

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not self.sample_counts:
            raise ValueError("sample_counts must not be empty")
        if any(count <= 0 for count in self.sample_counts):
            raise ValueError(f"sample_counts must be positive, got {self.sample_counts}")
        if list(self.sample_counts) != sorted(set(self.sample_counts)):
            raise ValueError(
                f"sample_counts must be strictly increasing, got {self.sample_counts}"
            )

    @property
    def max_samples(self) -> int:
        """The largest requested sample count."""
        return max(self.sample_counts)


@torch.no_grad()
def draw_prior_samples(
    model: ProbUNet,
    image: Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ``n_samples`` hard masks per image from the prior.

    The U-Net runs once; each sample re-runs only ``f_comb``.

    Args:
        model: The trained model.
        image: Image batch of shape ``(B, 1, H, W)``.
        n_samples: Samples per image.
        generator: CPU generator supplying the noise, for reproducibility.

    Returns:
        A uint8 mask tensor of shape ``(B, n_samples, H, W)``.
    """
    encoded = model.encode(image)
    return torch.stack(
        [
            logits_to_mask(model.reconstruct(encoded, reparameterize(encoded.prior, generator)))
            for _ in range(n_samples)
        ],
        dim=1,
    )


def _grader_areas(graders: Tensor) -> Tensor:
    """Median foreground area of a patch's non-empty grader masks.

    Args:
        graders: Grader masks of shape ``(B, m, H, W)``.

    Returns:
        Per-patch median area in pixels, shape ``(B,)``. NaN where every mask is empty.
    """
    areas = (graders != 0).flatten(start_dim=2).sum(dim=2).to(torch.float32)
    medians = []
    for row in areas:
        nonempty = row[row > 0]
        medians.append(
            torch.median(nonempty) if nonempty.numel() else torch.tensor(float("nan"))
        )
    return torch.stack(medians)


@torch.no_grad()
def collect_per_patch_metrics(
    variant: SegmentationVariant,
    loader: DataLoader,
    config: SamplingConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Evaluate every metric for every patch in a loader.

    Takes a :class:`~probunet.variants.SegmentationVariant`, not a model, so the same
    code serves the baseline, the modernized variant and the extension. There is exactly
    one variant-dependent branch: if ``variant.select`` returns indices, the metrics of
    the selected sample are recorded as well.

    The all-empty predictor is computed once rather than per sample count, because it is
    genuinely independent of ``n``: every self-distance is 0 and every cross-distance
    repeats, so all three GED components are unchanged by drawing more empty masks.

    Args:
        variant: The variant to evaluate, already on ``device``.
        loader: A deterministic, non-shuffled evaluation loader.
        config: Sampling configuration.
        device: Device to run on.

    Returns:
        A dict of flat per-patch arrays. Keys are ``index``, ``nonempty_count``,
        ``lesion_area_median``, the all-empty baseline (``empty_*``) and, for each
        requested count ``n``, entries suffixed ``@n``. When the variant selects,
        ``selected_dice@n`` and ``selected_ged@n`` appear too.
    """
    # A raw ProbUNet has a sample() too, but its signature is sample(encoded, n) --
    # passing one here would fail deep inside with an unhelpful message.
    if not (hasattr(variant, "sample") and hasattr(variant, "select")):
        raise TypeError(
            f"expected a SegmentationVariant with sample() and select(), got "
            f"{type(variant).__name__}. Wrap a model: ProbUNetVariant(model)."
        )
    if isinstance(getattr(variant, "model", None), torch.nn.Module):
        variant.model.eval()
    columns: dict[str, list[np.ndarray]] = {}

    def append(key: str, values: Tensor) -> None:
        columns.setdefault(key, []).append(values.detach().cpu().numpy())

    for batch in loader:
        image = batch["image"].to(device)
        graders = batch["masks"].to(device)
        append("index", batch["index"])
        append("nonempty_count", (graders != 0).flatten(start_dim=2).any(dim=2).sum(dim=1))
        append("lesion_area_median", _grader_areas(graders))

        samples = variant.sample(image, config.max_samples)

        for count in config.sample_counts:
            subset = samples[:, :count]
            ged = generalized_energy_distance(subset, graders)
            append(f"ged@{count}", ged["d_squared"])
            append(f"ged_ys@{count}", ged["d_ys"])
            append(f"ged_ss@{count}", ged["d_ss"])
            append(f"ged_yy@{count}", ged["d_yy"])
            append(f"oracle_dice@{count}", oracle_dice(subset, graders, config.aggregate))
            append(f"oracle_dice_per_grader@{count}", per_grader_oracle_dice(subset, graders))
            append(
                f"random_sample_dice@{count}",
                random_sample_dice(subset, graders, config.aggregate),
            )
            append(f"hungarian_iou@{count}", hungarian_matched_iou(subset, graders))
            append(
                f"emptiest_sample_dice@{count}",
                selected_sample_dice(
                    subset, graders, emptiest_sample_index(subset), config.aggregate
                ),
            )

            # The one variant-dependent branch in the whole evaluation path.
            chosen = variant.select(subset, image)
            if chosen is not None:
                append(
                    f"selected_dice@{count}",
                    selected_sample_dice(subset, graders, chosen, config.aggregate),
                )
                picked = subset.gather(
                    1,
                    chosen.to(torch.int64)
                    .view(-1, 1, 1, 1)
                    .expand(-1, 1, subset.shape[-2], subset.shape[-1]),
                )
                append(
                    f"selected_ged@{count}",
                    generalized_energy_distance(picked, graders)["d_squared"],
                )

        # Degenerate all-empty predictor, n-independent (see the docstring).
        empty = torch.zeros_like(graders[:, :1])
        empty_ged = generalized_energy_distance(empty, graders)
        append("empty_ged", empty_ged["d_squared"])
        append("empty_dice", random_sample_dice(empty, graders, config.aggregate))
        append("empty_oracle_dice", oracle_dice(empty, graders, config.aggregate))
        append("empty_hungarian_iou", hungarian_matched_iou(empty, graders))

    return {key: np.concatenate(parts) for key, parts in columns.items()}


def build_report(
    per_patch: dict[str, np.ndarray], config: SamplingConfig, buckets: tuple[int, ...] = (1, 2, 3, 4)
) -> dict[str, object]:
    """Summarize per-patch metrics aggregate and per ambiguity bucket.

    Aggregate numbers are dominated by the lesion-presence question, which the
    single-grader patches make easy to score well on for the wrong reasons. Per-bucket
    reporting is what exposes the shape-agreement question the extension is about.

    Args:
        per_patch: Output of :func:`collect_per_patch_metrics`.
        config: Sampling configuration.
        buckets: Non-empty-grader counts to report separately.

    Returns:
        A nested, JSON-serializable report.
    """
    counts = per_patch["nonempty_count"]
    metric_keys = [
        key
        for key in per_patch
        if key not in ("index", "nonempty_count", "lesion_area_median")
    ]

    def block(selector: np.ndarray) -> dict[str, object]:
        """Summarize every metric over a subset of patches."""
        summary: dict[str, object] = {
            "n_patches": int(selector.sum()),
            "lesion_area_median_px": (
                float(np.nanmedian(per_patch["lesion_area_median"][selector]))
                if selector.any()
                else None
            ),
        }
        for key in metric_keys:
            summary[key] = summarize(per_patch[key][selector])
        return summary

    everything = np.ones_like(counts, dtype=bool)
    report: dict[str, object] = {
        "sample_counts": list(config.sample_counts),
        "seed": config.seed,
        "aggregate": config.aggregate,
        "aggregate_over_all_patches": block(everything),
        "per_bucket": {
            str(bucket): block(counts == bucket)
            for bucket in buckets
            if (counts == bucket).any()
        },
    }
    return report
