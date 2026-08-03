"""Latent-space diagnostics: what the loss curve cannot show you.

Under the paper's reduction convention the KL term is about 0.0006% of the objective at
initialization, so the near-term risk is **not** posterior collapse but the opposite: an
unconstrained latent whose *prior* never learns to cover the grader variants. Training
loss falls happily while that happens, because training draws ``z`` from the posterior.
The failure only appears at inference, when ``z`` comes from the prior.

Read the four signatures together, never one alone:

=====================================  ==================================================
observation                            interpretation
=====================================  ==================================================
``prior_posterior_ce_ratio`` >> 1      Prior is not covering the variants. Posterior-z
                                       reconstructs well, prior-z does not.
``sample_diversity_iou`` -> 1.0 AND    Prior has genuinely **collapsed** to a point.
``nonempty_sample_fraction`` > 0
``sample_diversity_iou`` -> 1.0 AND    Model has not learned foreground **yet**. All
``nonempty_sample_fraction`` ~ 0       samples are empty, and two empty masks have IoU
                                       1.0 by the convention that makes GED reward
                                       agreement on lesion absence. Expected early:
                                       the background-to-foreground ratio is 176:1.
                                       **Not** a collapse alarm.
``prior_sigma_mean`` -> 0              Collapse again, seen from the sigma side.
``kl`` -> 0 while ``ce`` falls         Posterior collapse: the opposite failure.
=====================================  ==================================================

That third row is why ``nonempty_sample_fraction`` exists. Diversity alone cannot
distinguish "hasn't learned anything yet" from "collapsed", because both read 1.0.

The fixed diagnostic sets are **stratified over the ambiguity buckets** rather than
taken as the first N by index, so the panel always shows single-grader cases -- 33% of
the data and the hard case for the consensus-selection extension. The chosen indices are
recorded to disk so panels are identical across runs and comparable between phases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal, kl_divergence

from probunet.data.lidc import LidcDataset
from probunet.evaluation.metrics import binary_iou
from probunet.model.prob_unet import Encoded, ProbUNet

LOGGER = logging.getLogger(__name__)

N_GRADERS = 4
PANEL_PAD = 2
PANEL_SEPARATOR_VALUE = 0.35


@dataclass(frozen=True)
class DiagnosticSets:
    """Fixed image sets used by the diagnostics.

    Attributes:
        diversity: Global patch indices for the diversity measure.
        panel: Global patch indices shown in the qualitative panel.
        buckets: Ambiguity bucket of each panel index, for labelling.
    """

    diversity: np.ndarray
    panel: np.ndarray
    buckets: dict[str, list[int]]

    def to_dict(self) -> dict[str, object]:
        """Render as a JSON-serializable mapping."""
        return {
            "diversity": [int(i) for i in self.diversity],
            "panel": [int(i) for i in self.panel],
            "panel_buckets": self.buckets,
        }


def stratified_indices(
    dataset: LidcDataset, count: int, seed: int
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Pick ``count`` patch indices spread evenly over the ambiguity buckets.

    Args:
        dataset: The split to draw from.
        count: How many indices to pick.
        seed: Seed making the choice deterministic.

    Returns:
        A ``(indices, per_bucket)`` pair: the chosen global patch indices, sorted, and
        a mapping from bucket label to the indices chosen from it.

    Raises:
        ValueError: If the dataset has no patches.
    """
    if len(dataset) == 0:
        raise ValueError("cannot draw diagnostic indices from an empty dataset")

    buckets = {
        bucket: members
        for bucket, members in dataset.buckets().items()
        if members.size > 0
    }
    rng = np.random.default_rng(seed)
    ordered = sorted(buckets)
    # Spread requested slots over the populated buckets, remainder to the first ones.
    base, remainder = divmod(count, len(ordered))
    chosen: dict[str, list[int]] = {}
    picked: list[int] = []
    for position, bucket in enumerate(ordered):
        want = base + (1 if position < remainder else 0)
        members = np.sort(buckets[bucket])
        take = min(want, members.size)
        if take == 0:
            continue
        selection = rng.choice(members, size=take, replace=False)
        chosen[str(bucket)] = sorted(int(i) for i in selection)
        picked.extend(int(i) for i in selection)

    # If a small bucket could not fill its slots, top up from whatever is left.
    if len(picked) < count:
        remaining = np.setdiff1d(np.sort(dataset.indices), np.array(picked, dtype=np.int64))
        if remaining.size:
            extra = rng.choice(
                remaining, size=min(count - len(picked), remaining.size), replace=False
            )
            picked.extend(int(i) for i in extra)
    return np.array(sorted(picked), dtype=np.int64), chosen


def build_diagnostic_sets(
    dataset: LidcDataset, diversity_images: int, panel_images: int, seed: int
) -> DiagnosticSets:
    """Choose the fixed diversity and panel sets, both ambiguity-stratified.

    The panel is drawn as a subset of the diversity set where possible, so the two
    views describe the same images.

    Args:
        dataset: The validation split.
        diversity_images: Size of the diversity set.
        panel_images: Size of the panel.
        seed: Seed making the choice deterministic.

    Returns:
        The chosen :class:`DiagnosticSets`.
    """
    diversity, _ = stratified_indices(dataset, diversity_images, seed)
    panel, panel_buckets = stratified_indices(dataset, panel_images, seed + 1)
    return DiagnosticSets(diversity=diversity, panel=panel, buckets=panel_buckets)


def save_diagnostic_sets(sets: DiagnosticSets, path: Path) -> None:
    """Record the chosen indices so panels are comparable across runs.

    Args:
        sets: The chosen sets.
        path: Destination JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sets.to_dict(), indent=2) + "\n")


def reparameterize(
    distribution: Independent, generator: torch.Generator | None = None
) -> Tensor:
    """Draw a latent sample with an optionally seeded generator.

    ``Distribution.rsample`` accepts no generator, so the reparameterization is written
    out. The diagnostics need a fixed noise sequence: with only 8 to 64 images, sampling
    noise would otherwise make panels incomparable between epochs.

    Args:
        distribution: An ``Independent(Normal(...), 1)``.
        generator: Optional CPU generator supplying the noise.

    Returns:
        A sample of shape ``(batch, latent_dim)``.
    """
    base: Normal = distribution.base_dist  # type: ignore[assignment]
    if generator is None:
        return distribution.rsample()
    noise = torch.randn(
        base.loc.shape, generator=generator, dtype=base.loc.dtype
    ).to(base.loc.device)
    return base.loc + base.scale * noise


def sigma_stats(stats: tuple[Tensor, Tensor], prefix: str) -> dict[str, float]:
    """Summarize a latent distribution's predicted standard deviations.

    Args:
        stats: The ``(mu, logvar)`` pair the encoder produced.
        prefix: Metric name prefix, e.g. ``"prior"``.

    Returns:
        Mean and standard deviation of sigma, and the mean absolute mu.
    """
    mu, logvar = stats
    sigma = torch.exp(0.5 * logvar)
    return {
        f"{prefix}_sigma_mean": float(sigma.mean()),
        f"{prefix}_sigma_std": float(sigma.std()),
        f"{prefix}_sigma_min": float(sigma.min()),
        f"{prefix}_mu_abs_mean": float(mu.abs().mean()),
    }


def per_dim_kl(posterior: Independent, prior: Independent) -> Tensor:
    """KL per latent dimension, averaged over the batch.

    The wrapped ``Independent`` sums over latent dims, so the base distributions are
    used here to see which dimensions carry information. A dimension whose KL sits at
    zero is inactive.

    Args:
        posterior: ``Q(z | X, Y)``.
        prior: ``P(z | X)``.

    Returns:
        A tensor of shape ``(latent_dim,)``.
    """
    per_element = kl_divergence(posterior.base_dist, prior.base_dist)
    return per_element.mean(dim=0)


def mean_pairwise_iou(samples: Tensor) -> Tensor:
    """Mean pairwise IoU among sampled masks, averaged over images.

    Args:
        samples: Boolean or integer masks of shape ``(B, S, H, W)`` with ``S >= 2``.

    Returns:
        A scalar tensor in [0, 1]. Values near 1 mean the samples agree with each
        other -- which is either collapse or an all-empty model; read it together with
        :func:`nonempty_sample_fraction`.

    Raises:
        ValueError: If fewer than two samples are supplied.
    """
    if samples.dim() != 4:
        raise ValueError(f"expected (B, S, H, W), got {tuple(samples.shape)}")
    n_samples = samples.shape[1]
    if n_samples < 2:
        raise ValueError(f"need at least 2 samples to form a pair, got {n_samples}")
    scores = [
        binary_iou(samples[:, i], samples[:, j])
        for i in range(n_samples)
        for j in range(i + 1, n_samples)
    ]
    return torch.stack(scores, dim=0).mean()


def nonempty_sample_fraction(samples: Tensor) -> Tensor:
    """Fraction of sampled masks that contain at least one foreground pixel.

    Disambiguates a diversity score of 1.0: near zero means the model has not learned
    foreground yet, above zero means the prior really has collapsed.

    Args:
        samples: Masks of shape ``(B, S, H, W)``.

    Returns:
        A scalar tensor in [0, 1].
    """
    flat = samples.reshape(samples.shape[0], samples.shape[1], -1)
    return (flat != 0).any(dim=2).to(torch.float32).mean()


def make_panel(
    images: Tensor, grader_masks: Tensor, samples: Tensor
) -> Tensor:
    """Tile images, grader masks and prior samples into one greyscale grid.

    One row per image: the image, then its four grader masks, then the prior samples.
    Tiled by hand rather than with ``torchvision.utils.make_grid`` to avoid adding a
    dependency for fifteen lines of indexing.

    Args:
        images: Images of shape ``(B, 1, H, W)``, values in [0, 1].
        grader_masks: Grader masks of shape ``(B, 4, H, W)``.
        samples: Prior sample masks of shape ``(B, S, H, W)``.

    Returns:
        A tensor of shape ``(1, rows, cols)`` suitable for ``add_image``.
    """
    batch, _, height, width = images.shape
    columns = 1 + grader_masks.shape[1] + samples.shape[1]
    cell_h, cell_w = height + PANEL_PAD, width + PANEL_PAD
    panel = torch.full(
        (1, batch * cell_h, columns * cell_w),
        PANEL_SEPARATOR_VALUE,
        dtype=torch.float32,
    )

    def place(row: int, column: int, tile: Tensor) -> None:
        top = row * cell_h + PANEL_PAD // 2
        left = column * cell_w + PANEL_PAD // 2
        panel[0, top : top + height, left : left + width] = tile

    for row in range(batch):
        place(row, 0, images[row, 0].detach().to(torch.float32).cpu())
        for grader in range(grader_masks.shape[1]):
            place(row, 1 + grader, grader_masks[row, grader].detach().to(torch.float32).cpu())
        for sample in range(samples.shape[1]):
            place(
                row,
                1 + grader_masks.shape[1] + sample,
                samples[row, sample].detach().to(torch.float32).cpu(),
            )
    return panel.clamp(0.0, 1.0)


def logits_to_mask(logits: Tensor) -> Tensor:
    """Convert class logits to a binary foreground mask.

    Args:
        logits: Logits of shape ``(..., C, H, W)``.

    Returns:
        A uint8 mask of shape ``(..., H, W)``, 1 where the foreground class wins.
    """
    return logits.argmax(dim=-3).to(torch.uint8)


def prior_samples_for_images(
    model: ProbUNet,
    encoded: Encoded,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw prior samples as hard masks, re-running only ``f_comb``.

    Args:
        model: The model.
        encoded: State from ``model.encode``, whose U-Net pass is reused.
        n_samples: Samples per image.
        generator: Optional generator for reproducible noise.

    Returns:
        A uint8 mask tensor of shape ``(B, n_samples, H, W)``.
    """
    masks = []
    for _ in range(n_samples):
        z = reparameterize(encoded.prior, generator)
        masks.append(logits_to_mask(model.reconstruct(encoded, z)))
    return torch.stack(masks, dim=1)
