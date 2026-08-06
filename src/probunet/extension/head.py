"""The consensus-selection head and its frozen base, wired together.

**Stage 3 scope: structure and the freeze contract, not the objective.** This module owns
the head architecture and — more importantly — the *containment* that makes "the base is
frozen" true rather than merely intended. The scoring target, the regression objective and
the candidate-sampling loop are Stage 4 and are deliberately absent.

Two structural decisions carry the freeze guarantee, and both are here rather than in the
training loop, because a guarantee that lives in a loop is one refactor away from being
lost:

* **The base is a submodule, and the optimizer never sees it.** :meth:`SelectionHead.
  head_parameters` returns the scorer's parameters only. Handing ``model.parameters()`` to
  an optimizer is the failure mode that would silently invalidate every Phase 3 number —
  the base would drift, GED would move with it, and "distribution metrics unchanged"
  would be false while every log line still looked healthy.
* **``train()`` cannot thaw the base.** ``nn.Module.train()`` recurses into children, so
  the training loop's ordinary ``model.train()`` would put the frozen base back into
  training mode. :meth:`SelectionHead.train` overrides that and forces the base to
  ``eval()`` on every call. This is the standard way this breaks silently.

The head scores a *candidate mask in context*: it consumes the frozen U-Net's feature map
concatenated with the binarized candidate, and convolves over the pair **before** pooling,
so it can compute spatial agreement rather than a size prior. It is deliberately **not**
given mask summary statistics such as area — pooling recovers area implicitly, and handing
it over explicitly would make the size-prior shortcut easier to learn than the thing we
want. A mask-area-only regressor is kept as a diagnostic baseline instead (Stage 4).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from probunet.model.prob_unet import ProbUNet
from probunet.training.diagnostics import logits_to_mask, reparameterize
from probunet.training.freeze import assert_frozen, freeze_module

DEFAULT_SCORER_CHANNELS: tuple[int, ...] = (32, 64, 128)
"""Width of each strided convolution in the scorer tower.

At 128x128 input the three stride-2 convolutions take the map to 64, 32 and 16, after
which a global average pool leaves a 128-vector. Recorded as a constant rather than buried
in the constructor because CLAUDE.md forbids magic numbers in code.
"""


class MaskScorer(nn.Module):
    """Scores one candidate mask against a frozen feature map.

    Input is the feature map concatenated with the binarized candidate along the channel
    axis; output is one unbounded score per item.

    **Linear output, deliberately.** Not a sigmoid: the soft-consensus targets cluster near
    zero, which is exactly where the inverse sigmoid is steepest, so a sigmoid would drive
    pre-activations far negative and saturate. The score is only ever argmax'd *within* an
    image, so values outside ``[0, 1]`` are cosmetic — clamp for display, never before the
    loss.
    """

    def __init__(
        self,
        feature_channels: int,
        mask_channels: int = 1,
        channels: tuple[int, ...] = DEFAULT_SCORER_CHANNELS,
    ) -> None:
        """Build the scorer tower.

        Args:
            feature_channels: Channels in the frozen U-Net's output feature map.
            mask_channels: Channels used to carry the candidate mask.
            channels: Width of each stride-2 convolution.

        Raises:
            ValueError: If ``channels`` is empty.
        """
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty")

        layers: list[nn.Module] = []
        previous = feature_channels + mask_channels
        for width in channels:
            # Stride 2 with a 3x3 kernel: local spatial agreement is computed here, before
            # any pooling. Pooling early would leave only "how big is the mask".
            layers.append(nn.Conv2d(previous, width, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            previous = width
        self.tower = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # A HIDDEN LAYER, and the reason is structural rather than a guess about capacity.
        #
        # The target is a RATIO: soft Dice is 2*sum(s*c) / (sum(s) + sum(c)). Global
        # average pooling hands the readout spatial *averages* -- roughly an overlap term,
        # a candidate-area term and a consensus-area term -- and a single linear layer can
        # only form a weighted SUM of those, `a*overlap + b*area_s + c*area_c`. It can
        # never form their quotient. One ReLU hidden layer can approximate the division.
        #
        # This also bears directly on the size-prior shortcut (FINDINGS 4.4, the
        # pre-registered mean-centering fallback). Achievable scores are strongly
        # bucket-dependent -- ceilings run 0.40 to 0.89 -- so normalizing by the areas is
        # exactly what the task demands. A head that structurally *cannot* divide has no
        # way to express that normalization, and the nearest thing it can express is an
        # area prior: precisely the shortcut we are trying to avoid rewarding.
        self.project = nn.Sequential(
            nn.Linear(previous, previous),
            nn.ReLU(inplace=True),
            nn.Linear(previous, 1),
        )

    def forward(self, features: Tensor, candidate: Tensor) -> Tensor:
        """Score each candidate against its feature map.

        Args:
            features: Frozen U-Net features of shape ``(B, C, H, W)``.
            candidate: Binary candidate masks of shape ``(B, H, W)`` or
                ``(B, mask_channels, H, W)``.

        Returns:
            Unbounded scores of shape ``(B,)``.

        Raises:
            ValueError: If the shapes are incompatible.
        """
        if candidate.dim() == features.dim() - 1:
            candidate = candidate.unsqueeze(1)
        if candidate.dim() != features.dim():
            raise ValueError(
                f"candidate has {candidate.dim()} dims, expected {features.dim()}"
            )
        if candidate.shape[0] != features.shape[0]:
            raise ValueError(
                f"batch mismatch: features {features.shape[0]} vs candidate "
                f"{candidate.shape[0]}"
            )
        if candidate.shape[-2:] != features.shape[-2:]:
            raise ValueError(
                f"spatial mismatch: features {tuple(features.shape[-2:])} vs candidate "
                f"{tuple(candidate.shape[-2:])}"
            )
        stacked = torch.cat([features, candidate.to(features.dtype)], dim=1)
        pooled = self.pool(self.tower(stacked)).flatten(start_dim=1)
        return self.project(pooled).squeeze(-1)


class SelectionHead(nn.Module):
    """A trainable scorer on top of a **frozen** Probabilistic U-Net.

    The base is frozen in the constructor, and both structural guarantees described in the
    module docstring are enforced here: the optimizer is fed
    :meth:`head_parameters`, and :meth:`train` keeps the base in ``eval()``.
    """

    def __init__(
        self,
        base: ProbUNet,
        mask_channels: int = 1,
        channels: tuple[int, ...] = DEFAULT_SCORER_CHANNELS,
    ) -> None:
        """Freeze the base and build the scorer.

        Args:
            base: The trained Probabilistic U-Net. **Frozen here**, not by the caller, so
                that constructing a head and forgetting to freeze is not possible.
            mask_channels: Channels used to carry the candidate mask.
            channels: Width of each stride-2 convolution in the scorer.
        """
        super().__init__()
        self.base = base
        self.freeze_record = freeze_module(self.base, name="base Probabilistic U-Net")
        self.scorer = MaskScorer(
            feature_channels=base.unet.out_channels,
            mask_channels=mask_channels,
            channels=channels,
        )

    def train(self, mode: bool = True) -> SelectionHead:
        """Set training mode for the scorer while pinning the base to ``eval()``.

        ``nn.Module.train()`` recurses into children, so without this override the
        training loop's ordinary ``model.train()`` would silently thaw the base's mode.
        Nothing in this architecture has dropout or batch norm today, so the immediate
        numerical effect would be nil — which is precisely why it would go unnoticed until
        something mode-dependent was added and every Phase 3 number quietly changed.

        Args:
            mode: Whether the scorer is in training mode.

        Returns:
            self, matching ``nn.Module.train``.
        """
        super().train(mode)
        self.base.eval()
        return self

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """The **only** parameters an optimizer may be given.

        Yields:
            The scorer's parameters. The base's are deliberately absent: passing
            ``self.parameters()`` to an optimizer is the failure mode that would let the
            base drift and invalidate "distribution metrics unchanged".
        """
        return self.scorer.parameters()

    def assert_base_frozen(self) -> None:
        """Re-check the freeze contract at any point in training.

        Called after the last optimizer step of every epoch, not only before the first:
        an optimizer built over the wrong parameter set does its damage *during* the
        epoch, so a check that only ran at construction would pass on a broken run.

        Raises:
            RuntimeError: If the base has any trainable parameter or is in training mode.
        """
        assert_frozen(self.base, name="base Probabilistic U-Net")

    def forward(self, features: Tensor, candidate: Tensor) -> Tensor:
        """Score candidates against cached frozen features.

        Takes features rather than an image on purpose: Stage 4 scores many candidates per
        image, and the base's U-Net must run **once** per image regardless of how many
        candidates are scored.

        Args:
            features: Frozen U-Net features of shape ``(B, C, H, W)``.
            candidate: Binary candidate masks of shape ``(B, H, W)``.

        Returns:
            Unbounded scores of shape ``(B,)``.
        """
        return self.scorer(features, candidate)

    @torch.no_grad()
    def encode_base(self, image: Tensor) -> Tensor:
        """Run the frozen base's U-Net once and return its features.

        Under ``no_grad`` and on a frozen module, so nothing here can contribute a
        gradient to the base even if a caller forgets.

        Args:
            image: Image batch of shape ``(B, C, H, W)``.

        Returns:
            Feature map of shape ``(B, base_channels, H, W)``.
        """
        return self.base.encode(image).features

    @torch.no_grad()
    def sample_candidates(
        self, image: Tensor, n_samples: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draw ``n_samples`` **prior** candidates per image, plus the cached features.

        **Prior, never posterior.** Posterior samples have seen the ground-truth mask and
        are almost always good, so a head trained only on those never encounters a bad
        candidate and learns to emit a constant high score -- the failure CLAUDE.md names
        explicitly. Nothing in this method can reach the posterior: the base is encoded
        without a mask, so no posterior exists to sample from.

        The U-Net runs **once**; each candidate re-runs only ``f_comb``. Binarization goes
        through :func:`~probunet.training.diagnostics.logits_to_mask` -- argmax over the
        class axis -- which is the same convention Phase 1 evaluation already uses. The
        head consumes exactly the artifact that gets scored and delivered.

        Args:
            image: Image batch of shape ``(B, C, H, W)``.
            n_samples: Candidates per image.
            generator: Optional CPU generator. Supplied at validation so the candidate set
                is identical across epochs and arms; omitted during training so candidates
                are freshly resampled every epoch.

        Returns:
            ``(features, candidates)`` with shapes ``(B, C_f, H, W)`` and
            ``(B, n_samples, H, W)``, the candidates uint8.
        """
        encoded = self.base.encode(image)
        candidates = torch.stack(
            [
                logits_to_mask(
                    self.base.reconstruct(encoded, reparameterize(encoded.prior, generator))
                )
                for _ in range(n_samples)
            ],
            dim=1,
        )
        return encoded.features, candidates

    def score_candidates(self, features: Tensor, candidates: Tensor) -> Tensor:
        """Score every candidate against its image.

        Each candidate is scored **independently**, which is what lets the training and
        evaluation candidate counts differ.

        Args:
            features: Frozen features of shape ``(B, C, H, W)``.
            candidates: Masks of shape ``(B, n, H, W)``.

        Returns:
            Unbounded scores of shape ``(B, n)``.
        """
        scores = [
            self.scorer(features, candidates[:, index])
            for index in range(candidates.shape[1])
        ]
        return torch.stack(scores, dim=1)

    def select(self, features: Tensor, candidates: Tensor) -> Tensor:
        """Choose one candidate per image, **without ground truth**.

        Args:
            features: Frozen features of shape ``(B, C, H, W)``.
            candidates: Masks of shape ``(B, n, H, W)``.

        Returns:
            Chosen indices of shape ``(B,)``.
        """
        return self.score_candidates(features, candidates).argmax(dim=1)

    def parameter_counts(self) -> dict[str, int]:
        """Parameter count for the frozen base and the trainable scorer.

        Returns:
            Counts under ``base``, ``scorer`` and ``total``. The scorer's count is the one
            to quote for the head; the base's is not the head's capacity.
        """
        base = sum(p.numel() for p in self.base.parameters())
        scorer = sum(p.numel() for p in self.scorer.parameters())
        return {"base": base, "scorer": scorer, "total": base + scorer}