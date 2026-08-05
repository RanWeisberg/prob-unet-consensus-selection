"""Prior and posterior latent encoders.

Both predict a Gaussian over ``R^N`` -- axis-aligned by default, or with a full
covariance when ``model.latent_covariance: full`` (Phase 2). Per Appendix H.1 they share the
U-Net encoder's architecture exactly -- same number of scales, same three 3x3 convs
per scale, same average pooling -- with their own weights. The final feature map is
globally average pooled and a single 1x1 convolution predicts ``2N`` channels: the
mean and the **log-variance**.

The asymmetry between the two is the whole point of the model and the classic place
to get it wrong:

* :class:`PriorNet` sees the **image only**.
* :class:`PosteriorNet` sees the image **concatenated with a ground-truth mask**.

Log-variance is parameterized rather than sigma, so ``scale = exp(0.5 * logvar)``.
This keeps the scale positive without a clamp and keeps gradients well behaved.

The predicted parameters travel as a :class:`LatentStats` rather than a bare tuple, so
that adding a covariance factor later is a new *field* rather than a change of arity at
every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.distributions import Independent, MultivariateNormal, Normal

from probunet.model.unet import ConvBlock, channel_widths

LatentDistribution = Independent | MultivariateNormal
"""What a latent encoder returns.

``Independent(Normal, 1)`` for the paper's diagonal parameterization,
``MultivariateNormal(loc, scale_tril=L)`` for the Phase 2 full-covariance one. Both have
batch shape ``(B,)`` and event shape ``(N,)``, so ``kl_divergence`` returns one value per
batch element either way and the loss's reduction is a drop-in.
"""


@lru_cache(maxsize=8)
def strict_lower_mask(latent_dim: int, device: torch.device) -> Tensor:
    """Boolean mask selecting the strictly-lower triangle of an ``N x N`` matrix.

    Cached because it is a constant that would otherwise be rebuilt twice per training
    step (prior and posterior) for 240,000 steps.

    A module-level function rather than a cached method, because ``lru_cache`` on a method
    would keep ``self`` alive for the life of the process.

    ``device`` is part of the cache key, so a model moved between MPS and CUDA gets a mask
    on the right device rather than a stale one. ``dtype`` is not a parameter because a
    boolean index mask is invariantly ``torch.bool``. Note that ``torch.device("mps")`` and
    ``torch.device("mps", 0)`` compare unequal and so occupy separate entries; the cost is
    one extra entry, never a wrong device, and ``maxsize`` leaves room for cpu, mps and
    cuda in both spellings.

    Args:
        latent_dim: Dimensionality ``N``.
        device: Device the mask is needed on.

    Returns:
        A ``(N, N)`` boolean tensor, True strictly below the diagonal. Masked assignment
        fills it in row-major order, matching ``torch.tril_indices``; the ordering only
        has to be consistent, since the network learns whatever mapping it is given.
    """
    return torch.ones(latent_dim, latent_dim, dtype=torch.bool, device=device).tril(-1)


@dataclass(frozen=True)
class LatentStats:
    """The raw parameters a latent encoder predicts.

    A named container rather than a tuple. The parameters are consumed in four places
    (distribution construction, the sigma diagnostics, the latent-stats logging and the
    tests), and a tuple whose length depends on a config flag would make the arity
    implicit at every one of them -- including four ``*stats`` star-unpacks.

    Attributes:
        mu: Means of shape ``(B, latent_dim)``.
        logvar: Log-variances of shape ``(B, latent_dim)``. Log-variance rather than
            sigma or log-sigma; see DEVIATIONS.md entry 1.
        lower: The ``N(N-1)/2`` strictly-lower-triangular entries of the Cholesky factor,
            shape ``(B, N(N-1)/2)``, or None for the diagonal parameterization. **None is
            what selects the Phase 1 code path**, so the data itself says which
            parameterization it carries rather than a flag having to be threaded
            alongside it.
    """

    mu: Tensor
    logvar: Tensor
    lower: Tensor | None = None

    @property
    def sigma(self) -> Tensor:
        """Standard deviations, ``exp(0.5 * logvar)``.

        Positive by construction with no clamp, which is the reason for parameterizing
        the log-variance.
        """
        return torch.exp(0.5 * self.logvar)

    @property
    def latent_dim(self) -> int:
        """Dimensionality ``N`` of the latent space."""
        return int(self.mu.shape[-1])

    @property
    def is_full(self) -> bool:
        """Whether these parameters describe a full covariance."""
        return self.lower is not None

    @property
    def scale_tril(self) -> Tensor:
        """The Cholesky factor ``L`` of the covariance, shape ``(B, N, N)``.

        ``diag(L) = exp(0.5 * logvar) = sigma`` -- **exactly** the standard deviations the
        diagonal path produces, which is what makes the flag-off comparison meaningful
        rather than approximate. The strict lower triangle is used as predicted, with no
        transform: it is unconstrained, and only the diagonal needs to be positive.

        ``Sigma = L L^T`` is then positive-definite **by construction**, because ``L`` is
        triangular with a strictly positive diagonal. Nothing here builds ``Sigma`` and
        repairs it, and nothing calls ``torch.linalg.cholesky`` on a predicted matrix.

        Returns:
            The lower-triangular factor.

        Raises:
            ValueError: If these are diagonal parameters, which have no factor to build.
        """
        if self.lower is None:
            raise ValueError(
                "scale_tril is undefined for diagonal latent parameters: there is no "
                "predicted Cholesky factor. Check `is_full` first, or use `sigma`."
            )
        # Assemble out of place: the strict triangle is scattered into a fresh zeros
        # tensor and *added* to the diagonal, so no tensor that autograd needs is mutated.
        mask = strict_lower_mask(self.latent_dim, self.lower.device)
        strict = torch.zeros(
            *self.mu.shape[:-1],
            self.latent_dim,
            self.latent_dim,
            dtype=self.lower.dtype,
            device=self.lower.device,
        )
        strict[..., mask] = self.lower
        return torch.diag_embed(self.sigma) + strict


class LatentEncoder(nn.Module):
    """Encoder predicting the parameters of an axis-aligned Gaussian.

    Not used directly: see :class:`PriorNet` and :class:`PosteriorNet`, which fix
    the input channels and expose the correct call signature for each role.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 6,
        base_channels: int = 32,
        num_downs: int = 4,
        convs_per_scale: int = 3,
        max_channels: int | None = None,
        full_covariance: bool = False,
    ) -> None:
        """Build the encoder.

        Args:
            in_channels: Channels of the input tensor.
            latent_dim: Dimensionality ``N`` of the latent space.
            base_channels: Width of the shallowest scale.
            num_downs: Number of down-sampling operations.
            convs_per_scale: 3x3 convolutions per scale.
            max_channels: Optional channel ceiling (off by default, per the paper).
            full_covariance: Predict a full covariance's Cholesky factor rather than a
                diagonal. A plain bool rather than the config's string, because the
                encoder only needs to know which of two parameterizations to emit;
                :class:`~probunet.model.prob_unet.ProbUNetConfig` owns the vocabulary and
                its validation.

        Raises:
            ValueError: If ``latent_dim`` is not positive.
        """
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.full_covariance = full_covariance
        self.widths = channel_widths(base_channels, num_downs, max_channels)

        self.blocks = nn.ModuleList()
        previous = in_channels
        for width in self.widths:
            self.blocks.append(ConvBlock(previous, width, convs_per_scale))
            previous = width
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Predicts [mu | logvar | strictly-lower-triangular] along the channel axis.
        # 2N when diagonal; 2N + N(N-1)/2 == N + N(N+1)/2 when full (12 vs 27 at N=6).
        # The first 2N channels mean exactly the same thing in both cases, so the extra
        # ones are purely additive -- which, together with the zero-init below, is what
        # makes the full-covariance model start as an exact replica of the diagonal one.
        self.n_lower = latent_dim * (latent_dim - 1) // 2 if full_covariance else 0
        self.head_outputs = 2 * latent_dim + self.n_lower
        self.head = nn.Conv2d(self.widths[-1], self.head_outputs, kernel_size=1)

    @torch.no_grad()
    def zero_correlation_head(self) -> None:
        """Zero the head slice that predicts ``L``'s strictly-lower triangle.

        Called after the blanket He-normal initialization, so the full-covariance model
        begins with ``L`` exactly diagonal and therefore an **exact replica of the
        diagonal model's latent distribution**. Any correlation it later exhibits is
        demonstrably learned rather than inherited from initialization noise.

        The specific reason this matters here: FINDINGS.md 3.5 names weight
        initialization as a contributing suspect for the axis-wise posterior collapse
        that Phase 2 exists to test. Introducing a *second* initialization difference
        into exactly that mechanism would confound the result. Under He-normal the
        off-diagonals would start at roughly 6% of the diagonal, which is small but not
        nothing.

        Gradients are unaffected: ``dz_i/dL_ij = epsilon_j``, which is non-zero even when
        ``L_ij = 0``, and each of the ``N(N-1)/2`` outputs occupies a distinct position in
        ``L`` and so receives a distinct gradient -- there is no symmetry to break. Zero
        is a starting point, not a hyperparameter.
        """
        if not self.full_covariance:
            return
        self.head.weight[2 * self.latent_dim :].zero_()
        if self.head.bias is not None:
            self.head.bias[2 * self.latent_dim :].zero_()

    @property
    def first_conv(self) -> nn.Conv2d:
        """The first convolution, whose ``in_channels`` distinguishes the two roles."""
        return self.blocks[0].block[0]

    def forward(self, x: Tensor) -> LatentStats:
        """Predict the Gaussian parameters for an already-assembled input.

        Args:
            x: Input of shape ``(B, in_channels, H, W)``.

        Returns:
            The predicted :class:`LatentStats`, each field of shape
            ``(B, latent_dim)``.
        """
        for index, block in enumerate(self.blocks):
            if index > 0:
                x = self.pool(x)
            x = block(x)
        # Global average pooling over the spatial dimensions, then a 1x1 conv.
        pooled = x.mean(dim=(2, 3), keepdim=True)
        predicted = self.head(pooled).flatten(start_dim=1)
        n = self.latent_dim
        return LatentStats(
            mu=predicted[:, :n],
            logvar=predicted[:, n : 2 * n],
            # None for the diagonal parameterization, which is what routes every consumer
            # down the Phase 1 path. When diagonal the head is exactly 2N wide, so the
            # logvar slice above is the same tensor `predicted[:, n:]` produced before.
            lower=predicted[:, 2 * n :] if self.full_covariance else None,
        )

    def distribution_from_stats(self, stats: LatentStats) -> LatentDistribution:
        """Build the latent distribution from already-computed parameters.

        Two genuinely separate code paths, dispatched on whether a Cholesky factor was
        predicted:

        * **Diagonal** -- ``Independent(Normal, 1)``, byte-for-byte the Phase 1 path.
          ``Independent(..., 1)`` reinterprets the latent axis as part of the event shape,
          so ``kl_divergence`` sums over latent dimensions by construction and returns one
          value per batch element. The reduction rule required by CLAUDE.md -- sum over
          latent dims, mean over batch -- is therefore structural rather than hand-rolled.
        * **Full** -- ``MultivariateNormal(loc, scale_tril=L)``, which has the same batch
          and event shapes, so that same reduction is a drop-in.

        The diagonal case is deliberately **not** routed through a ``MultivariateNormal``
        with a diagonal factor. That would be algebraically equivalent but numerically
        different -- it samples and computes the KL through different kernels, and the two
        KLs measurably disagree by ~5e-7. Phase 2's entire comparison rests on flag-off
        reproducing Phase 1 exactly, so the two paths stay separate.

        **This is an instance method specifically so it can cross-check intent against
        data.** Dispatching on ``stats.lower is None`` is what keeps the diagonal branch
        untouched, but on its own it means a plumbing bug that dropped ``lower`` would
        silently train a *diagonal* model inside a run labelled full-covariance -- an
        undetectable false null, and the worst failure available to this project. Every
        construction path in the codebase goes through a module that knows its own
        configuration, so the mismatch is caught here rather than discovered in a report.

        Args:
            stats: The predicted parameters.

        Returns:
            A distribution with batch shape ``(B,)`` and event shape ``(latent_dim,)``.

        Raises:
            RuntimeError: If the encoder is configured for one parameterization and the
                parameters describe the other.
        """
        if self.full_covariance != stats.is_full:
            raise RuntimeError(
                f"latent parameterization mismatch: encoder was built with "
                f"full_covariance={self.full_covariance} but received parameters with "
                f"lower={'set' if stats.is_full else 'None'}. A full-covariance model "
                "must never fall back to the diagonal path -- that would train and report "
                "the wrong arm of the Phase 2 comparison."
            )
        if stats.lower is None:
            return Independent(Normal(loc=stats.mu, scale=stats.sigma), 1)
        return MultivariateNormal(loc=stats.mu, scale_tril=stats.scale_tril)

    def distribution_from_input(self, x: Tensor) -> LatentDistribution:
        """Build the latent distribution for an already-assembled input.

        Args:
            x: Input of shape ``(B, in_channels, H, W)``.

        Returns:
            The latent distribution over ``z``.
        """
        return self.distribution_from_stats(self.forward(x))


class PriorNet(LatentEncoder):
    """Latent encoder over the **image only**: ``P(z | X)``."""

    def __init__(self, image_channels: int = 1, **kwargs: object) -> None:
        """Build the prior net.

        Args:
            image_channels: Channels of the input image.
            **kwargs: Forwarded to :class:`LatentEncoder`.
        """
        super().__init__(in_channels=image_channels, **kwargs)  # type: ignore[arg-type]

    def distribution(self, image: Tensor) -> LatentDistribution:
        """Return ``P(z | X)``.

        Args:
            image: Image of shape ``(B, image_channels, H, W)``.

        Returns:
            The prior distribution over ``z``.
        """
        return self.distribution_from_input(image)


class PosteriorNet(LatentEncoder):
    """Latent encoder over the image **and a ground-truth mask**: ``Q(z | X, Y)``."""

    def __init__(
        self, image_channels: int = 1, mask_channels: int = 1, **kwargs: object
    ) -> None:
        """Build the posterior net.

        Args:
            image_channels: Channels of the input image.
            mask_channels: Channels used to encode the mask. One single 0/1 channel
                for this project's binary masks.
            **kwargs: Forwarded to :class:`LatentEncoder`.
        """
        super().__init__(  # type: ignore[arg-type]
            in_channels=image_channels + mask_channels, **kwargs
        )
        self.image_channels = image_channels
        self.mask_channels = mask_channels

    def assemble_input(self, image: Tensor, mask: Tensor) -> Tensor:
        """Concatenate image and mask along the channel axis.

        Args:
            image: Image of shape ``(B, image_channels, H, W)``.
            mask: Mask of shape ``(B, H, W)`` or ``(B, mask_channels, H, W)``, with
                any integer or floating dtype.

        Returns:
            Tensor of shape ``(B, image_channels + mask_channels, H, W)``.

        Raises:
            ValueError: If the mask's shape is not compatible with the image.
        """
        if mask.dim() == image.dim() - 1:
            mask = mask.unsqueeze(1)
        if mask.dim() != image.dim():
            raise ValueError(f"mask has {mask.dim()} dims, expected {image.dim()}")
        if mask.shape[1] != self.mask_channels:
            raise ValueError(
                f"mask has {mask.shape[1]} channels, expected {self.mask_channels}"
            )
        if mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with image shape "
                f"{tuple(image.shape)}"
            )
        return torch.cat([image, mask.to(image.dtype)], dim=1)

    def distribution(self, image: Tensor, mask: Tensor) -> LatentDistribution:
        """Return ``Q(z | X, Y)``.

        Args:
            image: Image of shape ``(B, image_channels, H, W)``.
            mask: Ground-truth mask for the same batch.

        Returns:
            The posterior distribution over ``z``.
        """
        return self.distribution_from_input(self.assemble_input(image, mask))