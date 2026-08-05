"""Prior and posterior latent encoders.

Both predict an axis-aligned Gaussian over ``R^N``. Per Appendix H.1 they share the
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

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from probunet.model.unet import ConvBlock, channel_widths


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
    """

    mu: Tensor
    logvar: Tensor

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
    ) -> None:
        """Build the encoder.

        Args:
            in_channels: Channels of the input tensor.
            latent_dim: Dimensionality ``N`` of the latent space.
            base_channels: Width of the shallowest scale.
            num_downs: Number of down-sampling operations.
            convs_per_scale: 3x3 convolutions per scale.
            max_channels: Optional channel ceiling (off by default, per the paper).

        Raises:
            ValueError: If ``latent_dim`` is not positive.
        """
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.widths = channel_widths(base_channels, num_downs, max_channels)

        self.blocks = nn.ModuleList()
        previous = in_channels
        for width in self.widths:
            self.blocks.append(ConvBlock(previous, width, convs_per_scale))
            previous = width
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Predicts [mu, logvar] stacked along the channel axis.
        self.head = nn.Conv2d(self.widths[-1], 2 * latent_dim, kernel_size=1)

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
        return LatentStats(
            mu=predicted[:, : self.latent_dim],
            logvar=predicted[:, self.latent_dim :],
        )

    @staticmethod
    def distribution_from_stats(stats: LatentStats) -> Independent:
        """Build the latent distribution from already-computed parameters.

        ``Independent(..., 1)`` reinterprets the latent axis as part of the event
        shape, so ``kl_divergence`` sums over latent dimensions by construction and
        returns one value per batch element. The reduction rule required by
        CLAUDE.md -- sum over latent dims, mean over batch -- is therefore
        structural here rather than hand-rolled in the loss.

        Args:
            stats: The predicted parameters.

        Returns:
            An ``Independent(Normal(mu, exp(0.5 * logvar)), 1)`` distribution with
            batch shape ``(B,)`` and event shape ``(latent_dim,)``.
        """
        return Independent(Normal(loc=stats.mu, scale=stats.sigma), 1)

    def distribution_from_input(self, x: Tensor) -> Independent:
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

    def distribution(self, image: Tensor) -> Independent:
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

    def distribution(self, image: Tensor, mask: Tensor) -> Independent:
        """Return ``Q(z | X, Y)``.

        Args:
            image: Image of shape ``(B, image_channels, H, W)``.
            mask: Ground-truth mask for the same batch.

        Returns:
            The posterior distribution over ``z``.
        """
        return self.distribution_from_input(self.assemble_input(image, mask))