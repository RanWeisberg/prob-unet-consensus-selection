"""The deterministic U-Net backbone of the Probabilistic U-Net.

Follows CLAUDE.md and Appendix H.1 of Kohl et al. (arXiv:1806.05034): four
down-sampling and four up-sampling operations, three 3x3 convolutions per scale
each followed by ReLU, base width 32 doubled per down-sampling step, average
pooling on the way down and bilinear interpolation on the way up. No normalization
layers and no dropout -- those belong to the later, flag-gated modernization phase.

The forward pass returns the **last activation**, not class logits: in the
Probabilistic U-Net the logits are produced by ``f_comb`` after the latent sample
has been concatenated. Note also that :meth:`UNet.forward` takes only an image, so
the latent ``z`` structurally cannot enter the encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def channel_widths(
    base_channels: int, num_downs: int, max_channels: int | None = None
) -> list[int]:
    """Per-scale channel widths, shallowest first.

    The paper specifies strict doubling per down-sampling step. The authors'
    released code caps the width for memory reasons; that is an optimization, not
    the specification, so ``max_channels`` defaults to None everywhere.

    Args:
        base_channels: Width of the shallowest scale.
        num_downs: Number of down-sampling operations; there are ``num_downs + 1``
            scales in total.
        max_channels: Optional ceiling applied to every scale.

    Returns:
        A list of ``num_downs + 1`` channel widths.

    Raises:
        ValueError: If ``base_channels`` or ``num_downs`` is not positive, or if
            ``max_channels`` is smaller than ``base_channels``.
    """
    if base_channels <= 0:
        raise ValueError(f"base_channels must be positive, got {base_channels}")
    if num_downs <= 0:
        raise ValueError(f"num_downs must be positive, got {num_downs}")
    if max_channels is not None and max_channels < base_channels:
        raise ValueError(
            f"max_channels {max_channels} < base_channels {base_channels}"
        )
    raw = [base_channels * 2**scale for scale in range(num_downs + 1)]
    return [min(width, max_channels) for width in raw] if max_channels else raw


class ConvBlock(nn.Module):
    """``num_convs`` successive 3x3 convolutions, each followed by ReLU.

    This is the per-scale unit used by the U-Net encoder, the U-Net decoder and the
    prior/posterior encoders alike, which is what makes the latent nets share the
    encoder's architecture exactly.
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 3) -> None:
        """Build the block.

        Args:
            in_channels: Channels of the input feature map.
            out_channels: Channels of every conv in the block.
            num_convs: Number of 3x3 convolutions.

        Raises:
            ValueError: If ``num_convs`` is not positive.
        """
        super().__init__()
        if num_convs <= 0:
            raise ValueError(f"num_convs must be positive, got {num_convs}")
        layers: list[nn.Module] = []
        for index in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block.

        Args:
            x: Input of shape ``(B, in_channels, H, W)``.

        Returns:
            Output of shape ``(B, out_channels, H, W)``.
        """
        return self.block(x)


class UNet(nn.Module):
    """Encoder/decoder with skip connections, returning its last activation."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_downs: int = 4,
        convs_per_scale: int = 3,
        max_channels: int | None = None,
        align_corners: bool = False,
    ) -> None:
        """Build the U-Net.

        Args:
            in_channels: Input image channels.
            base_channels: Width of the shallowest scale.
            num_downs: Number of down/up-sampling operations.
            convs_per_scale: 3x3 convolutions per scale.
            max_channels: Optional channel ceiling (off by default, per the paper).
            align_corners: Passed to the bilinear up-sampling. False is the modern
                default and avoids a half-pixel misalignment; the reference ports
                use True.
        """
        super().__init__()
        self.widths = channel_widths(base_channels, num_downs, max_channels)
        self.num_downs = num_downs
        self.align_corners = align_corners

        self.encoder_blocks = nn.ModuleList()
        previous = in_channels
        for width in self.widths:
            self.encoder_blocks.append(ConvBlock(previous, width, convs_per_scale))
            previous = width

        # Average pooling, per Appendix H.1. Parameter-free, so one instance is
        # reused for every down-sampling step.
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # One decoder block per up-sampling step, deepest first. Each consumes the
        # up-sampled deeper features concatenated with the skip connection.
        self.decoder_blocks = nn.ModuleList()
        current = self.widths[-1]
        for scale in range(num_downs - 1, -1, -1):
            skip = self.widths[scale]
            out = self.widths[scale]
            self.decoder_blocks.append(ConvBlock(current + skip, out, convs_per_scale))
            current = out

    @property
    def out_channels(self) -> int:
        """Channels of the returned last activation."""
        return self.widths[0]

    def encode(self, image: Tensor) -> list[Tensor]:
        """Run the encoder path and return one feature map per scale.

        Args:
            image: Input of shape ``(B, in_channels, H, W)``.

        Returns:
            Feature maps, shallowest first; the last entry is the bottleneck.
        """
        skips: list[Tensor] = []
        x = image
        for index, block in enumerate(self.encoder_blocks):
            if index > 0:
                x = self.pool(x)
            x = block(x)
            skips.append(x)
        return skips

    def forward(self, image: Tensor) -> Tensor:
        """Run the full U-Net.

        Args:
            image: Input of shape ``(B, in_channels, H, W)``.

        Returns:
            The last activation, of shape ``(B, base_channels, H, W)``. These are
            features, not logits: ``f_comb`` produces the logits once ``z`` has been
            concatenated.
        """
        skips = self.encode(image)
        x = skips[-1]
        for step, block in enumerate(self.decoder_blocks):
            scale = self.num_downs - 1 - step
            x = F.interpolate(
                x, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners
            )
            x = torch.cat([x, skips[scale]], dim=1)
            x = block(x)
        return x