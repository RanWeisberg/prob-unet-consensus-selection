"""``f_comb``: combines the U-Net's last activation with a latent sample.

This is where the **late latent injection** happens, and the reason drawing many
samples for one image is cheap. The latent vector ``z`` of length ``N`` is broadcast
into an ``N``-channel map at segmentation resolution, concatenated to the last U-Net
activation, and mapped to class logits by three successive 1x1 convolutions.

``z`` never enters the U-Net encoder: it is introduced here and nowhere else.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FComb(nn.Module):
    """Three successive 1x1 convolutions over ``[features, broadcast(z)]``."""

    def __init__(
        self,
        feature_channels: int = 32,
        latent_dim: int = 6,
        hidden_channels: int = 32,
        num_classes: int = 2,
        num_convs: int = 3,
    ) -> None:
        """Build f_comb.

        Args:
            feature_channels: Channels of the last U-Net activation.
            latent_dim: Dimensionality ``N`` of the latent space.
            hidden_channels: Width of the intermediate 1x1 convolutions. Defaults to
                the U-Net's base width, as in the authors' released code; the paper
                specifies only the number of convolutions.
            num_classes: Number of output logit channels.
            num_convs: Number of 1x1 convolutions. The paper specifies three.

        Raises:
            ValueError: If ``num_convs`` is less than two, which would leave no
                room for both an input projection and a class projection.
        """
        super().__init__()
        if num_convs < 2:
            raise ValueError(f"num_convs must be at least 2, got {num_convs}")
        self.latent_dim = latent_dim
        self.in_channels = feature_channels + latent_dim

        layers: list[nn.Module] = []
        previous = self.in_channels
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(previous, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            previous = hidden_channels
        # Final projection to logits: no activation, the softmax lives in the loss.
        layers.append(nn.Conv2d(previous, num_classes, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def broadcast_latent(self, z: Tensor, height: int, width: int) -> Tensor:
        """Broadcast a latent vector into an ``N``-channel map.

        Args:
            z: Latent of shape ``(B, latent_dim)``.
            height: Target height.
            width: Target width.

        Returns:
            Tensor of shape ``(B, latent_dim, height, width)``.

        Raises:
            ValueError: If ``z`` is not 2-D or has the wrong latent size.
        """
        if z.dim() != 2:
            raise ValueError(f"expected z of shape (B, latent_dim), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z has latent size {z.shape[1]}, expected {self.latent_dim}"
            )
        return z.view(z.shape[0], self.latent_dim, 1, 1).expand(-1, -1, height, width)

    def forward(self, features: Tensor, z: Tensor) -> Tensor:
        """Produce class logits from features and one latent sample per batch item.

        Args:
            features: Last U-Net activation, shape ``(B, feature_channels, H, W)``.
            z: Latent samples, shape ``(B, latent_dim)``.

        Returns:
            Logits of shape ``(B, num_classes, H, W)``.

        Raises:
            ValueError: If the batch sizes of ``features`` and ``z`` disagree.
        """
        if features.shape[0] != z.shape[0]:
            raise ValueError(
                f"batch mismatch: features {features.shape[0]} vs z {z.shape[0]}"
            )
        latent_map = self.broadcast_latent(z, features.shape[-2], features.shape[-1])
        return self.layers(torch.cat([features, latent_map], dim=1))