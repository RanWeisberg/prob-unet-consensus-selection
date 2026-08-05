"""The Probabilistic U-Net: U-Net + prior net + posterior net + ``f_comb``.

Assembles the four components and enforces the two properties that are easy to get
silently wrong:

* **Late injection.** The U-Net is called with the image alone. ``z`` reaches the
  network only inside ``f_comb``, concatenated to the last activation.
* **Cheap sampling.** :meth:`ProbUNet.encode` runs the U-Net exactly once and returns
  its features; :meth:`ProbUNet.sample` re-runs only ``f_comb`` per sample. A counter
  incremented at the point of the U-Net call makes that assertable in tests.

During training ``z`` is drawn from the posterior, which has seen the ground-truth
mask. At inference the posterior is discarded and ``z`` comes from the prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.distributions import Independent

from probunet.model.encoder import LatentEncoder, LatentStats, PosteriorNet, PriorNet
from probunet.model.fcomb import FComb
from probunet.model.unet import UNet

LATENT_COVARIANCE_MODES: tuple[str, ...] = ("diagonal", "full")
"""Supported latent covariance parameterizations.

* ``"diagonal"`` -- the paper's axis-aligned Gaussian (Appendix H.1). **Phase 1.**
* ``"full"`` -- a full covariance via its Cholesky factor. **Phase 2**, and the only
  change Phase 2 makes.
"""

DIAGONAL, FULL = LATENT_COVARIANCE_MODES


@dataclass(frozen=True)
class ProbUNetConfig:
    """Architecture configuration.

    Defaults follow CLAUDE.md and the paper. ``max_channels`` is None because strict
    channel doubling is the specification; capping is an optimization from the
    authors' released code and stays behind this flag.

    Attributes:
        latent_dim: Dimensionality ``N`` of the latent space. Held at 6 across both
            Phase 2 arms: the follow-up work tuned it per model, so changing the
            covariance *and* the dimension would confound two variables.
        latent_covariance: ``"diagonal"`` for the paper's axis-aligned Gaussian, or
            ``"full"`` for a full covariance parameterized by its Cholesky factor.
            This is **the** Phase 2 flag. ``"diagonal"`` takes the Phase 1 code path
            completely unchanged rather than a full-covariance object with a diagonal
            factor -- the latter samples through a different kernel and drifts the
            numbers (measured: the two KLs differ by ~5e-7).
        base_channels: Width of the shallowest scale.
        num_downs: Number of down/up-sampling operations.
        convs_per_scale: 3x3 convolutions per scale.
        num_classes: Output logit channels; 2 for binary segmentation with softmax
            cross-entropy.
        image_channels: Input image channels.
        mask_channels: Channels used to encode the mask for the posterior.
        fcomb_convs: Number of 1x1 convolutions in ``f_comb``.
        fcomb_channels: Hidden width of ``f_comb``; defaults to ``base_channels``.
        max_channels: Optional channel ceiling. None means the paper's strict
            doubling.
        align_corners: Bilinear up-sampling alignment.
        bias_init_std: Standard deviation of the truncated normal used for biases.
    """

    latent_dim: int = 6
    latent_covariance: str = DIAGONAL
    base_channels: int = 32
    num_downs: int = 4
    convs_per_scale: int = 3
    num_classes: int = 2
    image_channels: int = 1
    mask_channels: int = 1
    fcomb_convs: int = 3
    fcomb_channels: int | None = None
    max_channels: int | None = None
    align_corners: bool = False
    bias_init_std: float = 1e-3

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If ``latent_covariance`` is not a supported mode, or
                ``latent_dim`` is not positive. An unrecognized covariance mode must
                fail here rather than fall through to the diagonal default: a typo
                that silently trained the Phase 1 model under the name of the Phase 2
                arm would produce a comparison of the baseline against itself.
        """
        if self.latent_covariance not in LATENT_COVARIANCE_MODES:
            raise ValueError(
                f"latent_covariance must be one of {LATENT_COVARIANCE_MODES}, got "
                f"{self.latent_covariance!r}"
            )
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

    @property
    def resolved_fcomb_channels(self) -> int:
        """Hidden width of ``f_comb``, defaulting to the U-Net's base width."""
        return self.base_channels if self.fcomb_channels is None else self.fcomb_channels

    @property
    def full_covariance(self) -> bool:
        """Whether the latent Gaussians carry a full covariance."""
        return self.latent_covariance == FULL

    @property
    def latent_head_outputs(self) -> int:
        """Number of channels the prior/posterior 1x1 head predicts.

        ``2N`` for the diagonal parameterization (mean and log-variance), and
        ``N + N(N+1)/2`` for the full one -- 27 at ``N = 6``. The layout is
        ``[mu | logvar | strictly-lower-triangular]``, chosen so the **first 2N
        channels mean exactly what they mean in the diagonal case**; the extra
        ``N(N-1)/2`` are purely additive.
        """
        n = self.latent_dim
        return n + n * (n + 1) // 2 if self.full_covariance else 2 * n


@dataclass
class Encoded:
    """Everything produced by one U-Net pass plus the latent distributions.

    Attributes:
        features: Last U-Net activation, shape ``(B, base_channels, H, W)``.
        prior: ``P(z | X)``.
        prior_stats: The prior's raw predicted parameters.
        posterior: ``Q(z | X, Y)``, or None if no mask was supplied.
        posterior_stats: The posterior's raw predicted parameters, or None.
    """

    features: Tensor
    prior: Independent
    prior_stats: LatentStats
    posterior: Independent | None = None
    posterior_stats: LatentStats | None = None

    @property
    def batch_size(self) -> int:
        """Number of items in the batch."""
        return self.features.shape[0]


@dataclass
class ProbUNetOutput:
    """One training forward pass.

    Attributes:
        logits: Class logits, shape ``(B, num_classes, H, W)``.
        z: The latent sample used, shape ``(B, latent_dim)``.
        encoded: The :class:`Encoded` state the logits were produced from.
    """

    logits: Tensor
    z: Tensor
    encoded: Encoded = field(repr=False)

    @property
    def prior(self) -> Independent:
        """``P(z | X)``."""
        return self.encoded.prior

    @property
    def posterior(self) -> Independent | None:
        """``Q(z | X, Y)``."""
        return self.encoded.posterior


class ProbUNet(nn.Module):
    """The Probabilistic U-Net of Kohl et al. (NeurIPS 2018)."""

    def __init__(self, config: ProbUNetConfig | None = None) -> None:
        """Build the model and initialize its weights.

        Args:
            config: Architecture configuration; defaults to :class:`ProbUNetConfig`.
        """
        super().__init__()
        self.config = config or ProbUNetConfig()
        shared = {
            "base_channels": self.config.base_channels,
            "num_downs": self.config.num_downs,
            "convs_per_scale": self.config.convs_per_scale,
            "max_channels": self.config.max_channels,
        }

        self.unet = UNet(
            in_channels=self.config.image_channels,
            align_corners=self.config.align_corners,
            **shared,
        )
        self.prior_net = PriorNet(
            image_channels=self.config.image_channels,
            latent_dim=self.config.latent_dim,
            **shared,
        )
        self.posterior_net = PosteriorNet(
            image_channels=self.config.image_channels,
            mask_channels=self.config.mask_channels,
            latent_dim=self.config.latent_dim,
            **shared,
        )
        self.fcomb = FComb(
            feature_channels=self.unet.out_channels,
            latent_dim=self.config.latent_dim,
            hidden_channels=self.config.resolved_fcomb_channels,
            num_classes=self.config.num_classes,
            num_convs=self.config.fcomb_convs,
        )

        # Instrumentation, not state: counts U-Net invocations made through
        # encode(). Complements the forward-hook based tests -- the counter catches
        # a second unet call added inside encode(), the hook catches one added
        # anywhere else, e.g. in sample().
        self._unet_forward_calls = 0

        self.apply(self._init_module)

    def _init_module(self, module: nn.Module) -> None:
        """Initialize one module: He-normal weights, truncated-normal biases.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                std = self.config.bias_init_std
                # Truncate at +-2 sigma, matching TensorFlow's truncated_normal.
                # torch's default bounds (-2, 2) are absolute, so at sigma=1e-3 they
                # would mean no truncation at all.
                nn.init.trunc_normal_(
                    module.bias, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
                )

    @property
    def unet_forward_calls(self) -> int:
        """How many times :meth:`encode` has invoked the U-Net since the last reset."""
        return self._unet_forward_calls

    def reset_unet_forward_calls(self) -> None:
        """Reset the U-Net invocation counter."""
        self._unet_forward_calls = 0

    def encode(self, image: Tensor, mask: Tensor | None = None) -> Encoded:
        """Run the U-Net once and build the latent distributions.

        Args:
            image: Image batch of shape ``(B, image_channels, H, W)``.
            mask: Optional ground-truth mask. When given, the posterior is computed
                from the image concatenated with it; when omitted, only the prior is
                available.

        Returns:
            The :class:`Encoded` state, whose features can be reused for as many
            samples as wanted.
        """
        # The U-Net receives the image and nothing else: z cannot enter the encoder.
        features = self.unet(image)
        self._unet_forward_calls += 1

        # Compute the parameters once and build the distributions from them: calling
        # prior_net(image) and then prior_net.distribution(image) would run a 7.9 M
        # parameter encoder twice per step.
        prior_stats = self.prior_net(image)
        prior = LatentEncoder.distribution_from_stats(prior_stats)

        posterior = None
        posterior_stats = None
        if mask is not None:
            posterior_stats = self.posterior_net(
                self.posterior_net.assemble_input(image, mask)
            )
            posterior = LatentEncoder.distribution_from_stats(posterior_stats)

        return Encoded(
            features=features,
            prior=prior,
            prior_stats=prior_stats,
            posterior=posterior,
            posterior_stats=posterior_stats,
        )

    def reconstruct(self, encoded: Encoded, z: Tensor) -> Tensor:
        """Map cached features plus a latent sample to logits, re-running only f_comb.

        Args:
            encoded: State from :meth:`encode`.
            z: Latent samples of shape ``(B, latent_dim)``.

        Returns:
            Logits of shape ``(B, num_classes, H, W)``.
        """
        return self.fcomb(encoded.features, z)

    def forward(self, image: Tensor, mask: Tensor | None = None) -> ProbUNetOutput:
        """Training forward pass: ``z`` is drawn from the **posterior**.

        Args:
            image: Image batch of shape ``(B, image_channels, H, W)``.
            mask: Ground-truth mask, required. The posterior must see it -- that is
                what teaches the prior to cover the space of plausible variants.

        Returns:
            The :class:`ProbUNetOutput` for this pass.

        Raises:
            ValueError: If ``mask`` is None. Training on prior samples would train
                the model without the posterior and is never what is wanted; for
                inference use :meth:`encode` followed by :meth:`sample`.
        """
        if mask is None:
            raise ValueError(
                "mask is required: during training z is drawn from the posterior, "
                "which must see the ground-truth mask. For inference use "
                "encode() followed by sample(use_prior=True)."
            )
        encoded = self.encode(image, mask)
        assert encoded.posterior is not None  # guaranteed by the mask check above
        # rsample() keeps the path differentiable through mu and sigma.
        z = encoded.posterior.rsample()
        return ProbUNetOutput(
            logits=self.reconstruct(encoded, z), z=z, encoded=encoded
        )

    def sample(
        self,
        encoded: Encoded,
        n_samples: int = 1,
        use_prior: bool = True,
    ) -> Tensor:
        """Draw ``n_samples`` segmentations per image, re-running only ``f_comb``.

        The U-Net is not touched here: that is the architecture's explicit efficiency
        advantage. ``f_comb`` is applied once per sample rather than on one big
        batched tensor, because materializing ``(B * n_samples, C, H, W)`` features
        would dominate memory for large ``n_samples`` while buying nothing -- the
        1x1 convolutions are cheap either way.

        Args:
            encoded: State from :meth:`encode`.
            n_samples: Number of samples per image.
            use_prior: Draw from the prior (inference) or the posterior.

        Returns:
            Logits of shape ``(B, n_samples, num_classes, H, W)``.

        Raises:
            ValueError: If ``n_samples`` is not positive, or if the posterior was
                requested but is not available.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        distribution = encoded.prior if use_prior else encoded.posterior
        if distribution is None:
            raise ValueError(
                "posterior requested but unavailable: call encode() with a mask"
            )
        samples = [
            self.reconstruct(encoded, distribution.rsample()) for _ in range(n_samples)
        ]
        return torch.stack(samples, dim=1)

    def predict(
        self, image: Tensor, n_samples: int = 1, use_prior: bool = True
    ) -> Tensor:
        """Convenience inference path: encode once, then sample.

        Args:
            image: Image batch of shape ``(B, image_channels, H, W)``.
            n_samples: Number of samples per image.
            use_prior: Draw from the prior. Inference should leave this True.

        Returns:
            Logits of shape ``(B, n_samples, num_classes, H, W)``.
        """
        return self.sample(self.encode(image), n_samples=n_samples, use_prior=use_prior)

    def parameter_counts(self) -> dict[str, int]:
        """Parameter count per component, for reconciliation against the spec.

        Returns:
            Counts for each component plus the total.
        """
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in (
                ("unet", self.unet),
                ("prior_net", self.prior_net),
                ("posterior_net", self.posterior_net),
                ("fcomb", self.fcomb),
            )
        }
        counts["total"] = sum(counts.values())
        return counts