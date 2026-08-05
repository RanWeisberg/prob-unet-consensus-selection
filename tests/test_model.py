"""Tests for the baseline Probabilistic U-Net.

Beyond shape plumbing, these pin down the four properties that fail silently:

* the posterior sees the image concatenated with a ground-truth mask, the prior sees
  the image alone;
* ``z`` never enters the U-Net encoder;
* the U-Net runs exactly once no matter how many samples are drawn;
* log-variance, not sigma, is what the encoders predict.

Parameter counts are asserted against the arithmetic derived from Appendix H.1, so a
change to the channel schedule cannot pass unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import kl_divergence

from probunet.model import (
    ConvBlock,
    FComb,
    PosteriorNet,
    PriorNet,
    ProbUNet,
    ProbUNetConfig,
    UNet,
    channel_widths,
)

BATCH = 2
SIZE = 32  # smaller than the real 128 to keep the suite fast; 4 downsamplings still fit
LATENT_DIM = 6
NUM_CLASSES = 2

# Analytic counts for the real configuration (128x128, base 32, 4 downs, 3 convs per
# scale, latent 6, 2 classes), from scratch/param_arithmetic.py. A 3x3 conv with bias
# costs 9*in*out + out; a 1x1 conv costs in*out + out.
EXPECTED_PARAMS_PAPER = {
    "unet": 11_773_536,
    "prior_net": 7_861_452,
    "posterior_net": 7_861_740,
    "fcomb": 2_370,
    "total": 27_499_098,
}
EXPECTED_PARAMS_CAPPED = {
    "unet": 2_776_800,
    "prior_net": 1_367_244,
    "posterior_net": 1_367_532,
    "fcomb": 2_370,
    "total": 5_513_946,
}


@pytest.fixture
def config() -> ProbUNetConfig:
    """Default architecture config."""
    return ProbUNetConfig(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)


@pytest.fixture
def model(config: ProbUNetConfig) -> ProbUNet:
    """A model with reproducible weights."""
    torch.manual_seed(0)
    return ProbUNet(config)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """An image batch and a matching binary mask batch."""
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(BATCH, 1, SIZE, SIZE, generator=generator)
    mask = (torch.rand(BATCH, SIZE, SIZE, generator=generator) > 0.7).to(torch.int64)
    return image, mask


# --------------------------------------------------------------------------- #
# Channel schedule
# --------------------------------------------------------------------------- #
def test_channel_widths_follows_the_paper() -> None:
    """Strict doubling by default; capping only when explicitly asked for."""
    assert channel_widths(32, 4) == [32, 64, 128, 256, 512]
    assert channel_widths(32, 4, max_channels=128) == [32, 64, 128, 128, 128]


def test_channel_widths_validates() -> None:
    """Nonsensical schedules are rejected."""
    with pytest.raises(ValueError, match="base_channels"):
        channel_widths(0, 4)
    with pytest.raises(ValueError, match="num_downs"):
        channel_widths(32, 0)
    with pytest.raises(ValueError, match="max_channels"):
        channel_widths(32, 4, max_channels=16)


def test_capping_is_off_by_default() -> None:
    """The faithful baseline must not silently cap channels."""
    assert ProbUNetConfig().max_channels is None


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def test_conv_block_shape() -> None:
    """A block maps to its output width and preserves resolution."""
    block = ConvBlock(3, 16, num_convs=3)
    out = block(torch.zeros(BATCH, 3, SIZE, SIZE))
    assert out.shape == (BATCH, 16, SIZE, SIZE)
    # Three convs, each followed by a ReLU.
    assert sum(isinstance(m, nn.Conv2d) for m in block.block) == 3
    assert sum(isinstance(m, nn.ReLU) for m in block.block) == 3


def test_conv_block_validates() -> None:
    """A block with no convolutions is an error."""
    with pytest.raises(ValueError, match="num_convs"):
        ConvBlock(1, 1, num_convs=0)


def test_unet_returns_features_at_input_resolution() -> None:
    """The U-Net returns base_channels features at full resolution, not logits."""
    unet = UNet(in_channels=1, base_channels=32, num_downs=4)
    out = unet(torch.zeros(BATCH, 1, SIZE, SIZE))
    assert out.shape == (BATCH, 32, SIZE, SIZE)
    assert unet.out_channels == 32


def test_unet_encoder_scales() -> None:
    """The encoder halves resolution and doubles width per step."""
    unet = UNet(in_channels=1, base_channels=32, num_downs=4)
    skips = unet.encode(torch.zeros(BATCH, 1, SIZE, SIZE))
    assert [tuple(s.shape) for s in skips] == [
        (BATCH, 32, 32, 32),
        (BATCH, 64, 16, 16),
        (BATCH, 128, 8, 8),
        (BATCH, 256, 4, 4),
        (BATCH, 512, 2, 2),
    ]


def test_unet_uses_average_pooling_and_no_norm_layers() -> None:
    """Down-sampling is average pooling; the baseline has no normalization."""
    unet = UNet()
    assert isinstance(unet.pool, nn.AvgPool2d)
    assert not any(isinstance(m, nn.modules.batchnorm._NormBase) for m in unet.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in unet.modules())
    assert not any(isinstance(m, nn.Dropout) for m in unet.modules())


def test_latent_encoder_shapes() -> None:
    """Both encoders emit (mu, logvar) of shape (B, latent_dim)."""
    prior = PriorNet(image_channels=1, latent_dim=LATENT_DIM)
    mu, logvar = prior(torch.zeros(BATCH, 1, SIZE, SIZE))
    assert mu.shape == (BATCH, LATENT_DIM)
    assert logvar.shape == (BATCH, LATENT_DIM)

    distribution = prior.distribution(torch.zeros(BATCH, 1, SIZE, SIZE))
    assert distribution.batch_shape == (BATCH,)
    assert distribution.event_shape == (LATENT_DIM,)


def test_fcomb_shape_and_broadcast() -> None:
    """f_comb broadcasts z spatially and emits one logit map per class."""
    fcomb = FComb(feature_channels=32, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
    features = torch.zeros(BATCH, 32, SIZE, SIZE)
    z = torch.arange(BATCH * LATENT_DIM, dtype=torch.float32).view(BATCH, LATENT_DIM)

    latent_map = fcomb.broadcast_latent(z, SIZE, SIZE)
    assert latent_map.shape == (BATCH, LATENT_DIM, SIZE, SIZE)
    # Every spatial position carries the same latent vector.
    assert torch.equal(latent_map[:, :, 0, 0], z)
    assert torch.equal(latent_map[:, :, -1, -1], z)

    assert fcomb(features, z).shape == (BATCH, NUM_CLASSES, SIZE, SIZE)


def test_fcomb_is_three_1x1_convs() -> None:
    """f_comb is three 1x1 convolutions, with no activation on the last."""
    fcomb = FComb(feature_channels=32, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
    convs = [m for m in fcomb.layers if isinstance(m, nn.Conv2d)]
    assert len(convs) == 3
    assert all(conv.kernel_size == (1, 1) for conv in convs)
    assert convs[0].in_channels == 32 + LATENT_DIM
    assert convs[-1].out_channels == NUM_CLASSES
    assert isinstance(fcomb.layers[-1], nn.Conv2d)


def test_fcomb_validates() -> None:
    """Bad latent shapes and degenerate depths are rejected."""
    fcomb = FComb(feature_channels=8, latent_dim=LATENT_DIM)
    with pytest.raises(ValueError, match="at least 2"):
        FComb(num_convs=1)
    with pytest.raises(ValueError, match="latent size"):
        fcomb(torch.zeros(BATCH, 8, 4, 4), torch.zeros(BATCH, LATENT_DIM + 1))
    with pytest.raises(ValueError, match="shape"):
        fcomb(torch.zeros(BATCH, 8, 4, 4), torch.zeros(BATCH, LATENT_DIM, 1))
    with pytest.raises(ValueError, match="batch mismatch"):
        fcomb(torch.zeros(BATCH, 8, 4, 4), torch.zeros(BATCH + 1, LATENT_DIM))


def test_prob_unet_forward_shapes(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """A training pass yields logits, a latent sample and both distributions."""
    image, mask = batch
    output = model(image, mask)
    assert output.logits.shape == (BATCH, NUM_CLASSES, SIZE, SIZE)
    assert output.z.shape == (BATCH, LATENT_DIM)
    assert output.posterior is not None
    assert output.prior.event_shape == (LATENT_DIM,)


# --------------------------------------------------------------------------- #
# Prior vs posterior inputs
# --------------------------------------------------------------------------- #
def test_prior_takes_image_only_posterior_takes_image_and_mask(model: ProbUNet) -> None:
    """The first conv of each latent net encodes the input asymmetry."""
    assert model.prior_net.first_conv.in_channels == 1
    assert model.posterior_net.first_conv.in_channels == 2


def test_posterior_actually_receives_image_concatenated_with_mask(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Hook the posterior's first conv and check what it is really fed."""
    image, mask = batch
    seen: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen.append(inputs[0].detach().clone())

    handle = model.posterior_net.first_conv.register_forward_pre_hook(capture)
    try:
        model(image, mask)
    finally:
        handle.remove()

    assert len(seen) == 1
    received = seen[0]
    assert received.shape == (BATCH, 2, SIZE, SIZE)
    assert torch.equal(received[:, 0:1], image)
    assert torch.equal(received[:, 1], mask.to(image.dtype))


def test_prior_never_sees_the_mask(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The prior's input is exactly the image."""
    image, mask = batch
    seen: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen.append(inputs[0].detach().clone())

    handle = model.prior_net.first_conv.register_forward_pre_hook(capture)
    try:
        model(image, mask)
    finally:
        handle.remove()

    assert len(seen) == 1
    assert torch.equal(seen[0], image)


def test_posterior_rejects_mismatched_masks() -> None:
    """A mask that does not match the image is an error, not a broadcast."""
    posterior = PosteriorNet(image_channels=1, mask_channels=1, latent_dim=LATENT_DIM)
    image = torch.zeros(BATCH, 1, SIZE, SIZE)
    with pytest.raises(ValueError, match="channels"):
        posterior.assemble_input(image, torch.zeros(BATCH, 2, SIZE, SIZE))
    with pytest.raises(ValueError, match="incompatible"):
        posterior.assemble_input(image, torch.zeros(BATCH, 1, SIZE // 2, SIZE))
    with pytest.raises(ValueError, match="incompatible"):
        posterior.assemble_input(image, torch.zeros(BATCH + 1, 1, SIZE, SIZE))


def test_forward_requires_a_mask(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Training without a mask would silently bypass the posterior."""
    image, _ = batch
    with pytest.raises(ValueError, match="mask is required"):
        model(image)


# --------------------------------------------------------------------------- #
# Late injection: z must not enter the encoder
# --------------------------------------------------------------------------- #
def test_unet_receives_only_the_image(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The U-Net's input is the bare image: no latent channels appended."""
    image, mask = batch
    seen: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen.append(inputs[0].detach().clone())

    handle = model.unet.register_forward_pre_hook(capture)
    try:
        encoded = model.encode(image, mask)
        model.sample(encoded, n_samples=8, use_prior=True)
    finally:
        handle.remove()

    assert len(seen) == 1
    assert seen[0].shape == (BATCH, 1, SIZE, SIZE)
    assert torch.equal(seen[0], image)


def test_unet_forward_signature_has_no_latent(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The U-Net is callable with an image alone, so z structurally cannot enter."""
    image, _ = batch
    assert model.unet(image).shape == (BATCH, 32, SIZE, SIZE)


# --------------------------------------------------------------------------- #
# Efficient sampling: the U-Net runs once regardless of m
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_samples", [1, 4, 8, 16])
def test_unet_runs_once_per_encode_hook(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor], n_samples: int
) -> None:
    """Forward-hook count: drawing m samples must not re-run the U-Net."""
    image, _ = batch
    calls = {"unet": 0, "fcomb": 0}

    def count(key: str):
        def hook(*_args: object) -> None:
            calls[key] += 1

        return hook

    handles = [
        model.unet.register_forward_hook(count("unet")),
        model.fcomb.register_forward_hook(count("fcomb")),
    ]
    try:
        encoded = model.encode(image)
        model.sample(encoded, n_samples=n_samples)
    finally:
        for handle in handles:
            handle.remove()

    assert calls["unet"] == 1
    assert calls["fcomb"] == n_samples


@pytest.mark.parametrize("n_samples", [1, 4, 8, 16])
def test_unet_runs_once_per_encode_counter(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor], n_samples: int
) -> None:
    """Encode-level counter: complements the hook.

    The hook catches a stray U-Net call added anywhere in the module tree; this
    counter catches a second call added *inside* encode() itself, where a hook on
    the same module would simply see a higher count with no baseline to compare to.
    """
    image, _ = batch
    model.reset_unet_forward_calls()
    assert model.unet_forward_calls == 0

    encoded = model.encode(image)
    assert model.unet_forward_calls == 1

    model.sample(encoded, n_samples=n_samples)
    assert model.unet_forward_calls == 1, "sample() must not re-enter the U-Net"

    model.encode(image)
    assert model.unet_forward_calls == 2, "a second encode() must be counted"


def test_predict_shapes(model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """The inference convenience path returns one logit map per sample."""
    image, _ = batch
    model.reset_unet_forward_calls()
    logits = model.predict(image, n_samples=16)
    assert logits.shape == (BATCH, 16, NUM_CLASSES, SIZE, SIZE)
    assert model.unet_forward_calls == 1


def test_samples_differ_across_draws(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Different z produce different logits, so f_comb really consumes z."""
    image, _ = batch
    torch.manual_seed(3)
    logits = model.predict(image, n_samples=4)
    for index in range(1, 4):
        assert not torch.allclose(logits[:, 0], logits[:, index]), (
            "samples are identical; f_comb may be ignoring z"
        )


def test_reconstruct_is_deterministic_given_z(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Same features and same z give the same logits."""
    image, mask = batch
    encoded = model.encode(image, mask)
    z = torch.zeros(BATCH, LATENT_DIM)
    assert torch.equal(model.reconstruct(encoded, z), model.reconstruct(encoded, z))


def test_sample_validates(model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Zero samples, and asking for a posterior that was never computed, are errors."""
    image, _ = batch
    encoded = model.encode(image)
    with pytest.raises(ValueError, match="n_samples"):
        model.sample(encoded, n_samples=0)
    with pytest.raises(ValueError, match="posterior requested"):
        model.sample(encoded, use_prior=False)


# --------------------------------------------------------------------------- #
# Log-variance parameterization
# --------------------------------------------------------------------------- #
def test_scale_is_exp_half_logvar(model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """The distribution's scale is exp(0.5 * logvar), not the raw second output."""
    image, mask = batch
    encoded = model.encode(image, mask)

    mu, logvar = encoded.prior_stats
    assert torch.allclose(encoded.prior.base_dist.loc, mu)
    assert torch.allclose(encoded.prior.base_dist.scale, torch.exp(0.5 * logvar))

    assert encoded.posterior_stats is not None
    post_mu, post_logvar = encoded.posterior_stats
    assert torch.allclose(encoded.posterior.base_dist.loc, post_mu)
    assert torch.allclose(
        encoded.posterior.base_dist.scale, torch.exp(0.5 * post_logvar)
    )


def test_logvar_can_be_negative_and_scale_stays_positive() -> None:
    """A negative second output is legal and yields a small positive scale.

    Were the head interpreted as sigma directly, a negative value would be an
    invalid scale -- so this is the test that would catch that confusion.
    """
    prior = PriorNet(image_channels=1, latent_dim=LATENT_DIM)
    mu = torch.zeros(1, LATENT_DIM)
    logvar = torch.full((1, LATENT_DIM), -8.0)
    distribution = prior.distribution_from_stats(mu, logvar)
    assert torch.all(distribution.base_dist.scale > 0)
    assert torch.allclose(
        distribution.base_dist.scale, torch.full((1, LATENT_DIM), float(np.exp(-4.0)))
    )


def test_kl_reduces_over_latent_dims_only(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """kl_divergence returns one value per batch element, not per latent dim.

    This is what makes 'sum over latent dims, mean over batch' structural.
    """
    image, mask = batch
    encoded = model.encode(image, mask)
    assert encoded.posterior is not None
    kl = kl_divergence(encoded.posterior, encoded.prior)
    assert kl.shape == (BATCH,)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_forward_backward_smoke(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """A backward pass reaches the U-Net, the posterior and f_comb."""
    image, mask = batch
    output = model(image, mask)
    output.logits.sum().backward()

    for name, module in (
        ("unet", model.unet),
        ("posterior_net", model.posterior_net),
        ("fcomb", model.fcomb),
    ):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} received no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"{name} has non-finite grads"


def test_prior_gets_no_gradient_from_logits_alone(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The prior is reached only through the KL term, never through the logits.

    During training z comes from the posterior, so the reconstruction term cannot
    touch the prior. If this fails, z is being drawn from the wrong distribution.
    """
    image, mask = batch
    output = model(image, mask)
    output.logits.sum().backward()
    assert all(p.grad is None for p in model.prior_net.parameters())


def test_prior_gets_gradient_from_kl(
    model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The KL term does reach the prior, so the distributions are differentiable."""
    image, mask = batch
    encoded = model.encode(image, mask)
    assert encoded.posterior is not None
    kl_divergence(encoded.posterior, encoded.prior).mean().backward()
    grads = [p.grad for p in model.prior_net.parameters() if p.grad is not None]
    assert grads, "prior received no gradient from the KL term"


# --------------------------------------------------------------------------- #
# Initialization, determinism, parameter counts
# --------------------------------------------------------------------------- #
def test_bias_init_is_small_truncated_normal(config: ProbUNetConfig) -> None:
    """Biases start from a truncated normal with sigma 1e-3, bounded at 2 sigma."""
    torch.manual_seed(0)
    model = ProbUNet(config)
    biases = torch.cat(
        [
            module.bias.detach().flatten()
            for module in model.modules()
            if isinstance(module, nn.Conv2d) and module.bias is not None
        ]
    )
    assert biases.numel() > 1000
    assert biases.abs().max() <= 2.0 * config.bias_init_std + 1e-9
    assert biases.std().item() == pytest.approx(config.bias_init_std, rel=0.25)


def test_weight_init_is_he_normal(config: ProbUNetConfig) -> None:
    """Conv weights follow He-normal: std ~ sqrt(2 / fan_in)."""
    torch.manual_seed(0)
    model = ProbUNet(config)
    # A wide 3x3 conv gives enough samples for a tight check.
    conv = max(
        (m for m in model.unet.modules() if isinstance(m, nn.Conv2d)),
        key=lambda m: m.weight.numel(),
    )
    fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    expected = (2.0 / fan_in) ** 0.5
    assert conv.weight.std().item() == pytest.approx(expected, rel=0.05)


def test_same_seed_gives_identical_weights(config: ProbUNetConfig) -> None:
    """Construction is reproducible under a fixed seed."""
    torch.manual_seed(7)
    first = ProbUNet(config)
    torch.manual_seed(7)
    second = ProbUNet(config)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_parameter_counts_match_appendix_arithmetic() -> None:
    """Component parameter counts equal the values derived from Appendix H.1.

    Also asserts the structural claim that each latent net is the same size as the
    U-Net's encoder path, and that the two together exceed the whole U-Net.
    """
    model = ProbUNet(ProbUNetConfig())
    counts = model.parameter_counts()
    assert counts == EXPECTED_PARAMS_PAPER

    encoder_path = sum(
        p.numel() for p in model.unet.encoder_blocks.parameters()
    )
    assert encoder_path == 7_855_296
    # Each latent net is the encoder path plus only its 1x1 head (and, for the
    # posterior, one extra input channel).
    assert counts["prior_net"] - encoder_path == 6_156
    assert counts["posterior_net"] - counts["prior_net"] == 288
    assert counts["prior_net"] + counts["posterior_net"] > counts["unet"]


def test_capped_parameter_counts() -> None:
    """The optional channel cap reproduces the documented smaller model."""
    model = ProbUNet(ProbUNetConfig(max_channels=128))
    assert model.parameter_counts() == EXPECTED_PARAMS_CAPPED


def test_real_resolution_forward() -> None:
    """One pass at the real 128x128 resolution, batch 1, to catch shape drift."""
    torch.manual_seed(0)
    model = ProbUNet(ProbUNetConfig())
    image = torch.rand(1, 1, 128, 128)
    mask = (torch.rand(1, 128, 128) > 0.7).to(torch.int64)
    output = model(image, mask)
    assert output.logits.shape == (1, NUM_CLASSES, 128, 128)
    assert torch.isfinite(output.logits).all()


def test_float32_end_to_end(model: ProbUNet, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Everything stays float32: MPS does not support float64."""
    image, mask = batch
    output = model(image, mask)
    assert output.logits.dtype == torch.float32
    assert output.z.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in model.parameters())

# --------------------------------------------------------------------------- #
# Phase 2: the latent_covariance flag
# --------------------------------------------------------------------------- #
def test_latent_covariance_defaults_to_diagonal() -> None:
    """The default is the paper's axis-aligned Gaussian.

    Phase 2 is opt-in. A default of ``full`` would silently make every existing config
    a Phase 2 config.
    """
    assert ProbUNetConfig().latent_covariance == "diagonal"
    assert ProbUNetConfig().full_covariance is False


@pytest.mark.parametrize("mode", ["diagonal", "full"])
def test_latent_covariance_accepts_both_modes(mode: str) -> None:
    """Both supported modes construct."""
    assert ProbUNetConfig(latent_covariance=mode).latent_covariance == mode


@pytest.mark.parametrize("mode", ["Diagonal", "FULL", "dense", "tril", "", "none"])
def test_unknown_latent_covariance_is_rejected(mode: str) -> None:
    """An unrecognized mode must fail loudly, never fall back to the default.

    A typo that quietly trained the Phase 1 model under the Phase 2 config's name would
    produce a comparison of the baseline against itself.
    """
    with pytest.raises(ValueError, match="latent_covariance must be one of"):
        ProbUNetConfig(latent_covariance=mode)


def test_non_positive_latent_dim_is_rejected() -> None:
    """The config validates latent_dim, not only the encoder."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        ProbUNetConfig(latent_dim=0)


def test_latent_head_output_width_follows_the_flag() -> None:
    """2N when diagonal, N + N(N+1)/2 when full: 12 vs 27 at N = 6."""
    assert ProbUNetConfig(latent_dim=6).latent_head_outputs == 12
    assert ProbUNetConfig(latent_dim=6, latent_covariance="full").latent_head_outputs == 27
    # The extra entries are the strictly-lower triangle: 27 - 12 = 15 = 6*5/2.
    assert 27 - 12 == 6 * 5 // 2
    # And the arithmetic holds at other N, so nothing is hardcoded to 6.
    for n in (1, 2, 3, 8):
        assert ProbUNetConfig(latent_dim=n).latent_head_outputs == 2 * n
        full = ProbUNetConfig(latent_dim=n, latent_covariance="full")
        assert full.latent_head_outputs == n + n * (n + 1) // 2


def test_latent_covariance_flag_is_recorded_in_the_config_dict() -> None:
    """Phase 3 loads a frozen Phase 2 checkpoint, so the flag must survive to disk."""
    from probunet.training.config import ExperimentConfig

    config = ExperimentConfig(model=ProbUNetConfig(latent_covariance="full"))
    assert config.to_dict()["model"]["latent_covariance"] == "full"
    assert ExperimentConfig.from_dict(config.to_dict()).model.latent_covariance == "full"
