"""Tests for the paper-faithful training augmentation.

The properties here are the ones that fail *silently*, which is why each is pinned:

* a mask that stops being binary corrupts every loss and metric downstream;
* a non-empty mask that a transform empties corrupts the ambiguity buckets with no
  trace in the loss curve;
* an image pushed outside ``[0, 1]`` breaks the dataset's range invariant;
* augmentation that is not reproducible from the seed makes every comparison unfalsifiable;
* constant-zero border fill would make our augmentation harsher than the paper's, while
  still wearing the paper's parameter values.
"""

from __future__ import annotations

import numpy as np
import pytest

from probunet.data.transforms import (
    PAPER_TILE_PX,
    AugmentationConfig,
    AugmentationOutcome,
    AugmentationStats,
    apply_transform,
    augment_pair,
    sample_transform,
    source_coordinates,
)

SIZE = 32
# Same 180/128 ratio the paper's pipeline has, at a size that keeps the suite fast.
PAD = 45


def config(**overrides: object) -> AugmentationConfig:
    """Build an enabled augmentation config scaled to the test tile size.

    Args:
        **overrides: Fields to override.

    Returns:
        The configuration.
    """
    settings: dict[str, object] = {"enabled": True, "pad_to_px": PAD}
    settings.update(overrides)
    return AugmentationConfig(**settings)  # type: ignore[arg-type]


def identity_config(**overrides: object) -> AugmentationConfig:
    """Build a config whose transform is the exact identity.

    Args:
        **overrides: Fields to override.

    Returns:
        A configuration with no rotation, shear, scale, elastic or random crop.
    """
    settings: dict[str, object] = {
        "rotation_degrees": 0.0,
        "scale_range": (1.0, 1.0),
        "shear": 0.0,
        "elastic_alpha_px": 0.0,
        "random_crop": False,
    }
    settings.update(overrides)
    return config(**settings)


@pytest.fixture
def pair() -> tuple[np.ndarray, np.ndarray]:
    """A synthetic image and a paired mask with a compact central lesion."""
    rng = np.random.default_rng(0)
    image = rng.random((SIZE, SIZE), dtype=np.float32)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[13:19, 13:19] = 1
    return image, mask


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_augmentation_is_deterministic_under_a_fixed_seed(
    pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """The same (seed, epoch, position) reproduces the transform bit for bit."""
    image, mask = pair
    first = augment_pair(image, mask, config(), epoch=3, position=7)
    second = augment_pair(image, mask, config(), epoch=3, position=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_augmentation_varies_across_epochs(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """A fresh transform every epoch is the point of augmentation."""
    image, mask = pair
    first, _, _ = augment_pair(image, mask, config(), epoch=0, position=0)
    second, _, _ = augment_pair(image, mask, config(), epoch=1, position=0)
    assert not np.array_equal(first, second)


def test_augmentation_varies_across_positions(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Two samples in one epoch must not receive the same transform."""
    image, mask = pair
    first, _, _ = augment_pair(image, mask, config(), epoch=0, position=0)
    second, _, _ = augment_pair(image, mask, config(), epoch=0, position=1)
    assert not np.array_equal(first, second)


def test_a_different_seed_changes_the_transform(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """The augmentation seed is a real knob, not decoration."""
    image, mask = pair
    first, _, _ = augment_pair(image, mask, config(seed=1), epoch=0, position=0)
    second, _, _ = augment_pair(image, mask, config(seed=2), epoch=0, position=0)
    assert not np.array_equal(first, second)


def test_augmentation_does_not_touch_the_global_rng(
    pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Draws come from a derived generator, never numpy's global state.

    If augmentation consumed global randomness it would couple to the model's latent
    sampling, and changing an augmentation parameter would silently perturb ``z``.
    """
    image, mask = pair
    np.random.seed(12345)
    before = np.random.random()
    np.random.seed(12345)
    augment_pair(image, mask, config(), epoch=0, position=0)
    assert np.random.random() == before


# --------------------------------------------------------------------------- #
# Mask integrity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("epoch", range(6))
def test_mask_stays_strictly_binary(pair: tuple[np.ndarray, np.ndarray], epoch: int) -> None:
    """Nearest-neighbour sampling must leave the mask in {0, 1}."""
    image, mask = pair
    _, warped, _ = augment_pair(image, mask, config(), epoch=epoch, position=epoch)
    assert warped.dtype == mask.dtype
    assert set(np.unique(warped)).issubset({0, 1}), f"mask left {{0,1}}: {np.unique(warped)}"


def test_empty_mask_stays_empty(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """A grader who saw no lesion must still see none after augmentation."""
    image, _ = pair
    empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for position in range(20):
        _, warped, outcome = augment_pair(image, empty, config(), epoch=0, position=position)
        assert warped.sum() == 0
        # An empty mask cannot "lose" a lesion, so it must never trigger a redraw.
        assert outcome.redraws == 0
        assert not outcome.lesion_lost


def test_full_mask_stays_full(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """A fully-annotated mask survives, which also proves the border is not zero-filled."""
    image, _ = pair
    full = np.ones((SIZE, SIZE), dtype=np.uint8)
    for position in range(20):
        _, warped, _ = augment_pair(image, full, config(), epoch=0, position=position)
        assert warped.all(), "border fill leaked zeros into a fully-annotated mask"


def test_single_pixel_lesion_is_never_silently_emptied(
    pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """The dangerous case: neither empty nor full, but one pixel.

    Under ``order=0`` sampling a one-pixel region can receive no output pixel at all and
    simply vanish. That would turn a non-empty target into an empty one and corrupt the
    ambiguity buckets with no signal in the loss, so the transform is redrawn. Whatever
    happens, a lost lesion must be either recovered or *reported*.
    """
    image, _ = pair
    tiny = np.zeros((SIZE, SIZE), dtype=np.uint8)
    tiny[SIZE // 2, SIZE // 2] = 1

    lost = 0
    for position in range(300):
        _, warped, outcome = augment_pair(image, tiny, config(), epoch=0, position=position)
        assert warped.any() or outcome.lesion_lost, (
            "a one-pixel lesion was emptied without being reported"
        )
        if outcome.lesion_lost:
            lost += 1
            # The fallback returns the sample untransformed, so the lesion is intact.
            assert warped.sum() == 1
    # The redraw guard should make this rare; the counter exists to prove it.
    assert lost / 300 < 0.05, f"one-pixel lesions lost too often: {lost}/300"


def test_lesion_lost_fallback_returns_the_pair_untransformed() -> None:
    """When every redraw fails, the original pair is returned rather than a bad target."""
    image = np.full((SIZE, SIZE), 0.5, dtype=np.float32)
    tiny = np.zeros((SIZE, SIZE), dtype=np.uint8)
    tiny[0, 0] = 1  # a corner pixel, easy to rotate out of frame
    # Zero redraws and an aggressive shrink maximise the chance of the fallback firing.
    aggressive = config(max_redraws=0, scale_range=(0.5, 0.5), rotation_degrees=45.0)
    outcomes = [
        augment_pair(image, tiny, aggressive, epoch=0, position=p) for p in range(60)
    ]
    fallbacks = [o for _, _, o in outcomes if o.lesion_lost]
    if not fallbacks:
        pytest.skip("no fallback triggered in this sample; covered by the assertion above")
    for warped_image, warped_mask, outcome in outcomes:
        if outcome.lesion_lost:
            assert not outcome.augmented
            assert np.array_equal(warped_mask, tiny)
            assert np.array_equal(warped_image, image)


# --------------------------------------------------------------------------- #
# Image integrity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("epoch", range(6))
def test_image_stays_in_the_unit_range(pair: tuple[np.ndarray, np.ndarray], epoch: int) -> None:
    """Bilinear interpolation cannot overshoot, which keeps the dataset invariant.

    This is why the image is sampled with ``order=1`` rather than the reference's default
    cubic ``order_data=3``, which rings outside ``[0, 1]``.
    """
    image, mask = pair
    warped, _, _ = augment_pair(image, mask, config(), epoch=epoch, position=0)
    assert warped.dtype == np.float32
    assert warped.min() >= 0.0 and warped.max() <= 1.0


def test_border_is_mirrored_tissue_not_zeros() -> None:
    """Out-of-frame coordinates resolve to reflected content, never a constant.

    A uniform image must stay uniform under any transform. Under constant-0 fill the
    corners would drop to zero -- which is exactly the artifact the reflect-pad-to-180
    design exists to avoid, since at 128 with no margin a 22.5 deg rotation would zero
    almost 14% of the frame.
    """
    image = np.full((SIZE, SIZE), 0.7, dtype=np.float32)
    mask = np.ones((SIZE, SIZE), dtype=np.uint8)
    for position in range(20):
        warped, warped_mask, _ = augment_pair(
            image, mask, config(rotation_degrees=45.0, scale_range=(0.6, 0.6)), 0, position
        )
        assert np.allclose(warped, 0.7), "border fill was not reflection"
        assert warped_mask.all()


# --------------------------------------------------------------------------- #
# Identity and geometry
# --------------------------------------------------------------------------- #
def test_identity_transform_is_a_bit_exact_no_op(
    pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """A centred crop with no geometry must return the input unchanged.

    This pins the coordinate bookkeeping: the pad offset, the frame centre and the
    un-padding must cancel exactly, or every augmented sample carries a constant shift.
    """
    image, mask = pair
    warped_image, warped_mask, outcome = augment_pair(
        image, mask, identity_config(), epoch=0, position=0
    )
    assert np.array_equal(warped_image, image)
    assert np.array_equal(warped_mask, mask)
    assert outcome == AugmentationOutcome()


def test_random_crop_translates_the_content(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """With only the random crop active, the output is a translation of the input."""
    image, mask = pair
    translated = identity_config(random_crop=True)
    offsets = {
        sample_transform(translated, SIZE, np.random.default_rng([translated.seed, 0, p])
                         ).crop_offset
        for p in range(30)
    }
    assert len(offsets) > 1, "random_crop drew the same offset every time"
    # And the crop stays inside the padded frame.
    for row, col in offsets:
        assert 0 <= row <= PAD - SIZE and 0 <= col <= PAD - SIZE


def test_scaling_up_grows_the_lesion_by_the_square_of_the_factor() -> None:
    """Zooming in by s must grow a centred lesion's area by about s^2.

    This pins the *direction* of the affine matrix. Because resampling is a backward
    mapping, the matrix is the inverse of the visual transform, and an inverted sign here
    would be invisible in every other test -- rotation and shear ranges are symmetric, so
    they look identical either way.

    Only the zoom-in direction is asserted, because zooming in samples strictly inside the
    tile. Zooming out reaches past the border, where reflection applies; see
    :func:`test_scaling_down_pulls_in_mirrored_lesion_copies`.
    """
    image = np.zeros((SIZE, SIZE), dtype=np.float32)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[10:22, 10:22] = 1  # 144 px, centred
    settings = identity_config(scale_range=(2.0, 2.0))
    transform = sample_transform(settings, SIZE, np.random.default_rng(0))
    _, warped = apply_transform(image, mask, transform, settings)
    assert warped.sum() == pytest.approx(144 * 4, rel=0.1)


def test_scaling_down_pulls_in_mirrored_lesion_copies() -> None:
    """Documents the real cost of reflect-padding, so it cannot surprise us later.

    Shrinking the content reaches past the tile border, and there the frame is filled by
    reflection. The reflected region therefore contains *mirrored copies of the lesion*,
    which the mask marks as lesion too. The pair stays self-consistent -- mirrored tissue
    carries a mirrored label, so nothing is mislabelled -- but the copies are
    anatomically implausible in a way the paper's real 180x180 CT context was not. That
    is the trade we accepted in exchange for not zero-filling 36% of the frame at scale
    0.8. See DEVIATIONS.md entry 12.
    """
    image = np.zeros((SIZE, SIZE), dtype=np.float32)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    settings = identity_config(scale_range=(0.5, 0.5))
    transform = sample_transform(settings, SIZE, np.random.default_rng(0))
    _, warped = apply_transform(image, mask, transform, settings)

    # The centred lesion shrinks to ~36 px, but reflected copies push the total above it
    # and spread lesion pixels beyond where a single shrunken blob could reach.
    assert warped.sum() > 36
    assert set(np.unique(warped)).issubset({0, 1}), "reflection must not break binarity"
    columns = np.where(warped.any(axis=0))[0]
    assert columns.max() - columns.min() + 1 > 12, (
        "expected reflected copies to spread lesion pixels across the frame"
    )


def test_source_coordinates_have_the_output_shape(
    pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Coordinates are produced for the cropped output, not the padded frame."""
    settings = config()
    transform = sample_transform(settings, SIZE, np.random.default_rng(0))
    coordinates = source_coordinates(transform, SIZE, settings)
    assert coordinates.shape == (2, SIZE, SIZE)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_paper_tile_size_is_the_default() -> None:
    """The default frame is the paper's 180x180 tile."""
    assert AugmentationConfig().pad_to_px == PAPER_TILE_PX == 180


def test_margin_reproduces_the_papers_52_pixel_border() -> None:
    """128 padded to 180 gives the paper's 26-pixel margin per side."""
    assert AugmentationConfig().margin_px(128) == 26


def test_pad_smaller_than_the_tile_is_rejected() -> None:
    """A frame smaller than the data cannot supply a crop."""
    with pytest.raises(ValueError, match="smaller than the data tile"):
        AugmentationConfig(pad_to_px=64).margin_px(128)


def test_disabled_config_cannot_be_augmented_with(
    pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Calling the transform with augmentation off is a bug, not a no-op."""
    image, mask = pair
    with pytest.raises(ValueError, match="disabled AugmentationConfig"):
        augment_pair(image, mask, AugmentationConfig(enabled=False), epoch=0, position=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pad_to_px", 0, "pad_to_px must be positive"),
        ("rotation_degrees", -1.0, "rotation_degrees must be non-negative"),
        ("shear", -0.1, "shear must be non-negative"),
        ("scale_range", (1.2, 0.8), "low <= high"),
        ("scale_range", (0.0, 1.0), "low <= high"),
        ("elastic_alpha_px", -1.0, "elastic_alpha_px must be non-negative"),
        ("elastic_sigma_px", 0.0, "elastic_sigma_px must be positive"),
        ("max_redraws", -1, "max_redraws must be non-negative"),
    ],
)
def test_invalid_settings_are_rejected(field: str, value: object, message: str) -> None:
    """Malformed augmentation settings fail at construction, not mid-epoch."""
    with pytest.raises(ValueError, match=message):
        AugmentationConfig(**{field: value})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #
def test_stats_report_rates_and_reset() -> None:
    """The counters are what make a lesion-loss problem visible."""
    stats = AugmentationStats()
    assert stats.as_metrics() == {}, "no samples must report nothing, not a hollow zero"

    stats.record(AugmentationOutcome())
    stats.record(AugmentationOutcome(redraws=2))
    stats.record(AugmentationOutcome(redraws=3, lesion_lost=True, augmented=False))
    metrics = stats.as_metrics()
    assert metrics["aug_samples"] == 3
    assert metrics["aug_redraw_rate"] == pytest.approx(5 / 3)
    assert metrics["aug_lesion_lost_fraction"] == pytest.approx(1 / 3)

    stats.reset()
    assert stats.as_metrics() == {}