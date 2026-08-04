"""Paper-faithful data augmentation for the LIDC training split.

Appendix H.1 of Kohl et al. specifies the augmentation for the lung experiments in one
sentence, and it is worth quoting exactly because everything here follows from it:

    "During training image-grader pairs are drawn randomly. We apply augmentations to
    the image tiles (180x180 pixels size): random elastic deformation, rotation,
    shearing, scaling and a randomly translated crop that results in a tile size of
    128x128 pixels."

Three consequences, each of which is a deliberate decision recorded in DEVIATIONS.md:

1. **The list is exhaustive and it is short.** No mirroring and no intensity/gamma
   augmentation appear in it. The authors' released code *does* apply both, but only in
   ``data/cityscapes/data_loader.py`` -- there is no LIDC loader in that repository at
   all -- and the paper says Cityscapes "**additionally** impose random color
   augmentations". That word makes colour a Cityscapes-only addition. Adding either here
   would make our Phase 1 harsher than the paper's, so neither is implemented.

2. **The paper gives no numeric values whatsoever.** Rotation ``+-22.5 deg`` and scale
   ``(0.8, 1.2)`` are transferred from the reference's Cityscapes ``da_kwargs`` because
   those two quantities are resolution-independent. Shear is in neither the paper nor
   ``batchgenerators`` (which has no shear parameter) and is ours. The reference's
   elastic ``alpha=(0, 800)`` normalizes by an array-size-dependent L2 norm, so it does
   not transfer to 128x128; we re-parameterize elastic strength as a **peak displacement
   in pixels**, which is interpretable and directly testable.

3. **The 180x180 tile is load-bearing, not incidental.** Our preprocessed data is
   already 128x128, so there is no 180x180 tile to crop from. Applying the paper's
   magnitudes directly to a 128 frame is *not* faithful -- it is harsher, because the
   paper's 52-pixel margin is exactly what absorbs them:

   ==========================  =========================  ======================
   transform                   artifact-free up to        zero-fill at 128 bare
   ==========================  =========================  ======================
   rotation +-22.5 deg         ``180/(cos+sin) = 137.8``  13.97% of the frame
   scale 0.8                   ``180*0.8     = 144.0``    36.50% of the frame
   both at once                ``110.2``                  not covered even at 180
   ==========================  =========================  ======================

   So we reconstruct the margin: the 128 tile is **reflect-padded to 180**, the
   transform is sampled and applied in that padded frame, and a randomly translated
   128 crop is taken back out. That restores the paper's magnitudes honestly and makes
   its random translated crop reproducible. The cost -- the only real deviation left --
   is that the outer ring is *mirrored tissue rather than real CT*. Note the last row
   above: even the paper's own 180 margin does not cover maximum rotation combined with
   scale 0.8, so residual fill in that corner is what the paper had too; ours is
   mirrored tissue where theirs was zeros.

   The padding is never materialized. It is folded into the source-coordinate
   arithmetic and realized by sampling the original array with :data:`BORDER_MODE`,
   which keeps the whole operation to a single interpolation per array.

**Image and mask go through the same coordinates**, sampled bilinearly (``order=1``) and
nearest-neighbour (``order=0``) respectively. Nearest-neighbour is what keeps the mask
strictly binary *by construction*: it can only ever return a value already present in the
source. Bilinear for the image is a deliberate divergence from the reference's default
cubic ``order_data=3``, which overshoots outside ``[0, 1]`` and would break the dataset's
range invariant.

**Randomness is derived, never global.** Every draw comes from
``np.random.default_rng([seed, epoch, position])``, mirroring
:meth:`probunet.data.lidc.LidcDataset.set_epoch`. That makes augmentation reproducible
from the run seed, identical with or without DataLoader workers, correct across a resume
(the epoch is restored), and -- importantly -- independent of the global torch RNG, which
the model's latent sampling draws from. Coupling the two would make an augmentation
change silently perturb the ``z`` sequence.

**Tiny lesions are guarded.** Under ``order=0`` a very small mask can fail to receive any
output pixel and vanish, turning a non-empty target into an empty one. That would corrupt
the ambiguity buckets with no signal in the loss curve, so a transform that empties a
non-empty mask is redrawn up to :attr:`AugmentationConfig.max_redraws` times and, failing
that, the sample is returned untransformed. Both events are counted in
:class:`AugmentationStats` and logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

PAPER_TILE_PX: Final[int] = 180
"""Tile size the paper augments at, before its random crop down to 128 (Appendix H.1)."""

DEFAULT_AUGMENTATION_SEED: Final[int] = 5034
"""Seed for the augmentation draws.

Deliberately distinct from the split seed (1806, frozen forever) and the grader-pairing
seed (2018), so the three sources of randomness cannot alias one another. Overridden by
the training config.
"""

BORDER_MODE: Final[str] = "mirror"
"""How coordinates outside the source tile are resolved.

``scipy``'s ``"mirror"`` is whole-sample symmetric reflection, i.e. exactly
``numpy.pad(..., mode="reflect")`` -- it does not duplicate the edge pixel the way
scipy's own ``"reflect"`` does. Applied identically to image and mask, so the outer ring
is mirrored tissue carrying its own mirrored label rather than mislabelled data.
"""


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for the training-split augmentation.

    Defaults reproduce Appendix H.1 as closely as pre-cropped 128x128 data allows. See
    the module docstring for the provenance of every value: some are the paper's, some
    are transferred from the authors' released Cityscapes configuration because they are
    resolution-independent, and some are ours because neither source specifies them.

    Attributes:
        enabled: Master switch. When False, augmentation is a bit-exact no-op.
        seed: Seed for the per-sample draws.
        pad_to_px: Frame the transform is applied in, reflect-padded up from the data's
            own size. The paper's 180 restores the margin its magnitudes assume.
        rotation_degrees: In-plane rotation is drawn from ``+-rotation_degrees``.
        scale_range: Content scale factor, drawn uniformly. Below 1.0 shrinks the
            content, which is what consumes border margin.
        shear: Shear factor, drawn from ``+-shear``. Ours: absent from the paper and
            absent from ``batchgenerators`` entirely.
        elastic_alpha_px: Peak elastic displacement in pixels. The actual strength is
            drawn from ``(0, elastic_alpha_px)``, so the range includes the identity --
            the mechanism the reference uses to make its transforms stochastic without a
            separate per-sample probability.
        elastic_sigma_px: Smoothing scale of the elastic displacement field, in pixels.
        random_crop: Draw the 128 crop at a random offset in the padded frame, as the
            paper does. When False the crop is centred, which makes an identity
            transform an exact no-op and is useful in tests.
        max_redraws: How many times to redraw a transform that empties a non-empty
            mask before giving up and returning the sample untransformed.

    Raises:
        ValueError: If a value is out of range, or ``pad_to_px`` is too small to hold
            the data (which would make the crop impossible).
    """

    enabled: bool = False
    seed: int = DEFAULT_AUGMENTATION_SEED
    pad_to_px: int = PAPER_TILE_PX
    rotation_degrees: float = 22.5
    scale_range: tuple[float, float] = (0.8, 1.2)
    shear: float = 0.1
    elastic_alpha_px: float = 5.0
    elastic_sigma_px: float = 10.0
    random_crop: bool = True
    max_redraws: int = 3

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.pad_to_px <= 0:
            raise ValueError(f"pad_to_px must be positive, got {self.pad_to_px}")
        if self.rotation_degrees < 0:
            raise ValueError(
                f"rotation_degrees must be non-negative, got {self.rotation_degrees}"
            )
        if self.shear < 0:
            raise ValueError(f"shear must be non-negative, got {self.shear}")
        if len(self.scale_range) != 2:
            raise ValueError(f"scale_range must hold two values, got {self.scale_range}")
        low, high = self.scale_range
        if not 0 < low <= high:
            raise ValueError(f"scale_range must satisfy 0 < low <= high, got {self.scale_range}")
        if self.elastic_alpha_px < 0:
            raise ValueError(
                f"elastic_alpha_px must be non-negative, got {self.elastic_alpha_px}"
            )
        if self.elastic_alpha_px > 0 and self.elastic_sigma_px <= 0:
            raise ValueError(
                f"elastic_sigma_px must be positive when elastic deformation is on, "
                f"got {self.elastic_sigma_px}"
            )
        if self.max_redraws < 0:
            raise ValueError(f"max_redraws must be non-negative, got {self.max_redraws}")

    def margin_px(self, size_px: int) -> int:
        """Pixels of reflect-padding added on each side of a tile.

        Args:
            size_px: Side length of the data tile.

        Returns:
            The per-side pad width.

        Raises:
            ValueError: If ``pad_to_px`` is smaller than ``size_px``.
        """
        if self.pad_to_px < size_px:
            raise ValueError(
                f"pad_to_px={self.pad_to_px} is smaller than the data tile {size_px}; "
                "the random crop needs a frame at least as large as its output"
            )
        return (self.pad_to_px - size_px) // 2


@dataclass(frozen=True)
class SampledTransform:
    """The concrete geometry drawn for one sample.

    Exposed rather than kept internal so tests can assert determinism and inspect the
    individual components instead of only the resampled output.

    Attributes:
        angle_rad: In-plane rotation, radians.
        shear: Shear factor.
        scale: Content scale factor.
        crop_offset: ``(row, col)`` offset of the output crop within the padded frame.
        backward: 2x2 matrix mapping *output* offsets to *source* offsets, both relative
            to the padded frame's centre. It is the inverse of the visual transform,
            because resampling is a backward mapping.
        displacement: ``(2, pad, pad)`` elastic displacement field, or None when this
            sample drew no deformation.
    """

    angle_rad: float
    shear: float
    scale: float
    crop_offset: tuple[int, int]
    backward: np.ndarray
    displacement: np.ndarray | None


@dataclass(frozen=True)
class AugmentationOutcome:
    """What happened while augmenting one sample.

    Attributes:
        redraws: Transforms discarded because they emptied a non-empty mask.
        lesion_lost: True if every redraw failed and the sample was returned
            untransformed. Must be rare; if it is not, the magnitudes are too aggressive
            for this dataset's lesion sizes.
        augmented: False when the sample was returned untransformed.
    """

    redraws: int = 0
    lesion_lost: bool = False
    augmented: bool = True


@dataclass
class AugmentationStats:
    """Mutable per-epoch tallies, so augmentation cannot fail silently.

    A non-empty mask that a transform empties is invisible in the loss curve but
    corrupts the ambiguity buckets, so the rate is logged as a first-class metric rather
    than trusted to be zero.

    Note:
        These counters live in the process that runs ``__getitem__``. With
        ``num_workers > 0`` each worker keeps its own copy and the parent sees zeros;
        :class:`probunet.data.lidc.DataConfig` warns about that combination. Every
        shipped config uses ``num_workers: 0``.

    Attributes:
        samples: Samples passed through augmentation.
        redraws: Total transforms discarded across all samples.
        lesion_lost: Samples returned untransformed after exhausting the redraws.
    """

    samples: int = 0
    redraws: int = 0
    lesion_lost: int = 0

    def record(self, outcome: AugmentationOutcome) -> None:
        """Fold one sample's outcome into the tallies.

        Args:
            outcome: The outcome returned by :func:`augment_pair`.
        """
        self.samples += 1
        self.redraws += outcome.redraws
        self.lesion_lost += int(outcome.lesion_lost)

    def reset(self) -> None:
        """Zero the tallies, ready for the next epoch."""
        self.samples = 0
        self.redraws = 0
        self.lesion_lost = 0

    def as_metrics(self) -> dict[str, float]:
        """Render the tallies as loggable scalars.

        Returns:
            A mapping with the sample count and the two rates. Empty when no sample was
            augmented, so a non-augmenting run logs nothing rather than a misleading
            zero.
        """
        if not self.samples:
            return {}
        return {
            "aug_samples": float(self.samples),
            "aug_redraw_rate": self.redraws / self.samples,
            "aug_lesion_lost_fraction": self.lesion_lost / self.samples,
        }


def _elastic_displacement(
    config: AugmentationConfig, frame_px: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Draw a smooth elastic displacement field over the padded frame.

    Strength is drawn from ``(0, elastic_alpha_px)`` so the range includes the identity,
    which is how the reference makes its spatial transforms stochastic without a separate
    per-sample probability. Normalization is by peak absolute displacement, which is why
    the parameter is expressible in pixels; the reference normalizes by an L2 norm whose
    magnitude depends on the array size, and that is precisely why its ``alpha=(0, 800)``
    cannot be transferred to a different resolution.

    Args:
        config: Augmentation settings.
        frame_px: Side length of the padded frame.
        rng: The sample's generator.

    Returns:
        A ``(2, frame_px, frame_px)`` field of row/column displacements in pixels, or
        None when this draw produced no deformation.
    """
    alpha = float(rng.uniform(0.0, config.elastic_alpha_px))
    if alpha <= 0.0:
        return None
    noise = rng.uniform(-1.0, 1.0, size=(2, frame_px, frame_px))
    smoothed = np.stack(
        [
            gaussian_filter(noise[axis], config.elastic_sigma_px, mode="constant", cval=0.0)
            for axis in range(2)
        ]
    )
    peak = float(np.abs(smoothed).max())
    if peak <= 0.0:
        return None
    return smoothed / peak * alpha


def _backward_matrix(angle_rad: float, shear: float, scale: float) -> np.ndarray:
    """Build the backward-mapping matrix for a rotation, shear and scale.

    The visual (forward) transform is ``F = R @ H @ S`` in ``(row, col)`` coordinates.
    Resampling needs the inverse, since it asks "which source pixel does this output
    pixel come from".

    Args:
        angle_rad: Rotation in radians.
        shear: Shear factor along the column axis.
        scale: Content scale factor; below 1.0 shrinks the content.

    Returns:
        The 2x2 inverse of ``F``.
    """
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    shear_matrix = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float64)
    scale_matrix = np.array([[scale, 0.0], [0.0, scale]], dtype=np.float64)
    return np.linalg.inv(rotation @ shear_matrix @ scale_matrix)


def sample_transform(
    config: AugmentationConfig, size_px: int, rng: np.random.Generator
) -> SampledTransform:
    """Draw one sample's geometry.

    Every range is symmetric about, or starts at, the identity, so no separate
    per-transform probability is needed -- the same mechanism the reference relies on.

    Args:
        config: Augmentation settings.
        size_px: Side length of the data tile, e.g. 128.
        rng: The sample's generator.

    Returns:
        The drawn :class:`SampledTransform`.
    """
    margin = config.margin_px(size_px)
    span = config.pad_to_px - size_px

    angle_rad = float(np.deg2rad(rng.uniform(-config.rotation_degrees, config.rotation_degrees)))
    shear = float(rng.uniform(-config.shear, config.shear))
    scale = float(rng.uniform(*config.scale_range))

    if config.random_crop and span > 0:
        # Uniform over the full valid range, matching the reference's
        # rand_crop_dist = patch_size / 2 (which lets the crop centre reach anywhere at
        # least half a patch from the padded border).
        crop_offset = (int(rng.integers(0, span + 1)), int(rng.integers(0, span + 1)))
    else:
        crop_offset = (margin, margin)

    return SampledTransform(
        angle_rad=angle_rad,
        shear=shear,
        scale=scale,
        crop_offset=crop_offset,
        backward=_backward_matrix(angle_rad, shear, scale),
        displacement=_elastic_displacement(config, config.pad_to_px, rng),
    )


def source_coordinates(
    transform: SampledTransform, size_px: int, config: AugmentationConfig
) -> np.ndarray:
    """Map each output pixel to a coordinate in the original, unpadded tile.

    The chain, per output pixel ``(i, j)``:

    1. Into the padded frame: ``(i + oy, j + ox)`` for the drawn crop offset.
    2. Elastic: add the displacement field sampled at that padded location.
    3. Affine: apply the backward matrix about the **padded frame's** centre, which is
       where the paper's transform is centred -- this is what makes the 180 margin do its
       job.
    4. Back to the original tile: subtract the per-side pad width.

    The result is deliberately allowed to fall outside the tile; :data:`BORDER_MODE`
    resolves those coordinates as mirrored tissue, which is what stands in for the real
    CT the paper had out to 180 pixels.

    Args:
        transform: The drawn geometry.
        size_px: Side length of the data tile.
        config: Augmentation settings.

    Returns:
        A ``(2, size_px, size_px)`` array of ``(row, col)`` source coordinates, ready for
        ``scipy.ndimage.map_coordinates``.
    """
    margin = config.margin_px(size_px)
    centre = (config.pad_to_px - 1) / 2.0
    offset_row, offset_col = transform.crop_offset

    grid = np.mgrid[0:size_px, 0:size_px].astype(np.float64)
    rows = grid[0] + offset_row
    cols = grid[1] + offset_col

    if transform.displacement is not None:
        window = (
            slice(offset_row, offset_row + size_px),
            slice(offset_col, offset_col + size_px),
        )
        rows = rows + transform.displacement[0][window]
        cols = cols + transform.displacement[1][window]

    local_rows, local_cols = rows - centre, cols - centre
    matrix = transform.backward
    source_rows = matrix[0, 0] * local_rows + matrix[0, 1] * local_cols + centre - margin
    source_cols = matrix[1, 0] * local_rows + matrix[1, 1] * local_cols + centre - margin
    return np.stack([source_rows, source_cols])


def apply_transform(
    image: np.ndarray, mask: np.ndarray, transform: SampledTransform, config: AugmentationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an image and its paired mask through one transform.

    Both go through the *same* coordinates, so the pair stays registered; only the
    interpolation order differs.

    Args:
        image: ``(H, W)`` float32 image in ``[0, 1]``.
        mask: ``(H, W)`` integer mask in ``{0, 1}``.
        transform: The drawn geometry.
        config: Augmentation settings.

    Returns:
        The transformed ``(image, mask)``, with the input dtypes preserved. The image
        stays within ``[0, 1]`` because bilinear interpolation of values in ``[0, 1]``
        cannot overshoot, and the mask stays in ``{0, 1}`` because nearest-neighbour
        sampling can only return values already present.
    """
    coordinates = source_coordinates(transform, image.shape[-1], config)
    warped_image = map_coordinates(image, coordinates, order=1, mode=BORDER_MODE)
    warped_mask = map_coordinates(mask, coordinates, order=0, mode=BORDER_MODE)
    return warped_image.astype(image.dtype, copy=False), warped_mask.astype(
        mask.dtype, copy=False
    )


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: AugmentationConfig,
    epoch: int,
    position: int,
) -> tuple[np.ndarray, np.ndarray, AugmentationOutcome]:
    """Augment one (image, grader mask) pair.

    Called *after* the random image-grader pairing, so the posterior net sees a
    consistent augmented pair -- the order the paper states ("image-grader pairs are
    drawn randomly. We apply augmentations to the image tiles").

    The generator is derived from ``(seed, epoch, position)``, which makes the result
    reproducible from the run seed, stable across DataLoader workers, and correct after a
    resume, without consuming the global torch RNG the model samples ``z`` from.

    A transform that empties a non-empty mask is discarded and redrawn: the smallest
    lesion in this dataset is a single pixel, and under nearest-neighbour sampling a
    region that small can simply receive no output pixel. Silently emptying it would
    corrupt the ambiguity buckets with no trace in the loss.

    Args:
        image: ``(H, W)`` float32 image in ``[0, 1]``.
        mask: ``(H, W)`` integer mask in ``{0, 1}``.
        config: Augmentation settings. Must have ``enabled=True``.
        epoch: Current epoch, so the augmentation is redrawn each epoch.
        position: Index within the split, so samples differ from one another.

    Returns:
        ``(image, mask, outcome)``. On an unrecoverable draw the inputs are returned
        untransformed (as copies) and ``outcome.lesion_lost`` is True.

    Raises:
        ValueError: If ``config.enabled`` is False, or the tile is not square. A
            disabled config reaching here would mean the caller believes augmentation is
            active when it is not.
    """
    if not config.enabled:
        raise ValueError(
            "augment_pair called with a disabled AugmentationConfig; the caller should "
            "skip augmentation entirely rather than pass a disabled config"
        )
    if image.shape[-1] != image.shape[-2]:
        raise ValueError(f"expected a square tile, got {image.shape}")

    rng = np.random.default_rng([config.seed, epoch, position])
    had_lesion = bool(mask.any())

    for attempt in range(config.max_redraws + 1):
        transform = sample_transform(config, image.shape[-1], rng)
        warped_image, warped_mask = apply_transform(image, mask, transform, config)
        if not had_lesion or warped_mask.any():
            return warped_image, warped_mask, AugmentationOutcome(redraws=attempt)

    # Every draw lost the lesion. Returning the pair untransformed keeps the target
    # honest; the counter is what makes the event visible.
    return (
        image.copy(),
        mask.copy(),
        AugmentationOutcome(
            redraws=config.max_redraws, lesion_lost=True, augmented=False
        ),
    )
