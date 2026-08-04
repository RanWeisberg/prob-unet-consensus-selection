"""Dataset and DataLoaders over the converted LIDC-IDRI patches.

Reads ``data/processed/lidc.npz`` and takes its splits from ``data/splits/split.json``
via :func:`probunet.data.splits.load_split`. The split is never regenerated here.

Two modes, because training and evaluation need different things from the same data:

* ``"train"`` pairs each image with **one randomly chosen** grader mask, redrawn every
  epoch. This random pairing is what teaches the prior to cover the space of
  plausible variants; the four masks are never averaged or merged.
* ``"eval"`` returns **all four** grader masks, which the GED and oracle metrics need.

Empty masks are valid data and are never filtered: an empty mask is a grader's
genuine judgment that there is no lesion, and the model must be able to reproduce
lesion-absence samples. Uniform sampling over all four graders therefore includes the
empty ones.

Dtype contract, kept deliberately narrow (masks are stored as ``uint8``):

============================  ==========  ==================================
consumer                      dtype       why
============================  ==========  ==================================
train sample ``mask``         ``int64``   ``nn.CrossEntropyLoss`` class indices
eval sample ``masks``         ``uint8``   metrics convert to bool themselves
``image``                     ``float32`` as stored; MPS has no float64
posterior input               ``float32`` cast inside ``PosteriorNet``
============================  ==========  ==================================

**No normalization is applied.** The images are already in [0, 1] (see the Stage 0
findings) and the paper specifies no further normalization, so they are passed through
unchanged. :class:`DataConfig` exposes a ``normalization`` hook that *raises* rather than
silently no-ops, so a future phase cannot mistake "off" for "on".

**Augmentation is applied to the training split only**, and only when
``data.augmentation.enabled`` is set. Validation and test are returned untransformed in
every configuration, so a val/test number is never a function of the augmentation
settings. See :mod:`probunet.data.transforms` for the paper's specification and for what
we could and could not reproduce.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from probunet.data.splits import (
    DEFAULT_SPLIT_PATH,
    N_GRADERS,
    SPLIT_NAMES,
    load_split,
    nonempty_masks_per_patch,
)
from probunet.data.transforms import (
    AugmentationConfig,
    AugmentationStats,
    augment_pair,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_NPZ_PATH = Path("data/processed/lidc.npz")

DEFAULT_PAIRING_SEED: int = 2018
"""Seed for the epoch-varying grader pairing.

Deliberately different from the split seed (1806). The split seed is frozen forever;
this one is free to vary per run and is overridden by the training config.
"""

Mode = Literal["train", "eval"]
MODES: tuple[Mode, Mode] = ("train", "eval")

REQUIRED_ARRAYS = ("images", "masks", "series_uid")
IMAGE_MIN, IMAGE_MAX = 0.0, 1.0
_RANGE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DataConfig:
    """Configuration for building the LIDC datasets and loaders.

    Attributes:
        npz_path: Converted dataset.
        split_path: Fixed split file.
        batch_size: Batch size for every loader.
        num_workers: DataLoader workers. Defaults to 0; see :func:`build_data` for
            why more is rarely useful here.
        pin_memory: CUDA-only optimization; leave False on MPS and CPU.
        pairing_seed: Seed for the random grader pairing.
        normalization: Must be ``"none"``; the paper specifies none.
        augmentation: Training-split augmentation. Disabled by default; enabled in
            ``configs/baseline.yaml``, which is the paper's own configuration.
        validate_on_load: Check dtypes, ranges and binarity when loading.
        drop_last: Drop a trailing partial training batch.

    Raises:
        NotImplementedError: If ``normalization`` is not ``"none"``. It is a deliberate
            hook for the modernization phase; failing loudly stops a future caller from
            believing it is active when nothing applies it.
        ValueError: If ``batch_size`` or ``num_workers`` is invalid.
    """

    npz_path: Path = DEFAULT_NPZ_PATH
    split_path: Path = DEFAULT_SPLIT_PATH
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    pairing_seed: int = DEFAULT_PAIRING_SEED
    normalization: str = "none"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    validate_on_load: bool = True
    drop_last: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration, refusing silent no-ops."""
        if self.normalization != "none":
            raise NotImplementedError(
                f"normalization={self.normalization!r} is not implemented in the "
                "baseline phase. Images are already in [0, 1] and the paper "
                "specifies no further normalization. Add it as a flag-gated change "
                "in the modernization phase."
            )
        if self.augmentation.enabled and self.num_workers > 0:
            # The augmentation counters live in whichever process runs __getitem__, so
            # with workers the parent would read zeros and a lesion-loss problem would
            # look like a clean run. The augmentation itself is still correct.
            LOGGER.warning(
                "num_workers=%d with augmentation enabled: the augmentation counters "
                "(aug_lesion_lost_fraction, aug_redraw_rate) are accumulated per worker "
                "and will read as zero in the training log. Use num_workers=0 to see "
                "them.",
                self.num_workers,
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")


def nonempty_grader_counts(
    masks: np.ndarray, indices: np.ndarray | None = None
) -> np.ndarray:
    """Count non-empty grader masks per patch, optionally for a subset of patches.

    The ambiguity bucket of a patch -- how many of its four graders saw a lesion -- is
    needed in three places: the split stratification, per-bucket reporting in
    evaluation, and the consensus selection head, which must be able to slice its
    results by ambiguity. This is the shared primitive.

    Args:
        masks: Array of shape ``(N, 4, H, W)``.
        indices: Optional patch indices. When given, the returned array is aligned
            with ``indices`` rather than with ``masks``.

    Returns:
        Integer array of counts in ``0..4``, of length ``N`` or ``len(indices)``.
    """
    if indices is None:
        return nonempty_masks_per_patch(masks)
    indices = np.asarray(indices, dtype=np.int64)
    selected = masks[indices]
    return nonempty_masks_per_patch(selected)


def group_by_bucket(
    counts: np.ndarray, indices: np.ndarray | None = None
) -> dict[int, np.ndarray]:
    """Group patches by ambiguity bucket.

    Args:
        counts: Per-patch non-empty grader counts, as returned by
            :func:`nonempty_grader_counts`.
        indices: The patch indices ``counts`` was computed for. When omitted, the
            positions ``0..len(counts)-1`` are used.

    Returns:
        A mapping from bucket ``0..4`` to the indices in that bucket. Buckets with no
        members map to an empty array, so callers can iterate all five
        unconditionally.
    """
    counts = np.asarray(counts)
    base = np.arange(counts.size) if indices is None else np.asarray(indices)
    return {bucket: base[counts == bucket] for bucket in range(N_GRADERS + 1)}


@dataclass
class LidcArrays:
    """The converted dataset held in memory once and shared by every split.

    Roughly 1.9 GiB for the real data (989 MiB of float32 images plus 989 MiB of
    uint8 masks). Datasets hold *indices* into these arrays rather than slices,
    because fancy-indexing them would copy and multiply the memory by the number of
    splits.

    Attributes:
        images: ``(N, H, W)`` float32 in [0, 1].
        masks: ``(N, 4, H, W)`` uint8 in {0, 1}.
        series_uid: ``(N,)`` DICOM series UIDs.
        keys: ``(N,)`` original pickle keys, if the file carried them.
        source_index: ``(N,)`` row index each patch had in the FULL dataset, present
            only in a subset export. Its presence is what lets the same code address a
            subset and the full dataset by the same global indices.
    """

    images: np.ndarray
    masks: np.ndarray
    series_uid: np.ndarray
    keys: np.ndarray | None = None
    source_index: np.ndarray | None = None
    _nonempty_counts: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        """Number of patches."""
        return int(self.images.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """``(H, W)`` of the crops."""
        return int(self.images.shape[-2]), int(self.images.shape[-1])

    @classmethod
    def load(cls, npz_path: Path = DEFAULT_NPZ_PATH, validate: bool = True) -> LidcArrays:
        """Load the converted dataset.

        Args:
            npz_path: Path to ``lidc.npz``.
            validate: Run :meth:`validate` after loading.

        Returns:
            The loaded arrays.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If a required array is absent, or validation fails.
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found. Convert it once with: "
                "python scratch/convert_data.py"
            )
        with np.load(npz_path, allow_pickle=False) as handle:
            missing = [name for name in REQUIRED_ARRAYS if name not in handle.files]
            if missing:
                raise ValueError(
                    f"{npz_path} is missing required array(s) {missing}; "
                    f"found {sorted(handle.files)}"
                )
            arrays = cls(
                images=handle["images"],
                masks=handle["masks"],
                series_uid=handle["series_uid"],
                keys=handle["keys"] if "keys" in handle.files else None,
                source_index=(
                    handle["source_index"] if "source_index" in handle.files else None
                ),
            )
        if validate:
            arrays.validate()
        return arrays

    def validate(self) -> None:
        """Check shapes, dtypes, ranges and mask binarity, failing loudly.

        Raises:
            ValueError: On any violation, naming the offending values. Masks in
                particular must be strictly binary: a stray 255 (a 0/255 encoding)
                or a soft probability map would otherwise sail through and corrupt
                every loss and metric downstream.
        """
        if self.images.ndim != 3:
            raise ValueError(f"images must be (N, H, W), got {self.images.shape}")
        if self.masks.ndim != 4 or self.masks.shape[1] != N_GRADERS:
            raise ValueError(
                f"masks must be (N, {N_GRADERS}, H, W), got {self.masks.shape}"
            )
        if self.masks.shape[0] != self.images.shape[0]:
            raise ValueError(
                f"images/masks length mismatch: {self.images.shape[0]} vs "
                f"{self.masks.shape[0]}"
            )
        if self.masks.shape[-2:] != self.images.shape[-2:]:
            raise ValueError(
                f"images/masks spatial mismatch: {self.images.shape[-2:]} vs "
                f"{self.masks.shape[-2:]}"
            )
        if self.series_uid.shape[0] != self.images.shape[0]:
            raise ValueError(
                f"series_uid length {self.series_uid.shape[0]} != "
                f"{self.images.shape[0]}"
            )

        if self.images.dtype != np.float32:
            raise ValueError(f"images must be float32, got {self.images.dtype}")
        if not np.isfinite(self.images).all():
            raise ValueError("images contain non-finite values")
        low, high = float(self.images.min()), float(self.images.max())
        if low < IMAGE_MIN - _RANGE_TOLERANCE or high > IMAGE_MAX + _RANGE_TOLERANCE:
            raise ValueError(
                f"images outside [{IMAGE_MIN}, {IMAGE_MAX}]: found [{low}, {high}]"
            )

        if not np.issubdtype(self.masks.dtype, np.integer):
            raise ValueError(
                f"masks must have an integer dtype, got {self.masks.dtype}; a "
                "floating dtype suggests soft labels, which the baseline does not "
                "support"
            )
        mask_min, mask_max = int(self.masks.min()), int(self.masks.max())
        if mask_min < 0 or mask_max > 1:
            raise ValueError(
                f"masks must be strictly binary {{0, 1}}, found values in "
                f"[{mask_min}, {mask_max}] (0/255 encoding?)"
            )

    @property
    def is_subset(self) -> bool:
        """Whether this file is a subset export rather than the full dataset."""
        return self.source_index is not None

    def resolve_indices(self, global_indices: np.ndarray) -> np.ndarray:
        """Translate full-dataset row indices into rows of this file.

        For the full dataset this is the identity. For a subset export it maps through
        ``source_index``, so a caller holding indices from ``diagnostic_indices.json``
        addresses the same patches whichever file is loaded -- which is what keeps the
        qualitative panel free of a notebook-specific code path.

        Args:
            global_indices: Row indices in the full dataset.

        Returns:
            Row indices into this file's arrays.

        Raises:
            KeyError: If a requested patch is absent from this subset.
        """
        global_indices = np.asarray(global_indices, dtype=np.int64)
        if self.source_index is None:
            return global_indices
        lookup = {int(source): row for row, source in enumerate(self.source_index)}
        missing = [int(i) for i in global_indices if int(i) not in lookup]
        if missing:
            raise KeyError(
                f"patches {missing[:5]} are not in this subset export; load the full "
                "dataset or re-export the subset with those indices"
            )
        return np.array([lookup[int(i)] for i in global_indices], dtype=np.int64)

    def nonempty_counts(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Non-empty grader counts, computed once and cached.

        Args:
            indices: Optional patch indices to select.

        Returns:
            Counts in ``0..4``, aligned with ``indices`` when given.
        """
        if self._nonempty_counts is None:
            self._nonempty_counts = nonempty_masks_per_patch(self.masks)
        if indices is None:
            return self._nonempty_counts
        return self._nonempty_counts[np.asarray(indices, dtype=np.int64)]


class LidcDataset(Dataset):
    """One split of the LIDC patches, in either training or evaluation mode."""

    def __init__(
        self,
        arrays: LidcArrays,
        indices: np.ndarray,
        mode: Mode = "train",
        pairing_seed: int = DEFAULT_PAIRING_SEED,
        epoch: int = 0,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        """Build the dataset.

        Args:
            arrays: The shared in-memory arrays.
            indices: Patch indices belonging to this split.
            mode: ``"train"`` for one random grader mask, ``"eval"`` for all four.
            pairing_seed: Seed for the grader pairing.
            epoch: Initial epoch, so a train dataset is usable before the first
                :meth:`set_epoch` call.
            augmentation: Augmentation settings. Only meaningful in ``"train"`` mode;
                passing an *enabled* config with ``mode="eval"`` is an error rather than
                a silent no-op, because a transformed validation set would invalidate
                every checkpoint-selection decision.

        Raises:
            ValueError: If ``mode`` is unknown, ``indices`` are out of range, or
                augmentation is enabled on an evaluation dataset.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if augmentation is not None and augmentation.enabled and mode != "train":
            raise ValueError(
                f"augmentation is enabled on a mode={mode!r} dataset. Only the training "
                "split may be augmented: validation and test must stay untransformed or "
                "their numbers stop being comparable across configurations."
            )
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self.pairing_seed = pairing_seed
        self.epoch = epoch
        self.augmentation = augmentation
        self.aug_stats = AugmentationStats()
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= len(arrays)
        ):
            raise ValueError(
                f"indices out of range for {len(arrays)} patches: "
                f"[{self.indices.min()}, {self.indices.max()}]"
            )
        self._graders: np.ndarray | None = None
        if mode == "train":
            self.set_epoch(epoch)

    @property
    def augmenting(self) -> bool:
        """Whether this dataset actually transforms its samples."""
        return (
            self.mode == "train"
            and self.augmentation is not None
            and self.augmentation.enabled
        )

    def __len__(self) -> int:
        """Number of patches in this split."""
        return int(self.indices.size)

    def set_epoch(self, epoch: int) -> None:
        """Redraw the grader pairing for a new epoch.

        The whole epoch's assignment is drawn in one vectorized call, seeded by
        ``(pairing_seed, epoch)``. That makes it reproducible from the run seed, fresh
        every epoch, and safe with DataLoader workers -- workers are forked after this
        runs, so each inherits the same finished array. It does mean
        ``persistent_workers`` must stay False, since persistent workers would never
        observe a later call; :func:`build_data` never enables it.

        Sampling is **uniform over all four graders**, including those whose mask is
        empty. Skipping empties would remove exactly the lesion-absence cases the
        model is supposed to learn to reproduce.

        Args:
            epoch: Epoch index.

        Raises:
            ValueError: If called on an evaluation dataset, which has no pairing.
        """
        if self.mode != "train":
            raise ValueError("set_epoch is only meaningful for mode='train'")
        self.epoch = epoch
        generator = np.random.default_rng([self.pairing_seed, epoch])
        self._graders = generator.integers(0, N_GRADERS, size=len(self))

    @property
    def graders(self) -> np.ndarray:
        """The current epoch's grader assignment, one entry per patch.

        Raises:
            ValueError: If this is an evaluation dataset.
        """
        if self._graders is None:
            raise ValueError("no grader pairing: this is an evaluation dataset")
        return self._graders

    def nonempty_counts(self) -> np.ndarray:
        """Non-empty grader counts for this split, aligned with ``__getitem__`` order."""
        return self.arrays.nonempty_counts(self.indices)

    def buckets(self) -> dict[int, np.ndarray]:
        """Patch indices in this split grouped by ambiguity bucket."""
        return group_by_bucket(self.nonempty_counts(), self.indices)

    def __getitem__(self, position: int) -> dict[str, Tensor]:
        """Return one sample.

        Args:
            position: Position within this split, not a global patch index.

        Returns:
            In ``"train"`` mode: ``image`` (1, H, W) float32, ``mask`` (H, W) int64,
            ``grader`` scalar int64 and ``index`` scalar int64. In ``"eval"`` mode:
            ``image``, ``masks`` (4, H, W) uint8 and ``index``.
        """
        row = int(self.indices[position])
        sample: dict[str, Tensor] = {"index": torch.tensor(row, dtype=torch.int64)}

        if self.mode == "train":
            grader = int(self.graders[position])
            image_array = self.arrays.images[row]
            mask_array = self.arrays.masks[row, grader]
            if self.augmenting:
                # Augment AFTER the grader pairing, so the posterior net receives a
                # consistent (augmented image, augmented mask) pair. Both go through one
                # shared coordinate map inside augment_pair, which is what keeps them
                # registered. map_coordinates always allocates, so no .copy() is needed.
                assert self.augmentation is not None  # narrowed by self.augmenting
                image_array, mask_array, outcome = augment_pair(
                    image_array,
                    mask_array,
                    self.augmentation,
                    epoch=self.epoch,
                    position=position,
                )
                self.aug_stats.record(outcome)
            else:
                # .copy() detaches the sample from the shared arrays: torch.from_numpy
                # would otherwise return a view, and an in-place edit downstream would
                # silently corrupt the dataset for every other split.
                image_array, mask_array = image_array.copy(), mask_array.copy()
            sample["image"] = torch.from_numpy(image_array).unsqueeze(0)
            # int64 because nn.CrossEntropyLoss takes class indices, not one-hot.
            sample["mask"] = torch.from_numpy(mask_array.astype(np.int64))
            sample["grader"] = torch.tensor(grader, dtype=torch.int64)
        else:
            # Never augmented, in any configuration: see __init__.
            sample["image"] = torch.from_numpy(self.arrays.images[row].copy()).unsqueeze(0)
            # uint8 as stored: the metrics convert to bool themselves, and an
            # intermediate float cast would be pure waste.
            sample["masks"] = torch.from_numpy(self.arrays.masks[row].copy())
        return sample


def _worker_init_fn(worker_id: int) -> None:
    """Seed per-worker RNGs deterministically.

    The dataset itself draws no per-item randomness (the epoch's pairing is decided up
    front), so this is belt and braces for anything added later.

    Args:
        worker_id: Worker index, supplied by the DataLoader.
    """
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class LidcData:
    """Datasets and loaders for all three splits.

    Attributes:
        arrays: The shared in-memory arrays.
        datasets: Split name to dataset.
        loaders: Split name to DataLoader.
        config: The configuration used.
    """

    arrays: LidcArrays
    datasets: dict[str, LidcDataset]
    loaders: dict[str, DataLoader]
    config: DataConfig

    def set_epoch(self, epoch: int) -> None:
        """Redraw the training pairing, and the augmentation draw, for a new epoch.

        Args:
            epoch: Epoch index.
        """
        self.datasets["train"].set_epoch(epoch)

    def augmentation_metrics(self, reset: bool = True) -> dict[str, float]:
        """Collect and optionally clear the training split's augmentation counters.

        Args:
            reset: Zero the counters afterwards, so each epoch reports its own rates.

        Returns:
            Augmentation scalars, or an empty mapping when nothing was augmented.
        """
        train = self.datasets["train"]
        metrics = train.aug_stats.as_metrics()
        if reset:
            train.aug_stats.reset()
        return metrics

    def nonempty_counts(self, indices: np.ndarray) -> np.ndarray:
        """Ambiguity bucket per patch for arbitrary global indices.

        Evaluation collects the ``index`` field from each batch and calls this to
        report metrics per bucket.

        Args:
            indices: Global patch indices.

        Returns:
            Counts in ``0..4``.
        """
        return self.arrays.nonempty_counts(indices)


def build_data(config: DataConfig | None = None) -> LidcData:
    """Build datasets and loaders for train, val and test.

    ``train`` is in training mode (one random grader mask per epoch); ``val`` and
    ``test`` are in evaluation mode and expose all four masks. Validation *loss* is
    therefore computed by the training script over all four graders, which keeps it
    deterministic across epochs rather than depending on a pairing draw.

    On ``num_workers``: 0 is the default and usually the fastest choice here. The
    arrays are already resident in memory, so per-item work is a slice plus a cast and
    worker overhead dominates. More importantly, macOS and Windows spawn rather than
    fork worker processes, so each worker would receive its own ~1.9 GiB copy of the
    arrays; only Linux's copy-on-write fork makes ``num_workers > 0`` cheap.

    Args:
        config: Configuration; defaults to :class:`DataConfig`.

    Returns:
        The assembled :class:`LidcData`.
    """
    config = config or DataConfig()
    arrays = LidcArrays.load(config.npz_path, validate=config.validate_on_load)
    split = load_split(config.split_path, expected_n_patches=len(arrays))

    datasets = {
        name: LidcDataset(
            arrays,
            split.indices[name],
            mode="train" if name == "train" else "eval",
            pairing_seed=config.pairing_seed,
            # Train only. Val and test are constructed with no augmentation at all, so
            # there is no code path by which a transform could reach them.
            augmentation=config.augmentation if name == "train" else None,
        )
        for name in SPLIT_NAMES
    }

    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        shuffle = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            # Reproducible shuffling: the generator advances across epochs, so each
            # epoch sees a different order that still replays from the same seed.
            generator=(
                torch.Generator().manual_seed(config.pairing_seed) if shuffle else None
            ),
            num_workers=config.num_workers,
            # pin_memory only helps CUDA host-to-device copies; it is a no-op that
            # warns on MPS.
            pin_memory=config.pin_memory,
            drop_last=config.drop_last and shuffle,
            # Never True: persistent workers would keep a stale grader pairing from
            # before set_epoch(). Note also that on Windows (the RTX 3070 machine)
            # spawn re-imports __main__, so any script using num_workers > 0 must
            # guard its entry point with `if __name__ == "__main__":`.
            persistent_workers=False,
            worker_init_fn=_worker_init_fn if config.num_workers > 0 else None,
        )

    return LidcData(arrays=arrays, datasets=datasets, loaders=loaders, config=config)

def panel_batch(
    arrays: LidcArrays, global_indices: np.ndarray
) -> tuple[Tensor, Tensor]:
    """Load images and all four grader masks for a set of patches.

    The single path used for qualitative panels, by the training loop and by the
    notebook alike. It resolves ``global_indices`` through
    :meth:`LidcArrays.resolve_indices`, so it works unchanged against the full dataset or
    a subset export -- the choice is made by which ``npz_path`` the config points at,
    not by a separate code path.

    Args:
        arrays: The loaded dataset, full or subset.
        global_indices: Patch indices in the full dataset's numbering.

    Returns:
        An ``(image, masks)`` pair: float32 ``(B, 1, H, W)`` and uint8 ``(B, 4, H, W)``.
    """
    rows = arrays.resolve_indices(global_indices)
    images = torch.from_numpy(arrays.images[rows].copy()).unsqueeze(1)
    masks = torch.from_numpy(arrays.masks[rows].copy())
    return images, masks
