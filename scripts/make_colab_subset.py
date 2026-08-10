"""Build the tracked Colab demo subset and the split file that addresses it.

Runs **on the PC**, where ``data/processed/lidc.npz`` (454 MB, gitignored) is present, and
writes two small tracked files:

* ``data/processed/lidc_colab_demo.npz`` -- a few hundred patches in exactly the full
  dataset's format;
* ``data/splits/colab_demo_split.json`` -- a split whose indices address **that file**.

Together they let the submission notebook's Tier 2 train a tiny model from scratch on
Colab with nothing to download. ``configs/colab_demo.yaml`` points at both.

TEST CONTAMINATION IS THE PRIMARY RISK
--------------------------------------
Patches are drawn only from the ``train`` indices of ``data/splits/split.json``, plus a
small handful from ``val`` so the demo's validation loop and its ``val/total`` monitor have
something to run on. The exclusion is asserted on the **final selected indices**, not on
the candidate pool -- a pool-level check would pass while a later top-up or an off-by-one
still pulled a test patch in. The demo split file's test list is empty by construction.

THE INDEX RE-BASING TRAP
------------------------
``split.json`` indices address the full 15,096-patch array. Once the subset is a re-indexed
array of a few hundred rows those indices are wrong for it -- and wrong in the worst way,
because they are still valid integers addressing *some* patch. So this emits a separate
split file whose lists are indices **into the subset**, records every row's original
full-dataset index in the ``.npz``, and never touches ``split.json``.

FORMAT COMPATIBILITY
--------------------
The subset loads through the existing dataset class with **zero** loader changes: identical
keys, dtypes, layout and grader ordering, only fewer rows. Nothing is downcast --
``np.savez_compressed`` already takes the masks to 0.2% of raw. The script proves this
rather than claiming it: before exiting it reloads the file through
:class:`~probunet.data.lidc.LidcArrays`, reloads the split through
:func:`~probunet.data.splits.load_split`, builds the real loaders and pulls one batch from
each. If a loader change ever *were* needed, that is a finding to report, not something to
patch around here.

Usage::

    python scripts/make_colab_subset.py
    python scripts/make_colab_subset.py --train-patches 128 --val-patches 16 --seed 2018
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.data.colab_subset import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TRAIN_PATCHES,
    DEFAULT_VAL_PATCHES,
    DEMO_BUCKETS,
    SIZE_HARD_STOP_BYTES,
    SIZE_TARGET_BYTES,
    build_demo_split_document,
    render_report,
    select_demo_patches,
    sha256_file,
    size_arithmetic,
    write_demo_split,
    write_demo_subset,
)
from probunet.data.lidc import DataConfig, LidcArrays, build_data  # noqa: E402
from probunet.data.splits import load_split  # noqa: E402
from probunet.paths import (  # noqa: E402
    COLAB_DEMO_NPZ,
    COLAB_DEMO_SPLIT,
    FULL_NPZ,
    SPLIT_PATH,
)
from probunet.utils.runtime import git_revision  # noqa: E402

LOGGER = logging.getLogger("probunet.make_colab_subset")

SCHEMA_VERSION = 1
"""Bumped when a provenance key is renamed or removed."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=FULL_NPZ,
        help="the full converted dataset; gitignored, so this runs on the PC",
    )
    parser.add_argument(
        "--split", type=Path, default=SPLIT_PATH,
        help="the frozen main split, read for its train/val/test index lists",
    )
    parser.add_argument(
        "--train-patches", type=int, default=DEFAULT_TRAIN_PATCHES,
        help=(
            "demo train patches, spread evenly over the four ambiguity buckets. The "
            "default is 64 per bucket: at the MEASURED 30,349 compressed bytes/patch, "
            "256 + 32 = 288 patches project to 8.34 MiB, inside the 10 MiB target with "
            "16%% headroom (the target allows 345 patches, the hard stop 691)"
        ),
    )
    parser.add_argument(
        "--val-patches", type=int, default=DEFAULT_VAL_PATCHES,
        help="demo val patches, 8 per bucket; enough for the monitor to produce a number",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the draw")
    parser.add_argument("--out", type=Path, default=COLAB_DEMO_NPZ, help="output .npz")
    parser.add_argument(
        "--split-out", type=Path, default=COLAB_DEMO_SPLIT,
        help="output split file, whose indices address the SUBSET, not lidc.npz",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def verify_loads_unchanged(
    npz_path: Path, split_path: Path, selection_rows: np.ndarray
) -> dict[str, Any]:
    """Prove the emitted pair works through the real loader, rather than asserting it.

    Builds the actual :func:`~probunet.data.lidc.build_data` pipeline over the subset and
    pulls one batch from the train and val loaders. This is the check that would catch a
    format drift -- a renamed key, a changed dtype, a stale split -- at the moment it can
    still be fixed, instead of in a Colab session a grader is running.

    Args:
        npz_path: The written subset.
        split_path: The written demo split.
        selection_rows: The full-dataset index of every subset row, for a content check.

    Returns:
        A record of what was verified, for the console report.

    Raises:
        ValueError: If the subset does not present as a subset export, or a batch does not
            have the shape and dtype the trainer expects.
    """
    arrays = LidcArrays.load(npz_path)
    if not arrays.is_subset:
        raise ValueError(
            f"{npz_path} did not load as a subset export: its source_index array is "
            "missing, so nothing could map its rows back to the full dataset"
        )
    if not np.array_equal(arrays.source_index, selection_rows):
        raise ValueError(
            "the reloaded source_index does not match the selection that was written"
        )

    split = load_split(split_path, expected_n_patches=len(arrays))
    if split.indices["test"].size:
        raise ValueError(
            f"{split_path} carries {split.indices['test'].size} test indices; the demo "
            "split must have an empty test list"
        )

    config = DataConfig(npz_path=npz_path, split_path=split_path, batch_size=4, num_workers=0)
    data = build_data(config)
    checked: dict[str, Any] = {}
    for name, expected_keys in (("train", ("image", "mask")), ("val", ("image", "masks"))):
        batch = next(iter(data.loaders[name]))
        for key in expected_keys:
            if key not in batch:
                raise ValueError(f"{name} batch is missing {key!r}; got {sorted(batch)}")
        checked[name] = {
            "n_patches": len(data.datasets[name]),
            **{
                key: {"shape": list(batch[key].shape), "dtype": str(batch[key].dtype)}
                for key in expected_keys
            },
        }
    return {
        "loader": "probunet.data.lidc.build_data, unchanged",
        "batches_pulled": checked,
        "test_loader_n_patches": len(data.datasets["test"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Build the subset and its split file.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    if not arguments.source.exists():
        raise FileNotFoundError(
            f"{arguments.source} not found. This runs on the machine holding the full "
            "converted dataset; the file is gitignored and is not in a fresh clone."
        )

    projection = size_arithmetic(arguments.train_patches, arguments.val_patches)
    LOGGER.info(
        "planning %d + %d = %d patches -> %.2f MiB projected at the measured %d B/patch",
        arguments.train_patches, arguments.val_patches, projection["n_patches"],
        projection["projected_mib"], projection["bytes_per_patch_measured"],
    )
    if projection["projected_bytes"] > SIZE_HARD_STOP_BYTES:
        raise ValueError(
            f"{projection['n_patches']} patches project to "
            f"{projection['projected_mib']:.2f} MiB, past the "
            f"{SIZE_HARD_STOP_BYTES / 1024**2:.0f} MiB hard stop. The budget allows "
            f"{projection['max_patches_at_hard_stop']} patches; ask for fewer."
        )

    arrays = LidcArrays.load(arguments.source)
    if arrays.is_subset:
        raise ValueError(
            f"{arguments.source} is already a subset export. Point --source at the full "
            "dataset; drawing a demo subset from a subset would silently narrow the pool."
        )
    split = load_split(arguments.split, expected_n_patches=len(arrays))
    LOGGER.info(
        "source: %d patches, split seed %d, train/val/test = %d/%d/%d",
        len(arrays), split.seed, split.indices["train"].size,
        split.indices["val"].size, split.indices["test"].size,
    )

    selection, exclusion = select_demo_patches(
        nonempty_counts=arrays.nonempty_counts(),
        train_pool=split.indices["train"],
        val_pool=split.indices["val"],
        test_indices=split.indices["test"],
        train_patches=arguments.train_patches,
        val_patches=arguments.val_patches,
        seed=arguments.seed,
        buckets=DEMO_BUCKETS,
    )
    LOGGER.info(
        "test-split exclusion verified on the FINAL %d selected indices: %s",
        exclusion["n_selected"], exclusion["assertion"],
    )

    rows = selection.rows
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Colab Tier 2 demo subset: enough patches to train a tiny model from scratch "
            "in a notebook with no download. NOT a scientific artifact -- no number in the "
            "report is measured on this file."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_npz_path": str(arguments.source),
        "source_npz_sha256": sha256_file(arguments.source),
        "source_n_patches": len(arrays),
        "source_split_path": str(arguments.split),
        "source_split_seed": int(split.seed),
        "source_split_sha256": split.source_npz_sha256,
        "seed": int(arguments.seed),
        "n_patches": len(selection),
        "original_indices": {
            "all": [int(i) for i in rows],
            "train": [int(i) for i in selection.train_global],
            "val": [int(i) for i in selection.val_global],
        },
        "rebased_indices": {
            "train": [int(i) for i in selection.train_rows],
            "val": [int(i) for i in selection.val_rows],
            "note": (
                "Indices into THIS file. Row i holds full-dataset patch "
                "source_index[i]; the demo split file carries the same lists."
            ),
        },
        "per_bucket": {
            pool: {str(bucket): count for bucket, count in counts.items()}
            for pool, counts in selection.per_bucket.items()
        },
        "available_per_bucket": {
            pool: {str(bucket): count for bucket, count in counts.items()}
            for pool, counts in selection.available_per_bucket.items()
        },
        "test_split_exclusion": exclusion,
        "demo_split_path": str(arguments.split_out),
        # The PROJECTION only. The actual on-disk size cannot live in the file whose size
        # it describes without either a second write or a self-referential figure, so it
        # goes in the split document and the console report instead.
        "size": projection,
    }

    size = write_demo_subset(
        path=arguments.out,
        images=arrays.images[rows],
        masks=arrays.masks[rows],
        series_uid=arrays.series_uid[rows],
        source_index=rows.astype(np.int64),
        provenance=provenance,
        keys=None if arrays.keys is None else arrays.keys[rows],
    )

    document = build_demo_split_document(
        selection=selection,
        series_uid=arrays.series_uid,
        seed=arguments.seed,
        subset_npz_path=arguments.out,
        subset_npz_sha256=sha256_file(arguments.out),
        source_split_path=arguments.split,
        source_split_seed=int(split.seed),
        exclusion=exclusion,
    )
    document["source_npz"]["size_bytes"] = int(size)
    write_demo_split(arguments.split_out, document)

    verification = verify_loads_unchanged(arguments.out, arguments.split_out, rows)

    print()
    print(render_report(selection, size, projection))
    print()
    print("VERIFIED THROUGH THE REAL LOADER (no loader change was needed)")
    for name, block in verification["batches_pulled"].items():
        shapes = ", ".join(
            f"{key}{tuple(value['shape'])} {value['dtype']}"
            for key, value in block.items()
            if key != "n_patches"
        )
        print(f"  {name:<6} {block['n_patches']:>4} patches | {shapes}")
    print(f"  test   {verification['test_loader_n_patches']:>4} patches | EMPTY by construction")
    print()
    print(f"wrote {arguments.out} ({size / 1024**2:.2f} MiB)")
    print(f"wrote {arguments.split_out}")
    print(exclusion["statement"])

    if size > SIZE_HARD_STOP_BYTES:
        raise ValueError(
            f"{arguments.out} is {size / 1024**2:.2f} MiB, past the "
            f"{SIZE_HARD_STOP_BYTES / 1024**2:.0f} MiB hard stop. Do not commit it; "
            "re-run with fewer patches."
        )
    if size > SIZE_TARGET_BYTES:
        LOGGER.warning(
            "%s is %.2f MiB, above the %.0f MiB target (under the hard stop). It is a "
            "tracked file that has to survive a clone -- consider fewer patches.",
            arguments.out, size / 1024**2, SIZE_TARGET_BYTES / 1024**2,
        )
    return 0


if __name__ == "__main__":
    # Guard required on Windows, where DataLoader workers are spawned.
    raise SystemExit(main())
