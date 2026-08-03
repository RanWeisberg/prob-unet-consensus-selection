"""Export the diagnostic patches to a small, tracked ``.npz``.

The notebook needs images to sample from for its qualitative panels, but the full
converted dataset is ~450 MB compressed and is gitignored, so a Colab session that only
clones the repository has nothing to draw. This exports just the stratified
panel/diversity patches -- a few megabytes -- with **all four grader masks** each, and
commits them.

The export carries a ``source_index`` array recording each patch's row in the full
dataset. :meth:`probunet.data.lidc.LidcArrays.resolve_indices` uses it so that code
holding indices from ``diagnostic_indices.json`` addresses the same patches whether the
full dataset or the subset is loaded. The panel therefore has one code path, chosen by
which ``npz_path`` the config points at.

Usage::

    python scripts/export_subset.py                       # from configs/baseline.yaml
    python scripts/export_subset.py --indices runs/baseline/diagnostic_indices.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.data.lidc import LidcArrays, LidcDataset  # noqa: E402
from probunet.data.splits import load_split  # noqa: E402
from probunet.paths import CONFIGS_DIR, SUBSET_NPZ  # noqa: E402
from probunet.training.config import ExperimentConfig  # noqa: E402
from probunet.training.diagnostics import build_diagnostic_sets  # noqa: E402

LOGGER = logging.getLogger("probunet.export_subset")


def main(argv: list[str] | None = None) -> int:
    """Export the diagnostic subset.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "baseline.yaml",
        help="config supplying the dataset, split, seed and diagnostic set sizes",
    )
    parser.add_argument(
        "--indices",
        type=Path,
        default=None,
        help=(
            "a run's diagnostic_indices.json. Omit to recompute the same stratified "
            "sets from the config's seed, which is deterministic and needs no run."
        ),
    )
    parser.add_argument("--out", type=Path, default=SUBSET_NPZ, help="output .npz")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(message)s")

    config = ExperimentConfig.from_yaml(args.config)
    arrays = LidcArrays.load(config.data.npz_path)
    if arrays.is_subset:
        LOGGER.error("%s is already a subset export; point --config at the full dataset",
                     config.data.npz_path)
        return 1

    if args.indices is not None:
        recorded = json.loads(args.indices.read_text())
        wanted = sorted({*recorded["diversity"], *recorded["panel"]})
        LOGGER.info("using %d indices from %s", len(wanted), args.indices)
    else:
        split = load_split(config.data.split_path, expected_n_patches=len(arrays))
        dataset = LidcDataset(arrays, split.indices["val"], mode="eval")
        sets = build_diagnostic_sets(
            dataset,
            diversity_images=config.log.diversity_images,
            panel_images=config.log.panel_images,
            seed=config.run.seed,
        )
        wanted = sorted({*sets.diversity.tolist(), *sets.panel.tolist()})
        LOGGER.info(
            "recomputed %d indices from seed %d (diversity %d, panel %d)",
            len(wanted), config.run.seed, sets.diversity.size, sets.panel.size,
        )

    rows = np.array(wanted, dtype=np.int64)
    counts = arrays.nonempty_counts(rows)
    payload = {
        "images": arrays.images[rows],
        "masks": arrays.masks[rows],
        "series_uid": arrays.series_uid[rows],
        "source_index": rows,
    }
    if arrays.keys is not None:
        payload["keys"] = arrays.keys[rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    size_mib = args.out.stat().st_size / 1024**2

    LOGGER.info("wrote %s (%.2f MiB, %d patches)", args.out, size_mib, rows.size)
    LOGGER.info(
        "ambiguity buckets: %s",
        {int(bucket): int((counts == bucket).sum()) for bucket in range(5)},
    )
    # Reload through the normal path to prove the export is usable as-is.
    reloaded = LidcArrays.load(args.out)
    assert reloaded.is_subset
    assert np.array_equal(reloaded.resolve_indices(rows), np.arange(rows.size))
    LOGGER.info("verified: reloads as a subset and resolves the original indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())