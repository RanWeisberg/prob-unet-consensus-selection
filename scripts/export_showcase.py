"""Export everything the submission notebook needs to render its figures, with no model.

Runs **on the PC** -- the machine holding the four checkpoints, with CUDA -- and writes the
tracked ``data/processed/showcase.npz``. The notebook loads that one file and draws with
numpy and matplotlib alone: it loads no checkpoint, runs no model, and needs no GPU and no
full dataset. Every array a figure needs is therefore in this file or it does not exist.

THE CENTRAL CORRECTNESS REQUIREMENT
-----------------------------------
The candidates exported here **must be the same candidates that produced**
``results/consensus_selection_test.json``. So this script does not contain a sampling loop
of its own for the published numbers: it imports and calls
:func:`probunet.evaluation.headroom.measure_selection` and
:func:`probunet.evaluation.sampling.collect_per_patch_metrics` -- the exact functions
``scripts/consensus_headroom.py`` and ``scripts/evaluate.py`` call -- and then **verifies
rather than assumes**:

1. It recomputes the full aggregate selection table over all test patches and asserts every
   published figure (random, area, head, oracle, ceil, edge -- per bucket and aggregate)
   matches ``results/consensus_selection_test.json`` for the Phase 1 arm and
   ``results/consensus_selection_modernized_test.json`` for the modernized arm, **exactly**.
2. It does the same for the Phase 2 GED tables against
   ``results/evaluation_test_baseline-short.json`` and
   ``results/evaluation_test_modernized-short.json``.

No tolerance. These are recomputed on the same device from the same weights through the
same code path, so they are bit-reproducible; a tolerance would let a divergent sampling
path pass. This is a **value**-recompute check -- a shape assertion could not detect the
failure it exists to detect.

The measurement functions do not return the candidates themselves, so a second pass
replays the identical loop with a freshly seeded generator to recapture them, and every
per-image column that replay recomputes is asserted bit-equal to the verified first pass.

TWO THINGS THAT MUST NOT BE "TIDIED"
------------------------------------
* **Batch size is never overridden.** The sampling generator is a CPU generator drawing a
  ``(batch, latent_dim)`` noise tensor per candidate, so a different batch size is a
  different RNG stream and every figure above would stop matching.
* **cuDNN flags are set per pass to mirror the script that produced each published file.**
  ``consensus_headroom.py`` calls ``seed_everything(..., deterministic=True)``, which sets
  ``cudnn.deterministic``; ``evaluate.py`` calls no such thing and leaves torch's defaults.
  ``cudnn.deterministic`` can select a different convolution algorithm and therefore
  different bits, so running both passes in one process without restoring the flags would
  break one of the two verifications. See :func:`cudnn_flags`.

CASE SELECTION IS MECHANICAL AND PRE-REGISTERED
-----------------------------------------------
Two sets of three, each at the 5th percentile, the median and the 95th percentile of a
per-image difference -- percentile-nearest, never ``argmin``/``argmax``, because
single-pixel lesions make the extremes unstable. Set B's 5th percentile is a **failure**
case where the head loses to the largest-candidate rule; it is exported deliberately.

THE SET A DISPLAY GUARDS WERE ADDED AFTER THE FACT, AND THAT IS DISCLOSED
------------------------------------------------------------------------
The first export's Set A median, case ``a1``, was rendered and found **illegible**: a
bucket-1 patch whose only non-empty grader marked 4 pixels, so the soft-consensus tile it
is judged against was invisible. Two display guards were added *in response*
(``--min-consensus-footprint``, ``--min-nonempty-samples``), and the order is recorded in
the manifest under ``set_a_display_guard`` rather than presented as a pre-registration.

The guards read the grader masks and sample emptiness, never a score, and apply the same
threshold to both arms, so neither can prefer an arm. But a guard declared after seeing a
rendered case is indistinguishable from a cherry-pick unless the sequence is stated, so it
is stated -- here, in the manifest, and in the notebook. Recorded alongside: the originally
specified guard ("at least one non-empty sample per arm") would **not** have caught ``a1``,
because it was aimed at the wrong failure mode. The real one is *the target is too small to
see*, not *both arms are empty*.

Consequence, also recorded: Set A percentiles are now computed over the **legible
subpopulation**, not over the whole test split.

Usage (PowerShell, one line -- see the report command at the bottom of this docstring)::

    python scripts/export_showcase.py --split test

Every checkpoint argument has a default under ``runs/`` and **every one is required to
exist**: a missing checkpoint raises :class:`FileNotFoundError` rather than falling back to
a stand-in model or quietly exporting fewer variants.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import platform
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probunet.data.lidc import build_data  # noqa: E402
from probunet.evaluation.headroom import measure_selection, per_bucket  # noqa: E402
from probunet.evaluation.metrics import (  # noqa: E402
    consensus,
    consensus_ceiling,
    consensus_scores,
    consensus_selected,
    generalized_energy_distance,
)
from probunet.evaluation.runner import load_variant  # noqa: E402
from probunet.evaluation.sampling import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    DEFAULT_SAMPLE_COUNTS,
    SamplingConfig,
    build_report,
    collect_per_patch_metrics,
)
from probunet.evaluation.showcase import (  # noqa: E402
    CASE_PERCENTILES,
    MAX_SHOWCASE_BYTES,
    SCHEMA_VERSION,
    SET_A_GUARD_PROVENANCE,
    SET_A_MIN_CONSENSUS_FOOTPRINT_PX,
    SET_A_MIN_NONEMPTY_SAMPLES,
    assert_arrays_identical,
    check_published_figures,
    check_published_ged,
    duplicate_case_positions,
    render_case_table,
    select_cases,
    selection_eligibility,
    set_a_eligibility,
    write_showcase,
)
from probunet.extension.ablation import (  # noqa: E402
    ablation_signature,
    assert_comparable,
    assert_not_a_smoke_run,
)
from probunet.extension.head import load_selection_head  # noqa: E402
from probunet.paths import RESULTS_DIR, SHOWCASE_NPZ  # noqa: E402
from probunet.training.checkpoint import load_checkpoint  # noqa: E402
from probunet.training.config import ExperimentConfig  # noqa: E402
from probunet.utils.runtime import (  # noqa: E402
    describe_device,
    git_revision,
    seed_everything,
    select_device,
)

LOGGER = logging.getLogger("probunet.export_showcase")

# ---------------------------------------------------------------------------------
# Checkpoint defaults and the provenance they are asserted against
# ---------------------------------------------------------------------------------
#
# THESE PATHS ARE NOT DERIVED FROM THE CONFIGS, AND THAT IS DELIBERATE.
# `run.name` in configs/extension.yaml is "extension" and in
# configs/extension_modernized.yaml is "extension_modernized", but the on-disk run
# directories are runs/selection-head and runs/selection-head-modernized. Deriving a
# checkpoint path from `run.name` + `out_dir` would have produced a path that does not
# exist -- and, worse, in a future run might exist and hold something else. That is the
# same "believed wired, wasn't" failure mode as the shadowed _selection_head_step bug.
# Nothing in this script derives a path from a config.
#
# A wrong-but-existing path is the dangerous case, so the paths are not trusted on their
# own either: each head checkpoint records the sha256 of its frozen base's parameter values
# and that base's parameter count, and both are asserted below.

EXPECTED_BASES: dict[str, dict[str, Any]] = {
    "selection_head": {
        "sha256_prefix": "b1887d8d242f",
        "base_parameter_count": 27499098,
        "describes": "runs/baseline, epoch 129, git ad5443c, diagonal latent",
    },
    "selection_head_modernized": {
        "sha256_prefix": "459ff5c84aee",
        "base_parameter_count": 27514488,
        "describes": "runs/modernized-short, epoch 50, git 31a50e6-dirty, full covariance",
    },
}
"""Which frozen base each head checkpoint must carry.

The parameter counts also separate the two latent parameterizations -- 27,499,098 diagonal
against 27,514,488 full-covariance -- which is the check that correctly prevented the wrong
ablation from being launched. A count mismatch means the latent parameterization is wrong,
whatever the filename says.
"""

DEFAULT_CHECKPOINTS: dict[str, Path] = {
    "selection_head": Path("runs/selection-head/checkpoints/best.pt"),
    "selection_head_modernized": Path("runs/selection-head-modernized/checkpoints/best.pt"),
    "baseline_short": Path("runs/baseline-short/checkpoints/best.pt"),
    "modernized_short": Path("runs/modernized-short/checkpoints/best.pt"),
}

PUBLISHED_SELECTION: dict[str, Path] = {
    "selection_head": RESULTS_DIR / "consensus_selection_test.json",
    "selection_head_modernized": RESULTS_DIR / "consensus_selection_modernized_test.json",
}

PUBLISHED_GED: dict[str, Path] = {
    "baseline_short": RESULTS_DIR / "evaluation_test_baseline-short.json",
    "modernized_short": RESULTS_DIR / "evaluation_test_modernized-short.json",
}

SET_A_CRITERION = (
    "per-image GED at n=16, baseline-short minus modernized-short "
    "(positive = modernized better)"
)
SET_B_CRITERION = (
    "per-image soft-consensus Dice of the head's pick minus the area control's pick, "
    "Phase 1 base (negative = the head lost to the largest-candidate rule)"
)

GED_SAMPLE_COUNT = 16
"""Sample count the Set A criterion is read at. The paper's largest, and the one the
follow-up literature reports."""

MIN_ELIGIBLE_FOR_PERCENTILES = 200
"""Floor on the Set A eligible pool. Below this a percentile describes nothing, so the
export refuses rather than quietly reporting one."""


# ---------------------------------------------------------------------------------
# Environment plumbing
# ---------------------------------------------------------------------------------


@contextlib.contextmanager
def cudnn_flags(deterministic: bool, benchmark: bool) -> Iterator[None]:
    """Set cuDNN's determinism flags for one pass and restore them afterwards.

    Each published results file was produced by a script with its own flag state, and
    ``cudnn.deterministic`` can change which convolution algorithm is selected and hence
    the bits a forward pass produces. Running both kinds of pass in one process therefore
    requires restoring the flags between them, or one of the two exact verifications would
    fail for a reason that has nothing to do with sampling.

    Args:
        deterministic: Value for ``torch.backends.cudnn.deterministic``.
        benchmark: Value for ``torch.backends.cudnn.benchmark``.

    Yields:
        None, for the duration of the pass.
    """
    previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous


class DataCache:
    """Builds each distinct data pipeline once.

    The full converted dataset is ~450 MB and every arm in this export points at the same
    ``.npz`` and the same split, so rebuilding it per checkpoint would load it four times
    for no reason. Keyed on the resolved data config, so two arms that genuinely differ
    still get their own pipeline rather than silently sharing one.
    """

    def __init__(self) -> None:
        """Start with an empty cache."""
        self._entries: dict[str, Any] = {}

    def get(self, config: ExperimentConfig) -> Any:
        """Return the :class:`~probunet.data.lidc.LidcData` for a config.

        Args:
            config: The experiment config whose ``data`` block describes the pipeline.

        Returns:
            The assembled data object.
        """
        key = json.dumps(dataclasses.asdict(config.data), sort_keys=True, default=str)
        if key not in self._entries:
            self._entries[key] = build_data(config.data)
        return self._entries[key]


def require_checkpoint(path: Path, role: str) -> Path:
    """Fail loudly on a missing checkpoint.

    There is no fallback, no stand-in model and no "export the variants that happen to be
    present": a showcase assembled from three of four arms would look complete and be
    wrong.

    Args:
        path: The checkpoint path.
        role: What it is for, named in the error.

    Returns:
        The path.

    Raises:
        FileNotFoundError: If the checkpoint is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint for {role} not found: {path}. Run this script on the machine "
            "that trained it. There is no fallback -- a partial showcase would be "
            "indistinguishable from a complete one."
        )
    return path


def require_published(path: Path, role: str) -> dict[str, Any]:
    """Load a published results file that this export must reproduce.

    Args:
        path: The JSON file.
        role: What it is for, named in the error.

    Returns:
        The parsed record.

    Raises:
        FileNotFoundError: If the file is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"published results for {role} not found: {path}. This export verifies itself "
            "against that file; without it nothing would establish that the exported "
            "candidates are the published ones."
        )
    return json.loads(path.read_text())


def assert_expected_base(head_state: Any, arm: str, live_base_parameters: int) -> dict[str, Any]:
    """Assert a head checkpoint carries the frozen base it is supposed to carry.

    A wrong path that happens to exist is the dangerous case -- it loads, it runs, and it
    produces a plausible table for the wrong arm. The head checkpoint records the sha256 of
    its base's parameter *values* at construction, so the base's identity is verifiable
    from the artifact rather than asserted by a filename.

    The parameter count is checked twice: once as recorded, once as actually constructed in
    this process. The second is what distinguishes the latent parameterizations
    (27,499,098 diagonal against 27,514,488 full covariance), so a config that built the
    wrong latent cannot pass by carrying the right recorded number.

    Args:
        head_state: The loaded :class:`~probunet.training.checkpoint.CheckpointState`.
        arm: Key into :data:`EXPECTED_BASES`.
        live_base_parameters: Parameter count of the base actually built here.

    Returns:
        The checkpoint's ``base_provenance``, for the manifest.

    Raises:
        ValueError: On any mismatch, reporting the expected and the found value.
    """
    expected = EXPECTED_BASES[arm]
    provenance = head_state.base_provenance
    if not provenance:
        raise ValueError(
            f"{arm}: the head checkpoint carries no base_provenance, so which base "
            "produced it cannot be verified from the artifact. Refusing to export a "
            "figure whose base is unknown."
        )

    found_sha = str(provenance.get("parameter_sha256", ""))
    if found_sha[:12] != expected["sha256_prefix"]:
        raise ValueError(
            f"{arm}: WRONG FROZEN BASE. Expected base sha256 starting "
            f"{expected['sha256_prefix']} ({expected['describes']}), found "
            f"{found_sha[:12] or '<empty>'} (recorded base: "
            f"{provenance.get('checkpoint')}, epoch {provenance.get('epoch')}, git "
            f"{provenance.get('git_revision')}). The checkpoint path exists but holds a "
            "head trained on a different base."
        )

    for label, found in (
        ("recorded", int(provenance.get("frozen_parameters", -1))),
        ("constructed", int(live_base_parameters)),
    ):
        if found != expected["base_parameter_count"]:
            raise ValueError(
                f"{arm}: WRONG LATENT PARAMETERIZATION. Expected "
                f"{expected['base_parameter_count']:,} base parameters "
                f"({expected['describes']}), found {found:,} ({label}). A parameter-count "
                "mismatch here means the diagonal and full-covariance arms have been "
                "crossed."
            )
    return dict(provenance)


# ---------------------------------------------------------------------------------
# Replay passes: recover the candidates behind the verified numbers
# ---------------------------------------------------------------------------------


@torch.no_grad()
def replay_selection(
    head: Any,
    loader: Any,
    device: torch.device,
    n_samples: int,
    seed: int,
    wanted: set[int],
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """Replay the verified selection pass, keeping the candidates and the scores.

    :func:`~probunet.evaluation.headroom.measure_selection` returns per-image summaries and
    discards the candidates, so this walks the identical loop -- same loader, same batch
    size, same freshly seeded CPU generator, same calls in the same order -- and keeps what
    the showcase needs. Its per-image columns are asserted bit-equal to the verified pass
    by the caller; that assertion is what makes the replay trustworthy.

    Args:
        head: The loaded selection head.
        loader: The evaluation loader for the split.
        device: Device to run on.
        n_samples: Candidates per image.
        seed: The candidate-draw seed, ``config.head.eval_seed``.
        wanted: Global patch indices whose full candidate sets to keep.

    Returns:
        ``(columns, captured)``. ``columns`` holds per-image and per-candidate arrays over
        the whole split; ``captured`` maps each wanted patch index to its image, masks,
        consensus, candidates, scores, areas and picks.
    """
    generator = torch.Generator().manual_seed(seed)
    parts: dict[str, list[np.ndarray]] = {}
    captured: dict[int, dict[str, np.ndarray]] = {}

    def add(key: str, value: torch.Tensor) -> None:
        """Accumulate one batch's column."""
        parts.setdefault(key, []).append(value.detach().cpu().numpy())

    for batch in loader:
        image = batch["image"].to(device)
        graders = batch["masks"].to(device)

        # The published call, verbatim: one shared candidate set per image, drawn from the
        # PRIOR, with the base frozen.
        features, candidates = head.sample_candidates(image, n_samples, generator)
        scores = consensus_scores(candidates, graders)
        predicted = head.score_candidates(features, candidates)
        area_scores = head.score_by_area(candidates)
        areas = (candidates != 0).flatten(start_dim=2).sum(dim=2)

        pick_head = predicted.argmax(dim=1)
        pick_area = area_scores.argmax(dim=1)
        pick_oracle = scores.argmax(dim=1)
        ceilings = consensus_ceiling(graders)

        # Per-image columns, the same definitions measure_selection uses.
        add("head", consensus_selected(candidates, graders, pick_head))
        add("area_only", consensus_selected(candidates, graders, pick_area))
        add("random", scores.mean(dim=1))
        add("oracle", scores.amax(dim=1))
        add("ceiling", ceilings)
        add("n_nonempty", (graders.flatten(start_dim=2).sum(dim=2) > 0).sum(dim=1))
        add("index", batch["index"])
        add("pick_head", pick_head)
        add("pick_area", pick_area)
        add("pick_oracle", pick_oracle)
        # Per-candidate columns, for the Set C scatter.
        add("candidate_true", scores)
        add("candidate_pred", predicted)
        add("candidate_area", areas)

        present = [
            (row, index)
            for row, index in enumerate(batch["index"].tolist())
            if index in wanted
        ]
        if not present:
            continue
        soft = consensus(graders)
        for row, index in present:
            captured[int(index)] = {
                "image": image[row, 0].detach().cpu().numpy().astype(np.float32),
                "masks": graders[row].detach().cpu().numpy().astype(np.uint8),
                "consensus": soft[row].detach().cpu().numpy().astype(np.float32),
                "candidates": candidates[row].detach().cpu().numpy().astype(np.uint8),
                "true_scores": scores[row].detach().cpu().numpy().astype(np.float32),
                "pred_scores": predicted[row].detach().cpu().numpy().astype(np.float32),
                "areas": areas[row].detach().cpu().numpy().astype(np.int32),
                "ceiling": np.asarray(ceilings[row].item(), dtype=np.float32),
            }

    columns = {key: np.concatenate(values) for key, values in parts.items()}
    return columns, captured


@torch.no_grad()
def replay_ged(
    variant: Any,
    loader: Any,
    device: torch.device,
    n_samples: int,
    wanted: set[int],
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """Replay the verified GED pass, keeping the samples.

    The Set A analogue of :func:`replay_selection`. ``variant`` must be freshly loaded so
    its generator is at the start of the same sequence
    :func:`~probunet.evaluation.sampling.collect_per_patch_metrics` consumed.

    Args:
        variant: A freshly loaded variant, its generator seeded as the published run's was.
        loader: The evaluation loader for the split.
        device: Device to run on.
        n_samples: Samples per image; must equal the published run's largest sample count,
            since that is how many draws the generator supplied per batch.
        wanted: Global patch indices whose samples to keep.

    Returns:
        ``(columns, captured)`` with per-image ``ged`` and ``index``, and the kept samples.
    """
    parts: dict[str, list[np.ndarray]] = {}
    captured: dict[int, dict[str, np.ndarray]] = {}

    for batch in loader:
        image = batch["image"].to(device)
        graders = batch["masks"].to(device)
        samples = variant.sample(image, n_samples)
        distance = generalized_energy_distance(samples, graders)["d_squared"]

        parts.setdefault("ged", []).append(distance.detach().cpu().numpy())
        parts.setdefault("index", []).append(batch["index"].detach().cpu().numpy())
        # Spelled exactly as collect_per_patch_metrics spells "nonempty_count", so the
        # equality check below compares like with like rather than two definitions that
        # happen to agree.
        parts.setdefault("n_nonempty", []).append(
            (graders != 0)
            .flatten(start_dim=2)
            .any(dim=2)
            .sum(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        nonempty = (samples.flatten(start_dim=2) != 0).any(dim=2)
        parts.setdefault("nonempty_frac", []).append(
            nonempty.to(torch.float32).mean(dim=1).detach().cpu().numpy()
        )
        # Per-image COUNT, not just the fraction: the Set A display guard is expressed as
        # "at least N non-empty samples per arm", and a fraction would hide the count
        # behind the sample size.
        parts.setdefault("nonempty_samples", []).append(
            nonempty.sum(dim=1).detach().cpu().numpy()
        )
        # The soft-consensus map's footprint -- the pixels the consensus tile actually
        # paints. This is what the legibility guard thresholds; see
        # showcase.SET_A_MIN_CONSENSUS_FOOTPRINT_PX.
        parts.setdefault("consensus_footprint", []).append(
            (graders != 0).any(dim=1).flatten(start_dim=1).sum(dim=1)
            .detach().cpu().numpy()
        )

        present = [
            (row, index)
            for row, index in enumerate(batch["index"].tolist())
            if index in wanted
        ]
        if not present:
            continue
        soft = consensus(graders)
        for row, index in present:
            captured[int(index)] = {
                "image": image[row, 0].detach().cpu().numpy().astype(np.float32),
                "masks": graders[row].detach().cpu().numpy().astype(np.uint8),
                "consensus": soft[row].detach().cpu().numpy().astype(np.float32),
                "samples": samples[row].detach().cpu().numpy().astype(np.uint8),
                "ged": np.asarray(distance[row].item(), dtype=np.float32),
            }

    columns = {key: np.concatenate(values) for key, values in parts.items()}
    return columns, captured


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


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
        "--split", default="test", choices=("val", "test"),
        help="split the published figures were measured on",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_EVAL_SEED,
        help="evaluation sampling seed; must equal head.eval_seed in both head configs",
    )
    parser.add_argument("--out", type=Path, default=SHOWCASE_NPZ, help="output .npz")
    parser.add_argument(
        "--head-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["selection_head"],
        help="the selection head on the PHASE 1 base; also supplies that base",
    )
    parser.add_argument(
        "--head-checkpoint-modernized", type=Path,
        default=DEFAULT_CHECKPOINTS["selection_head_modernized"],
        help="the selection head on the modernized base; verified, not exported",
    )
    parser.add_argument(
        "--checkpoint-baseline-short", type=Path,
        default=DEFAULT_CHECKPOINTS["baseline_short"],
        help="Phase 1 short run, the first arm of the Set A GED comparison",
    )
    parser.add_argument(
        "--checkpoint-modernized-short", type=Path,
        default=DEFAULT_CHECKPOINTS["modernized_short"],
        help="Phase 2 short run, the second arm of the Set A GED comparison",
    )
    parser.add_argument(
        "--min-consensus-footprint", type=int,
        default=SET_A_MIN_CONSENSUS_FOOTPRINT_PX,
        help=(
            "SET A LEGIBILITY GUARD, ADDED AFTER case a1 was rendered and found "
            "illegible. Minimum union area of the four grader masks -- the footprint the "
            "soft-consensus tile actually paints -- for a Set A case to be eligible. The "
            "default is a 5x5-equivalent region, the mildest threshold that clears a1's "
            "4 px by a wide margin; measured on test it retains 79.7%% of patches with "
            "every bucket populated. Applied identically to both arms and reading no "
            "model output, so it cannot favour an arm"
        ),
    )
    parser.add_argument(
        "--min-nonempty-samples", type=int, default=SET_A_MIN_NONEMPTY_SAMPLES,
        help=(
            "Set A display guard: non-empty samples each arm must offer. The originally "
            "specified value was 1, which case a1 SATISFIED while still being unreadable "
            "-- that guard was aimed at the wrong failure mode. Symmetric between arms"
        ),
    )
    parser.add_argument("--device", default=None, help="override device selection")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def load_head_arm(
    checkpoint: Path, arm: str, device: torch.device
) -> tuple[Any, ExperimentConfig, Any, dict[str, Any]]:
    """Load one selection-head checkpoint and verify which base it was frozen on.

    Args:
        checkpoint: Path to the head checkpoint.
        arm: Key into :data:`EXPECTED_BASES`.
        device: Device to place the head on.

    Returns:
        ``(head, config, state, base_provenance)``.
    """
    state = load_checkpoint(checkpoint, restore_rng=False)
    assert_not_a_smoke_run(state.config, checkpoint)
    config = ExperimentConfig.from_dict(state.config)
    head = load_selection_head(checkpoint, config, device)
    provenance = assert_expected_base(state, arm, head.parameter_counts()["base"])
    LOGGER.info(
        "%s: %s (epoch %d, git %s) | base %s sha256 %s | %s latent",
        arm, checkpoint, state.epoch, state.git_revision,
        provenance.get("checkpoint"), str(provenance.get("parameter_sha256", ""))[:12],
        config.model.latent_covariance,
    )
    return head, config, state, provenance


def variant_record(
    checkpoint: Path,
    state: Any,
    config: ExperimentConfig,
    parameter_counts: dict[str, int],
    base_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one variant's manifest entry.

    Args:
        checkpoint: The checkpoint path as given on the command line.
        state: Its :class:`~probunet.training.checkpoint.CheckpointState`.
        config: Its resolved configuration.
        parameter_counts: ``{"total": ..., "base": ...}``; ``base`` may be None.
        base_provenance: The frozen base's record, or None for an ELBO checkpoint.

    Returns:
        A JSON-serializable mapping carrying every key the notebook prints.
    """
    return {
        "checkpoint": str(checkpoint),
        "epoch": int(state.epoch),
        "checkpoint_git_revision": state.git_revision,
        "checkpoint_device": state.device,
        "checkpoint_torch_version": state.torch_version,
        "checkpoint_monitor": state.monitor,
        "checkpoint_best_metric": state.best_metric,
        "latent_covariance": config.model.latent_covariance,
        "parameter_count": int(parameter_counts["total"]),
        "base_parameter_count": (
            None if parameter_counts.get("base") is None else int(parameter_counts["base"])
        ),
        "base_parameter_sha256": (
            None if base_provenance is None else base_provenance.get("parameter_sha256")
        ),
        "base_checkpoint": (
            None if base_provenance is None else base_provenance.get("checkpoint")
        ),
        "base_epoch": None if base_provenance is None else base_provenance.get("epoch"),
        "base_git_revision": (
            None if base_provenance is None else base_provenance.get("git_revision")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the export.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    checkpoints = {
        "selection_head": require_checkpoint(
            arguments.head_checkpoint, "the head on the Phase 1 base"
        ),
        "selection_head_modernized": require_checkpoint(
            arguments.head_checkpoint_modernized, "the head on the modernized base"
        ),
        "baseline_short": require_checkpoint(
            arguments.checkpoint_baseline_short, "the Phase 1 short run"
        ),
        "modernized_short": require_checkpoint(
            arguments.checkpoint_modernized_short, "the Phase 2 short run"
        ),
    }
    published = {
        **{
            arm: require_published(path, f"the {arm} selection table")
            for arm, path in PUBLISHED_SELECTION.items()
        },
        **{
            arm: require_published(path, f"the {arm} GED table")
            for arm, path in PUBLISHED_GED.items()
        },
    }

    device = select_device(arguments.device or "auto")
    LOGGER.info("device: %s", describe_device(device))
    data_cache = DataCache()
    assertions: list[dict[str, Any]] = []
    variants: dict[str, dict[str, Any]] = {}

    # =============================================================================
    # SETS B and C -- the selection arms. Mirrors scripts/consensus_headroom.py, which
    # calls seed_everything(config.run.seed, deterministic=config.run.deterministic)
    # before measuring; that sets the cuDNN flags, so the pass runs inside cudnn_flags.
    # =============================================================================
    heads: dict[str, Any] = {}
    head_configs: dict[str, ExperimentConfig] = {}
    signatures: dict[str, dict[str, Any]] = {}
    verified_selection: dict[str, dict[str, np.ndarray]] = {}

    for arm in ("selection_head", "selection_head_modernized"):
        head, config, state, provenance = load_head_arm(checkpoints[arm], arm, device)
        if config.head.eval_seed != arguments.seed:
            raise ValueError(
                f"{arm}: head.eval_seed is {config.head.eval_seed} but --seed is "
                f"{arguments.seed}. The published table was drawn with the config's seed; "
                "an export that used a different one would not reproduce it."
            )
        heads[arm] = head
        head_configs[arm] = config
        signatures[arm] = ablation_signature(state.config)
        variants[arm] = variant_record(
            checkpoints[arm], state, config, head.parameter_counts(), provenance
        )

        loader = data_cache.get(config).loaders[arguments.split]
        # seed_everything sets exactly these two flags when deterministic is on, so taking
        # them from the same config value reproduces the published run in both cases and
        # restores whatever was there before.
        with cudnn_flags(config.run.deterministic, False):
            seed_everything(config.run.seed, deterministic=config.run.deterministic)
            LOGGER.info(
                "%s: recomputing the published selection table over %s (%d candidates, "
                "seed %d)", arm, arguments.split, config.head.eval_samples,
                config.head.eval_seed,
            )
            results = measure_selection(
                head, loader, device, config.head.eval_samples, config.head.eval_seed
            )
        verified_selection[arm] = results
        assertions.append(
            check_published_figures(
                per_bucket(results),
                published[arm]["buckets"],
                arm=arm,
                source=str(PUBLISHED_SELECTION[arm]),
            )
        )
        LOGGER.info("%s: MATCHES %s exactly", arm, PUBLISHED_SELECTION[arm])

    # The two head arms must be comparable in everything but the base, or the modernized
    # verification says nothing about the Phase 1 arm it is being checked alongside.
    assert_comparable(signatures)

    # ---- Set B / Set C source data: replay the Phase 1 arm and keep the candidates ----
    phase1_config = head_configs["selection_head"]
    phase1_loader = data_cache.get(phase1_config).loaders[arguments.split]
    reference = verified_selection["selection_head"]

    # THREE passes over the Phase 1 arm, and each one earns its cost:
    #   1. measure_selection    -- the published path, verified against the results JSON;
    #   2. this replay          -- recovers the per-CANDIDATE scores and areas that
    #                             measure_selection discards, which the guards and the
    #                             Set C scatter both need. Checked against pass 1.
    #   3. the replay below     -- keeps the full candidate masks for the three patches
    #                             the guards and the percentiles then chose. Checked too.
    # Passes 2 and 3 cannot be merged: which patches to keep is not known until pass 2's
    # numbers have been through the guards, and keeping every candidate mask for the whole
    # split would be ~800 MB of uint8.
    LOGGER.info("Phase 1 arm: replaying the verified pass to recover per-candidate scores")
    with cudnn_flags(phase1_config.run.deterministic, False):
        seed_everything(phase1_config.run.seed, deterministic=phase1_config.run.deterministic)
        columns, _ = replay_selection(
            heads["selection_head"], phase1_loader, device,
            phase1_config.head.eval_samples, phase1_config.head.eval_seed, wanted=set(),
        )
    assert_arrays_identical(
        {key: columns[key] for key in ("head", "area_only", "random", "oracle", "ceiling",
                                       "n_nonempty", "index")},
        reference,
        context="Phase 1 selection",
    )
    LOGGER.info("Phase 1 selection replay reproduced the verified pass bit-for-bit")

    true_scores = columns["candidate_true"]
    pred_scores = columns["candidate_pred"]
    candidate_areas = columns["candidate_area"]
    selection_guards = selection_eligibility(true_scores, candidate_areas)
    LOGGER.info("Set B/C eligibility: %s", selection_guards.as_dict())

    set_b_criterion = (columns["head"] - columns["area_only"]).astype(np.float64)
    set_b_cases = select_cases(
        values=set_b_criterion,
        eligible=selection_guards.eligible,
        patch_indices=columns["index"],
        buckets=columns["n_nonempty"],
        set_name="B",
        criterion_name=SET_B_CRITERION,
    )

    # Second replay, now keeping the full candidate sets for the three chosen patches. Its
    # per-image columns are re-checked for the same reason the first replay was.
    wanted_b = {case.patch_index for case in set_b_cases}
    LOGGER.info("Phase 1 arm: replaying again to keep the candidates for patches %s",
                sorted(wanted_b))
    with cudnn_flags(phase1_config.run.deterministic, False):
        seed_everything(phase1_config.run.seed, deterministic=phase1_config.run.deterministic)
        recheck, captured_b = replay_selection(
            heads["selection_head"], phase1_loader, device,
            phase1_config.head.eval_samples, phase1_config.head.eval_seed, wanted=wanted_b,
        )
    assert_arrays_identical(recheck, columns, context="Set B capture")

    # =============================================================================
    # SET A -- the Phase 2 GED comparison. Mirrors scripts/evaluate.py, which calls NO
    # seed_everything, so torch's default cuDNN flags apply.
    # =============================================================================
    sampling = SamplingConfig(sample_counts=DEFAULT_SAMPLE_COUNTS, seed=arguments.seed)
    per_patch: dict[str, dict[str, np.ndarray]] = {}

    for arm in ("baseline_short", "modernized_short"):
        with cudnn_flags(False, False):
            variant, config, state = load_variant(
                checkpoints[arm], device, seed=sampling.seed
            )
            loader = data_cache.get(config).loaders[arguments.split]
            LOGGER.info(
                "%s: recomputing the published GED table over %s (%d samples, seed %d)",
                arm, arguments.split, sampling.max_samples, sampling.seed,
            )
            metrics = collect_per_patch_metrics(variant, loader, sampling, device)
        per_patch[arm] = metrics
        variants[arm] = variant_record(
            checkpoints[arm],
            state,
            config,
            {"total": sum(p.numel() for p in variant.model.parameters()), "base": None},
            None,
        )
        assertions.append(
            check_published_ged(
                build_report(metrics, sampling),
                published[arm],
                arm=arm,
                source=str(PUBLISHED_GED[arm]),
                sample_counts=DEFAULT_SAMPLE_COUNTS,
            )
        )
        LOGGER.info("%s: MATCHES %s exactly", arm, PUBLISHED_GED[arm])

    ged_key = f"ged@{GED_SAMPLE_COUNT}"
    if not np.array_equal(per_patch["baseline_short"]["index"],
                          per_patch["modernized_short"]["index"]):
        raise ValueError(
            "the two Phase 2 arms did not traverse the split in the same order, so a "
            "per-image difference between them would pair the wrong patches"
        )

    # ---- Set A display guards need per-image sample counts and the consensus footprint,
    # neither of which collect_per_patch_metrics returns. So the Set A arms follow the same
    # three-pass shape the selection arm already does: verify, replay to recover what the
    # guards need, then replay again to keep the chosen cases. Each replay is checked
    # bit-for-bit against the verified pass.
    replayed: dict[str, dict[str, np.ndarray]] = {}
    for arm in ("baseline_short", "modernized_short"):
        with cudnn_flags(False, False):
            variant, config, _ = load_variant(checkpoints[arm], device, seed=sampling.seed)
            loader = data_cache.get(config).loaders[arguments.split]
            LOGGER.info("%s: replaying to recover sample counts and grader footprints", arm)
            replayed[arm], _ = replay_ged(
                variant, loader, device, sampling.max_samples, wanted=set()
            )
        assert_arrays_identical(
            {"ged": replayed[arm]["ged"], "index": replayed[arm]["index"]},
            {"ged": per_patch[arm][ged_key], "index": per_patch[arm]["index"]},
            context=f"{arm} guard replay",
        )

    ged_guards = set_a_eligibility(
        per_variant_ged={
            arm: per_patch[arm][ged_key] for arm in ("baseline_short", "modernized_short")
        },
        per_variant_nonempty_samples={
            arm: replayed[arm]["nonempty_samples"]
            for arm in ("baseline_short", "modernized_short")
        },
        # Identical across arms by construction -- it is a property of the grader masks --
        # so either arm's copy will do. Asserted rather than assumed.
        consensus_footprint=replayed["baseline_short"]["consensus_footprint"],
        min_footprint=arguments.min_consensus_footprint,
        min_nonempty_samples=arguments.min_nonempty_samples,
    )
    if not np.array_equal(replayed["baseline_short"]["consensus_footprint"],
                          replayed["modernized_short"]["consensus_footprint"]):
        raise ValueError(
            "the grader-union footprint differs between the two arms; it depends on the "
            "grader masks alone and must be identical, so the loaders are not aligned"
        )
    # Recorded, not excluded: the all-samples-empty counts remain informative in their own
    # right even though the display guard now removes those patches from SELECTION.
    diagnostics: dict[str, int] = {
        f"n_all_samples_empty_{arm}": int((replayed[arm]["nonempty_frac"] == 0).sum())
        for arm in ("baseline_short", "modernized_short")
    }
    LOGGER.info("Set A eligibility (with the display guards): %s", ged_guards.as_dict())
    if ged_guards.n_eligible < MIN_ELIGIBLE_FOR_PERCENTILES:
        raise ValueError(
            f"only {ged_guards.n_eligible} Set A patches survived the display guards, "
            f"below the floor of {MIN_ELIGIBLE_FOR_PERCENTILES}. A percentile over that "
            "few patches describes nothing. Lower --min-consensus-footprint or "
            "--min-nonempty-samples and record that you did."
        )

    set_a_criterion = (
        per_patch["baseline_short"][ged_key] - per_patch["modernized_short"][ged_key]
    ).astype(np.float64)
    set_a_cases = select_cases(
        values=set_a_criterion,
        eligible=ged_guards.eligible,
        patch_indices=per_patch["baseline_short"]["index"],
        buckets=per_patch["baseline_short"]["nonempty_count"],
        set_name="A",
        criterion_name=SET_A_CRITERION,
    )

    wanted_a = {case.patch_index for case in set_a_cases}
    captured_a: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for arm in ("baseline_short", "modernized_short"):
        with cudnn_flags(False, False):
            # Freshly loaded, so its generator starts where the verified pass started.
            variant, config, _ = load_variant(checkpoints[arm], device, seed=sampling.seed)
            loader = data_cache.get(config).loaders[arguments.split]
            recaptured, captured = replay_ged(
                variant, loader, device, sampling.max_samples, wanted=wanted_a
            )
        assert_arrays_identical(
            {"ged": recaptured["ged"], "index": recaptured["index"],
             "n_nonempty": recaptured["n_nonempty"],
             "nonempty_samples": recaptured["nonempty_samples"]},
            {"ged": per_patch[arm][ged_key], "index": per_patch[arm]["index"],
             "n_nonempty": per_patch[arm]["nonempty_count"],
             "nonempty_samples": replayed[arm]["nonempty_samples"]},
            context=f"{arm} GED capture",
        )
        captured_a[arm] = captured
        LOGGER.info("%s: GED replay reproduced the verified pass bit-for-bit", arm)

    # =============================================================================
    # Assemble the payload
    # =============================================================================
    arrays: dict[str, np.ndarray] = {}
    keys: dict[str, list[str]] = {"set_a": [], "set_b": [], "set_c": []}

    for case in set_a_cases:
        base = captured_a["baseline_short"][case.patch_index]
        modern = captured_a["modernized_short"][case.patch_index]
        entries = {
            f"{case.key_prefix}_image": base["image"],
            f"{case.key_prefix}_masks": base["masks"],
            f"{case.key_prefix}_consensus": base["consensus"],
            f"{case.key_prefix}_samples_baseline_short": base["samples"],
            f"{case.key_prefix}_samples_modernized_short": modern["samples"],
            f"{case.key_prefix}_ged_baseline_short": base["ged"],
            f"{case.key_prefix}_ged_modernized_short": modern["ged"],
            f"{case.key_prefix}_bucket": np.asarray(case.bucket, dtype=np.int64),
            f"{case.key_prefix}_patch_index": np.asarray(case.patch_index, dtype=np.int64),
            f"{case.key_prefix}_criterion": np.asarray(case.criterion, dtype=np.float32),
        }
        arrays.update(entries)
        keys["set_a"] += list(entries)

    for case in set_b_cases:
        kept = captured_b[case.patch_index]
        position = case.position
        entries = {
            f"{case.key_prefix}_image": kept["image"],
            f"{case.key_prefix}_masks": kept["masks"],
            f"{case.key_prefix}_consensus": kept["consensus"],
            f"{case.key_prefix}_candidates": kept["candidates"],
            f"{case.key_prefix}_true_scores": kept["true_scores"],
            f"{case.key_prefix}_pred_scores": kept["pred_scores"],
            f"{case.key_prefix}_areas": kept["areas"],
            f"{case.key_prefix}_ceiling": kept["ceiling"],
            f"{case.key_prefix}_pick_head": np.asarray(
                columns["pick_head"][position], dtype=np.int64
            ),
            f"{case.key_prefix}_pick_area": np.asarray(
                columns["pick_area"][position], dtype=np.int64
            ),
            f"{case.key_prefix}_pick_oracle": np.asarray(
                columns["pick_oracle"][position], dtype=np.int64
            ),
            # EXPORTED BUT NO LONGER RENDERED. Candidate index 0 was drawn as a contrast
            # tile in an earlier version of the notebook figure. On case b2 that index is
            # also the head's pick and the oracle, so a tile labelled as an unselected draw
            # was in fact the best candidate in the set -- the opposite of its purpose. The
            # figure now shows only the three selection rules, and the honest random
            # quantity is the published `random` column, which is E[score] over all 16
            # candidates (exported as `_mean_score`), never a single index. The key is kept
            # so this writer still matches the tracked showcase.npz and its schema test;
            # nothing reads it.
            f"{case.key_prefix}_arbitrary_unselected": np.asarray(0, dtype=np.int64),
            f"{case.key_prefix}_mean_score": np.asarray(
                columns["random"][position], dtype=np.float32
            ),
            f"{case.key_prefix}_bucket": np.asarray(case.bucket, dtype=np.int64),
            f"{case.key_prefix}_patch_index": np.asarray(case.patch_index, dtype=np.int64),
            f"{case.key_prefix}_criterion": np.asarray(case.criterion, dtype=np.float32),
        }
        arrays.update(entries)
        keys["set_b"] += list(entries)

    eligible = selection_guards.eligible
    set_c = {
        # Per candidate, three parallel arrays flattened image-major.
        "c_pred_scores": pred_scores[eligible].reshape(-1).astype(np.float32),
        "c_true_scores": true_scores[eligible].reshape(-1).astype(np.float32),
        "c_areas": candidate_areas[eligible].reshape(-1).astype(np.int32),
        "c_n_candidates": np.asarray(true_scores.shape[1], dtype=np.int64),
        # Per image.
        "c_random_scores": columns["random"][eligible].astype(np.float32),
        "c_area_scores": columns["area_only"][eligible].astype(np.float32),
        "c_head_scores": columns["head"][eligible].astype(np.float32),
        "c_oracle_scores": columns["oracle"][eligible].astype(np.float32),
        # The ceiling travels with the scores: soft-consensus values are low by design and
        # must never be read against 1.0.
        "c_ceiling": columns["ceiling"][eligible].astype(np.float32),
        "c_buckets": columns["n_nonempty"][eligible].astype(np.int8),
        "c_patch_index": columns["index"][eligible].astype(np.int64),
    }
    arrays.update(set_c)
    keys["set_c"] = list(set_c)

    all_cases = set_a_cases + set_b_cases
    duplicates = {
        "set_a": duplicate_case_positions(set_a_cases),
        "set_b": duplicate_case_positions(set_b_cases),
    }
    for set_name, positions in duplicates.items():
        if positions:
            LOGGER.warning(
                "%s: two percentiles landed on the same patch at position(s) %s, so two "
                "panels will be identical. Recorded in the manifest, not deduplicated.",
                set_name, positions,
            )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_git_revision": git_revision(),
        "split": arguments.split,
        "n_patches": int(columns["index"].size),
        "eval_seed": int(arguments.seed),
        "n_samples": int(phase1_config.head.eval_samples),
        "ged_sample_counts": list(DEFAULT_SAMPLE_COUNTS),
        "ged_sample_count_reported": GED_SAMPLE_COUNT,
        "torch_version": torch.__version__,
        "device": describe_device(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cuda_version": torch.version.cuda,
        },
        "variants": variants,
        "case_percentiles": list(CASE_PERCENTILES),
        "selection_record": [case.as_dict() for case in all_cases],
        "duplicate_case_positions": duplicates,
        "guards": {
            "set_a": {
                **ged_guards.as_dict(), **diagnostics,
                "per_bucket_eligible": {
                    str(bucket): int(
                        ((per_patch["baseline_short"]["nonempty_count"] == bucket)
                         & ged_guards.eligible).sum()
                    )
                    for bucket in (1, 2, 3, 4)
                },
                "note": (
                    "Three guards: GED defined in both arms (the correctness guard), at "
                    "least min_nonempty_samples_per_arm non-empty samples per arm, and a "
                    "grader-union footprint of at least min_consensus_footprint_px. The "
                    "last two are DISPLAY guards added after the fact -- see "
                    "set_a_display_guard below. Percentiles are therefore computed over "
                    "the legible subpopulation, not the whole split."
                ),
            },
            "set_b_and_c": selection_guards.as_dict(),
        },
        # THE DISCLOSURE. Travels with the export so the sequence cannot be lost.
        "set_a_display_guard": {
            **SET_A_GUARD_PROVENANCE,
            "min_consensus_footprint_px": int(arguments.min_consensus_footprint),
            "min_nonempty_samples_per_arm": int(arguments.min_nonempty_samples),
            "n_eligible_after": int(ged_guards.n_eligible),
            "n_total": int(ged_guards.eligible.size),
        },
        "assertions": assertions,
        "keys": keys,
        "criteria": {"A": SET_A_CRITERION, "B": SET_B_CRITERION},
    }

    size = write_showcase(arguments.out, arrays, manifest)
    LOGGER.info("wrote %s (%.2f MiB, %d arrays)", arguments.out, size / 1024**2, len(arrays))
    if size > MAX_SHOWCASE_BYTES:
        LOGGER.warning(
            "%s is %.2f MiB, above the %.0f MiB budget for a tracked file. It has to "
            "survive a clone and a Colab checkout -- trim the Set C arrays or the number "
            "of exported samples before committing it.",
            arguments.out, size / 1024**2, MAX_SHOWCASE_BYTES / 1024**2,
        )

    print()
    print(render_case_table(all_cases))
    print()
    print(
        "Buckets above are printed because a BUCKET-BLIND prediction was registered "
        "before this run: which ambiguity buckets the mechanical percentile rule landed "
        "on is itself a result, and it has to be visible rather than buried in the file."
    )
    print(
        f"Set A guards: {ged_guards.as_dict()}\n"
        f"Set B/C guards: {selection_guards.as_dict()}"
    )
    print(
        "Verified against: "
        + ", ".join(record["source"] for record in assertions)
        + " -- every figure matched exactly."
    )
    return 0


if __name__ == "__main__":
    # Guard required on Windows, where DataLoader workers are spawned and __main__ is
    # re-imported.
    raise SystemExit(main())
