"""Tests for the variant interface, train.mode dispatch, the freeze contract and exports.

These cover the scaffolding that Phases 2 and 3 will build on: the protocol every variant
satisfies, the assertion that a base model really is frozen, and the weights-only export.
No Phase 2 improvement or Phase 3 head logic exists yet, and two tests assert exactly
that -- a scaffold must fail loudly rather than quietly behave like the baseline.
"""

from __future__ import annotations

import logging
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.distributions import MultivariateNormal

from probunet.data.lidc import DataConfig, LidcArrays, panel_batch
from probunet.data.splits import generate_split
from probunet.evaluation.metrics import consensus_scores
from probunet.evaluation.sampling import SamplingConfig, collect_per_patch_metrics
from probunet.losses.elbo import elbo_loss, kl_term
from probunet.model.encoder import LatentStats, PriorNet
from probunet.model.prob_unet import ProbUNet, ProbUNetConfig
from probunet.training.checkpoint import (
    export_weights,
    is_weights_only,
    load_checkpoint,
    save_checkpoint,
)
from probunet.training.config import TRAIN_MODES, ExperimentConfig, RunConfig, TrainConfig
from probunet.training.freeze import assert_frozen, freeze_module
from probunet.extension.head import MaskScorer, SelectionHead
from probunet.training.freeze import parameter_fingerprint
from probunet.training.trainer import Trainer
from probunet.variants import ProbUNetVariant, SegmentationVariant, SelectionHeadVariant

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
SIZE = 16
N_GRADERS = 4


def write_npz(path: Path, n_patches: int = 24, seed: int = 0) -> None:
    """Write a synthetic dataset spanning all four ambiguity buckets."""
    rng = np.random.default_rng(seed)
    series = np.array([f"s{i // 4:03d}" for i in range(n_patches)], dtype=np.str_)
    images = rng.random((n_patches, SIZE, SIZE), dtype=np.float32)
    masks = np.zeros((n_patches, N_GRADERS, SIZE, SIZE), dtype=np.uint8)
    for row in range(n_patches):
        for slot in range((row % N_GRADERS) + 1):
            masks[row, slot, : 2 + slot, : 2 + (row % 3)] = 1
    np.savez_compressed(path, images=images, masks=masks, series_uid=series)


@pytest.fixture
def tiny_experiment(tmp_path: Path) -> ExperimentConfig:
    """A small but complete experiment config over synthetic data."""
    npz = tmp_path / "lidc.npz"
    write_npz(npz)
    split = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=split)
    return ExperimentConfig(
        run=RunConfig(name="test", seed=123, device="cpu", out_dir=tmp_path / "runs"),
        model=ProbUNetConfig(latent_dim=2, base_channels=4, num_downs=2, convs_per_scale=1),
        data=DataConfig(npz_path=npz, split_path=split, batch_size=4),
        train=TrainConfig(epochs=1, limit_train_batches=2, limit_val_batches=1),
    )


@pytest.fixture
def tiny_model() -> ProbUNet:
    """A small model with reproducible weights."""
    torch.manual_seed(0)
    return ProbUNet(ProbUNetConfig(latent_dim=2, base_channels=4, num_downs=2, convs_per_scale=1))


# --------------------------------------------------------------------------- #
# The variant protocol
# --------------------------------------------------------------------------- #
def test_probunet_variant_satisfies_the_protocol(tiny_model: ProbUNet) -> None:
    """The adapter is a SegmentationVariant at runtime."""
    variant = ProbUNetVariant(tiny_model, name="baseline")
    assert isinstance(variant, SegmentationVariant)
    assert variant.name == "baseline"


def test_variant_sample_shapes_are_batched(tiny_model: ProbUNet) -> None:
    """sample() returns (B, n, H, W): batched, so one U-Net pass serves the batch."""
    variant = ProbUNetVariant(tiny_model)
    image = torch.rand(3, 1, SIZE, SIZE)
    samples = variant.sample(image, 5)
    assert samples.shape == (3, 5, SIZE, SIZE)
    assert samples.dtype == torch.uint8
    assert set(torch.unique(samples).tolist()) <= {0, 1}


def test_variant_sample_is_reproducible_with_a_generator(tiny_model: ProbUNet) -> None:
    """A seeded generator makes sampling repeatable."""
    image = torch.rand(2, 1, SIZE, SIZE)
    first = ProbUNetVariant(tiny_model, generator=torch.Generator().manual_seed(1)).sample(image, 4)
    second = ProbUNetVariant(tiny_model, generator=torch.Generator().manual_seed(1)).sample(image, 4)
    third = ProbUNetVariant(tiny_model, generator=torch.Generator().manual_seed(2)).sample(image, 4)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_variant_runs_the_unet_once_per_sample_call(tiny_model: ProbUNet) -> None:
    """Drawing n samples must not re-run the U-Net: that is the architecture's point."""
    variant = ProbUNetVariant(tiny_model)
    calls = {"n": 0}
    handle = tiny_model.unet.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    try:
        variant.sample(torch.rand(2, 1, SIZE, SIZE), 8)
    finally:
        handle.remove()
    assert calls["n"] == 1


def test_probunet_variant_does_not_select(tiny_model: ProbUNet) -> None:
    """A plain Probabilistic U-Net has no principled way to choose one sample."""
    variant = ProbUNetVariant(tiny_model)
    image = torch.rand(2, 1, SIZE, SIZE)
    assert variant.select(variant.sample(image, 4), image) is None


def test_variant_rejects_nonpositive_sample_counts(tiny_model: ProbUNet) -> None:
    """Zero samples is an error."""
    with pytest.raises(ValueError, match="n_samples"):
        ProbUNetVariant(tiny_model).sample(torch.rand(1, 1, SIZE, SIZE), 0)


def test_evaluation_accepts_a_variant(tiny_experiment: ExperimentConfig, tiny_model: ProbUNet) -> None:
    """collect_per_patch_metrics is driven by the protocol, not by a model type."""
    from probunet.data.lidc import build_data

    data = build_data(tiny_experiment.data)
    variant = ProbUNetVariant(tiny_model, generator=torch.Generator().manual_seed(0))
    per_patch = collect_per_patch_metrics(
        variant, data.loaders["val"], SamplingConfig(sample_counts=(1, 2)), torch.device("cpu")
    )
    assert "ged@2" in per_patch
    # No selection, so no selected_* metrics appear.
    assert not [key for key in per_patch if key.startswith("selected_")]


def test_selecting_variant_adds_selected_metrics(
    tiny_experiment: ExperimentConfig, tiny_model: ProbUNet
) -> None:
    """A variant that selects gets selected_dice and selected_ged, with no other change.

    Stands in for the Phase 3 head: any object satisfying the protocol and returning an
    index gets scored, so the evaluation code needs no per-variant branching.
    """
    from probunet.data.lidc import build_data

    class AlwaysFirst(ProbUNetVariant):
        """A stand-in selector that always picks sample 0."""

        def select(self, samples: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
            """Pick the first sample for every patch."""
            return torch.zeros(samples.shape[0], dtype=torch.int64)

    data = build_data(tiny_experiment.data)
    variant = AlwaysFirst(tiny_model, name="stub", generator=torch.Generator().manual_seed(0))
    assert isinstance(variant, SegmentationVariant)
    per_patch = collect_per_patch_metrics(
        variant, data.loaders["val"], SamplingConfig(sample_counts=(2,)), torch.device("cpu")
    )
    assert "selected_dice@2" in per_patch
    assert "selected_ged@2" in per_patch
    assert per_patch["selected_dice@2"].shape == per_patch["ged@2"].shape


# --------------------------------------------------------------------------- #
# The freeze contract
# --------------------------------------------------------------------------- #
def test_freeze_module_freezes_and_reports(tiny_model: ProbUNet) -> None:
    """Freezing zeroes the trainable count, switches to eval, and returns a record."""
    assert any(p.requires_grad for p in tiny_model.parameters())
    record = freeze_module(tiny_model, name="base")
    assert record["trainable_parameters"] == 0
    assert record["frozen_parameters"] == sum(p.numel() for p in tiny_model.parameters())
    assert record["training_mode"] is False
    assert not any(p.requires_grad for p in tiny_model.parameters())
    assert_frozen(tiny_model)


def test_assert_frozen_catches_a_thawed_model(tiny_model: ProbUNet) -> None:
    """The check is independent of the freeze, so a later thaw is caught."""
    freeze_module(tiny_model)
    next(iter(tiny_model.parameters())).requires_grad_(True)
    with pytest.raises(RuntimeError, match="not frozen"):
        assert_frozen(tiny_model)


def test_assert_frozen_catches_training_mode(tiny_model: ProbUNet) -> None:
    """A frozen-but-training module would still update norm statistics in later phases."""
    freeze_module(tiny_model)
    tiny_model.train()
    with pytest.raises(RuntimeError, match="training mode"):
        assert_frozen(tiny_model)


# --------------------------------------------------------------------------- #
# train.mode dispatch
# --------------------------------------------------------------------------- #
def test_train_modes_are_validated() -> None:
    """Only the two documented modes exist."""
    assert TRAIN_MODES == ("elbo", "selection_head")
    assert TrainConfig().mode == "elbo"
    with pytest.raises(ValueError, match="train.mode"):
        TrainConfig(mode="selection")


def test_selection_head_mode_requires_a_base_checkpoint(
    tiny_experiment: ExperimentConfig
) -> None:
    """The head is fitted on top of a trained model, so the base is mandatory."""
    config = dataclasses.replace(
        tiny_experiment, train=dataclasses.replace(tiny_experiment.train, mode="selection_head")
    )
    with pytest.raises(ValueError, match="requires --base-checkpoint"):
        Trainer(config)


def base_checkpoint_for(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> tuple[Path, ExperimentConfig]:
    """Write a base checkpoint and return it with a selection_head config to use it.

    Args:
        tiny_experiment: The synthetic experiment fixture.
        tmp_path: Temporary directory.

    Returns:
        The checkpoint path and a config in ``selection_head`` mode.
    """
    base = Trainer(tiny_experiment)
    checkpoint = tmp_path / "base.pt"
    save_checkpoint(
        checkpoint,
        model=base.model,
        optimizer=base.optimizer,
        scheduler=base.scheduler,
        epoch=1,
        global_step=1,
        config=tiny_experiment.to_dict(),
        seed=tiny_experiment.run.seed,
        device="cpu",
        monitor="val/total",
        best_metric=1.0,
        metrics={},
    )
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="head"),
        train=dataclasses.replace(tiny_experiment.train, mode="selection_head"),
    )
    return checkpoint, config


def test_selection_head_builds_a_frozen_base_and_a_trainable_scorer(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Stage 3: the mode no longer raises, and the freeze contract is real."""
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    assert trainer.head is not None
    assert isinstance(trainer.model, SelectionHead)
    assert trainer.base_record is not None
    assert trainer.base_record["trainable_parameters"] == 0
    assert trainer.base_record["training_mode"] is False
    assert trainer.base_record["scorer_parameters"] > 0
    assert trainer.base_fingerprint == parameter_fingerprint(trainer.head.base)

    # Every base parameter frozen, every scorer parameter trainable.
    assert not any(p.requires_grad for p in trainer.head.base.parameters())
    assert all(p.requires_grad for p in trainer.head.scorer.parameters())


def test_the_optimizer_never_receives_the_base(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """THE structural guarantee: the optimizer holds scorer parameters and nothing else.

    Handing it ``model.parameters()`` is the failure mode that would let the base drift
    and make "distribution metrics unchanged" false while every log line looked fine.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    optimized = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    scorer = {id(p) for p in trainer.head.scorer.parameters()}
    area = {id(p) for p in trainer.head.area_baseline.parameters()}
    base = {id(p) for p in trainer.head.base.parameters()}

    # The scorer AND its size-prior control -- both trainable, both disjoint from the base.
    assert optimized == scorer | area, "the optimizer is not exactly the trainable head"
    assert optimized.isdisjoint(base), "the optimizer can reach the frozen base"
    assert scorer.isdisjoint(area), "control and head share parameters; it is not a control"


def test_train_cannot_thaw_the_base(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """``model.train()`` must not put the frozen base back into training mode.

    ``nn.Module.train()`` recurses into children, so without the override the training
    loop's ordinary call would thaw the base's mode. Nothing here has dropout or batch
    norm today, so the immediate numerical effect would be nil -- which is exactly why it
    would go unnoticed until something mode-dependent was added.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    trainer.model.train()
    assert trainer.head.scorer.training is True
    assert trainer.head.base.training is False, "train() thawed the frozen base"

    trainer.model.train(True)
    assert trainer.head.base.training is False
    trainer.head.assert_base_frozen()


def test_a_full_epoch_leaves_the_base_bit_identical(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The check that catches a misconfigured optimizer: compare the BASE before/after.

    ``requires_grad`` and ``eval()`` are intent; this measures the outcome. The scorer
    must move -- otherwise "the base did not move" would pass vacuously because nothing
    trained at all.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    before_base = parameter_fingerprint(trainer.head.base)
    before_scorer = parameter_fingerprint(trainer.head.scorer)

    metrics = trainer.train_epoch(0)

    assert parameter_fingerprint(trainer.head.base) == before_base, "the base moved"
    assert parameter_fingerprint(trainer.head.scorer) != before_scorer, (
        "the scorer did not move, so the freeze check passed vacuously"
    )
    assert np.isfinite(metrics["total"])
    assert trainer.global_step > 0


def test_a_drifting_base_is_caught_at_the_epoch_boundary(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Simulate the failure: perturb the base mid-run and confirm the epoch check fires.

    Without this, the fingerprint comparison is only ever exercised on the passing path,
    and a check that has never failed is not known to work.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    with torch.no_grad():
        next(iter(trainer.head.base.parameters())).add_(1e-3)

    with pytest.raises(RuntimeError, match="not actually frozen"):
        trainer.train_epoch(0)


def test_head_trains_and_monitors_the_deliverable(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Stage 4: the monitored metric exists, so best.pt is written on the real quantity.

    ``val/selected_consensus_dice`` is the deliverable itself -- the consensus score of the
    sample the head picked -- so it cannot be gamed by a constant predictor, which
    degenerates to a fixed arbitrary pick and scores about the same as random selection.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    config = dataclasses.replace(
        config,
        train=dataclasses.replace(config.train, epochs=2),
        checkpoint=dataclasses.replace(
            config.checkpoint, monitor="val/selected_consensus_dice", mode="max"
        ),
    )
    trainer = Trainer(config, base_checkpoint=checkpoint)
    summary = trainer.train()

    assert (trainer.checkpoint_dir / "best.pt").exists()
    assert trainer.best_metric is not None
    last = summary["history"][-1]
    for key in (
        "val/selected_consensus_dice",
        "val/random_consensus_dice",
        "val/oracle_consensus_dice",
        "val/ceiling",
        "val/huber",
        "val/spearman",
        "val/selected_fraction_of_ceiling",
    ):
        assert key in last, key
    # Never reported against 1.0: soft-consensus scores are bounded by the ceiling.
    assert last["val/selected_consensus_dice"] <= last["val/ceiling"] + 1e-6
    # The selected sample is one of the candidates, so it cannot beat the oracle.
    assert last["val/selected_consensus_dice"] <= last["val/oracle_consensus_dice"] + 1e-6
    # The ELBO is not computed at all under a frozen base.
    assert "val/total" not in last


def test_a_missing_monitor_is_still_loud(
    tiny_experiment: ExperimentConfig, tmp_path: Path, caplog
) -> None:
    """A monitor that names nothing must warn and write no best.pt, in any mode."""
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    config = dataclasses.replace(
        config,
        checkpoint=dataclasses.replace(config.checkpoint, monitor="val/does_not_exist"),
    )
    trainer = Trainer(config, base_checkpoint=checkpoint)
    with caplog.at_level(logging.WARNING):
        trainer.train()
    assert not (trainer.checkpoint_dir / "best.pt").exists()
    assert "val/does_not_exist" in " ".join(r.message for r in caplog.records)


def test_the_head_never_touches_the_posterior(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Candidates come from the PRIOR. A posterior-trained head learns a constant.

    Posterior samples have seen the ground-truth mask and are almost always good, so a head
    trained on them never meets a bad candidate. Asserted by watching the posterior net for
    any call at all during a training epoch.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    calls = {"n": 0}
    handle = trainer.head.base.posterior_net.register_forward_hook(
        lambda *_: calls.__setitem__("n", calls["n"] + 1)
    )
    try:
        trainer.train_epoch(0)
    finally:
        handle.remove()
    assert calls["n"] == 0, "the posterior net was invoked during head training"


def test_the_regression_target_is_data_not_a_gradient_path(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Targets are detached: gradients flow only through the head's prediction."""
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    batch = next(iter(trainer.train_loader))

    graders = batch["masks"].to(trainer.device)
    _, candidates = trainer.head.sample_candidates(
        batch["image"].to(trainer.device), 3
    )
    targets = trainer._consensus_targets(graders, candidates)
    assert not targets.requires_grad
    assert targets.grad_fn is None
    # And they are the real soft-consensus scores, not a stand-in.
    assert torch.allclose(targets, consensus_scores(candidates, graders))


def test_validation_candidates_are_a_fixed_shared_set(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The candidate set is identical across validation passes, so epochs are comparable.

    Re-seeded at the start of every pass. Without this an epoch-to-epoch change in the
    selected score would mix the head improving with the draw changing.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    first = trainer.validate()
    second = trainer.validate()
    for key in ("random_consensus_dice", "oracle_consensus_dice", "ceiling"):
        assert first[key] == pytest.approx(second[key], rel=1e-9), key


def test_head_training_uses_all_four_graders(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The head trains on the EVAL-mode four-mask shape, not the single-grader pairing.

    The consensus needs every grader; a target built from one quarter of the evidence
    would be a different quantity. DEVIATIONS entry 13.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    batch = next(iter(trainer.train_loader))
    assert "masks" in batch and batch["masks"].shape[1] == N_GRADERS
    assert "mask" not in batch, "still on the single-grader training path"
    # And it is the TRAIN split, not val.
    assert len(trainer.train_loader.dataset) == len(trainer.data.datasets["train"])


def test_selection_head_refuses_augmentation(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Augmentation is off for this phase, and the refusal is explicit."""
    from probunet.data.transforms import AugmentationConfig

    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            augmentation=AugmentationConfig(enabled=True, pad_to_px=SIZE + 4),
        ),
    )
    with pytest.raises(ValueError, match="augmentation.enabled=false"):
        Trainer(config, base_checkpoint=checkpoint)


def test_mean_centered_targets_are_available_as_the_pre_registered_fallback(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The fallback for the image-only shortcut exists and does what it says.

    Recorded in advance (FINDINGS 4.4) so that switching it on later is a planned
    contingency rather than an unexplained pivot. Centering within each image removes the
    between-image component a shortcut would exploit.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    config = dataclasses.replace(
        config, head=dataclasses.replace(config.head, mean_centered_targets=True)
    )
    trainer = Trainer(config, base_checkpoint=checkpoint)
    batch = next(iter(trainer.train_loader))
    _, candidates = trainer.head.sample_candidates(
        batch["image"].to(trainer.device), 4
    )
    targets = trainer._consensus_targets(batch["masks"].to(trainer.device), candidates)
    assert torch.allclose(
        targets.mean(dim=1), torch.zeros(targets.shape[0]), atol=1e-6
    )


def test_spearman_exclusions_are_reported_not_dropped(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Degenerate images are counted per bucket -- the fraction is itself a finding.

    It measures how often the sampler offered no real choice at all.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    metrics = trainer.validate()

    assert "spearman_excluded_fraction" in metrics
    assert 0.0 <= metrics["spearman_excluded_fraction"] <= 1.0
    assert metrics["spearman_images"] >= 0
    per_bucket = [k for k in metrics if k.startswith("spearman_excluded_fraction_bucket")]
    assert per_bucket, "no per-bucket exclusion bookkeeping"
    for key in per_bucket:
        assert 0.0 <= metrics[key] <= 1.0


def test_the_freeze_record_reaches_the_checkpoint(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The freeze record is evidence, so it has to survive into the artifact."""
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    trainer.train_epoch(0)
    trainer._checkpoint(0, {"train/total": 1.0})

    state = load_checkpoint(trainer.checkpoint_dir / "last.pt")
    assert state.metrics["freeze/frozen_parameters"] > 0
    assert state.metrics["freeze/trainable_parameters"] == 0
    assert state.metrics["freeze/scorer_parameters"] > 0


def test_the_scorer_can_represent_a_ratio() -> None:
    """The projection has a hidden layer because the target is a RATIO, not a sum.

    Soft Dice is ``2*sum(s*c) / (sum(s) + sum(c))``. Global average pooling hands the
    readout spatial averages -- an overlap term and two area terms -- and a single linear
    layer can only form a weighted SUM of those, never their quotient. One ReLU hidden
    layer can approximate the division.

    Structural, not a capacity guess, so it is asserted structurally: there must be a
    non-linearity between the pooled features and the scalar output.
    """
    scorer = MaskScorer(feature_channels=32)
    layers = list(scorer.project)
    assert len(layers) == 3, "projection collapsed back to a single linear readout"
    assert isinstance(layers[0], torch.nn.Linear)
    assert isinstance(layers[1], torch.nn.ReLU), "no non-linearity: cannot form a quotient"
    assert isinstance(layers[2], torch.nn.Linear)
    assert layers[2].out_features == 1


def test_head_capacity_is_the_recorded_size() -> None:
    """~118.5k trainable parameters at the real feature width."""
    scorer = MaskScorer(feature_channels=32)
    total = sum(p.numel() for p in scorer.parameters())
    assert 118_000 <= total <= 119_000, total


def test_head_checkpoint_identifies_which_base_it_came_from(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The artifact must attribute the head to its base, verifiably.

    head-on-Phase1 versus head-on-Phase2 is only a meaningful ablation if each head
    checkpoint says which base it was frozen on top of -- and a filename is not a
    guarantee. The parameter hash makes the identity checkable rather than asserted, which
    also replaces the integrity signal given up by not computing the ELBO on a frozen base.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    trainer.train_epoch(0)
    trainer._checkpoint(0, {"train/total": 1.0})

    state = load_checkpoint(trainer.checkpoint_dir / "last.pt")
    provenance = state.base_provenance
    assert provenance is not None
    assert provenance["checkpoint"] == str(checkpoint)
    for key in ("git_revision", "device", "torch_version", "parameter_sha256", "epoch"):
        assert provenance[key] is not None, key
    assert provenance["latent_covariance"] == config.model.latent_covariance

    # VERIFIABLE, not merely recorded: the hash must match the base actually in the
    # checkpoint, and it is the same hash the epoch-boundary check compares against.
    assert provenance["parameter_sha256"] == trainer.base_fingerprint
    assert provenance["parameter_sha256"] == parameter_fingerprint(trainer.head.base)

    # A different base gives a different hash, so the field discriminates.
    other = ProbUNet(config.model)
    assert parameter_fingerprint(other) != provenance["parameter_sha256"]


def test_elbo_checkpoints_carry_no_base_provenance(
    tiny_experiment: ExperimentConfig,
) -> None:
    """Nothing is frozen in ELBO mode, so the field stays None rather than empty-but-present."""
    trainer = Trainer(tiny_experiment)
    assert trainer.base_provenance is None
    trainer._checkpoint(0, {"val/total": 1.0})
    state = load_checkpoint(trainer.checkpoint_dir / "last.pt")
    assert state.base_provenance is None


def test_elbo_mode_is_unaffected(tiny_experiment: ExperimentConfig) -> None:
    """The default mode still trains normally."""
    trainer = Trainer(tiny_experiment)
    assert trainer.base_record is None
    assert "mode          : elbo" in trainer.describe()
    metrics = trainer.train_epoch(0)
    assert np.isfinite(metrics["total"])


# --------------------------------------------------------------------------- #
# Weights-only export
# --------------------------------------------------------------------------- #
def test_export_weights_is_smaller_and_traceable(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The export drops optimizer state but keeps provenance."""
    trainer = Trainer(tiny_experiment)
    trainer.train_epoch(0)  # populate Adam moment buffers
    full = tmp_path / "full.pt"
    save_checkpoint(
        full,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=7,
        global_step=42,
        config=tiny_experiment.to_dict(),
        seed=tiny_experiment.run.seed,
        device="cpu",
        monitor="val/total",
        best_metric=0.5,
        metrics={"val/total": 0.5},
    )

    export = tmp_path / "weights.pt"
    summary = export_weights(full, export)
    assert summary["destination_bytes"] < summary["source_bytes"]
    assert is_weights_only(export)
    assert not is_weights_only(full)

    payload = torch.load(export, map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert "scheduler" not in payload
    assert "rng" not in payload
    # Still traceable to the run that produced it.
    assert payload["epoch"] == 7
    assert payload["git_revision"]
    assert json.loads(payload["config_json"])["model"]["latent_dim"] == 2


def test_export_can_be_loaded_for_evaluation(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """An export is enough to rebuild and evaluate a model."""
    trainer = Trainer(tiny_experiment)
    full = tmp_path / "full.pt"
    save_checkpoint(
        full,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=3,
        global_step=9,
        config=tiny_experiment.to_dict(),
        seed=123,
        device="cpu",
        monitor="val/total",
        best_metric=1.0,
        metrics={},
    )
    export = tmp_path / "weights.pt"
    export_weights(full, export)

    from probunet.evaluation.runner import load_variant

    variant, config, state = load_variant(export, torch.device("cpu"))
    assert state.epoch == 3
    assert config.model.latent_dim == 2
    samples = variant.sample(torch.rand(2, 1, SIZE, SIZE), 3)
    assert samples.shape == (2, 3, SIZE, SIZE)
    for saved, loaded in zip(trainer.model.parameters(), variant.model.parameters(), strict=True):
        assert torch.equal(saved.detach(), loaded.detach())


def test_export_missing_source(tmp_path: Path) -> None:
    """Exporting a checkpoint that is not there fails clearly."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        export_weights(tmp_path / "absent.pt", tmp_path / "out.pt")


# --------------------------------------------------------------------------- #
# The subset export path
# --------------------------------------------------------------------------- #
def test_subset_resolves_global_indices(tmp_path: Path) -> None:
    """A subset addresses the same patches by their full-dataset indices."""
    npz = tmp_path / "lidc.npz"
    write_npz(npz, n_patches=20)
    full = LidcArrays.load(npz)
    assert not full.is_subset

    rows = np.array([3, 11, 17], dtype=np.int64)
    subset_path = tmp_path / "subset.npz"
    np.savez_compressed(
        subset_path,
        images=full.images[rows],
        masks=full.masks[rows],
        series_uid=full.series_uid[rows],
        source_index=rows,
    )
    subset = LidcArrays.load(subset_path)
    assert subset.is_subset
    assert len(subset) == 3
    assert subset.resolve_indices(np.array([17, 3])).tolist() == [2, 0]


def test_panel_batch_is_identical_from_full_and_subset(tmp_path: Path) -> None:
    """The panel has ONE code path: same indices, same output, either source file.

    This is what lets the notebook run on Colab from the tracked few-MB subset while the
    training loop uses the full dataset, without a notebook-specific branch.
    """
    npz = tmp_path / "lidc.npz"
    write_npz(npz, n_patches=20)
    full = LidcArrays.load(npz)

    rows = np.array([2, 9, 14], dtype=np.int64)
    subset_path = tmp_path / "subset.npz"
    np.savez_compressed(
        subset_path,
        images=full.images[rows],
        masks=full.masks[rows],
        series_uid=full.series_uid[rows],
        source_index=rows,
    )
    subset = LidcArrays.load(subset_path)

    from_full = panel_batch(full, rows)
    from_subset = panel_batch(subset, rows)
    assert torch.equal(from_full[0], from_subset[0])
    assert torch.equal(from_full[1], from_subset[1])
    assert from_full[0].shape == (3, 1, SIZE, SIZE)
    assert from_full[1].shape == (3, N_GRADERS, SIZE, SIZE)


def test_subset_rejects_absent_patches(tmp_path: Path) -> None:
    """Asking a subset for a patch it does not hold is an error, not a wrong image."""
    npz = tmp_path / "lidc.npz"
    write_npz(npz, n_patches=20)
    full = LidcArrays.load(npz)
    rows = np.array([1, 2], dtype=np.int64)
    subset_path = tmp_path / "subset.npz"
    np.savez_compressed(
        subset_path,
        images=full.images[rows],
        masks=full.masks[rows],
        series_uid=full.series_uid[rows],
        source_index=rows,
    )
    with pytest.raises(KeyError, match="not in this subset"):
        LidcArrays.load(subset_path).resolve_indices(np.array([19]))


# --------------------------------------------------------------------------- #
# The three shipped configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["baseline", "modernized", "extension"])
def test_shipped_configs_parse(name: str) -> None:
    """All three variant configs load."""
    config = ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml")
    assert config.run.name == name


@pytest.mark.parametrize("name", ["baseline", "modernized", "extension", "smoke"])
def test_no_shipped_config_enables_amp(name: str) -> None:
    """AMP is CUDA-only and raises elsewhere, so it must never ship enabled.

    Otherwise a config that works on the CUDA machine would fail immediately on macOS.
    """
    raw = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    assert raw.get("train", {}).get("amp", False) is False


@pytest.mark.parametrize("name", ["baseline", "modernized", "extension"])
def test_no_shipped_config_caps_channels(name: str) -> None:
    """Capping is a memory optimization, never a variant under test."""
    raw = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    assert raw["model"]["max_channels"] is None


def test_variants_differ_only_by_the_improvement_flag() -> None:
    """baseline and modernized differ in EXACTLY ``latent_covariance``, which is ``full``.

    Replaces the former ``baseline.model == modernized.model`` assertion, which could not
    survive Phase 2 introducing a real architectural flag. The claim is now the sharper
    one: the two configs differ in **exactly one field**, so the comparison isolates a
    single variable. Any second difference fails here by name.

    **The equality is deliberate and was tightened from a subset check in Stage 5.** Written
    as ``differing <= {"latent_covariance"}`` this passes when ``differing`` is *empty* --
    that is, when the flag was never flipped and both arms are the baseline under two
    different run names. Nothing else in the suite would have caught it: the bit-identity
    test only asserts flag-off reproduces Phase 1, and
    ``test_shipped_configs_agree_with_the_models_they_build`` is deliberately flag-agnostic.
    A green suite would then have accompanied a comparison of the baseline against itself --
    the same undetectable false null the Stage 3 positive control exists to prevent, one
    level up at the config layer. Hence both halves below: exactly one field differs, **and**
    it holds the value Phase 2 is about.
    """
    baseline = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml")
    modernized = ExperimentConfig.from_yaml(CONFIGS / "modernized.yaml")

    differing = {
        field.name
        for field in dataclasses.fields(baseline.model)
        if getattr(baseline.model, field.name) != getattr(modernized.model, field.name)
    }
    assert differing == {"latent_covariance"}, (
        f"modernized.yaml differs from baseline.yaml in {sorted(differing)}; Phase 2 is "
        "exactly one flag. An empty set means the flag was never flipped and both arms "
        "would train the baseline; anything beyond latent_covariance confounds the "
        "comparison."
    )
    assert baseline.model.latent_covariance == "diagonal"
    assert modernized.model.latent_covariance == "full", (
        "modernized.yaml must actually enable the Phase 2 improvement -- without this the "
        "headline comparison is the baseline against itself"
    )

    assert baseline.optim == modernized.optim
    assert baseline.schedule == modernized.schedule
    assert baseline.loss == modernized.loss
    assert baseline.train.iterations == modernized.train.iterations, "budgets must match"
    assert baseline.data == modernized.data, "the data pipeline must be identical"
    assert baseline.run.seed == modernized.run.seed


def test_flag_off_is_bit_identical_to_the_baseline() -> None:
    """modernized.yaml with the improvement flag off reproduces baseline.yaml EXACTLY.

    This is the load-bearing test of Phase 2. The whole comparison rests on flag-off being
    Phase 1 rather than something numerically near it, which is why the diagonal case takes
    the ``Independent(Normal)`` path unchanged instead of a MultivariateNormal with a
    diagonal factor -- the latter is algebraically identical but reaches the answer through
    different kernels and disagrees at ~5e-7.

    Bit-exact equality under a fixed seed, on logits, z and every loss term.
    """
    baseline = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml").model
    modernized = ExperimentConfig.from_yaml(CONFIGS / "modernized.yaml").model
    flag_off = dataclasses.replace(modernized, latent_covariance="diagonal")
    assert baseline == flag_off, "with the flag off the two architectures must be equal"

    generator = torch.Generator().manual_seed(11)
    image = torch.rand(2, 1, 32, 32, generator=generator)
    mask = (torch.rand(2, 32, 32, generator=generator) > 0.6).to(torch.int64)

    outputs = []
    for architecture in (baseline, flag_off):
        # Rebuilt under the same seed, so initialization is identical too.
        torch.manual_seed(4242)
        model = ProbUNet(
            dataclasses.replace(architecture, base_channels=8, num_downs=2, convs_per_scale=1)
        )
        torch.manual_seed(99)
        output = model(image, mask)
        terms = elbo_loss(output.logits, mask, output.posterior, output.prior, beta=1.0)
        outputs.append(
            (output.logits, output.z, {k: v.detach().clone() for k, v in terms.items()})
        )

    (left_logits, left_z, left_terms), (right_logits, right_z, right_terms) = outputs
    assert torch.equal(left_logits, right_logits), "logits differ with the flag off"
    assert torch.equal(left_z, right_z), "latent samples differ with the flag off"
    for key in sorted(left_terms):
        assert torch.equal(left_terms[key], right_terms[key]), f"{key} differs"


def test_full_covariance_flag_is_actually_engaged() -> None:
    """POSITIVE CONTROL: the deliberate complement to the bit-identity test.

    Distribution construction dispatches on ``stats.lower is None``, so a plumbing bug that
    dropped ``lower`` would make a full-configured model train **diagonally** while every
    label, config dump and report said otherwise. The bit-identity test cannot catch that:
    it only asserts flag-off is identical, never that flag-on is different. The result would
    be an undetectable false null -- the worst outcome available to this project -- so the
    engaged path is asserted directly, through the real config-to-model wiring rather than a
    hand-constructed encoder.
    """
    config = ExperimentConfig.from_dict(
        {"model": {"latent_dim": 6, "latent_covariance": "full"}}
    )
    assert config.model.full_covariance is True
    model = ProbUNet(
        dataclasses.replace(
            config.model, base_channels=8, num_downs=2, convs_per_scale=1
        )
    )

    # (a) The architecture really is the wide-head one.
    for net in (model.prior_net, model.posterior_net):
        assert net.full_covariance is True
        assert net.head.out_channels == 27, "head is not N + N(N+1)/2 wide"
        assert net.n_lower == 15

    generator = torch.Generator().manual_seed(3)
    image = torch.rand(2, 1, 32, 32, generator=generator)
    mask = (torch.rand(2, 32, 32, generator=generator) > 0.6).to(torch.int64)
    encoded = model.encode(image, mask)

    # (b) The parameters carry a factor, and the distributions are the full family.
    assert encoded.prior_stats.lower is not None
    assert encoded.posterior_stats.lower is not None
    assert isinstance(encoded.prior, MultivariateNormal)
    assert isinstance(encoded.posterior, MultivariateNormal)

    # (b, belt and braces) A full-configured encoder handed diagonal parameters refuses,
    # rather than silently building the Phase 1 distribution.
    with pytest.raises(RuntimeError, match="latent parameterization mismatch"):
        model.prior_net.distribution_from_stats(
            LatentStats(encoded.prior_stats.mu, encoded.prior_stats.logvar)
        )
    # And the converse, so neither direction can drift.
    diagonal_net = ProbUNet(
        dataclasses.replace(
            config.model,
            latent_covariance="diagonal",
            base_channels=8,
            num_downs=2,
            convs_per_scale=1,
        )
    ).prior_net
    with pytest.raises(RuntimeError, match="latent parameterization mismatch"):
        diagonal_net.distribution_from_stats(encoded.prior_stats)


def test_full_covariance_diverges_from_diagonal_once_correlations_are_nonzero() -> None:
    """The other half of the positive control: the paths differ where they should.

    Zero-init deliberately makes the two agree at step 0, and a separate test asserts that.
    Here the correlations are set non-zero, and z and the KL must then diverge from what the
    diagonal path produces from the same mu and logvar. Without this, "full covariance"
    could be a no-op that every other test happily passes.
    """
    torch.manual_seed(0)
    batch, dims = 4, 6
    mu = torch.randn(batch, dims)
    logvar = torch.randn(batch, dims) * 0.3
    prior_mu = torch.randn(batch, dims)
    prior_logvar = torch.randn(batch, dims) * 0.3
    correlations = torch.randn(batch, dims * (dims - 1) // 2)

    diagonal_net = PriorNet(image_channels=1, latent_dim=dims)
    full_net = PriorNet(image_channels=1, latent_dim=dims, full_covariance=True)

    diagonal_posterior = diagonal_net.distribution_from_stats(LatentStats(mu, logvar))
    diagonal_prior = diagonal_net.distribution_from_stats(
        LatentStats(prior_mu, prior_logvar)
    )
    full_posterior = full_net.distribution_from_stats(
        LatentStats(mu, logvar, lower=correlations)
    )
    full_prior = full_net.distribution_from_stats(
        LatentStats(prior_mu, prior_logvar, lower=torch.zeros_like(correlations))
    )

    # The KL must move: correlations change the geometry, not just the marginals.
    diagonal_kl = kl_term(diagonal_posterior, diagonal_prior)
    full_kl = kl_term(full_posterior, full_prior)
    assert not torch.isclose(diagonal_kl, full_kl, atol=1e-4), (
        "the KL is unchanged by non-zero correlations: the full path is a no-op"
    )

    # And the samples must move, under the same noise.
    torch.manual_seed(7)
    diagonal_z = diagonal_posterior.rsample()
    torch.manual_seed(7)
    full_z = full_posterior.rsample()
    assert not torch.allclose(diagonal_z, full_z, atol=1e-5), (
        "z is unchanged by non-zero correlations: L is not reaching the sample"
    )
    # The first coordinate is unaffected by construction -- row 0 of L has no strict lower
    # entries -- which is a useful check that the factor is lower- and not upper-triangular.
    assert torch.allclose(diagonal_z[:, 0], full_z[:, 0], atol=1e-6)


def test_shipped_configs_agree_with_the_models_they_build() -> None:
    """Whatever each shipped config declares, the built model must actually be that.

    Flag-agnostic on purpose, so it keeps its meaning as modernized.yaml flips to ``full``
    in Stage 5 rather than needing an edit at the moment it matters most.
    """
    for name in ("baseline", "modernized", "extension", "ablation_no_augmentation"):
        declared = ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml").model
        model = ProbUNet(
            dataclasses.replace(
                declared, base_channels=8, num_downs=2, convs_per_scale=1
            )
        )
        expected_width = declared.latent_head_outputs
        for net in (model.prior_net, model.posterior_net):
            assert net.full_covariance is declared.full_covariance, name
            assert net.head.out_channels == expected_width, (
                f"{name}: config declares latent_covariance={declared.latent_covariance!r} "
                f"(head width {expected_width}) but the model built a "
                f"{net.head.out_channels}-wide head"
            )


def test_extension_config_uses_selection_head_mode() -> None:
    """The extension config is the only one in selection_head mode."""
    extension = ExperimentConfig.from_yaml(CONFIGS / "extension.yaml")
    assert extension.train.mode == "selection_head"
    for other in ("baseline", "modernized"):
        assert ExperimentConfig.from_yaml(CONFIGS / f"{other}.yaml").train.mode == "elbo"


def _notebook_code() -> str:
    """Every code cell of the submission notebook, concatenated."""
    notebook = json.loads((REPO_ROOT / "notebooks" / "submission.ipynb").read_text())
    assert notebook["nbformat"] == 4
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_keeps_its_logic_in_the_package() -> None:
    """The notebook is a narrative layer: markdown, a call, a figure.

    Every helper it needs lives in :mod:`probunet.notebook`, so a cell may not define a
    function or a class. That is what keeps the notebook reviewable as *argument* rather
    than as a second, untested copy of the library.
    """
    joined = _notebook_code()
    assert "class " not in joined, "model/metric definitions belong in the package"
    assert "def " not in joined, "logic belongs in the package, not in cells"
    assert "from probunet.notebook import" in joined, (
        "the display helpers must come from probunet.notebook, not from a cell"
    )
    assert "Trainer(" not in joined, (
        "no in-notebook training loop: the demo goes through scripts/train.py, the same "
        "entry point every reported run used"
    )
    assert ".train()" not in joined


def test_notebook_installs_nothing() -> None:
    """A fresh Colab already has torch, numpy, matplotlib and scipy.

    Installing anything is the most common way a grader's run breaks, and the notebook's
    first cell promises it does not. This is that promise, asserted.
    """
    joined = _notebook_code()
    for form in ("!pip", "%pip", "!apt", "pip3 install", "install("):
        assert form not in joined, f"the notebook must not install anything (found {form!r})"


def test_notebook_reads_tracked_artifacts_and_reports_nothing_it_computed() -> None:
    """Tier 1 renders recorded predictions; Tier 2 trains but keeps nothing.

    **CLAUDE.md's original rule was "the notebook must not train".** That rule existed so
    no *reported* number could come from a notebook run on a different device, where seeds
    do not reproduce. The notebook now carries a Tier 2 executability demo -- the course
    requires a working implementation runnable in Colab -- and the rule's purpose is
    preserved by construction instead: Tier 2 trains on a tracked 288-patch subset through
    its own config, reports nothing, and deletes its run directory. Every reported number
    still comes from a tracked file.
    """
    joined = _notebook_code()
    # Tier 1: the recorded predictions and the published tables.
    assert "showcase.npz" in joined, "Tier 1 must read the tracked showcase export"
    assert "consensus_selection_test.json" in joined
    assert "comparison_test.json" in joined
    # Tier 2: its own config, its own subset, and no surviving weights.
    assert "colab_demo.yaml" in joined, "the demo must use its own config"
    assert "scripts/train.py" in joined, "the demo must use the real training entry point"
    assert "rmtree" in joined, (
        "Tier 2 must discard its weights: nothing it produces may be retained, reported "
        "or committed"
    )


# --------------------------------------------------------------------------- #
# Stage 5: selection through the protocol, the results table, the ablation guard
# --------------------------------------------------------------------------- #
def test_ged_is_bit_identical_with_and_without_the_head(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """THE first half of the extension's claim: distribution metrics unchanged.

    Attaching the head must not move GED, diversity, or anything else computed from the
    sample *set* -- the head only chooses among samples the frozen base already produced.
    Asserted as **bit-identical**, not close: sampling is literally the same code on the
    same frozen weights with the same seed, so any difference at all means something is
    reaching the generative path that should not be.

    Also: the distribution metrics keep using the four SEPARATE grader masks. Consensus is
    the selection target only; collapsing the graders for GED would discard exactly the
    spread Phase 1 measures.
    """
    from probunet.data.lidc import build_data

    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    data = build_data(config.data)
    sampling = SamplingConfig(sample_counts=(2, 4))

    plain = ProbUNetVariant(
        trainer.head.base, name="base", generator=torch.Generator().manual_seed(2018)
    )
    with_head = SelectionHeadVariant(
        trainer.head, name="extension", generator=torch.Generator().manual_seed(2018)
    )

    without = collect_per_patch_metrics(
        plain, data.loaders["val"], sampling, torch.device("cpu")
    )
    with_ = collect_per_patch_metrics(
        with_head, data.loaders["val"], sampling, torch.device("cpu")
    )

    for key, values in without.items():
        assert np.array_equal(np.asarray(values), np.asarray(with_[key])), (
            f"{key} moved when the head was attached"
        )
    # The head adds its own columns and changes nothing else.
    assert "selected_dice@4" in with_ and "selected_dice@4" not in without


def test_only_the_extension_selects(tiny_model: ProbUNet) -> None:
    """One variant-dependent branch in evaluation: did select() return an index?"""
    head = SelectionHead(tiny_model)
    image = torch.rand(2, 1, SIZE, SIZE)
    plain = ProbUNetVariant(tiny_model)
    extension = SelectionHeadVariant(head)

    samples = plain.sample(image, 4)
    assert plain.select(samples, image) is None
    chosen = extension.select(samples, image)
    assert isinstance(chosen, torch.Tensor)
    assert chosen.shape == (2,)
    assert bool(((chosen >= 0) & (chosen < 4)).all())
    assert isinstance(extension, SegmentationVariant)


def test_the_area_control_selects_on_area_alone(tiny_model: ProbUNet) -> None:
    """The control must not be able to see the image -- otherwise it is not a control."""
    head = SelectionHead(tiny_model)
    control = SelectionHeadVariant(head, by_area=True)
    samples = torch.zeros(1, 3, SIZE, SIZE, dtype=torch.uint8)
    samples[0, 0, :2, :2] = 1
    samples[0, 1, :4, :4] = 1
    samples[0, 2, :6, :6] = 1

    first = control.select(samples, torch.rand(1, 1, SIZE, SIZE))
    second = control.select(samples, torch.rand(1, 1, SIZE, SIZE))
    assert torch.equal(first, second), "the area control's pick depended on the image"


def test_selection_table_carries_every_requested_column(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The Stage 5 table: head beside its control, the gap fraction, and the empty rows."""
    from probunet.data.lidc import build_data
    from probunet.evaluation.headroom import measure_selection, per_bucket, render_selection

    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    data = build_data(config.data)

    results = measure_selection(
        trainer.head, data.loaders["val"], torch.device("cpu"), 6, config.head.eval_seed
    )
    report = per_bucket(results)
    row = report["all"]

    for column in ("head", "area_only", "random", "oracle", "ceiling", "emptiest"):
        assert column in row, column
    # The number the report leads with, and its control immediately beside it.
    assert "gap_captured_head" in row
    assert "gap_captured_area_only" in row
    # The pre-registered rows.
    assert 0.0 <= row["all_candidates_empty_fraction"] <= 1.0
    if row["all_candidates_empty_fraction"] < 1.0:
        assert "oracle_where_offered" in row
        assert row["n_offered"] <= row["n"]

    # Bounds that must hold whatever the head learned.
    assert row["head"]["mean"] <= row["oracle"]["mean"] + 1e-6
    assert row["area_only"]["mean"] <= row["oracle"]["mean"] + 1e-6
    assert row["oracle"]["mean"] <= row["ceiling"]["mean"] + 1e-6

    # The head's edge over its size-prior control, reported against BOTH denominators,
    # because the bucket trend reverses depending on which is used (FINDINGS 4.5).
    assert "head_edge_over_area" in row
    assert "head_edge_frac_of_total_headroom" in row
    assert "head_edge_frac_of_headroom_area_left" in row
    # The two diagnostics that separate "ranks non-empty above empty" from
    # "discriminates among non-empty".
    assert "area_spearman" in row and "head_spearman" in row
    assert "pred_spread_within_image" in row and "pred_std_within_image" in row
    # Whether the control collapses to the deterministic largest-area rule -- measured,
    # never assumed: the scorer is a ReLU MLP and is not guaranteed monotone.
    assert 0.0 <= row["area_picks_largest"]["mean"] <= 1.0

    text = render_selection(report)
    for token in ("head", "area", "edge", "e/tot", "e/left", "rho_h", "rho_a", "lrgst"):
        assert token in text


def test_all_candidates_empty_is_counted_not_averaged_away() -> None:
    """Images where the sampler offered nothing are reported, and oracle is split.

    Those images score exactly 0 under every rule, so they cap what any selector can
    achieve. Separating them makes "the sampler offered nothing" distinguishable from
    "the sampler offered something poorly localized".
    """
    from probunet.evaluation.headroom import per_bucket

    results = {
        "head": np.array([0.0, 0.4, 0.0, 0.6]),
        "random": np.array([0.0, 0.2, 0.0, 0.3]),
        "oracle": np.array([0.0, 0.5, 0.0, 0.7]),
        "area_only": np.array([0.0, 0.3, 0.0, 0.4]),
        "all_empty": np.zeros(4),
        "ceiling": np.array([0.4, 0.4, 1.0, 1.0]),
        # Two images where every candidate was empty.
        "nonempty_frac": np.array([0.0, 0.5, 0.0, 1.0]),
        "n_nonempty": np.array([1, 1, 4, 4]),
        "index": np.arange(4),
    }
    report = per_bucket(results)
    assert report["all"]["all_candidates_empty_fraction"] == pytest.approx(0.5)
    assert report["1 grader"]["all_candidates_empty_fraction"] == pytest.approx(0.5)
    # Oracle over everything is dragged down by the zeros; restricted to images that
    # offered something it is not.
    assert report["all"]["oracle"]["mean"] == pytest.approx(0.3)
    assert report["all"]["oracle_where_offered"] == pytest.approx(0.6)
    assert report["all"]["n_offered"] == 2


def test_ablation_guard_refuses_mismatched_head_runs() -> None:
    """A refusal, not a warning: a warning at the end of a long run gets read past."""
    import copy

    from probunet.extension.ablation import ablation_signature, assert_comparable

    arm = {
        "head": {
            "train_samples": 8, "eval_samples": 16, "eval_seed": 2018,
            "huber_delta": 0.1, "scorer_channels": [32, 64, 128],
            "mean_centered_targets": False,
        },
        "model": {"base_channels": 32, "latent_covariance": "diagonal"},
        "optim": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0,
                  "betas": [0.9, 0.999], "eps": 1e-8},
        "schedule": {"name": "constant"},
        "train": {"epochs": 30, "iterations": None, "mode": "selection_head"},
        "data": {"batch_size": 32, "split_path": "s.json"},
        "run": {"seed": 2018},
    }

    # The variable under test may differ; nothing else may.
    other = copy.deepcopy(arm)
    other["model"]["latent_covariance"] = "full"
    assert_comparable({"p1": ablation_signature(arm), "p2": ablation_signature(other)})

    for path, value in (
        (("head", "eval_seed"), 99),
        (("head", "train_samples"), 4),
        (("head", "eval_samples"), 8),
        (("head", "huber_delta"), 1.0),
        (("head", "scorer_channels"), [16, 32]),
        (("model", "base_channels"), 16),
        (("optim", "lr"), 1e-4),
        (("train", "epochs"), 20),
        (("data", "batch_size"), 16),
        (("run", "seed"), 7),
    ):
        broken = copy.deepcopy(other)
        broken[path[0]][path[1]] = value
        with pytest.raises(ValueError, match="REFUSING"):
            assert_comparable(
                {"p1": ablation_signature(arm), "p2": ablation_signature(broken)}
            )


def test_ablation_guard_treats_a_missing_field_as_a_difference() -> None:
    """A checkpoint predating a field must not silently match one that has it."""
    from probunet.extension.ablation import ablation_signature, assert_comparable

    complete = {"head": {"eval_seed": 2018}, "train": {"mode": "selection_head"}}
    missing = {"train": {"mode": "selection_head"}}
    with pytest.raises(ValueError, match="eval_seed"):
        assert_comparable(
            {"a": ablation_signature(complete), "b": ablation_signature(missing)}
        )


def test_a_single_arm_is_trivially_comparable() -> None:
    """One arm cannot disagree with itself; the guard must not block ordinary evaluation."""
    from probunet.extension.ablation import ablation_signature, assert_comparable

    assert_comparable({"only": ablation_signature({"head": {"eval_seed": 2018}})})
    assert_comparable({})


def test_the_training_step_really_is_the_consensus_regression(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The live objective must BE Huber-on-soft-consensus, recomputed independently.

    This test exists because it was missing. A Stage 3 placeholder objective
    (``scores.pow(2).mean()``) survived as a second ``_selection_head_step`` definition
    later in the class body, shadowed the Stage 4 implementation, and ran on the CUDA
    smoke gate -- while every existing test passed, because they only checked that the
    validation record contained the right *keys*. Keys come from ``validate``; the
    objective comes from here. Asserting the loss VALUE against an independent
    recomputation is the only version of this test that could have caught it.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    batch = next(iter(trainer.train_loader))

    torch.manual_seed(0)
    terms = trainer._selection_head_step(batch)

    # The Stage 4 objective reports both losses; the placeholder reported score_mean.
    assert set(terms) == {
        "total", "head_huber", "area_huber", "target_mean", "predicted_mean"
    }, "the live step is not the Stage 4 objective"
    assert "score_mean" not in terms

    # Recompute independently from the same seed and compare the value.
    torch.manual_seed(0)
    image = batch["image"].to(trainer.device)
    graders = batch["masks"].to(trainer.device)
    features, candidates = trainer.head.sample_candidates(
        image, config.head.train_samples
    )
    expected_target = consensus_scores(candidates, graders)
    expected_head = torch.nn.functional.huber_loss(
        trainer.head.score_candidates(features, candidates),
        expected_target,
        delta=config.head.huber_delta,
    )
    expected_area = torch.nn.functional.huber_loss(
        trainer.head.score_by_area(candidates),
        expected_target,
        delta=config.head.huber_delta,
    )
    # detach before float(): `total` is still attached to the graph, and reading a
    # requires_grad tensor as a scalar warns.
    assert float(terms["head_huber"]) == pytest.approx(float(expected_head.detach()), rel=1e-6)
    assert float(terms["area_huber"]) == pytest.approx(float(expected_area.detach()), rel=1e-6)
    assert float(terms["total"].detach()) == pytest.approx(
        float((expected_head + expected_area).detach()), rel=1e-6
    )
    assert float(terms["target_mean"]) == pytest.approx(
        float(expected_target.mean()), rel=1e-6
    )
    # The candidates really are the K the config asks for.
    assert candidates.shape[1] == config.head.train_samples


def test_no_method_is_defined_twice_in_the_trainer() -> None:
    """A shadowed duplicate is silent in Python and cost this project a wasted run.

    A second definition of the same method later in a class body simply wins, with no
    warning from the interpreter, no failure from the type checker and no test failure
    unless a test asserts behaviour rather than shape.
    """
    import ast

    source = (REPO_ROOT / "src" / "probunet" / "training" / "trainer.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        names = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        duplicates = {name for name in names if names.count(name) > 1}
        assert not duplicates, f"{node.name} defines {sorted(duplicates)} more than once"


def test_smoke_checkpoints_are_refused_as_results() -> None:
    """A gate run's best.pt must never reach a results table.

    It is written under a legitimate-looking monitor name, and once it sits in ``runs/``
    nothing about the file says it came from four epochs on eight batches. One such
    checkpoint was produced by the Stage 3 placeholder objective before that bug was found.
    """
    from probunet.extension.ablation import assert_not_a_smoke_run

    for name in ("smoke", "smoke_head", "smoke_full_cov"):
        with pytest.raises(ValueError, match="REFUSING"):
            assert_not_a_smoke_run({"run": {"name": name}}, "runs/x/best.pt")
    for name in ("baseline", "modernized", "extension", "baseline-short"):
        assert_not_a_smoke_run({"run": {"name": name}}, "runs/x/best.pt")


def test_the_gate_refuses_a_missing_base_rather_than_substituting_one() -> None:
    """A gate that invented its own base could not detect what it exists to detect."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "head_smoke_gate", REPO_ROOT / "scripts" / "head_smoke_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(FileNotFoundError, match="REFUSING to run"):
        module.run_gate(Path("runs/definitely/not/here.pt"), None, None)


def test_the_gate_reports_skips_not_passes_when_behavioural_criteria_are_absent() -> None:
    """PLUMBING ONLY must be unmistakable, and must not read as a pass."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "head_smoke_gate", REPO_ROOT / "scripts" / "head_smoke_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    verdict = module.Verdict()
    verdict.check(True, "1 completes", "ok")
    verdict.skip("4 scores in band", "stand-in base")
    text = verdict.render(behavioural=False)

    assert "PLUMBING ONLY - BEHAVIOURAL CRITERIA SKIPPED" in text
    assert "GATE PASSED" not in text, "a plumbing run must not read as a passing gate"
    assert verdict.skipped and not verdict.failed


# --------------------------------------------------------------------------- #
# Loading a trained head back, and the validation leak audit
# --------------------------------------------------------------------------- #
def test_a_trained_head_round_trips_through_the_evaluation_entry_point(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """END TO END: train a head, save it, load it the way evaluate.py does, select with it.

    This is the test that was missing. ``load_variant`` built a bare ``ProbUNet`` and
    strict-loaded, so a head checkpoint -- whose keys are ``base.*`` plus ``scorer.*`` and
    ``area_baseline.*`` -- failed on every key. A key-presence test would not have caught
    it, and neither would anything that stopped short of the real entry point: the head was
    built, trained and saved correctly, and only the *reader* was missing.

    So this goes through ``load_variant`` itself, and then asserts the loaded scorer weights
    equal the saved ones -- because a strict load that silently dropped the scorer would
    still produce a variant that selects, and would produce a plausible meaningless number.
    """
    from probunet.evaluation.runner import load_variant

    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)
    trainer.train_epoch(0)
    trainer._checkpoint(0, {"val/selected_consensus_dice": 0.5})

    head_path = trainer.checkpoint_dir / "last.pt"
    saved_scorer = {
        name: parameter.detach().clone()
        for name, parameter in trainer.head.scorer.named_parameters()
    }
    saved_area = {
        name: parameter.detach().clone()
        for name, parameter in trainer.head.area_baseline.named_parameters()
    }

    variant, loaded_config, state = load_variant(head_path, torch.device("cpu"))

    # Dispatched on the recorded config, and it is the selecting variant.
    assert loaded_config.train.mode == "selection_head"
    assert isinstance(variant, SelectionHeadVariant)
    assert isinstance(variant, SegmentationVariant)

    # The weights really arrived -- not merely "a model was constructed".
    for name, parameter in variant.head.scorer.named_parameters():
        assert torch.equal(parameter.detach(), saved_scorer[name]), f"scorer.{name} drifted"
    for name, parameter in variant.head.area_baseline.named_parameters():
        assert torch.equal(parameter.detach(), saved_area[name]), f"area.{name} drifted"

    # And it can actually do the job.
    image = torch.rand(2, 1, SIZE, SIZE)
    samples = variant.sample(image, 4)
    chosen = variant.select(samples, image)
    assert chosen.shape == (2,)
    assert bool(((chosen >= 0) & (chosen < 4)).all())

    # The base it was frozen on is stated by the artifact.
    assert state.base_provenance is not None
    assert state.base_provenance["parameter_sha256"]


def test_a_plain_checkpoint_still_loads_as_a_non_selecting_variant(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """The elbo branch is unchanged: dispatch must not capture ordinary checkpoints."""
    from probunet.evaluation.runner import load_variant

    checkpoint, _ = base_checkpoint_for(tiny_experiment, tmp_path)
    variant, config, _ = load_variant(checkpoint, torch.device("cpu"))
    assert config.train.mode == "elbo"
    assert isinstance(variant, ProbUNetVariant)
    assert not isinstance(variant, SelectionHeadVariant)
    image = torch.rand(2, 1, SIZE, SIZE)
    assert variant.select(variant.sample(image, 3), image) is None


def test_validation_candidates_come_from_the_prior(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """LEAK AUDIT: the posterior must not be reachable during VALIDATION either.

    Stage 4 asserted prior-only for training with a hooked posterior net. Validation is a
    separate code path and was never covered by that assertion; this closes it. A posterior
    candidate has seen the ground-truth mask, so a validation score computed from one would
    be optimistic in a way no downstream check would reveal.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    calls = {"n": 0}
    handle = trainer.head.base.posterior_net.register_forward_hook(
        lambda *_: calls.__setitem__("n", calls["n"] + 1)
    )
    try:
        trainer.validate()
    finally:
        handle.remove()
    assert calls["n"] == 0, "the posterior net was invoked during validation"


def test_the_selected_sample_is_chosen_by_the_HEAD_not_by_the_truth(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """LEAK AUDIT: ``selected`` must come from PREDICTED scores, never the true ones.

    If the argmax were taken over the true soft-consensus scores, ``selected`` would equal
    ``oracle`` exactly and the reported number would be the oracle wearing the head's name.

    Forced by construction: the candidates are replaced with a set whose true scores are
    known and DISTINCT -- candidate 0 reproduces a grader mask exactly, the rest are empty
    -- and the scorer is replaced with one that always ranks the last candidate first. The
    truth then says "pick 0" and the head says "pick the last", so ``selected`` and
    ``oracle`` must differ. Constructing the candidates matters: an untrained tiny model
    emits all-empty candidates that score 0.0 alike, and the assertion would pass on a
    degeneracy rather than on the property.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    trainer = Trainer(config, base_checkpoint=checkpoint)

    def staged(image, n_samples, generator=None):
        """Candidate 0 is a perfect grader match; the rest are empty."""
        batch = image.shape[0]
        candidates = torch.zeros(
            batch, n_samples, SIZE, SIZE, dtype=torch.uint8, device=image.device
        )
        # Grader 0's mask for each image, which scores well against the consensus.
        graders = staged.graders
        candidates[:, 0] = graders[:, 0]
        return trainer.head.encode_base(image), candidates

    class PrefersLast(torch.nn.Module):
        """A scorer whose ranking is fixed, and deliberately wrong."""

        def forward(self, features: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
            """Rank later candidates higher, regardless of content."""
            self.calls = getattr(self, "calls", 0) + 1
            return torch.full((features.shape[0],), float(self.calls))

    # Capture the graders of each batch so the staged candidates can match them.
    original_scores = trainer.validate
    batch = next(iter(trainer.data.loaders["val"]))
    staged.graders = batch["masks"].to(trainer.device)

    trainer.head.sample_candidates = staged
    trainer.head.scorer = PrefersLast()
    trainer.config = dataclasses.replace(
        config, train=dataclasses.replace(config.train, limit_val_batches=1)
    )
    metrics = trainer.validate()

    # The oracle found candidate 0; the head was forced onto an empty one.
    assert metrics["oracle_consensus_dice"] > 0.0, "the staged candidates are degenerate"
    assert metrics["selected_consensus_dice"] < metrics["oracle_consensus_dice"], (
        "selected tracked oracle: the choice is being made with the ground truth"
    )
    assert metrics["selected_consensus_dice"] == pytest.approx(0.0, abs=1e-9)


def test_validation_uses_the_val_split_and_the_eval_candidate_count(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """LEAK AUDIT: held-out split, and eval_samples rather than train_samples.

    A different candidate count would change what the oracle should be compared against,
    so the count the run actually used has to be the configured evaluation one.
    """
    checkpoint, config = base_checkpoint_for(tiny_experiment, tmp_path)
    config = dataclasses.replace(
        config,
        head=dataclasses.replace(config.head, train_samples=2, eval_samples=5),
        train=dataclasses.replace(config.train, limit_val_batches=None),
    )
    trainer = Trainer(config, base_checkpoint=checkpoint)

    seen = {"n": 0, "counts": set()}
    original = trainer.head.sample_candidates

    def watched(image, n_samples, generator=None):
        """Record the candidate count and image total each call."""
        seen["counts"].add(n_samples)
        seen["n"] += image.shape[0]
        return original(image, n_samples, generator)

    trainer.head.sample_candidates = watched
    trainer.validate()

    assert seen["counts"] == {5}, f"validation used {seen['counts']} candidates, not eval_samples"
    assert seen["n"] == len(trainer.data.datasets["val"]), "not the whole held-out val split"
    # And the val split is disjoint from what the head trained on.
    train_indices = set(trainer.data.datasets["train"].indices.tolist())
    val_indices = set(trainer.data.datasets["val"].indices.tolist())
    assert train_indices.isdisjoint(val_indices)


def test_the_two_head_arms_compare_cleanly() -> None:
    """The ablation pair differs in exactly the variable under test, and nothing else.

    ``configs/extension_modernized.yaml`` exists only so the Phase 2 arm can load a
    full-covariance base. If the 19-field refusal in ``extension/ablation.py`` rejected this
    pair, the ablation could never be compared -- so that is asserted here rather than
    discovered after two head runs have been spent.
    """
    from probunet.extension.ablation import (
        VARYING_FIELD,
        ablation_signature,
        assert_comparable,
    )

    phase1 = ExperimentConfig.from_yaml(CONFIGS / "extension.yaml")
    phase2 = ExperimentConfig.from_yaml(CONFIGS / "extension_modernized.yaml")

    # Exactly one architectural difference, and it is the one under test.
    differing = {
        field.name
        for field in dataclasses.fields(phase1.model)
        if getattr(phase1.model, field.name) != getattr(phase2.model, field.name)
    }
    assert differing == {"latent_covariance"}
    assert phase1.model.latent_covariance == "diagonal"
    assert phase2.model.latent_covariance == "full"
    assert VARYING_FIELD == "model.latent_covariance"

    # Everything the head's own comparison depends on is identical.
    assert phase1.head == phase2.head
    assert phase1.optim == phase2.optim
    assert phase1.train == phase2.train
    assert phase1.data == phase2.data
    assert phase1.schedule == phase2.schedule
    assert phase1.checkpoint == phase2.checkpoint

    # The run names differ, so the two arms cannot overwrite each other under runs/.
    assert phase1.run.name != phase2.run.name

    # And the refusal accepts the pair.
    assert_comparable(
        {
            "head_on_phase1": ablation_signature(phase1.to_dict()),
            "head_on_phase2": ablation_signature(phase2.to_dict()),
        }
    )


def test_the_phase1_arm_still_defaults_to_diagonal() -> None:
    """The Phase 2 flag must not have leaked into the Phase 1 arm.

    Its headline result (+0.0589 edge over the largest-area rule) is recorded against a
    diagonal base, so an override reaching ``extension.yaml`` would silently invalidate it.
    """
    phase1 = ExperimentConfig.from_yaml(CONFIGS / "extension.yaml")
    assert phase1.model.latent_covariance == "diagonal"
    raw = yaml.safe_load((CONFIGS / "extension.yaml").read_text())
    assert "latent_covariance" not in raw.get("model", {}), (
        "extension.yaml now sets latent_covariance explicitly; the Phase 1 arm is supposed "
        "to take the default, and the ablation arm lives in extension_modernized.yaml"
    )
