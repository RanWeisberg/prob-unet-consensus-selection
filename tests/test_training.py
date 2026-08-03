"""Tests for configuration, checkpointing, diagnostics and the training loop.

The loop is exercised end to end on a synthetic dataset with a deliberately tiny model,
so the whole file runs in seconds while still covering every code path a real run takes:
validation over all four graders, the latent diagnostics, the image panel, checkpoint
selection on validation loss, and resume.

The shipped ``configs/baseline.yaml`` is also asserted directly -- in particular that
channel capping is **off** there, so nobody can quietly turn a capped run into "the
baseline".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from probunet.data.lidc import DataConfig, LidcArrays, LidcDataset
from probunet.data.splits import generate_split
from probunet.losses.elbo import ElboConfig
from probunet.model.prob_unet import ProbUNetConfig
from probunet.training.checkpoint import (
    is_improvement,
    load_checkpoint,
    save_checkpoint,
)
from probunet.training.config import (
    CheckpointConfig,
    ExperimentConfig,
    LogConfig,
    OptimConfig,
    RunConfig,
    ScheduleConfig,
    TrainConfig,
)
from probunet.training.diagnostics import (
    build_diagnostic_sets,
    logits_to_mask,
    make_panel,
    mean_pairwise_iou,
    nonempty_sample_fraction,
    stratified_indices,
)
from probunet.training.trainer import Trainer
from probunet.utils.runtime import git_revision, select_device

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
N_GRADERS = 4
SIZE = 16


def write_npz(path: Path, sizes: list[int], seed: int = 0) -> None:
    """Write a synthetic lidc.npz spanning several ambiguity buckets."""
    uids: list[str] = []
    for index, size in enumerate(sizes):
        uids.extend([f"series-{index:04d}"] * size)
    series = np.array(uids, dtype=np.str_)
    rng = np.random.default_rng(seed)
    images = rng.random((series.size, SIZE, SIZE), dtype=np.float32)
    masks = np.zeros((series.size, N_GRADERS, SIZE, SIZE), dtype=np.uint8)
    for row in range(series.size):
        # Cycle bucket sizes 1..4 so every bucket is populated.
        for slot in range((row % N_GRADERS) + 1):
            masks[row, slot, : 2 + slot, : 2 + (row % 3)] = 1
    np.savez_compressed(path, images=images, masks=masks, series_uid=series)


@pytest.fixture
def tiny_experiment(tmp_path: Path) -> ExperimentConfig:
    """A complete experiment config over synthetic data with a very small model."""
    npz = tmp_path / "lidc.npz"
    write_npz(npz, [7, 5, 3, 11, 2, 9, 4, 6, 8, 1])
    split = tmp_path / "split.json"
    generate_split(npz_path=npz, out_path=split)
    return ExperimentConfig(
        run=RunConfig(name="test", seed=123, device="cpu", out_dir=tmp_path / "runs"),
        model=ProbUNetConfig(latent_dim=2, base_channels=4, num_downs=2, convs_per_scale=1),
        loss=ElboConfig(beta=1.0),
        data=DataConfig(npz_path=npz, split_path=split, batch_size=4),
        optim=OptimConfig(lr=1e-3),
        schedule=ScheduleConfig(name="constant"),
        train=TrainConfig(epochs=2, limit_train_batches=3, limit_val_batches=2),
        log=LogConfig(
            diagnostics_every_n_epochs=1,
            prior_samples_for_ce=2,
            diversity_samples=3,
            diversity_images=6,
            panel_images=4,
            panel_samples=2,
            log_every_n_steps=1,
        ),
        checkpoint=CheckpointConfig(),
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_shipped_baseline_config_is_uncapped() -> None:
    """configs/baseline.yaml must not cap channels: capped runs are not the baseline."""
    raw = yaml.safe_load((CONFIGS / "baseline.yaml").read_text())
    assert raw["model"]["max_channels"] is None
    config = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml")
    assert config.model.max_channels is None
    # And the paper's schedule really is what that produces.
    from probunet.model.unet import channel_widths

    assert channel_widths(
        config.model.base_channels, config.model.num_downs, config.model.max_channels
    ) == [32, 64, 128, 256, 512]


def test_shipped_baseline_config_matches_the_paper() -> None:
    """Spot-check the values CLAUDE.md pins down."""
    config = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml")
    assert config.model.latent_dim == 6
    assert config.model.num_downs == 4
    assert config.model.convs_per_scale == 3
    assert config.loss.beta == 1.0
    assert config.data.batch_size == 32
    assert config.optim.name == "adam"
    assert config.optim.lr == pytest.approx(1e-4)
    assert config.optim.weight_decay == pytest.approx(1e-5)
    assert config.data.augment is False
    assert config.data.normalization == "none"
    assert config.checkpoint.monitor.startswith("val/")


def test_shipped_smoke_config_is_small_and_valid() -> None:
    """configs/smoke.yaml must stay tiny enough to run before every long run."""
    config = ExperimentConfig.from_yaml(CONFIGS / "smoke.yaml")
    assert config.train.epochs <= 3
    assert config.train.limit_train_batches is not None
    assert config.train.limit_val_batches is not None
    # Diagnostics must run in the smoke test, or they are never exercised before a
    # long run reaches them.
    assert config.log.diagnostics_every_n_epochs == 1


def test_config_yaml_round_trip(tmp_path: Path, tiny_experiment: ExperimentConfig) -> None:
    """to_yaml -> from_yaml preserves the configuration."""
    path = tmp_path / "round.yaml"
    path.write_text(tiny_experiment.to_yaml())
    reloaded = ExperimentConfig.from_yaml(path)
    assert reloaded.to_dict() == tiny_experiment.to_dict()


def test_unknown_section_rejected(tmp_path: Path) -> None:
    """A typo'd section is an error, not a silently ignored block."""
    path = tmp_path / "bad.yaml"
    path.write_text("modle:\n  latent_dim: 6\n")
    with pytest.raises(ValueError, match="unknown config section"):
        ExperimentConfig.from_yaml(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    """A typo'd key would otherwise leave the default silently in place."""
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  latentdim: 6\n")
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentConfig.from_yaml(path)


def test_monitor_must_be_a_validation_metric() -> None:
    """Selecting checkpoints on test would bias every reported number."""
    with pytest.raises(ValueError, match="val/"):
        CheckpointConfig(monitor="test/total")
    with pytest.raises(ValueError, match="val/"):
        CheckpointConfig(monitor="train/total")


def test_adamw_is_rejected() -> None:
    """AdamW's decoupled decay is not the reference's L2 regularizer."""
    with pytest.raises(NotImplementedError, match="AdamW"):
        OptimConfig(name="adamw")


def test_schedule_validation() -> None:
    """Malformed schedules are caught at construction."""
    with pytest.raises(ValueError, match="one of"):
        ScheduleConfig(name="cosine")
    with pytest.raises(ValueError, match="no milestones"):
        ScheduleConfig(name="constant", milestones=(0.5,), values=(1e-4, 1e-5))
    with pytest.raises(ValueError, match="len\\(values\\)"):
        ScheduleConfig(name="piecewise", milestones=(0.5,), values=(1e-4,))
    with pytest.raises(ValueError, match="fractions in"):
        ScheduleConfig(name="piecewise", milestones=(5000.0,), values=(1e-4, 1e-5))
    with pytest.raises(ValueError, match="increasing"):
        ScheduleConfig(name="piecewise", milestones=(0.7, 0.3), values=(1e-4, 1e-5, 1e-6))


def test_piecewise_first_value_must_match_lr() -> None:
    """Two sources of truth for the initial learning rate is a trap."""
    with pytest.raises(ValueError, match="single source of truth"):
        ExperimentConfig(
            optim=OptimConfig(lr=1e-4),
            schedule=ScheduleConfig(name="piecewise", milestones=(0.5,), values=(1e-3, 1e-5)),
        )


def test_pairing_seed_follows_run_seed(tmp_path: Path) -> None:
    """One knob controls run randomness; an explicit pairing_seed still wins."""
    path = tmp_path / "c.yaml"
    path.write_text("run:\n  seed: 4242\n")
    assert ExperimentConfig.from_yaml(path).data.pairing_seed == 4242

    path.write_text("run:\n  seed: 4242\ndata:\n  pairing_seed: 7\n")
    assert ExperimentConfig.from_yaml(path).data.pairing_seed == 7


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_stratified_indices_span_buckets(tiny_experiment: ExperimentConfig) -> None:
    """Diagnostic sets cover the ambiguity buckets rather than the first N indices."""
    arrays = LidcArrays.load(tiny_experiment.data.npz_path)
    from probunet.data.splits import load_split

    split = load_split(tiny_experiment.data.split_path)
    dataset = LidcDataset(arrays, split.indices["val"], mode="eval")

    indices, per_bucket = stratified_indices(dataset, count=4, seed=0)
    assert indices.size == 4
    assert len(per_bucket) >= 2, "diagnostic set collapsed into one bucket"
    assert set(indices.tolist()) <= set(dataset.indices.tolist())
    # Deterministic given the seed.
    again, _ = stratified_indices(dataset, count=4, seed=0)
    assert indices.tolist() == again.tolist()


def test_diagnostic_sets_are_recorded(tiny_experiment: ExperimentConfig) -> None:
    """The chosen indices are written out so panels are comparable across runs."""
    trainer = Trainer(tiny_experiment)
    recorded = json.loads((trainer.run_dir / "diagnostic_indices.json").read_text())
    assert recorded["diversity"] == [int(i) for i in trainer.diagnostic_sets.diversity]
    assert recorded["panel"] == [int(i) for i in trainer.diagnostic_sets.panel]
    assert recorded["panel_buckets"]


def test_diversity_of_identical_samples_is_one() -> None:
    """Identical samples agree perfectly, which is the collapse signature."""
    samples = torch.ones(3, 4, 8, 8, dtype=torch.uint8)
    assert mean_pairwise_iou(samples).item() == pytest.approx(1.0)
    assert nonempty_sample_fraction(samples).item() == pytest.approx(1.0)


def test_diversity_of_all_empty_samples_is_also_one() -> None:
    """The false alarm this diagnostic pair exists to disambiguate.

    All-empty samples score diversity 1.0 because two empty masks agree perfectly. With
    a 176:1 class imbalance that is expected early in training, so the non-empty
    fraction is what separates "hasn't learned foreground yet" from "prior collapsed".
    """
    samples = torch.zeros(3, 4, 8, 8, dtype=torch.uint8)
    assert mean_pairwise_iou(samples).item() == pytest.approx(1.0)
    assert nonempty_sample_fraction(samples).item() == pytest.approx(0.0)


def test_diversity_of_disjoint_samples_is_zero() -> None:
    """Maximally disagreeing samples score 0."""
    samples = torch.zeros(1, 2, 4, 4, dtype=torch.uint8)
    samples[0, 0, 0, 0] = 1
    samples[0, 1, 3, 3] = 1
    assert mean_pairwise_iou(samples).item() == pytest.approx(0.0)
    assert nonempty_sample_fraction(samples).item() == pytest.approx(1.0)


def test_diversity_needs_two_samples() -> None:
    """A single sample has no pair."""
    with pytest.raises(ValueError, match="at least 2"):
        mean_pairwise_iou(torch.zeros(2, 1, 4, 4, dtype=torch.uint8))


def test_logits_to_mask_takes_argmax() -> None:
    """Foreground wins where its logit is larger."""
    logits = torch.zeros(2, 2, 3, 3)
    logits[:, 1, 0, 0] = 5.0
    mask = logits_to_mask(logits)
    assert mask.shape == (2, 3, 3)
    assert mask.dtype == torch.uint8
    assert mask[0, 0, 0].item() == 1
    assert mask[0, 1, 1].item() == 0


def test_panel_layout() -> None:
    """The panel is one row per image and one column per view."""
    images = torch.rand(3, 1, 8, 8)
    graders = torch.ones(3, 4, 8, 8, dtype=torch.uint8)
    samples = torch.zeros(3, 2, 8, 8, dtype=torch.uint8)
    panel = make_panel(images, graders, samples)
    assert panel.shape[0] == 1
    assert panel.shape[1] == 3 * (8 + 2)
    assert panel.shape[2] == (1 + 4 + 2) * (8 + 2)
    assert float(panel.min()) >= 0.0 and float(panel.max()) <= 1.0


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def test_is_improvement() -> None:
    """Improvement respects the monitor's direction."""
    assert is_improvement(1.0, None, "min")
    assert is_improvement(0.5, 1.0, "min")
    assert not is_improvement(1.5, 1.0, "min")
    assert is_improvement(1.5, 1.0, "max")
    assert not is_improvement(0.5, 1.0, "max")


def test_checkpoint_round_trip(tmp_path: Path, tiny_experiment: ExperimentConfig) -> None:
    """A checkpoint restores model and optimizer state, and carries provenance."""
    trainer = Trainer(tiny_experiment)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=3,
        global_step=42,
        config=tiny_experiment.to_dict(),
        seed=tiny_experiment.run.seed,
        device="cpu",
        monitor="val/total",
        best_metric=1.25,
        metrics={"val/total": 1.25},
    )

    fresh = Trainer(
        dataclasses.replace(
            tiny_experiment, run=dataclasses.replace(tiny_experiment.run, name="fresh")
        )
    )
    # Both trainers share a seed, so fresh starts out identical to trainer. Perturb it
    # first, otherwise "loading worked" and "loading did nothing" look the same.
    with torch.no_grad():
        for parameter in fresh.model.parameters():
            parameter.add_(1.0)
    assert any(
        not torch.equal(a.detach(), b.detach())
        for a, b in zip(trainer.model.parameters(), fresh.model.parameters(), strict=True)
    )

    state = load_checkpoint(path, model=fresh.model, optimizer=fresh.optimizer)

    assert state.epoch == 3
    assert state.global_step == 42
    assert state.best_metric == 1.25
    assert state.seed == tiny_experiment.run.seed
    assert state.git_revision  # "unknown" is acceptable, empty is not
    assert state.config["model"]["latent_dim"] == 2
    for saved, loaded in zip(trainer.model.parameters(), fresh.model.parameters(), strict=True):
        assert torch.equal(saved.detach(), loaded.detach())


def test_missing_checkpoint_is_an_error(tmp_path: Path) -> None:
    """Loading a checkpoint that is not there fails clearly."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pt")


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def test_training_runs_end_to_end(tiny_experiment: ExperimentConfig) -> None:
    """Two epochs produce finite metrics, checkpoints, a panel and a summary."""
    trainer = Trainer(tiny_experiment)
    summary = trainer.train()

    assert summary["epochs"] == 2
    assert summary["global_step"] == 2 * tiny_experiment.train.limit_train_batches
    assert summary["best_metric"] is not None
    assert len(summary["history"]) == 2

    for record in summary["history"]:
        for key in ("train/total", "train/ce", "train/kl", "val/total", "val/ce", "val/kl"):
            assert np.isfinite(record[key]), key
        # Both diagnostics that must be read together are present.
        assert "diag/sample_diversity_iou" in record
        assert "diag/nonempty_sample_fraction" in record
        assert "diag/prior_posterior_ce_ratio" in record
        assert "diag/gap_total" in record

    assert (trainer.checkpoint_dir / "last.pt").exists()
    assert (trainer.checkpoint_dir / "best.pt").exists()
    assert (trainer.run_dir / "config.resolved.yaml").exists()
    assert list((trainer.run_dir / "tb").glob("events*"))


def test_all_four_graders_used_in_validation(tiny_experiment: ExperimentConfig) -> None:
    """Validation averages over four graders, so it does not depend on the pairing."""
    trainer = Trainer(tiny_experiment)
    first = trainer.validate()
    # The pairing changes the training targets but must not change validation.
    trainer.data.set_epoch(99)
    second = trainer.validate()
    assert first["ce"] == pytest.approx(second["ce"], rel=0.3)
    # Latent diagnostics come along with validation.
    assert "prior_sigma_mean" in first
    assert "kl_dim_0" in first


def test_per_dim_kl_covers_every_latent_dimension(tiny_experiment: ExperimentConfig) -> None:
    """One KL scalar per latent dimension, so a dead dimension is visible."""
    trainer = Trainer(tiny_experiment)
    metrics = trainer.validate()
    for dimension in range(tiny_experiment.model.latent_dim):
        assert f"kl_dim_{dimension}" in metrics


def test_best_checkpoint_tracks_validation_loss(tiny_experiment: ExperimentConfig) -> None:
    """The monitored metric drives best-checkpoint selection."""
    trainer = Trainer(tiny_experiment)
    trainer._checkpoint(0, {"val/total": 10.0})
    assert trainer.best_metric == 10.0
    trainer._checkpoint(1, {"val/total": 12.0})
    assert trainer.best_metric == 10.0, "a worse value must not become the best"
    trainer._checkpoint(2, {"val/total": 5.0})
    assert trainer.best_metric == 5.0


def test_limit_batches_respected(tiny_experiment: ExperimentConfig) -> None:
    """Smoke runs really do stop early."""
    config = dataclasses.replace(
        tiny_experiment, train=dataclasses.replace(tiny_experiment.train, limit_train_batches=2)
    )
    trainer = Trainer(config)
    assert trainer.steps_per_epoch == 2
    trainer.train_epoch(0)
    assert trainer.global_step == 2


def test_same_seed_gives_identical_first_epoch(tiny_experiment: ExperimentConfig) -> None:
    """A run is reproducible from its seed on a fixed device."""
    first = Trainer(tiny_experiment).train_epoch(0)
    second = Trainer(
        dataclasses.replace(tiny_experiment, run=dataclasses.replace(tiny_experiment.run, name="b"))
    ).train_epoch(0)
    assert first["total"] == pytest.approx(second["total"], rel=1e-6)


def test_different_seed_changes_the_run(tiny_experiment: ExperimentConfig) -> None:
    """A different seed gives a different trajectory."""
    first = Trainer(tiny_experiment).train_epoch(0)
    other = dataclasses.replace(
        tiny_experiment, run=dataclasses.replace(tiny_experiment.run, seed=999, name="c")
    )
    assert first["total"] != pytest.approx(Trainer(other).train_epoch(0)["total"], rel=1e-9)


def test_resume_continues_the_same_trajectory(tiny_experiment: ExperimentConfig) -> None:
    """Interrupting and resuming matches an uninterrupted run.

    This is what the saved RNG and DataLoader generator states buy: a resumed run
    replays the exact sequence rather than a statistically similar one.
    """
    uninterrupted = Trainer(tiny_experiment)
    uninterrupted.train()
    reference = [p.detach().clone() for p in uninterrupted.model.parameters()]

    # Same config, but stop after one epoch, then resume for the second.
    first_half = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="half"),
        train=dataclasses.replace(tiny_experiment.train, epochs=1),
    )
    stopped = Trainer(first_half)
    stopped.train()

    resumed = Trainer(
        dataclasses.replace(
            tiny_experiment, run=dataclasses.replace(tiny_experiment.run, name="resumed")
        )
    )
    resumed.resume(stopped.checkpoint_dir / "last.pt")
    assert resumed.epoch == 1
    resumed.train()

    for expected, actual in zip(reference, resumed.model.parameters(), strict=True):
        assert torch.allclose(expected, actual.detach(), atol=1e-6), (
            "resumed run diverged from the uninterrupted one"
        )


def test_amp_rejected_off_cuda(tiny_experiment: ExperimentConfig) -> None:
    """AMP is CUDA-only; the flag must not silently do nothing."""
    config = dataclasses.replace(
        tiny_experiment, train=dataclasses.replace(tiny_experiment.train, amp=True)
    )
    if torch.cuda.is_available():
        pytest.skip("CUDA present, so amp is supported here")
    with pytest.raises(NotImplementedError, match="amp=true requires CUDA"):
        Trainer(config)


def test_piecewise_schedule_decays_at_fractional_milestones(
    tiny_experiment: ExperimentConfig
) -> None:
    """Milestones are fractions of total steps, so a schedule scales with the budget."""
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="sched"),
        optim=dataclasses.replace(tiny_experiment.optim, lr=1e-3),
        schedule=ScheduleConfig(name="piecewise", milestones=(0.5,), values=(1e-3, 1e-4)),
    )
    trainer = Trainer(config)
    assert trainer.total_steps == 6  # 2 epochs x 3 batches
    seen = []
    for _ in range(trainer.total_steps):
        seen.append(trainer.optimizer.param_groups[0]["lr"])
        # Mirror the real loop's order (optimizer then scheduler), which also avoids
        # torch's "scheduler.step() before optimizer.step()" warning.
        trainer.optimizer.step()
        trainer.scheduler.step()
    assert seen[0] == pytest.approx(1e-3)
    assert seen[-1] == pytest.approx(1e-4)


def test_startup_banner_flags_a_capped_model(tiny_experiment: ExperimentConfig) -> None:
    """A capped run announces itself, so it cannot be mistaken for the baseline."""
    assert "CAPPED" not in Trainer(tiny_experiment).describe()
    capped = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="capped"),
        model=dataclasses.replace(tiny_experiment.model, max_channels=4),
    )
    assert "CAPPED - not baseline" in Trainer(capped).describe()


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #
def test_select_device_honours_explicit_cpu() -> None:
    """An explicit device is respected."""
    assert select_device("cpu").type == "cpu"


def test_select_device_refuses_unavailable_accelerator() -> None:
    """Falling back silently would mislabel a run's device."""
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA is not available"):
            select_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="MPS is not available"):
            select_device("mps")


def test_git_revision_is_a_string() -> None:
    """Provenance is best-effort but never blank."""
    revision = git_revision()
    assert isinstance(revision, str) and revision