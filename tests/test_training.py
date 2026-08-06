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
from torch.distributions import Independent, MultivariateNormal, Normal, kl_divergence

from probunet.data.lidc import DataConfig, LidcArrays, LidcDataset
from probunet.data.splits import generate_split
from probunet.data.transforms import AugmentationConfig
from probunet.losses.elbo import ElboConfig
from probunet.model.prob_unet import ProbUNet, ProbUNetConfig
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
    EffectiveRankAccumulator,
    build_diagnostic_sets,
    effective_rank,
    logits_to_mask,
    make_panel,
    mean_pairwise_iou,
    nonempty_sample_fraction,
    per_dim_kl,
    stratified_indices,
    whitened_kl_decomposition,
)
from probunet.training.trainer import Trainer
from probunet.utils.runtime import git_revision, rng_state, select_device, set_rng_state

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
    assert config.data.normalization == "none"
    assert config.checkpoint.monitor.startswith("val/")


def test_shipped_baseline_config_reproduces_the_papers_augmentation() -> None:
    """Phase 1 is only a faithful reproduction if the paper's augmentation is on.

    Augmentation is the paper's own technique (Appendix H.1), so omitting it here and
    reintroducing it later as a "modernization" would misattribute their work to us.
    """
    augmentation = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml").data.augmentation
    assert augmentation.enabled is True
    # The paper's tile size, which is what makes its rotation and scale magnitudes safe.
    assert augmentation.pad_to_px == 180
    assert augmentation.random_crop is True
    assert augmentation.rotation_degrees == pytest.approx(22.5)
    assert augmentation.scale_range == (0.8, 1.2)
    assert augmentation.elastic_alpha_px > 0


def test_shipped_baseline_config_uses_the_papers_budget_and_schedule() -> None:
    """240k iterations and the five-step decay from 1e-4 to 1e-6."""
    config = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml")
    # The budget is expressed in iterations, as the paper states it; the epoch count is
    # derived by the Trainer from the train split size.
    assert config.train.iterations == 240000
    assert config.train.epochs is None

    schedule = config.schedule
    assert schedule.name == "piecewise"
    # "lowered to 1e-6 in 5 steps" -> five decay events, so six levels.
    assert len(schedule.milestones) == 5
    assert len(schedule.values) == 6
    assert schedule.values[0] == pytest.approx(1e-4)
    assert schedule.values[-1] == pytest.approx(1e-6)
    # Geometric ladder: a constant ratio per step injects no arbitrary round numbers.
    ratios = [b / a for a, b in zip(schedule.values, schedule.values[1:], strict=False)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-4)
    assert ratios[0] == pytest.approx(10 ** (-2 / 5), rel=1e-4)


def test_shipped_ablation_differs_from_baseline_in_exactly_one_setting() -> None:
    """The no-augmentation run is a control, so it must isolate augmentation alone."""
    baseline = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml")
    ablation = ExperimentConfig.from_yaml(CONFIGS / "ablation_no_augmentation.yaml")
    assert ablation.data.augmentation.enabled is False
    assert baseline.data.augmentation.enabled is True
    # Everything that would confound the comparison must match.
    assert ablation.train.iterations == baseline.train.iterations
    assert ablation.schedule == baseline.schedule
    assert ablation.optim == baseline.optim
    assert ablation.model == baseline.model
    assert ablation.loss == baseline.loss
    assert ablation.data.batch_size == baseline.data.batch_size
    assert ablation.run.seed == baseline.run.seed
    assert ablation.run.name != baseline.run.name, "the run name must say what it is"


def test_modernized_config_inherits_the_baseline_data_pipeline() -> None:
    """Phase 2 must not also change the augmentation or the budget.

    Augmentation is explicitly NOT a Phase 2 candidate: it is the paper's own method and
    belongs to Phase 1. If modernized.yaml toggled it, a Phase-1-vs-Phase-2 comparison
    would change more than one variable and support no claim.
    """
    baseline = ExperimentConfig.from_yaml(CONFIGS / "baseline.yaml")
    modernized = ExperimentConfig.from_yaml(CONFIGS / "modernized.yaml")
    assert modernized.data.augmentation == baseline.data.augmentation
    assert modernized.train.iterations == baseline.train.iterations
    assert modernized.schedule == baseline.schedule


def test_smoke_config_disables_augmentation() -> None:
    """The smoke run stays a fast check of the loop's plumbing."""
    config = ExperimentConfig.from_yaml(CONFIGS / "smoke.yaml")
    assert config.data.augmentation.enabled is False


def test_no_shipped_config_still_uses_the_removed_augment_key() -> None:
    """The old boolean hook is gone; a stale key must fail loudly, not be ignored."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        assert "augment" not in (raw.get("data") or {}), f"{path.name} uses the old key"


def test_nested_augmentation_typos_are_rejected(tmp_path: Path) -> None:
    """Unknown-key rejection must work at nested depth too, not only top level."""
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  augmentation:\n    enabled: true\n    rotation_degrese: 22.5\n")
    with pytest.raises(ValueError, match="rotation_degrese"):
        ExperimentConfig.from_yaml(path)


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
# Latent geometry: the prior-whitened KL decomposition (Stage 4)
# --------------------------------------------------------------------------- #
WHITEN_BATCH = 5
WHITEN_DIM = 6


def whiten_pair(
    dtype: torch.dtype, seed: int = 0, full: bool = True
) -> tuple[object, object]:
    """Build a (posterior, prior) pair of the requested family and dtype.

    Constructed **in** the target dtype rather than cast afterwards: casting a float32 KL
    to float64 buys no accuracy, and the point of the two-dtype tests is to measure what
    each precision can actually deliver end to end.

    Args:
        dtype: Element type for every parameter.
        seed: Seed for the parameters.
        full: Full covariance if True, diagonal otherwise.

    Returns:
        The posterior and prior.
    """
    generator = torch.Generator().manual_seed(seed)

    def draw(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype) * scale

    def build(mean_scale: float, tril_scale: float) -> object:
        mu = draw((WHITEN_BATCH, WHITEN_DIM), mean_scale)
        sigma = torch.exp(0.5 * draw((WHITEN_BATCH, WHITEN_DIM), 0.5))
        if not full:
            return Independent(Normal(mu, sigma), 1)
        strict = torch.tril(draw((WHITEN_BATCH, WHITEN_DIM, WHITEN_DIM), tril_scale), -1)
        return MultivariateNormal(mu, scale_tril=strict + torch.diag_embed(sigma))

    return build(1.0, 0.4), build(0.3, 0.3)


@pytest.mark.parametrize("full", [True, False])
def test_whitened_decomposition_sums_to_the_exact_kl_in_float64(full: bool) -> None:
    """float64: an ABSOLUTE tolerance, because the arithmetic really is that good.

    ``sum_i kl_i == KL(Q || P)`` is the property that makes this a decomposition rather
    than a set of loosely related numbers, and it is what licenses logging
    ``kl_snapshot_total`` next to the parts as a reconciliation partner. In float64 the
    residual is ~1e-15, so an absolute bound catches any real algebra error; a relative
    bound here would be needlessly loose.
    """
    posterior, prior = whiten_pair(torch.float64, full=full)
    decomposition = whitened_kl_decomposition(posterior, prior)
    exact = kl_divergence(posterior, prior)
    assert torch.allclose(decomposition.total, exact, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        decomposition.per_direction.sum(dim=-1), exact, atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("full", [True, False])
def test_whitened_decomposition_sums_to_the_exact_kl_in_float32(full: bool) -> None:
    """float32: a RELATIVE tolerance, because that is the honest bound in the real dtype.

    Production runs in float32 -- MPS has no float64 at all -- and there the reference KL
    is itself only accurate to ~1e-7 relative. The absolute residual scales with the KL
    (it reaches ~4e-6 on these inputs), so an absolute bound would either be vacuous or
    fail on a larger KL. This is the same identity as the float64 test, stated in the
    strongest form float32 can actually support.
    """
    posterior, prior = whiten_pair(torch.float32, full=full)
    decomposition = whitened_kl_decomposition(posterior, prior)
    exact = kl_divergence(posterior, prior)
    assert torch.allclose(decomposition.total, exact, rtol=1e-5, atol=0.0)
    assert torch.allclose(
        decomposition.per_direction.sum(dim=-1), exact, rtol=1e-5, atol=0.0
    )


def test_whitened_decomposition_reduces_to_per_dim_kl_when_diagonal() -> None:
    """On the diagonal path the rotation-invariant view must recover the Phase 1 series.

    The eigenbasis of a diagonal-vs-diagonal problem IS the coordinate basis, so the two
    measurements have to agree as sorted multisets. This is what makes the Phase 2 table
    comparable with FINDINGS 2.3 rather than a differently-defined number that happens to
    sit nearby.
    """
    posterior, prior = whiten_pair(torch.float64, full=False)
    decomposition = whitened_kl_decomposition(posterior, prior)
    axis_wise = kl_divergence(posterior.base_dist, prior.base_dist)
    assert torch.allclose(
        decomposition.per_direction,
        axis_wise.sort(dim=-1, descending=True).values,
        atol=1e-12,
        rtol=0.0,
    )


def test_whitened_terms_are_non_negative_and_descending() -> None:
    """Each direction contributes a non-negative amount, and they come out sorted."""
    for full in (True, False):
        decomposition = whitened_kl_decomposition(*whiten_pair(torch.float64, full=full))
        assert (decomposition.per_direction >= 0).all()
        differences = decomposition.per_direction[:, :-1] - decomposition.per_direction[:, 1:]
        assert (differences >= 0).all(), "per-direction values are not descending"
        eigen_differences = decomposition.eigenvalues[:, :-1] - decomposition.eigenvalues[:, 1:]
        assert (eigen_differences >= 0).all(), "eigenvalues are not descending"


def test_whitened_decomposition_survives_the_collapsed_regime() -> None:
    """lambda ~ 1 is the dead-direction case, and it must not be lost to cancellation.

    This is why the implementation uses ``u - log1p(u)`` rather than
    ``lambda - 1 - log(lambda)``. With the posterior a hair wider than the prior in every
    direction, the true per-direction KL is ~1e-8; the naive form subtracts two nearly
    equal numbers and returns noise of the wrong sign.
    """
    offset = 1e-4
    generator = torch.Generator().manual_seed(3)
    mu = torch.randn(WHITEN_BATCH, WHITEN_DIM, generator=generator, dtype=torch.float64)
    sigma = torch.exp(
        0.5 * torch.randn(WHITEN_BATCH, WHITEN_DIM, generator=generator, dtype=torch.float64)
    )
    prior = Independent(Normal(mu, sigma), 1)
    posterior = Independent(Normal(mu.clone(), sigma * (1 + offset)), 1)

    decomposition = whitened_kl_decomposition(posterior, prior)
    # 1/2 (u - log1p(u)) with u = (1+offset)^2 - 1, i.e. ~offset^2 per direction.
    expected = 0.5 * ((1 + offset) ** 2 - 1 - 2 * np.log1p(offset))
    assert (decomposition.per_direction > 0).all(), "a dead direction went non-positive"
    assert torch.allclose(
        decomposition.per_direction,
        torch.full_like(decomposition.per_direction, expected),
        rtol=1e-6,
    )
    assert torch.allclose(
        decomposition.total, kl_divergence(posterior, prior), atol=1e-16, rtol=0.0
    )


def test_eigengap_flags_a_degenerate_spectrum() -> None:
    """An isotropic posterior has no well-defined directions, and the gap says so.

    The total stays exact -- degeneracy costs the *split*, not the sum -- so the companion
    is what tells a reader which of the two to trust.
    """
    identity = torch.eye(WHITEN_DIM, dtype=torch.float64).expand(
        WHITEN_BATCH, WHITEN_DIM, WHITEN_DIM
    )
    mu = torch.randn(
        WHITEN_BATCH, WHITEN_DIM, generator=torch.Generator().manual_seed(4), dtype=torch.float64
    )
    posterior = MultivariateNormal(mu, scale_tril=identity * 1.3)
    prior = MultivariateNormal(torch.zeros_like(mu), scale_tril=identity)

    degenerate = whitened_kl_decomposition(posterior, prior)
    assert float(degenerate.min_eigenvalue_gap.max()) == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(
        degenerate.total, kl_divergence(posterior, prior), atol=1e-12, rtol=0.0
    )

    # A non-degenerate spectrum reports a gap comfortably above zero, so the signal
    # discriminates rather than always firing.
    anisotropic = whitened_kl_decomposition(*whiten_pair(torch.float64))
    assert float(anisotropic.min_eigenvalue_gap.min()) > 1e-6


# --------------------------------------------------------------------------- #
# Effective rank -- and the inversion that motivates the whole design
# --------------------------------------------------------------------------- #
def test_effective_rank_endpoints() -> None:
    """exp(H) is 1 for a one-hot vector, N for a uniform one, and N-invariant to scale."""
    assert float(effective_rank(torch.tensor([1.0, 0.0, 0.0, 0.0]))) == pytest.approx(1.0)
    assert float(effective_rank(torch.ones(6))) == pytest.approx(6.0)
    # Scale-free: only the shape of the distribution matters.
    assert float(effective_rank(torch.tensor([2.0, 1.0]))) == pytest.approx(
        float(effective_rank(torch.tensor([200.0, 100.0])))
    )
    # Batched, and two active directions read as 2.
    batched = effective_rank(torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert batched.shape == (2,)
    assert float(batched[0]) == pytest.approx(2.0)
    assert float(batched[1]) == pytest.approx(1.0)


def test_effective_rank_of_an_all_zero_vector_argues_against_us() -> None:
    """A vanished total KL reports rank 1, never rank N.

    Documented in :func:`effective_rank`: the degenerate case must not be able to
    manufacture support for the Phase 2 hypothesis that the rank rose.
    """
    assert float(effective_rank(torch.zeros(6))) == pytest.approx(1.0)


def test_effective_rank_inverts_on_the_raw_spectrum() -> None:
    """THE measurement behind this design: the naive input reports the OPPOSITE answer.

    Phase 1's measured geometry, idealized: a unit prior, and a posterior that has
    contracted to sigma ~= 0.42 along exactly one direction while sitting on the prior in
    the other five. That is a latent using **one** direction -- the finding of
    FINDINGS 2.3, where one of six dimensions carried 98.8% of the KL.

    Feed the raw eigenvalues of Sigma_q to the same entropy formula and it reports ~5.5
    of 6 directions active, because five near-equal eigenvalues look beautifully isotropic
    and the entropy cannot tell "matches the prior" from "carries information". Feed it
    the prior-whitened per-direction KL and it reports 1.0, which is the truth.

    The gap is not a subtlety, it is an inversion: the naive metric is maximized by
    exactly the collapse the project is trying to detect. This test exists so that a
    future simplification back to the raw spectrum fails loudly instead of quietly
    reversing the headline result.
    """
    contracted = 0.42
    scales = torch.tensor([contracted] + [1.0] * (WHITEN_DIM - 1), dtype=torch.float64)
    factor = torch.diag_embed(scales).unsqueeze(0)
    mean = torch.zeros(1, WHITEN_DIM, dtype=torch.float64)
    posterior = MultivariateNormal(mean, scale_tril=factor)
    prior = MultivariateNormal(
        mean, scale_tril=torch.eye(WHITEN_DIM, dtype=torch.float64).unsqueeze(0)
    )

    # The naive reading: entropy of the raw covariance spectrum.
    raw = float(effective_rank(scales**2))
    # The measurement actually used: entropy of the normalized whitened KL.
    decomposition = whitened_kl_decomposition(posterior, prior)
    whitened = float(decomposition.effective_rank[0])

    assert raw == pytest.approx(5.50, abs=0.05), f"raw spectrum read {raw}"
    assert whitened == pytest.approx(1.00, abs=0.01), f"whitened read {whitened}"
    assert raw > 5.0 > whitened, "the inversion this design exists to avoid has vanished"

    # And the single active direction carries the entire KL, as it must.
    parts = decomposition.per_direction[0]
    assert float(parts[0]) > 0.4
    assert torch.allclose(parts[1:], torch.zeros_like(parts[1:]), atol=1e-12)


def test_effective_rank_rises_when_directions_share_the_load() -> None:
    """The metric moves in the direction the Phase 2 hypothesis predicts.

    Three equally contracted directions must read ~3, against ~1 for one. Without this,
    "effective rank" could be a constant that the inversion test alone would not catch.
    """
    identity = torch.eye(WHITEN_DIM, dtype=torch.float64).unsqueeze(0)
    mean = torch.zeros(1, WHITEN_DIM, dtype=torch.float64)
    prior = MultivariateNormal(mean, scale_tril=identity)

    def rank_with(active: int) -> float:
        scales = torch.tensor(
            [0.42] * active + [1.0] * (WHITEN_DIM - active), dtype=torch.float64
        )
        posterior = MultivariateNormal(mean, scale_tril=torch.diag_embed(scales).unsqueeze(0))
        return float(whitened_kl_decomposition(posterior, prior).effective_rank[0])

    assert rank_with(1) == pytest.approx(1.0, abs=0.01)
    assert rank_with(3) == pytest.approx(3.0, abs=0.01)
    assert rank_with(6) == pytest.approx(6.0, abs=0.01)


def test_effective_rank_accumulator_pools_and_summarizes() -> None:
    """The full-val accumulator reports mean, spread, quartiles and its sample count."""
    accumulator = EffectiveRankAccumulator()
    assert accumulator.metrics() == {}, "an empty accumulator must merge as a no-op"

    for seed in range(3):
        accumulator.update(*whiten_pair(torch.float32, seed=seed))
    metrics = accumulator.metrics()

    assert metrics["effrank_val_count"] == 3 * WHITEN_BATCH
    assert 1.0 <= metrics["effrank_val_mean"] <= WHITEN_DIM
    assert metrics["effrank_val_std"] > 0.0, "a dispersion estimate of zero is not evidence"
    assert (
        metrics["effrank_val_p25"] <= metrics["effrank_val_mean"] <= metrics["effrank_val_p75"]
        or metrics["effrank_val_p25"] <= metrics["effrank_val_p75"]
    )
    assert metrics["kl_val_total_mean"] > 0.0


def test_accumulator_measures_each_image_not_the_mean_covariance() -> None:
    """Per-image is the deliberate choice, and it is not the same number.

    Two images, each rank-1 but informative along *different* axes. Averaging the
    covariances first would report a two-dimensional population; measuring each posterior
    and then averaging reports 1.0, which is the honest answer to "how many directions
    does a posterior use".
    """
    mean = torch.zeros(2, WHITEN_DIM, dtype=torch.float64)
    first = torch.tensor([0.42] + [1.0] * (WHITEN_DIM - 1), dtype=torch.float64)
    second = torch.tensor([1.0, 0.42] + [1.0] * (WHITEN_DIM - 2), dtype=torch.float64)
    posterior = MultivariateNormal(mean, scale_tril=torch.diag_embed(torch.stack([first, second])))
    prior = MultivariateNormal(
        mean, scale_tril=torch.eye(WHITEN_DIM, dtype=torch.float64).expand(2, -1, -1)
    )

    accumulator = EffectiveRankAccumulator()
    accumulator.update(posterior, prior)
    assert accumulator.metrics()["effrank_val_mean"] == pytest.approx(1.0, abs=0.01)


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


def test_snapshot_geometry_reconciles_with_its_own_total(
    tiny_experiment: ExperimentConfig,
) -> None:
    """``kl_whitened_*`` must sum to ``kl_snapshot_total`` in the logged scalars.

    FINDINGS 2.3 had to caveat that per-dimension KL does not reconcile with ``val/kl``,
    because the two are computed over different populations. Logging the snapshot's own
    total next to the snapshot's own parts resolves that: the identity is now checkable
    straight from the metrics, and the mismatch with ``val/kl`` becomes an expected
    difference in scope rather than an unexplained one.
    """
    trainer = Trainer(tiny_experiment)
    metrics = trainer.validate()

    parts = [
        metrics[f"kl_whitened_{index}"]
        for index in range(tiny_experiment.model.latent_dim)
    ]
    assert sum(parts) == pytest.approx(metrics["kl_snapshot_total"], rel=1e-5)
    assert parts == sorted(parts, reverse=True), "not logged descending"
    assert "whitened_eigengap_min" in metrics
    assert 1.0 <= metrics["effrank_snapshot"] <= tiny_experiment.model.latent_dim
    # The snapshot is deliberately NOT val/kl: different population, and saying so in a
    # test stops a future reader from "fixing" the discrepancy.
    assert "kl" in metrics


def test_full_validation_geometry_runs_on_the_diagnostics_cadence(
    tiny_experiment: ExperimentConfig,
) -> None:
    """``effrank_val_*`` appears only when asked for, and pools far more samples."""
    trainer = Trainer(tiny_experiment)

    cheap = trainer.validate()
    assert not any(key.startswith("effrank_val") for key in cheap)
    assert "effrank_snapshot" in cheap, "the per-epoch snapshot must survive either way"

    full = trainer.validate(full_latent=True)
    assert full["effrank_val_count"] > 0
    # Every validation image at all four graders, against the snapshot's one batch at
    # grader 0 -- the whole reason the second family exists.
    batch_size = tiny_experiment.data.batch_size
    assert full["effrank_val_count"] >= N_GRADERS * batch_size
    assert full["effrank_val_std"] >= 0.0
    assert full["effrank_val_p25"] <= full["effrank_val_p75"]
    assert 1.0 <= full["effrank_val_mean"] <= tiny_experiment.model.latent_dim


def test_cadence_misalignment_is_flagged(tiny_experiment: ExperimentConfig) -> None:
    """A diagnostics cadence that misses validation epochs must not fail silently.

    ``effrank_val_*`` is computed inside ``validate``, so misaligned cadences would make
    the headline Phase 2 series sparse for a reason nothing in the logs would explain.
    """
    aligned = Trainer(
        dataclasses.replace(
            tiny_experiment,
            train=dataclasses.replace(tiny_experiment.train, val_every_n_epochs=2),
            log=dataclasses.replace(tiny_experiment.log, diagnostics_every_n_epochs=4),
        )
    )
    assert aligned._latent_cadences_align

    misaligned = Trainer(
        dataclasses.replace(
            tiny_experiment,
            train=dataclasses.replace(tiny_experiment.train, val_every_n_epochs=2),
            log=dataclasses.replace(tiny_experiment.log, diagnostics_every_n_epochs=3),
        )
    )
    assert not misaligned._latent_cadences_align
    assert "WARNING" in misaligned.describe()

    # Every shipped config gets this right.
    for name in ("baseline", "modernized", "extension", "ablation_no_augmentation"):
        config = ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml")
        assert (
            config.log.diagnostics_every_n_epochs % config.train.val_every_n_epochs == 0
        ), name


def test_a_full_covariance_run_completes_end_to_end(
    tiny_experiment: ExperimentConfig,
) -> None:
    """THE Stage 4 deliverable: a full-covariance run survives a whole training loop.

    Before Stage 4 this raised ``NotImplementedError`` from ``per_dim_kl`` at the end of
    the first validation pass, and again from ``reparameterize`` in the diagnostics --
    which is why Stage 3 shipped with an explicit refusal rather than a latent crash.
    Every latent diagnostic must now produce finite numbers on the full-covariance path,
    because a Phase 2 run that dies at epoch 1 of 848 is discovered a day late.
    """
    config = dataclasses.replace(
        tiny_experiment,
        model=dataclasses.replace(tiny_experiment.model, latent_covariance="full"),
    )
    trainer = Trainer(config)
    assert trainer.model.prior_net.full_covariance

    summary = trainer.train()
    assert summary["epochs"] == 2

    for record in summary["history"]:
        for key, value in record.items():
            assert np.isfinite(value), f"{key} is not finite on the full-covariance path"
        for key in (
            "val/kl_dim_0",
            "val/kl_whitened_0",
            "val/kl_snapshot_total",
            "val/whitened_eigengap_min",
            "val/effrank_snapshot",
            "val/effrank_val_mean",
            "val/effrank_val_std",
            "diag/sample_diversity_iou",
            "diag/prior_posterior_ce_ratio",
        ):
            assert key in record, f"{key} missing from a full-covariance run"

    # At zero-init the correlations start at zero, so the geometry must still be sane
    # rather than degenerate-by-construction.
    last = summary["history"][-1]
    assert 1.0 <= last["val/effrank_val_mean"] <= config.model.latent_dim


def test_per_dim_kl_marginals_do_not_pretend_to_decompose() -> None:
    """Under a full covariance, ``kl_dim_*`` is a marginal and says so by not summing.

    Kept as an explicit test because the tempting misreading -- treating the coordinate
    series as a breakdown of the total, exactly as Phase 1 legitimately did -- would
    silently misattribute the KL once the covariance is full. The whitened decomposition
    is the one that adds up.
    """
    posterior, prior = whiten_pair(torch.float64, full=True)
    marginal_total = float(per_dim_kl(posterior, prior).sum())
    exact_total = float(kl_divergence(posterior, prior).mean())
    assert marginal_total != pytest.approx(exact_total, rel=1e-3), (
        "the marginals summed to the total, so this test no longer proves anything"
    )
    # The whitened decomposition, on the same pair, does reconcile.
    whitened = whitened_kl_decomposition(posterior, prior)
    assert float(whitened.total.mean()) == pytest.approx(exact_total, rel=1e-9)

    # On the diagonal path the coordinate series IS separable, and still is.
    diagonal_posterior, diagonal_prior = whiten_pair(torch.float64, full=False)
    assert float(per_dim_kl(diagonal_posterior, diagonal_prior).sum()) == pytest.approx(
        float(kl_divergence(diagonal_posterior, diagonal_prior).mean()), rel=1e-9
    )


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


def test_budget_in_iterations_derives_the_epoch_count(
    tiny_experiment: ExperimentConfig,
) -> None:
    """Epochs are derived from the split size, never hardcoded."""
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="iters"),
        train=dataclasses.replace(tiny_experiment.train, epochs=None, iterations=7),
    )
    trainer = Trainer(config)
    # 3 batches per epoch, so 7 iterations rounds to the nearest whole epoch: 2.
    assert trainer.steps_per_epoch == 3
    assert trainer.planned_epochs == round(7 / 3) == 2
    assert trainer.total_steps == 6


def test_iteration_budget_rounds_to_the_nearest_epoch() -> None:
    """Nearest, not ceiling: for the real config that is a 0.007% miss, not 0.11%.

    9,056 train patches at batch 32 give 283 steps/epoch, and 240,000 / 283 = 848.06.
    Rounding to nearest yields 848 epochs = 239,984 steps (16 short); rounding up would
    yield 849 = 240,267 (267 over). The learning-rate milestones are fractions of the
    realized total, so the schedule spans exactly the run performed.
    """
    steps_per_epoch, iterations = 283, 240000
    assert round(iterations / steps_per_epoch) == 848
    assert 848 * steps_per_epoch == 239984
    assert abs(239984 - iterations) < abs(849 * steps_per_epoch - iterations)


def test_budget_must_be_given_exactly_once() -> None:
    """Expressing the budget twice invites the two from drifting apart."""
    with pytest.raises(ValueError, match="exactly one of train.iterations"):
        TrainConfig(epochs=100, iterations=240000)
    with pytest.raises(ValueError, match="exactly one of train.iterations"):
        TrainConfig(epochs=None, iterations=None)
    # And each on its own is fine.
    assert TrainConfig(epochs=None, iterations=240000).iterations == 240000
    assert TrainConfig(epochs=10).epochs == 10


def test_five_step_decay_visits_every_level(tiny_experiment: ExperimentConfig) -> None:
    """The paper's schedule shape: six learning rates separated by five decays."""
    values = (1.0e-4, 3.9810717e-5, 1.5848932e-5, 6.3095734e-6, 2.5118864e-6, 1.0e-6)
    milestones = (1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6)
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="ladder"),
        train=dataclasses.replace(tiny_experiment.train, epochs=12),
        optim=dataclasses.replace(tiny_experiment.optim, lr=values[0]),
        schedule=ScheduleConfig(name="piecewise", milestones=milestones, values=values),
    )
    trainer = Trainer(config)
    assert trainer.total_steps == 36

    seen = []
    for _ in range(trainer.total_steps):
        seen.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.optimizer.step()
        trainer.scheduler.step()

    # Every configured level is actually visited, in order, and the run ends at 1e-6.
    assert sorted(set(seen), reverse=True) == pytest.approx(sorted(values, reverse=True))
    assert seen[0] == pytest.approx(values[0])
    assert seen[-1] == pytest.approx(values[-1])
    assert seen == sorted(seen, reverse=True), "the learning rate must never increase"
    # Decays land on the fractional boundaries, not on absolute step counts.
    assert seen[5] == pytest.approx(values[0]) and seen[6] == pytest.approx(values[1])


def test_history_survives_a_resume(tiny_experiment: ExperimentConfig) -> None:
    """A multi-day run may resume several times; the loss curve must not be truncated.

    Without this, ``summary.json`` would hold only the epochs since the last resume and
    the report's curve would silently start partway through.
    """
    first = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="hist"),
        train=dataclasses.replace(tiny_experiment.train, epochs=2),
    )
    stopped = Trainer(first)
    stopped.train()
    assert len(stopped.history) == 2

    resumed = Trainer(
        dataclasses.replace(
            tiny_experiment,
            run=dataclasses.replace(tiny_experiment.run, name="hist2"),
            train=dataclasses.replace(tiny_experiment.train, epochs=4),
        )
    )
    resumed.resume(stopped.checkpoint_dir / "last.pt")
    assert len(resumed.history) == 2, "history was not restored from the checkpoint"
    summary = resumed.train()
    assert len(summary["history"]) == 4
    assert [entry["epoch"] for entry in summary["history"]] == [0.0, 1.0, 2.0, 3.0]


def test_augmentation_metrics_reach_the_training_record(
    tiny_experiment: ExperimentConfig,
) -> None:
    """The lesion-loss counter must actually be logged, or the guard is unobservable."""
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="augmetrics"),
        data=dataclasses.replace(
            tiny_experiment.data,
            augmentation=AugmentationConfig(
                enabled=True, pad_to_px=SIZE + 4, elastic_sigma_px=2.0
            ),
        ),
    )
    trainer = Trainer(config)
    metrics = trainer.train_epoch(0)
    assert metrics["aug_samples"] > 0
    assert "aug_lesion_lost_fraction" in metrics
    assert "aug_redraw_rate" in metrics
    # And the banner says augmentation is on, so a run cannot be mislabelled.
    assert "augmentation  : on" in trainer.describe()


def test_banner_flags_a_run_without_augmentation(tiny_experiment: ExperimentConfig) -> None:
    """An ablation must announce itself; the baseline is the augmented configuration."""
    assert "OFF - not the paper's configuration" in Trainer(tiny_experiment).describe()


def test_banner_reports_the_derived_budget(tiny_experiment: ExperimentConfig) -> None:
    """A multi-day run must not be a surprise, so the banner states what it will do."""
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="banner"),
        train=dataclasses.replace(tiny_experiment.train, epochs=None, iterations=7),
    )
    banner = Trainer(config).describe()
    assert "7 iterations" in banner
    assert "planned epochs: 2" in banner


def test_runtime_estimate_does_not_disturb_the_run(
    tiny_experiment: ExperimentConfig,
) -> None:
    """The timing probe must leave the trajectory bit-identical.

    It runs on a deep copy with a throwaway optimizer and restores the RNG state, so a
    run that printed an estimate must train exactly like one that did not.
    """
    reference = Trainer(
        dataclasses.replace(
            tiny_experiment, run=dataclasses.replace(tiny_experiment.run, name="noprobe")
        )
    )
    reference.train_epoch(0)

    probed = Trainer(
        dataclasses.replace(
            tiny_experiment, run=dataclasses.replace(tiny_experiment.run, name="probe")
        )
    )
    estimate = probed.estimate_seconds_per_epoch(probe_steps=2)
    if estimate is not None:
        assert estimate > 0
    probed.train_epoch(0)

    for expected, actual in zip(
        reference.model.parameters(), probed.model.parameters(), strict=True
    ):
        assert torch.allclose(expected.detach(), actual.detach(), atol=1e-6), (
            "the timing probe perturbed the training trajectory"
        )


def test_duration_formatting_is_readable() -> None:
    """A multi-day estimate must read as days, not as 118800 seconds."""
    assert Trainer.format_duration(45) == "45.0 s"
    assert Trainer.format_duration(600) == "10.0 min"
    assert Trainer.format_duration(7200) == "2.0 h"
    assert Trainer.format_duration(118800) == "1 d 9.0 h"


# --------------------------------------------------------------------------- #
# RNG state: CPU, CUDA and MPS
# --------------------------------------------------------------------------- #
def test_rng_state_covers_every_available_backend() -> None:
    """A resume replays the exact sequence only if every generator is captured.

    ``torch.get_rng_state()`` is the CPU generator alone, but this model samples ``z`` on
    the training device every step -- CUDA for the long run, MPS in development. Saving
    only the CPU state silently turns "resumes exactly" into "resumes approximately".
    """
    state = rng_state()
    assert {"torch", "numpy", "python"} <= set(state)
    assert ("cuda" in state) == torch.cuda.is_available()
    assert ("mps" in state) == torch.backends.mps.is_available()


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_rng_state_round_trip_reproduces_draws(device: str) -> None:
    """Restoring the captured state must reproduce the next draw on each backend."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS device available")

    state = rng_state()
    first = torch.randn(8, device=device)
    set_rng_state(state)
    second = torch.randn(8, device=device)
    assert torch.equal(first, second), f"{device} RNG state was not restored"


def accelerator() -> str | None:
    """Return an available accelerator device string, or None.

    Returns:
        ``"cuda"``, ``"mps"`` or None.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def test_rng_state_survives_a_device_mapped_checkpoint_load(
    tmp_path: Path, tiny_experiment: ExperimentConfig
) -> None:
    """Restoring RNG state from a checkpoint loaded onto an accelerator must work.

    Regression test. Checkpoints are loaded with ``map_location=<training device>``, which
    moves *every* storage in the payload -- including the RNG state tensors -- onto the
    accelerator, and ``torch.set_rng_state`` rejects a non-CPU tensor with
    ``TypeError: RNG state must be a torch.ByteTensor``. Resuming on CPU worked while
    resuming on MPS or CUDA raised, and a CPU-only test cannot tell the two apart.
    """
    device = accelerator()
    if device is None:
        pytest.skip("no accelerator available")

    model = ProbUNet(tiny_experiment.model)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "rng.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        global_step=1,
        config={},
        seed=0,
        device=device,
        monitor="val/total",
        best_metric=None,
        metrics={},
    )
    expected = torch.randn(4)
    # restore_rng=True is the path that used to raise.
    load_checkpoint(path, map_location=device, restore_rng=True)
    assert torch.equal(torch.randn(4), expected)


def test_resume_matches_uninterrupted_on_the_accelerator(
    tiny_experiment: ExperimentConfig,
) -> None:
    """The exact-replay guarantee must hold on the device the long run actually uses.

    The 240k-iteration run happens on an accelerator and will very likely be resumed, so
    "resumes exactly" has to be true there and not only on CPU. Augmentation is enabled
    here because its draws are derived from the restored epoch rather than stored.
    """
    device = accelerator()
    if device is None:
        pytest.skip("no accelerator available")

    def configure(name: str, epochs: int) -> ExperimentConfig:
        return dataclasses.replace(
            tiny_experiment,
            run=dataclasses.replace(tiny_experiment.run, name=name, device=device),
            data=dataclasses.replace(
                tiny_experiment.data,
                augmentation=AugmentationConfig(
                    enabled=True, pad_to_px=SIZE + 4, elastic_sigma_px=2.0
                ),
            ),
            train=dataclasses.replace(tiny_experiment.train, epochs=epochs),
        )

    uninterrupted = Trainer(configure("acc_full", 4))
    uninterrupted.train()
    reference = [p.detach().cpu().clone() for p in uninterrupted.model.parameters()]

    stopped = Trainer(configure("acc_half", 2))
    stopped.train()
    resumed = Trainer(configure("acc_resumed", 4))
    resumed.resume(stopped.checkpoint_dir / "last.pt")
    resumed.train()

    for expected, actual in zip(reference, resumed.model.parameters(), strict=True):
        assert torch.allclose(expected, actual.detach().cpu(), atol=1e-6), (
            f"resume diverged from the uninterrupted run on {device}"
        )


def test_checkpoint_is_written_every_epoch(tiny_experiment: ExperimentConfig) -> None:
    """An interruption must cost at most one epoch, which requires a save per epoch."""
    config = dataclasses.replace(
        tiny_experiment,
        run=dataclasses.replace(tiny_experiment.run, name="cadence"),
        train=dataclasses.replace(tiny_experiment.train, epochs=3, val_every_n_epochs=2),
    )
    trainer = Trainer(config)
    seen: list[int] = []
    original = trainer._checkpoint

    def spy(epoch: int, record: dict[str, float]) -> None:
        seen.append(epoch)
        original(epoch, record)

    trainer._checkpoint = spy  # type: ignore[method-assign]
    trainer.train()
    # Every epoch, even the ones that did not validate: last.pt is what a resume reads.
    assert seen == [0, 1, 2]
    state = load_checkpoint(trainer.checkpoint_dir / "last.pt", restore_rng=False)
    assert state.epoch == 3


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