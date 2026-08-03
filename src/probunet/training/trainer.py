"""The baseline training loop.

Training draws ``z`` from the posterior, which has seen one randomly chosen grader mask
for this epoch. Validation is deliberately different: it averages the objective over
**all four graders**, so the validation number does not depend on which pairing the
epoch happened to draw. Checkpoints are selected on validation loss only -- never test.

The validation pass shares work the way the architecture intends: one U-Net pass and one
prior pass per batch, reused across four posterior passes and four ``f_comb`` passes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from probunet.data.lidc import LidcDataset, build_data, panel_batch
from probunet.losses.elbo import elbo_loss
from probunet.model.prob_unet import ProbUNet
from probunet.training.checkpoint import (
    BEST_NAME,
    LAST_NAME,
    is_improvement,
    load_checkpoint,
    save_checkpoint,
)
from probunet.training.config import ExperimentConfig
from probunet.training.freeze import freeze_module
from probunet.training.diagnostics import (
    build_diagnostic_sets,
    logits_to_mask,
    make_panel,
    mean_pairwise_iou,
    nonempty_sample_fraction,
    per_dim_kl,
    prior_samples_for_images,
    reparameterize,
    save_diagnostic_sets,
    sigma_stats,
)
from probunet.utils.runtime import (
    describe_device,
    seed_everything,
    select_device,
)

LOGGER = logging.getLogger(__name__)

N_GRADERS = 4
DIAGNOSTIC_SEED_OFFSET = 7919


class Trainer:
    """Owns the model, data, optimizer and logging for one run."""

    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device | None = None,
        base_checkpoint: Path | None = None,
    ) -> None:
        """Build everything a run needs.

        Args:
            config: The resolved experiment configuration.
            device: Override for the device; normally selected from the config.
            base_checkpoint: Required when ``config.train.mode == "selection_head"``:
                the trained Probabilistic U-Net the head is fitted on top of. It is
                frozen, and the freeze is asserted and logged.

        Raises:
            NotImplementedError: If AMP is requested on a non-CUDA device (autocast on
                MPS is not exercised here and ``GradScaler`` is CUDA-centric, so
                silently ignoring the flag would be worse than refusing it), or if
                ``selection_head`` mode is requested -- the head is Phase 3.
            ValueError: If ``selection_head`` mode is requested without a base
                checkpoint.
        """
        self.config = config
        self.base_checkpoint = base_checkpoint
        seed_everything(config.run.seed, deterministic=config.run.deterministic)
        self.device = device or select_device(config.run.device)

        if config.train.amp and self.device.type != "cuda":
            raise NotImplementedError(
                f"amp=true requires CUDA, but the device is {self.device.type}. Mixed "
                "precision on MPS is untested here and GradScaler is CUDA-specific; "
                "run with amp=false or on the CUDA machine."
            )

        self.data = build_data(config.data)
        self.model = ProbUNet(config.model).to(self.device)
        self.base_record: dict[str, object] | None = None
        if config.train.mode == "selection_head":
            self._prepare_selection_head()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=tuple(config.optim.betas),
            eps=config.optim.eps,
        )

        self.steps_per_epoch = self._steps_per_epoch()
        self.total_steps = self.steps_per_epoch * config.train.epochs
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler(enabled=config.train.amp)

        self.run_dir = config.run.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.writer: SummaryWriter | None = (
            SummaryWriter(str(self.run_dir / "tb")) if config.log.tensorboard else None
        )

        # Fixed, ambiguity-stratified diagnostic sets, recorded so the panel is the
        # same images in every run and comparable across phases.
        self.diagnostic_sets = build_diagnostic_sets(
            self.data.datasets["val"],
            diversity_images=config.log.diversity_images,
            panel_images=config.log.panel_images,
            seed=config.run.seed,
        )
        save_diagnostic_sets(self.diagnostic_sets, self.run_dir / "diagnostic_indices.json")
        self._diversity_dataset = LidcDataset(
            self.data.arrays, self.diagnostic_sets.diversity, mode="eval"
        )
        # Deterministic order, built once: the diagnostics must compare the same images
        # in the same order at every epoch and across runs.
        self._diagnostic_loader = torch.utils.data.DataLoader(
            self._diversity_dataset, batch_size=config.data.batch_size, shuffle=False
        )

        self.epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ setup
    def _steps_per_epoch(self) -> int:
        """Training batches per epoch, respecting ``limit_train_batches``."""
        batches = len(self.data.loaders["train"])
        limit = self.config.train.limit_train_batches
        return min(batches, limit) if limit else batches

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the learning-rate schedule.

        Milestones are fractions of the total step count, so a schedule keeps its shape
        when the budget changes.

        Returns:
            A per-step ``LambdaLR``.
        """
        schedule = self.config.schedule
        if schedule.name == "constant":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _step: 1.0)

        boundaries = [int(round(m * self.total_steps)) for m in schedule.milestones]
        factors = [value / self.config.optim.lr for value in schedule.values]

        def factor_at(step: int) -> float:
            """Multiplier for the base learning rate at a given step."""
            index = sum(1 for boundary in boundaries if step >= boundary)
            return factors[index]

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor_at)

    def _prepare_selection_head(self) -> None:
        """Load and freeze the base model for selection-head training.

        The head is Phase 3, so this stops after the part that is defined now: the base
        checkpoint is loaded, frozen, and the freeze is asserted and logged. Doing the
        freeze here rather than inside a future head implementation means the contract
        the extension depends on is already covered by tests.

        Raises:
            ValueError: If no base checkpoint was supplied.
            NotImplementedError: Always, once the base is frozen -- the head itself does
                not exist yet.
        """
        if self.base_checkpoint is None:
            raise ValueError(
                "train.mode='selection_head' requires --base-checkpoint: the head is "
                "trained on top of an already-trained, frozen Probabilistic U-Net."
            )
        load_checkpoint(
            self.base_checkpoint,
            model=self.model,
            map_location=self.device,
            restore_rng=False,
        )
        self.base_record = freeze_module(self.model, name="base Probabilistic U-Net")
        LOGGER.info("base model frozen: %s", self.base_record)
        raise NotImplementedError(
            "the consensus-selection head is Phase 3 and is not implemented yet. The "
            "base model loads and freezes correctly (see the log line above); what is "
            "missing is the head module, its scoring target and its training step."
        )

    def describe(self) -> str:
        """Return a startup banner: config, device, seed and provenance."""
        parameters = sum(p.numel() for p in self.model.parameters())
        lines = [
            f"run           : {self.config.run.name}",
            f"mode          : {self.config.train.mode}",
            f"device        : {describe_device(self.device)}",
            f"seed          : {self.config.run.seed}",
            f"parameters    : {parameters:,}",
            f"channels      : {self.model.unet.widths}"
            + ("" if self.config.model.max_channels is None else "  [CAPPED - not baseline]"),
            f"train patches : {len(self.data.datasets['train'])}",
            f"val patches   : {len(self.data.datasets['val'])}",
            f"steps/epoch   : {self.steps_per_epoch}",
            f"total steps   : {self.total_steps}",
            f"beta          : {self.config.loss.beta}",
            f"lr            : {self.config.optim.lr} ({self.config.schedule.name})",
            f"monitor       : {self.config.checkpoint.monitor} ({self.config.checkpoint.mode})",
            f"run dir       : {self.run_dir}",
        ]
        return "\n".join(lines)

    # --------------------------------------------------------------- training
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Epoch index, used to redraw the grader pairing.

        Returns:
            Mean training metrics for the epoch.
        """
        self.model.train()
        # Fresh grader pairing for this epoch, reproducible from (pairing_seed, epoch).
        self.data.set_epoch(epoch)
        loader = self.data.loaders["train"]
        limit = self.config.train.limit_train_batches

        totals: dict[str, float] = {}
        count = 0
        started = time.perf_counter()

        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.config.train.amp):
                output = self.model(image, mask)
                terms = elbo_loss(
                    output.logits, mask, output.posterior, output.prior, beta=self.config.loss.beta
                )
            self.scaler.scale(terms["total"]).backward()
            if self.config.train.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            # detach before converting: these tensors are still attached to the graph,
            # and float() on a requires_grad tensor warns and keeps it alive.
            scalars = {key: float(value.detach()) for key, value in terms.items()}
            for key, value in scalars.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1

            if self.writer and self.global_step % self.config.log.log_every_n_steps == 0:
                for key, value in scalars.items():
                    self.writer.add_scalar(f"step/{key}", value, self.global_step)
                self.writer.add_scalar(
                    "step/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        metrics["seconds"] = time.perf_counter() - started
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate the objective over **all four graders** per image.

        Averaging over graders removes the dependence on the epoch's random pairing, so
        the number that selects checkpoints is comparable from epoch to epoch. One U-Net
        pass and one prior pass are shared across the four posteriors.

        Returns:
            Mean validation metrics.
        """
        self.model.eval()
        loader = self.data.loaders["val"]
        limit = self.config.train.limit_val_batches

        totals: dict[str, float] = {}
        count = 0
        latent: dict[str, float] = {}
        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            image = batch["image"].to(self.device, non_blocking=True)
            masks = batch["masks"].to(self.device, non_blocking=True)

            encoded = self.model.encode(image)
            for grader in range(N_GRADERS):
                target = masks[:, grader].to(torch.int64)
                posterior = self.model.posterior_net.distribution(image, target)
                logits = self.model.reconstruct(encoded, posterior.rsample())
                terms = elbo_loss(
                    logits, target, posterior, encoded.prior, beta=self.config.loss.beta
                )
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                count += 1

            if index == 0:
                # Latent statistics from the first batch: cheap, and enough to spot a
                # collapsing sigma or a dead latent dimension.
                latent = self._latent_stats(image, masks)

        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        metrics.update(latent)
        return metrics

    @torch.no_grad()
    def _latent_stats(self, image: Tensor, masks: Tensor) -> dict[str, float]:
        """Sigma statistics and per-dimension KL, from one batch.

        Args:
            image: Image batch.
            masks: All four grader masks.

        Returns:
            Latent diagnostics as flat scalars.
        """
        target = masks[:, 0].to(torch.int64)
        prior_stats = self.model.prior_net(image)
        posterior_stats = self.model.posterior_net(
            self.model.posterior_net.assemble_input(image, target)
        )
        prior = self.model.prior_net.distribution_from_stats(*prior_stats)
        posterior = self.model.posterior_net.distribution_from_stats(*posterior_stats)

        stats = sigma_stats(prior_stats, "prior")
        stats.update(sigma_stats(posterior_stats, "posterior"))
        for dimension, value in enumerate(per_dim_kl(posterior, prior)):
            stats[f"kl_dim_{dimension}"] = float(value)
        return stats

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def run_diagnostics(self, epoch: int) -> dict[str, float]:
        """Compare posterior-z against prior-z, and measure sample diversity.

        Returns:
            Diagnostic scalars. See :mod:`probunet.training.diagnostics` for how to read
            them; in particular ``sample_diversity_iou`` must always be read together
            with ``nonempty_sample_fraction``.
        """
        self.model.eval()
        generator = torch.Generator().manual_seed(
            self.config.run.seed + DIAGNOSTIC_SEED_OFFSET
        )
        config = self.config.log

        posterior_ce = 0.0
        prior_ce_mean = 0.0
        prior_ce_best = 0.0
        pairs = 0
        diversity_total = 0.0
        nonempty_total = 0.0
        batches = 0

        # One pass over the fixed diversity set: the U-Net encoding is shared between
        # the reconstruction comparison and the diversity measure.
        for batch in self._diagnostic_loader:
            image = batch["image"].to(self.device)
            masks = batch["masks"].to(self.device)
            encoded = self.model.encode(image)

            prior_logits = [
                self.model.reconstruct(encoded, reparameterize(encoded.prior, generator))
                for _ in range(config.prior_samples_for_ce)
            ]
            for grader in range(N_GRADERS):
                target = masks[:, grader].to(torch.int64)
                posterior = self.model.posterior_net.distribution(image, target)
                logits = self.model.reconstruct(
                    encoded, reparameterize(posterior, generator)
                )
                posterior_ce += float(
                    elbo_loss(logits, target, posterior, encoded.prior)["ce"]
                )
                sample_ces = [
                    float(elbo_loss(sample, target, posterior, encoded.prior)["ce"])
                    for sample in prior_logits
                ]
                prior_ce_mean += sum(sample_ces) / len(sample_ces)
                prior_ce_best += min(sample_ces)
                pairs += 1

            samples = prior_samples_for_images(
                self.model, encoded, config.diversity_samples, generator
            )
            diversity_total += float(mean_pairwise_iou(samples))
            nonempty_total += float(nonempty_sample_fraction(samples))
            batches += 1

        posterior_ce /= max(pairs, 1)
        prior_ce_mean /= max(pairs, 1)
        prior_ce_best /= max(pairs, 1)

        metrics = {
            "ce_posterior_z": posterior_ce,
            "ce_prior_z_mean": prior_ce_mean,
            "ce_prior_z_best": prior_ce_best,
            "prior_posterior_ce_ratio": prior_ce_mean / max(posterior_ce, 1e-12),
            "sample_diversity_iou": diversity_total / max(batches, 1),
            "nonempty_sample_fraction": nonempty_total / max(batches, 1),
        }

        if self.writer:
            self._log_panel(epoch, generator)
        return metrics

    @torch.no_grad()
    def _log_panel(self, epoch: int, generator: torch.Generator) -> None:
        """Write the qualitative panel to TensorBoard.

        Args:
            epoch: Epoch index, used as the image step.
            generator: Generator supplying reproducible sample noise.
        """
        # Same path the notebook uses, so a panel works from the full dataset or from
        # the tracked subset export without a second code path.
        panel_images, panel_masks = panel_batch(
            self.data.arrays, self.diagnostic_sets.panel
        )
        image = panel_images.to(self.device)
        masks = panel_masks.to(self.device)
        encoded = self.model.encode(image)
        samples = prior_samples_for_images(
            self.model, encoded, self.config.log.panel_samples, generator
        )
        panel = make_panel(image.cpu(), masks.cpu(), samples.cpu())
        assert self.writer is not None
        self.writer.add_image("panel/image_graders_priorsamples", panel, epoch)

    # ------------------------------------------------------------------- loop
    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            A summary with the best monitored value, the epoch it occurred at, and the
            per-epoch history.
        """
        LOGGER.info("starting training\n%s", self.describe())
        (self.run_dir / "config.resolved.yaml").write_text(self.config.to_yaml())

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config.train.epochs):
            train_metrics = self.train_epoch(epoch)
            record = {f"train/{k}": v for k, v in train_metrics.items()}

            if (epoch + 1) % self.config.train.val_every_n_epochs == 0:
                val_metrics = self.validate()
                record.update({f"val/{k}": v for k, v in val_metrics.items()})
                # The overfitting gap as a first-class number: 27.5M parameters on
                # 9,056 patches with no augmentation will diverge, and the report needs
                # the magnitude quantified rather than eyeballed off two curves.
                record["diag/gap_total"] = record["val/total"] - record["train/total"]
                record["diag/gap_ce"] = record["val/ce"] - record["train/ce"]

            if (epoch + 1) % self.config.log.diagnostics_every_n_epochs == 0:
                record.update(
                    {f"diag/{k}": v for k, v in self.run_diagnostics(epoch).items()}
                )

            record["lr"] = self.optimizer.param_groups[0]["lr"]
            self.epoch = epoch + 1
            self._log_epoch(epoch, record)
            self.history.append({"epoch": float(epoch), **record})
            self._checkpoint(epoch, record)

        summary = {
            "best_metric": self.best_metric,
            "monitor": self.config.checkpoint.monitor,
            "epochs": self.epoch,
            "global_step": self.global_step,
            "history": self.history,
        }
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return summary

    def _log_epoch(self, epoch: int, record: dict[str, float]) -> None:
        """Emit one epoch's metrics to TensorBoard and the log.

        Args:
            epoch: Epoch index.
            record: Flat metric mapping.
        """
        if self.writer:
            for key, value in record.items():
                if key.endswith("seconds"):
                    continue
                self.writer.add_scalar(key, value, epoch)
        summary = " ".join(
            f"{key}={record[key]:.4f}"
            for key in ("train/total", "train/ce", "train/kl", "val/total", "diag/gap_total")
            if key in record
        )
        LOGGER.info("epoch %d/%d %s", epoch + 1, self.config.train.epochs, summary)

    def _checkpoint(self, epoch: int, record: dict[str, float]) -> None:
        """Save last, and best when the monitored metric improves.

        Args:
            epoch: Epoch index.
            record: Flat metric mapping for this epoch.
        """
        policy = self.config.checkpoint
        generator = self.data.loaders["train"].generator
        common = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "seed": self.config.run.seed,
            "device": str(self.device),
            "monitor": policy.monitor,
            "metrics": {k: v for k, v in record.items()},
            "loader_generator_state": generator.get_state() if generator is not None else None,
        }
        if policy.save_last:
            save_checkpoint(
                self.checkpoint_dir / LAST_NAME, best_metric=self.best_metric, **common
            )

        candidate = record.get(policy.monitor)
        if candidate is None:
            return
        if is_improvement(candidate, self.best_metric, policy.mode):
            self.best_metric = candidate
            if policy.save_best:
                save_checkpoint(
                    self.checkpoint_dir / BEST_NAME, best_metric=self.best_metric, **common
                )
            LOGGER.info("new best %s=%.5f at epoch %d", policy.monitor, candidate, epoch + 1)

    def resume(self, path: Path) -> None:
        """Restore a run from a checkpoint, continuing the exact sequence.

        Args:
            path: Checkpoint file, normally ``last.pt``.
        """
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
            restore_rng=True,
        )
        self.epoch = state.epoch
        self.global_step = state.global_step
        self.best_metric = state.best_metric
        from probunet.training.checkpoint import loader_generator_state

        saved = loader_generator_state(path)
        generator = self.data.loaders["train"].generator
        if saved is not None and generator is not None:
            generator.set_state(saved)
        LOGGER.info(
            "resumed from %s at epoch %d (step %d, best %s=%s, git %s)",
            path,
            state.epoch,
            state.global_step,
            state.monitor,
            state.best_metric,
            state.git_revision,
        )
