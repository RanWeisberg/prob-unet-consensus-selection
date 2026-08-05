"""The baseline training loop.

Training draws ``z`` from the posterior, which has seen one randomly chosen grader mask
for this epoch. Validation is deliberately different: it averages the objective over
**all four graders**, so the validation number does not depend on which pairing the
epoch happened to draw. Checkpoints are selected on validation loss only -- never test.

The validation pass shares work the way the architecture intends: one U-Net pass and one
prior pass per batch, reused across four posterior passes and four ``f_comb`` passes.
"""

from __future__ import annotations

import copy
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
    rng_state,
    seed_everything,
    select_device,
    set_rng_state,
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
        self.planned_epochs, self.total_steps = self._resolve_budget()
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

    def _resolve_budget(self) -> tuple[int, int]:
        """Turn the configured budget into an epoch count and a total step count.

        The paper states its budget in iterations (240k at batch 32), so that is what the
        config expresses and the epoch count is *derived* from the train split size rather
        than hardcoded. An iteration budget rarely divides evenly into whole epochs, and
        the loop's unit is an epoch, so it is rounded to the **nearest** whole epoch. For
        the real configuration that is a far better fit than rounding up: 240,000 / 283 =
        848.06, so nearest gives 848 epochs = 239,984 steps (16 short, 0.007%) where
        rounding up would give 849 = 240,267 (267 over).

        The realized count is what the learning-rate milestones are taken as fractions
        of, so the schedule always spans exactly the run that is actually performed.

        Returns:
            ``(planned_epochs, total_steps)``.
        """
        train = self.config.train
        if train.iterations is not None:
            epochs = max(1, round(train.iterations / self.steps_per_epoch))
            total = epochs * self.steps_per_epoch
            LOGGER.info(
                "budget: %d iterations requested -> %d epochs x %d steps = %d steps "
                "(%+d, %+.3f%%)",
                train.iterations,
                epochs,
                self.steps_per_epoch,
                total,
                total - train.iterations,
                100.0 * (total - train.iterations) / train.iterations,
            )
            return epochs, total
        assert train.epochs is not None  # guaranteed by TrainConfig validation
        return train.epochs, train.epochs * self.steps_per_epoch

    def estimate_seconds_per_epoch(self, probe_steps: int = 3) -> float | None:
        """Measure the cost of a training step, without disturbing the run.

        A 240k-iteration budget is a multi-day job on some of the hardware this runs on,
        so the startup banner should say so rather than let the user discover it. The
        measurement is a real forward/backward/step on real batches -- including the
        augmentation cost, since that is paid in ``__getitem__`` -- but it is performed on
        a **deep copy** of the model with a throwaway optimizer, and the global RNG state
        is snapshotted and restored around it. Nothing about the actual trajectory
        changes.

        Batches are assembled directly from the dataset rather than drawn from the train
        DataLoader, because iterating that loader would advance its generator and change
        the batch order the run replays.

        Args:
            probe_steps: Steps to time. The first is discarded as warm-up (lazy kernel
                compilation on MPS and cuDNN autotuning both land there).

        Returns:
            Estimated seconds per epoch, or None if the probe could not run -- it is a
            convenience, so any failure is logged and swallowed rather than killing a run
            that was about to start.
        """
        dataset = self.data.datasets["train"]
        batch_size = self.config.data.batch_size
        if len(dataset) < batch_size or probe_steps < 2:
            return None

        snapshot = rng_state()
        try:
            model = copy.deepcopy(self.model)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.optim.lr)
            durations: list[float] = []
            for step in range(probe_steps):
                start = step * batch_size % max(len(dataset) - batch_size, 1)
                samples = [dataset[start + offset] for offset in range(batch_size)]
                image = torch.stack([s["image"] for s in samples]).to(self.device)
                mask = torch.stack([s["mask"] for s in samples]).to(self.device)

                began = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                output = model(image, mask)
                terms = elbo_loss(
                    output.logits,
                    mask,
                    output.posterior,
                    output.prior,
                    beta=self.config.loss.beta,
                )
                terms["total"].backward()
                optimizer.step()
                self._synchronize()
                durations.append(time.perf_counter() - began)
            # Drop the warm-up step.
            per_step = sum(durations[1:]) / len(durations[1:])
            return per_step * self.steps_per_epoch
        except Exception as error:  # noqa: BLE001 - a timing estimate must never be fatal
            LOGGER.warning("could not estimate runtime (%s); continuing without it", error)
            return None
        finally:
            # The probe consumed RNG draws (latent sampling, and augmentation if it were
            # global). Restoring makes the run's sequence identical to a run that never
            # probed, which the reproducibility tests depend on.
            set_rng_state(snapshot)
            del dataset
            self.data.datasets["train"].aug_stats.reset()

    def _synchronize(self) -> None:
        """Wait for queued accelerator work, so a timing measurement is not a lie."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Render a duration as a human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            A compact string such as ``"1 d 8.4 h"`` or ``"12.3 min"``.
        """
        if seconds < 90:
            return f"{seconds:.1f} s"
        if seconds < 5400:
            return f"{seconds / 60:.1f} min"
        # Switch to days at 24 h, not 48 h: the paper's budget lands around 33 h on some
        # of this project's hardware, and "1 d 9.0 h" is the phrasing that makes a
        # multi-day commitment obvious where "33.0 h" invites a shrug.
        if seconds < 86400:
            return f"{seconds / 3600:.1f} h"
        days, remainder = divmod(seconds, 86400)
        return f"{int(days)} d {remainder / 3600:.1f} h"

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

    def describe(self, estimate_runtime: bool = False) -> str:
        """Return a startup banner: config, device, seed and provenance.

        Args:
            estimate_runtime: Time a few probe steps and include a projected wall-clock
                total. Off by default so tests and short runs pay nothing for it.

        Returns:
            The banner text.
        """
        parameters = sum(p.numel() for p in self.model.parameters())
        train_config = self.config.train
        budget = (
            f"{train_config.iterations} iterations (paper: 240000)"
            if train_config.iterations is not None
            else f"{train_config.epochs} epochs"
        )
        augmentation = self.config.data.augmentation
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
            f"augmentation  : "
            + (
                f"on  (pad {augmentation.pad_to_px}px, rot +-{augmentation.rotation_degrees} deg, "
                f"scale {augmentation.scale_range}, shear +-{augmentation.shear}, "
                f"elastic <={augmentation.elastic_alpha_px}px)"
                if self.data.datasets["train"].augmenting
                else "OFF - not the paper's configuration; label this run as an ablation"
            ),
            f"budget        : {budget}",
            f"steps/epoch   : {self.steps_per_epoch}",
            f"planned epochs: {self.planned_epochs}",
            f"total steps   : {self.total_steps}",
            f"beta          : {self.config.loss.beta}",
            f"lr            : {self.config.optim.lr} ({self.config.schedule.name})",
            f"validation    : every {train_config.val_every_n_epochs} epoch(s)",
            f"diagnostics   : every {self.config.log.diagnostics_every_n_epochs} epoch(s)",
            f"monitor       : {self.config.checkpoint.monitor} ({self.config.checkpoint.mode})",
            f"run dir       : {self.run_dir}",
        ]

        if estimate_runtime:
            seconds_per_epoch = self.estimate_seconds_per_epoch()
            if seconds_per_epoch is not None:
                remaining = max(self.planned_epochs - self.epoch, 0)
                training = seconds_per_epoch * remaining
                lines += [
                    f"measured      : {seconds_per_epoch:.1f} s/epoch "
                    f"({seconds_per_epoch / self.steps_per_epoch * 1000:.0f} ms/step)",
                    f"ESTIMATED RUN : {self.format_duration(training)} for {remaining} epoch(s), "
                    "excluding validation and diagnostics",
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
        # Augmentation counters for the epoch just finished, then reset. Empty when the
        # run is not augmenting, so an ablation logs nothing rather than a hollow zero.
        metrics.update(self.data.augmentation_metrics())
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
        prior = self.model.prior_net.distribution_from_stats(prior_stats)
        posterior = self.model.posterior_net.distribution_from_stats(posterior_stats)

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
        LOGGER.info("starting training\n%s", self.describe(estimate_runtime=True))
        (self.run_dir / "config.resolved.yaml").write_text(self.config.to_yaml())

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.planned_epochs):
            train_metrics = self.train_epoch(epoch)
            record = {f"train/{k}": v for k, v in train_metrics.items()}

            if (epoch + 1) % self.config.train.val_every_n_epochs == 0:
                val_metrics = self.validate()
                record.update({f"val/{k}": v for k, v in val_metrics.items()})
                # The overfitting gap as a first-class number: 27.5M parameters on
                # 9,056 patches is a regime where the gap is the thing to watch, and the
                # report needs its magnitude quantified rather than eyeballed off two
                # curves. With augmentation on, this is also how we see what the
                # augmentation bought.
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
            "planned_epochs": self.planned_epochs,
            "iterations_requested": self.config.train.iterations,
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
        # A live ETA from the epochs actually observed. On a multi-day run this is the
        # number that matters more than the startup estimate, because it includes
        # validation and diagnostics as they really fall.
        # _log_epoch runs before the record is appended to history, so include it here to
        # get an ETA from the very first epoch rather than the second.
        eta = ""
        elapsed = [entry.get("train/seconds", 0.0) for entry in self.history[-9:]]
        elapsed.append(record.get("train/seconds", 0.0))
        if any(elapsed) and self.planned_epochs > epoch + 1:
            mean_seconds = sum(elapsed) / len(elapsed)
            eta = f" eta={self.format_duration(mean_seconds * (self.planned_epochs - epoch - 1))}"
        LOGGER.info("epoch %d/%d %s%s", epoch + 1, self.planned_epochs, summary, eta)

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
            # The record for this epoch is appended to self.history after _log_epoch but
            # before _checkpoint, so history already includes it.
            "history": self.history,
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

        Everything a continuation depends on is restored: weights, optimizer moments, the
        scheduler's position, the epoch and step counters, the best-so-far metric, the
        per-epoch history, the DataLoader's shuffle generator and the RNG state of every
        backend. Because a checkpoint is written at the end of every epoch, an
        interruption costs at most one epoch of work.

        The augmentation needs nothing stored: its draws are derived from
        ``(seed, epoch, position)``, and the epoch is restored here, so a resumed run
        reproduces the same transforms it would have applied uninterrupted.

        Args:
            path: Checkpoint file, normally ``last.pt``.

        Warns:
            If the checkpoint was written on a different device, or under a different
            planned budget. Both silently change what a "continued" run means: seeds do
            not reproduce across backends, and the learning-rate milestones are fractions
            of the total step count.
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
        self.history = list(state.history)
        from probunet.training.checkpoint import loader_generator_state

        saved = loader_generator_state(path)
        generator = self.data.loaders["train"].generator
        if saved is not None and generator is not None:
            generator.set_state(saved)

        if state.device != str(self.device):
            LOGGER.warning(
                "checkpoint was written on %s but this run is on %s: the sampling "
                "sequence will not match, and numbers from before and after this resume "
                "are not strictly comparable",
                state.device,
                self.device,
            )
        saved_train = state.config.get("train", {})
        if saved_train.get("iterations") != self.config.train.iterations or saved_train.get(
            "epochs"
        ) != self.config.train.epochs:
            LOGGER.warning(
                "budget changed across the resume (checkpoint: iterations=%s epochs=%s; "
                "now: iterations=%s epochs=%s). Learning-rate milestones are fractions of "
                "the total step count, so the schedule shape has shifted.",
                saved_train.get("iterations"),
                saved_train.get("epochs"),
                self.config.train.iterations,
                self.config.train.epochs,
            )
        LOGGER.info(
            "resumed from %s at epoch %d/%d (step %d/%d, best %s=%s, %d history entries, "
            "git %s)",
            path,
            state.epoch,
            self.planned_epochs,
            state.global_step,
            self.total_steps,
            state.monitor,
            state.best_metric,
            len(self.history),
            state.git_revision,
        )
