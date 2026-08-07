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
from probunet.evaluation.metrics import (
    consensus_ceiling,
    consensus_scores,
    consensus_selected,
    spearman_per_image,
)
from probunet.evaluation.sampling import DEFAULT_EVAL_SEED
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
from probunet.extension.head import SelectionHead
from probunet.training.freeze import (
    assert_unchanged,
    freeze_module,
    parameter_fingerprint,
)
from probunet.training.diagnostics import (
    EffectiveRankAccumulator,
    build_diagnostic_sets,
    logits_to_mask,
    make_panel,
    mean_pairwise_iou,
    nonempty_sample_fraction,
    per_dim_kl,
    prior_samples_for_images,
    prior_spectrum,
    reparameterize,
    save_diagnostic_sets,
    sigma_stats,
    whitened_kl_decomposition,
    whitened_kl_snapshot,
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
        self.head: SelectionHead | None = None
        self.head_train_loader: torch.utils.data.DataLoader | None = None
        self.base_fingerprint: str | None = None
        self.base_provenance: dict[str, Any] | None = None
        if config.train.mode == "selection_head":
            self._prepare_selection_head()
        # THE line that decides whether the base stays frozen. In selection_head mode the
        # optimizer is given the scorer's parameters ONLY; handing it self.model
        # .parameters() would include the frozen base and let it drift.
        trainable = (
            self.head.head_parameters() if self.head is not None else self.model.parameters()
        )
        self.optimizer = torch.optim.Adam(
            trainable,
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
    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        """The loader this run trains on.

        In ``selection_head`` mode that is the eval-mode four-mask loader over the train
        split, not ``data.loaders["train"]`` -- the head needs every grader to form the
        consensus, and the single-grader pairing an ELBO run uses would give it a target
        computed from one quarter of the evidence.
        """
        return (
            self.head_train_loader if self.head is not None else self.data.loaders["train"]
        )

    def _steps_per_epoch(self) -> int:
        """Training batches per epoch, respecting ``limit_train_batches``."""
        batches = len(self.train_loader)
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
        if self.head is not None:
            # The probe below is an ELBO forward/backward, which is not this run's step at
            # all. Skipping is honest; letting it raise into the catch-all would log
            # "could not estimate runtime" and imply something went wrong.
            LOGGER.info("no runtime estimate in selection_head mode: the probe is an ELBO step")
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
        """Load the base checkpoint, freeze it, and wrap it in the selection head.

        **Stage 3 scope.** The head's structure and the freeze contract exist; its scoring
        target, regression objective and candidate-sampling loop are Stage 4. Training in
        this mode today runs a deliberately meaningless placeholder objective
        (:meth:`_selection_head_step`) whose only purpose is to make the freeze machinery
        run against a real optimizer step.

        After this returns, ``self.model`` is the :class:`SelectionHead`, and
        ``self.head`` is the same object under a name that says what it is. The optimizer
        is built over :meth:`SelectionHead.head_parameters` -- the scorer alone.

        Raises:
            ValueError: If no base checkpoint was supplied.
        """
        if self.base_checkpoint is None:
            raise ValueError(
                "train.mode='selection_head' requires --base-checkpoint: the head is "
                "trained on top of an already-trained, frozen Probabilistic U-Net."
            )
        base_state = load_checkpoint(
            self.base_checkpoint,
            model=self.model,
            map_location=self.device,
            restore_rng=False,
        )
        # SelectionHead freezes the base in its own constructor, so a head cannot be
        # built around an unfrozen base even by mistake.
        self.head = SelectionHead(self.model).to(self.device)
        self.model = self.head
        self.base_record = dict(self.head.freeze_record)
        # The fingerprint the whole stage exists to defend: taken once, here, before any
        # optimizer exists, and re-checked after the last step of every epoch.
        self.base_fingerprint = parameter_fingerprint(self.head.base)
        self.base_record["parameter_fingerprint"] = self.base_fingerprint
        self.base_record["scorer_parameters"] = self.head.parameter_counts()["scorer"]

        # WHICH base, not merely that a freeze happened. head-on-Phase1 versus
        # head-on-Phase2 is only a meaningful ablation if each head checkpoint is
        # attributable to its base from the artifact alone, and a filename is not a
        # guarantee. The parameter hash makes the identity verifiable rather than
        # asserted, and it is the same hash the epoch-boundary check compares against.
        self.base_provenance = {
            "checkpoint": str(self.base_checkpoint),
            "epoch": base_state.epoch,
            "git_revision": base_state.git_revision,
            "device": base_state.device,
            "torch_version": base_state.torch_version,
            "seed": base_state.seed,
            "latent_covariance": self.config.model.latent_covariance,
            "parameter_sha256": self.base_fingerprint,
            "frozen_parameters": self.head.freeze_record["frozen_parameters"],
        }
        LOGGER.info(
            "base frozen and wrapped: %s trainable scorer parameters; base = %s "
            "(epoch %s, git %s, %s, torch %s), sha256 %s",
            f"{self.base_record['scorer_parameters']:,}",
            self.base_checkpoint,
            base_state.epoch,
            base_state.git_revision,
            base_state.device,
            base_state.torch_version,
            self.base_fingerprint[:12],
        )
        # The head needs ALL FOUR grader masks per image to form the consensus, which is
        # the eval-mode dataset shape. With augmentation off (DEVIATIONS 13) that shape is
        # usable directly for training, so there is no third dataset mode: the same
        # LidcDataset class, mode="eval", over the train split's indices.
        head_dataset = LidcDataset(
            self.data.arrays, self.data.datasets["train"].indices, mode="eval"
        )
        self.head_train_loader = torch.utils.data.DataLoader(
            head_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.data.pairing_seed),
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=self.config.data.drop_last,
            persistent_workers=False,
        )
        if self.data.datasets["train"].augmenting:
            raise ValueError(
                "selection_head requires data.augmentation.enabled=false. The head trains "
                "on the eval-mode four-mask path, and an augmented eval dataset is refused "
                "by construction; see DEVIATIONS.md entry 13 for why augmentation is off "
                "for this phase rather than extended to carry four masks."
            )

    @property
    def base_model(self) -> ProbUNet:
        """The Probabilistic U-Net, whichever mode this run is in.

        In ``selection_head`` mode ``self.model`` is the :class:`SelectionHead` wrapper, so
        anything that needs the generative model itself -- the channel schedule, the
        latent diagnostics, sampling -- has to reach through it rather than assume
        ``self.model`` is a ``ProbUNet``.
        """
        return self.head.base if self.head is not None else self.model

    @property
    def is_selection_head(self) -> bool:
        """Whether this run trains the selection head rather than the ELBO."""
        return self.config.train.mode == "selection_head"

    def _elbo_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """One ELBO training step: z from the posterior, CE + beta*KL.

        Extracted unchanged from the training loop so ``train_epoch`` can dispatch between
        objectives without an inline branch. Phases 1 and 2 take this path.

        Args:
            batch: A training batch carrying ``image`` and one paired grader ``mask``.

        Returns:
            The ELBO terms; ``total`` is what is minimized.
        """
        image = batch["image"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        output = self.model(image, mask)
        return elbo_loss(
            output.logits, mask, output.posterior, output.prior, beta=self.config.loss.beta
        )

    def _consensus_targets(self, graders: Tensor, candidates: Tensor) -> Tensor:
        """Soft-consensus Dice of each candidate: the head's regression target.

        Computed under ``no_grad`` and returned detached -- it is **data**, not a
        differentiable objective. Gradients must flow only through the head's prediction.

        Args:
            graders: All four grader masks, shape ``(B, 4, H, W)``.
            candidates: Binary candidates, shape ``(B, n, H, W)``.

        Returns:
            Targets of shape ``(B, n)``.
        """
        with torch.no_grad():
            targets = consensus_scores(candidates, graders)
            if self.config.head.mean_centered_targets:
                # The pre-registered fallback (FINDINGS 4.4): remove the between-image
                # component so the head cannot score well by predicting each image's
                # typical value while ignoring the candidate.
                targets = targets - targets.mean(dim=1, keepdim=True)
        return targets.detach()

    def _selection_head_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """One head training step: draw candidates, score them, regress onto the target.

        Draws ``head.train_samples`` **prior** candidates per image, scores each against
        the soft consensus of the four graders, and fits the head's prediction to that
        score under a Huber loss.

        **Prior candidates, never posterior.** Posterior samples have seen the ground-truth
        mask and are almost always good, so a head trained on them would never meet a bad
        candidate and would learn a constant high score.

        **Candidates are redrawn every step**, which is where the head's data variety comes
        from -- and the reason augmentation is off for this phase (DEVIATIONS 13): the head
        sees fresh candidates on every pass, so its effective dataset is already far larger
        than the patch count.

        Args:
            batch: An eval-mode batch carrying ``image`` and all four ``masks``.

        Returns:
            ``total`` (the Huber loss), plus the mean predicted and target scores for
            monitoring drift between them.
        """
        image = batch["image"].to(self.device, non_blocking=True)
        graders = batch["masks"].to(self.device, non_blocking=True)

        # sample_candidates is @torch.no_grad on a frozen base: no gradient path exists
        # back into the generative model.
        features, candidates = self.head.sample_candidates(
            image, self.config.head.train_samples
        )
        targets = self._consensus_targets(graders, candidates)
        predicted = self.head.score_candidates(features, candidates)

        delta = self.config.head.huber_delta
        loss = torch.nn.functional.huber_loss(predicted, targets, delta=delta)
        # The size-prior control, fitted on the same candidates and the same targets under
        # the same loss. Its parameters are disjoint from the scorer's, so adding the two
        # objectives optimizes each independently -- neither can borrow the other's
        # gradient, which is what keeps the control honest.
        area_loss = torch.nn.functional.huber_loss(
            self.head.score_by_area(candidates), targets, delta=delta
        )
        return {
            "total": loss + area_loss,
            "head_huber": loss.detach(),
            "area_huber": area_loss.detach(),
            "target_mean": targets.mean().detach(),
            "predicted_mean": predicted.mean().detach(),
        }

    @torch.no_grad()
    def _validate_selection_head(self) -> dict[str, float]:
        """Validate the head on what it is actually for: the sample it selects.

        **One shared candidate set per image.** The candidates are drawn once, at a fixed
        seed re-set at the start of every validation pass, and the head-selected, random
        and oracle scores are all computed from that same set. Scoring them from
        independent draws would confound the head's contribution with sampling noise, and
        re-seeding each pass makes the candidate set identical across epochs and across
        arms, so an epoch-to-epoch change is the head changing rather than the draw.

        Returns:
            The monitored ``selected_consensus_dice`` plus the baselines it must beat, the
            regression loss, and the rank correlation with its exclusion bookkeeping.
        """
        self.model.eval()
        config = self.config.head
        limit = self.config.train.limit_val_batches
        generator = torch.Generator().manual_seed(DEFAULT_EVAL_SEED)

        totals: dict[str, float] = {}
        images = 0
        rho_sum = 0.0
        rho_valid = 0
        rho_by_bucket: dict[int, list[int]] = {}

        for index, batch in enumerate(self.data.loaders["val"]):
            if limit and index >= limit:
                break
            image = batch["image"].to(self.device, non_blocking=True)
            graders = batch["masks"].to(self.device, non_blocking=True)

            features, candidates = self.head.sample_candidates(
                image, config.eval_samples, generator
            )
            scores = consensus_scores(candidates, graders)
            predicted = self.head.score_candidates(features, candidates)
            chosen = predicted.argmax(dim=1)
            by_area = self.head.select_by_area(candidates)

            batch_totals = {
                # THE deliverable, and the monitored metric.
                "selected_consensus_dice": consensus_selected(candidates, graders, chosen),
                # The size-prior control, reported ADJACENT so the comparison cannot be
                # overlooked: if the head barely beats this, it learned area, not overlap.
                "area_only_consensus_dice": consensus_selected(candidates, graders, by_area),
                "random_consensus_dice": scores.mean(dim=1),
                "oracle_consensus_dice": scores.amax(dim=1),
                "ceiling": consensus_ceiling(graders),
                "huber": torch.nn.functional.huber_loss(
                    predicted,
                    self._consensus_targets(graders, candidates),
                    delta=config.huber_delta,
                    reduction="none",
                ).mean(dim=1),
            }
            for key, value in batch_totals.items():
                totals[key] = totals.get(key, 0.0) + float(value.sum())
            images += image.shape[0]

            rho, valid = spearman_per_image(predicted, scores)
            rho_sum += float(rho[valid].sum()) if bool(valid.any()) else 0.0
            rho_valid += int(valid.sum())
            # DEVICE: spearman_per_image returns CPU tensors because float64 does not
            # exist on MPS (see the comment in metrics.spearman_per_image). Indexing a CPU
            # mask with an accelerator-resident selector raises, so the bucket selector
            # must follow the ranks to the CPU. DO NOT drop this .cpu() to "keep things on
            # device" -- it only fails on MPS, and only at validation.
            buckets = (graders.flatten(start_dim=2).sum(dim=2) > 0).sum(dim=1).cpu()
            for bucket in range(1, N_GRADERS + 1):
                selector = buckets == bucket
                if not bool(selector.any()):
                    continue
                counts = rho_by_bucket.setdefault(bucket, [0, 0])
                counts[0] += int(selector.sum())
                counts[1] += int(valid[selector].sum())

        metrics = {key: value / max(images, 1) for key, value in totals.items()}
        # Report the deliverable against what it could have reached, never against 1.0:
        # soft-consensus scores are bounded well below 1 by construction.
        metrics["selected_fraction_of_ceiling"] = metrics["selected_consensus_dice"] / max(
            metrics["ceiling"], 1e-12
        )
        # THE number to lead with: scale-free across buckets whose ceilings run 0.40 to
        # 0.89, so it is the only one comparable between them. Computed from the MEANS
        # rather than per image, because the per-image denominator (oracle - random) is
        # zero whenever every candidate scores alike, which on bucket 1 is common.
        gap = max(
            metrics["oracle_consensus_dice"] - metrics["random_consensus_dice"], 1e-12
        )
        metrics["headroom_captured"] = (
            metrics["selected_consensus_dice"] - metrics["random_consensus_dice"]
        ) / gap
        metrics["headroom_captured_area_only"] = (
            metrics["area_only_consensus_dice"] - metrics["random_consensus_dice"]
        ) / gap

        # Spearman over the images where it is DEFINED, with the exclusions counted rather
        # than silently dropped. The excluded fraction is itself a finding: it measures how
        # often the sampler offered no real choice, and on bucket 1 -- where empty
        # candidates all score exactly 0.000 -- it is expected to be substantial.
        metrics["spearman"] = rho_sum / max(rho_valid, 1)
        metrics["spearman_images"] = float(rho_valid)
        metrics["spearman_excluded_fraction"] = 1.0 - rho_valid / max(images, 1)
        for bucket, (seen, valid_count) in sorted(rho_by_bucket.items()):
            metrics[f"spearman_excluded_fraction_bucket{bucket}"] = 1.0 - valid_count / max(
                seen, 1
            )
        return metrics

    @property
    def _latent_cadences_align(self) -> bool:
        """Whether every diagnostics epoch is also a validation epoch.

        The full-validation latent geometry (``effrank_val_*``) is computed *inside*
        ``validate``, because that is where the four posteriors per image already exist.
        So it can only be emitted on an epoch where both schedules fire. When the
        diagnostics cadence is not a multiple of the validation cadence the two schedules
        intersect rarely -- or, if they are coprime, only at their least common multiple --
        and the headline Phase 2 series would come out sparse or empty for a reason
        invisible in the logs. Every shipped config satisfies this (validation every
        epoch), so the check exists to catch an override rather than a shipped mistake.
        """
        return (
            self.config.log.diagnostics_every_n_epochs
            % self.config.train.val_every_n_epochs
            == 0
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
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
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
            f"parameters    : {parameters:,}"
            + (f"  ({trainable:,} trainable)" if trainable != parameters else ""),
            f"channels      : {self.base_model.unet.widths}"
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
            f"diagnostics   : every {self.config.log.diagnostics_every_n_epochs} epoch(s)"
            + ("" if self._latent_cadences_align else "  [see WARNING above]"),
            f"monitor       : {self.config.checkpoint.monitor} ({self.config.checkpoint.mode})",
            f"run dir       : {self.run_dir}",
        ]
        if self.head is not None:
            # Describe the objective ACTUALLY IN USE. This banner is written into
            # config.resolved.yaml and the run log, both of which the report cites, so a
            # stale line here becomes a false claim in the write-up. It was stale once
            # already -- and, worse, it was *correct* at the time, because a duplicate
            # method definition had silently shadowed the real objective.
            counts = self.head.parameter_counts()
            head_config = self.config.head
            lines[2:2] = [
                f"head objective: Huber(delta={head_config.huber_delta}) regression onto "
                "soft-consensus Dice",
                f"head candidates: {head_config.train_samples} train / "
                f"{head_config.eval_samples} eval (prior), eval seed {head_config.eval_seed}"
                + ("  [MEAN-CENTERED targets]" if head_config.mean_centered_targets else ""),
                f"head parameters: {counts['scorer']:,} scorer + "
                f"{counts['area_baseline']:,} area control "
                f"= {counts['scorer'] + counts['area_baseline']:,} trainable",
                f"frozen base    : {counts['base']:,} parameters",
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
        # In selection_head mode this is SelectionHead.train(), which pins the base to
        # eval() no matter what mode is requested. That override is load-bearing: the
        # ordinary recursive train() would otherwise thaw the frozen base's mode here.
        self.model.train()
        if self.head is not None:
            # The freeze must hold at the START of the epoch too, so a failure is
            # attributed to the right epoch rather than to the next one.
            self.head.assert_base_frozen()
        if self.head is None:
            # Fresh grader pairing for this epoch, reproducible from (pairing_seed, epoch).
            # The head has no pairing: it sees all four masks every epoch, and its
            # epoch-to-epoch variety comes from freshly drawn candidates instead.
            self.data.set_epoch(epoch)
        loader = self.train_loader
        limit = self.config.train.limit_train_batches

        totals: dict[str, float] = {}
        count = 0
        started = time.perf_counter()

        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.config.train.amp):
                terms = (
                    self._selection_head_step(batch)
                    if self.head is not None
                    else self._elbo_step(batch)
                )
            self.scaler.scale(terms["total"]).backward()
            if self.config.train.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                # Clip only what is being optimized. Passing self.model.parameters() here
                # would reach into the frozen base -- harmless today because its grads are
                # None, but it would silently start clipping the base the moment anything
                # gave it gradients.
                torch.nn.utils.clip_grad_norm_(
                    self.head.head_parameters() if self.head is not None
                    else self.model.parameters(),
                    self.config.train.grad_clip,
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

        if self.head is not None:
            # AFTER the last optimizer step, not only before the first. An optimizer built
            # over the wrong parameter set does its damage DURING the epoch, so a check
            # that ran only at construction would pass on a thoroughly broken run.
            self.head.assert_base_frozen()
            assert self.base_fingerprint is not None
            assert_unchanged(
                self.head.base,
                self.base_fingerprint,
                name="base Probabilistic U-Net",
                context=f"epoch {epoch + 1}",
            )

        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        metrics["seconds"] = time.perf_counter() - started
        # Augmentation counters for the epoch just finished, then reset. Empty when the
        # run is not augmenting, so an ablation logs nothing rather than a hollow zero.
        metrics.update(self.data.augmentation_metrics())
        return metrics

    @torch.no_grad()
    def validate(self, full_latent: bool = False) -> dict[str, float]:
        """Evaluate the objective over **all four graders** per image.

        Averaging over graders removes the dependence on the epoch's random pairing, so
        the number that selects checkpoints is comparable from epoch to epoch. One U-Net
        pass and one prior pass are shared across the four posteriors.

        Args:
            full_latent: Also accumulate the per-image effective rank over the **whole**
                validation set at all four graders, emitting the ``effrank_val_*`` family.
                Off by default and driven by the diagnostics cadence: the per-batch cost
                is one batched ``6 x 6`` SVD on distributions this pass already builds,
                but it moves small tensors to the CPU on every posterior, and that has no
                business on the per-epoch path of a multi-day run. The ``*_snapshot``
                family is emitted either way.

        Returns:
            Mean validation metrics.
        """
        if self.head is not None:
            # With a frozen base the ELBO is a constant, so computing it would burn a full
            # validation pass to log a flat line, and monitoring it would mean selecting
            # checkpoints on noise. The head has its own metrics.
            return self._validate_selection_head()

        self.model.eval()
        loader = self.data.loaders["val"]
        limit = self.config.train.limit_val_batches

        totals: dict[str, float] = {}
        count = 0
        latent: dict[str, float] = {}
        accumulator = EffectiveRankAccumulator() if full_latent else None
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
                if accumulator is not None:
                    accumulator.update(posterior, encoded.prior, grader=grader)

            if index == 0:
                # Latent statistics from the first batch: cheap, and enough to spot a
                # collapsing sigma or a dead latent dimension.
                latent = self._latent_stats(image, masks)

        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        metrics.update(latent)
        if accumulator is not None:
            metrics.update(accumulator.metrics())
        return metrics

    @torch.no_grad()
    def _latent_stats(self, image: Tensor, masks: Tensor) -> dict[str, float]:
        """Sigma statistics and the latent-geometry snapshot, from one batch.

        Grader 0 of the first validation batch only -- 32 images. Cheap enough to run
        every epoch, and enough to spot a collapsing sigma or a dead latent dimension,
        but **not** enough to support a claim about a small shift in effective rank; that
        is what ``validate(full_latent=True)`` is for. The ``_snapshot`` suffix on the
        keys carries the distinction into the logs and any table built from them.

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
        # Coordinate-indexed, for continuity with the Phase 1 series (FINDINGS 2.3).
        for dimension, value in enumerate(per_dim_kl(posterior, prior)):
            stats[f"kl_dim_{dimension}"] = float(value)
        # Rotation-invariant, and an exact decomposition of kl_snapshot_total.
        stats.update(whitened_kl_snapshot(whitened_kl_decomposition(posterior, prior)))
        # The frame the whitened numbers are measured in: lambda ~ 1 means "matches the
        # prior", which is only "isotropic" if the prior itself is.
        stats.update(prior_spectrum(prior))
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
        if not self._latent_cadences_align:
            LOGGER.warning(
                "diagnostics_every_n_epochs=%d is not a multiple of val_every_n_epochs=%d, "
                "so the full-validation latent geometry (effrank_val_*) will only be "
                "recorded on epochs where both schedules fire. That is the headline Phase 2 "
                "series -- make the cadences align unless this is deliberate.",
                self.config.log.diagnostics_every_n_epochs,
                self.config.train.val_every_n_epochs,
            )
        LOGGER.info("starting training\n%s", self.describe(estimate_runtime=True))
        (self.run_dir / "config.resolved.yaml").write_text(self.config.to_yaml())

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.planned_epochs):
            train_metrics = self.train_epoch(epoch)
            record = {f"train/{k}": v for k, v in train_metrics.items()}

            diagnostics_due = (
                epoch + 1
            ) % self.config.log.diagnostics_every_n_epochs == 0

            if (epoch + 1) % self.config.train.val_every_n_epochs == 0:
                # The full-validation latent geometry rides along with the validation
                # pass it reuses, on the diagnostics schedule.
                val_metrics = self.validate(full_latent=diagnostics_due)
                record.update({f"val/{k}": v for k, v in val_metrics.items()})
                # The overfitting gap as a first-class number: 27.5M parameters on
                # 9,056 patches is a regime where the gap is the thing to watch, and the
                # report needs its magnitude quantified rather than eyeballed off two
                # curves. With augmentation on, this is also how we see what the
                # augmentation bought.
                # Absent in selection_head mode, where validate() returns no ELBO terms.
                if "val/total" in record:
                    record["diag/gap_total"] = record["val/total"] - record["train/total"]
                    record["diag/gap_ce"] = record["val/ce"] - record["train/ce"]

            # The latent diagnostics describe the BASE, which is frozen here -- they
            # would repeat the same numbers every epoch. They belong to the run that
            # produced the base checkpoint, not to the head's run.
            if diagnostics_due and self.head is None:
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

    def _freeze_metrics(self) -> dict[str, float]:
        """The freeze record as numeric metrics, for storage in a checkpoint.

        Returns:
            ``freeze/``-prefixed counts, or empty in ELBO mode where nothing is frozen.
            The parameter fingerprint is a hex digest rather than a number, so it is not
            included here; it is re-derivable from the stored weights.
        """
        if self.base_record is None:
            return {}
        return {
            f"freeze/{key}": float(value)
            for key, value in self.base_record.items()
            if isinstance(value, (int, float, bool))
        }

    def _checkpoint(self, epoch: int, record: dict[str, float]) -> None:
        """Save last, and best when the monitored metric improves.

        Args:
            epoch: Epoch index.
            record: Flat metric mapping for this epoch.
        """
        policy = self.config.checkpoint
        generator = self.train_loader.generator
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
            # The freeze record travels WITH the artifact, not just in the log. "The base
            # was frozen" is a claim the report makes about this checkpoint, so it has to
            # be checkable from the checkpoint rather than from a log file that may be
            # long gone. Numeric-only, matching the metrics dict's contract.
            "metrics": {**{k: v for k, v in record.items()}, **self._freeze_metrics()},
            "loader_generator_state": generator.get_state() if generator is not None else None,
            # Non-numeric, so it cannot ride in `metrics`; it gets its own field.
            "base_provenance": self.base_provenance,
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
            # Loud, not silent, and deliberately no best.pt. In Stage 3 the configured
            # monitor (val/selected_consensus_dice) does not exist yet, and the objective
            # is a placeholder -- so any best.pt written here would be a scaffold
            # checkpoint that a later stage could mistake for a real one. Refusing to
            # write it is the safe failure; writing one selected on some other metric,
            # flat or otherwise, is not.
            LOGGER.warning(
                "no best.pt at epoch %d: monitored metric %r is absent from this epoch's "
                "record. %s",
                epoch + 1,
                policy.monitor,
                "selection_head is a Stage 3 scaffold with a placeholder objective; the "
                "monitor lands with the head's real metrics in Stage 4."
                if self.head is not None
                else "Check that the metric name matches what validate() emits.",
            )
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
        generator = self.train_loader.generator
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
