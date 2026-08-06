"""Experiment configuration: nested dataclasses loaded from YAML.

Every knob a run depends on lives here, and the resolved configuration is written next
to the checkpoints so a result can always be traced back to the settings that produced
it. Unknown keys are **rejected** rather than ignored: a typo in a YAML key would
otherwise silently leave the default in place, which is the worst kind of
configuration bug.

There is deliberately no way to express a modernization or extension setting. Those
phases add their own flags when they arrive.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from probunet.data.lidc import DataConfig
from probunet.data.transforms import AugmentationConfig
from probunet.losses.elbo import ElboConfig
from probunet.model.prob_unet import ProbUNetConfig

SCHEDULE_NAMES = ("constant", "piecewise")
OPTIMIZER_NAMES = ("adam",)
TRAIN_MODES = ("elbo", "selection_head")


@dataclass(frozen=True)
class RunConfig:
    """Identity and environment of a run.

    Attributes:
        name: Run name; also the sub-directory under ``out_dir``.
        seed: Global seed. Also the default for the data pairing seed, so one knob
            controls run randomness while the split seed stays frozen.
        device: ``"auto"`` (cuda -> mps -> cpu) or an explicit device.
        out_dir: Root directory for run artifacts.
        deterministic: Request deterministic cuDNN kernels.
    """

    name: str = "baseline"
    seed: int = 2018
    device: str = "auto"
    out_dir: Path = Path("runs")
    deterministic: bool = True

    @property
    def run_dir(self) -> Path:
        """Directory holding this run's artifacts."""
        return Path(self.out_dir) / self.name


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer settings.

    Plain **Adam**, not AdamW: the reference applies L2 as ``1e-5 * sum(w^2)/2`` over
    weights *and* biases, which ``torch.optim.Adam(weight_decay=1e-5)`` reproduces
    exactly. AdamW's decoupled decay is a different objective.

    Attributes:
        name: Optimizer name; only ``"adam"`` in the baseline phase.
        lr: Learning rate.
        weight_decay: L2 coefficient.
        betas: Adam's exponential decay rates.
        eps: Adam's numerical floor.

    Raises:
        NotImplementedError: If a non-Adam optimizer is requested.
        ValueError: If a value is out of range.
    """

    name: str = "adam"
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate the optimizer settings."""
        if self.name not in OPTIMIZER_NAMES:
            raise NotImplementedError(
                f"optimizer {self.name!r} is not supported in the baseline phase; "
                f"expected one of {OPTIMIZER_NAMES}. Note that AdamW is NOT "
                "equivalent to the reference's L2 regularizer."
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


@dataclass(frozen=True)
class ScheduleConfig:
    """Learning-rate schedule.

    Milestones are **fractions of the total training steps**, not absolute step
    counts. The paper's LIDC schedule decays 1e-4 to 1e-6 in five steps over 240,000
    iterations; transplanting those absolute boundaries into a run that is ~12% as
    long would place the decay outside its design regime, so the fractions scale with
    whatever budget is configured. See DEVIATIONS.md.

    Attributes:
        name: ``"constant"`` or ``"piecewise"``.
        milestones: Strictly increasing fractions in (0, 1).
        values: Learning rates, one more than there are milestones. The first must
            equal ``OptimConfig.lr`` so there is a single source of truth.

    Raises:
        ValueError: If the schedule is malformed.
    """

    name: str = "constant"
    milestones: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the schedule."""
        if self.name not in SCHEDULE_NAMES:
            raise ValueError(f"schedule must be one of {SCHEDULE_NAMES}, got {self.name!r}")
        if self.name == "constant":
            if self.milestones or self.values:
                raise ValueError("a constant schedule takes no milestones or values")
            return
        if len(self.values) != len(self.milestones) + 1:
            raise ValueError(
                f"piecewise needs len(values) == len(milestones) + 1, got "
                f"{len(self.values)} values and {len(self.milestones)} milestones"
            )
        if not self.milestones:
            raise ValueError("piecewise needs at least one milestone")
        if any(not 0.0 < m < 1.0 for m in self.milestones):
            raise ValueError(f"milestones must be fractions in (0, 1), got {self.milestones}")
        if list(self.milestones) != sorted(set(self.milestones)):
            raise ValueError(f"milestones must be strictly increasing, got {self.milestones}")
        if any(v <= 0 for v in self.values):
            raise ValueError(f"values must be positive, got {self.values}")


@dataclass(frozen=True)
class TrainConfig:
    """Training loop settings.

    The budget is expressed **either** in iterations or in epochs, never both. The paper
    states its budget in iterations (240k at batch 32, Appendix H.1), and an iteration
    count is the quantity that is actually comparable across datasets and batch sizes --
    an epoch count silently means something different the moment the split size changes.
    :class:`Trainer` derives the epoch count from ``iterations`` and the train split size.

    Attributes:
        mode: ``"elbo"`` trains the Probabilistic U-Net itself. ``"selection_head"``
            trains the consensus-selection head on top of a **frozen** base model and
            requires ``--base-checkpoint``; the head itself is Phase 3 and not
            implemented yet.
        iterations: Optimizer steps to train for. The paper's LIDC budget is 240,000.
            Mutually exclusive with ``epochs``.
        epochs: Number of epochs, for runs where an iteration count is not the natural
            unit (the smoke config). Mutually exclusive with ``iterations``.
        amp: Mixed precision. **CUDA only** -- see :class:`Trainer`.
        grad_clip: Optional gradient-norm clip.
        val_every_n_epochs: Validation cadence.
        limit_train_batches: Cap on training batches per epoch, for smoke runs.
        limit_val_batches: Cap on validation batches, for smoke runs.

    Raises:
        ValueError: If a value is out of range, or if the budget is not given by exactly
            one of ``iterations`` and ``epochs``.
    """

    mode: str = "elbo"
    iterations: int | None = None
    epochs: int | None = 100
    amp: bool = False
    grad_clip: float | None = None
    val_every_n_epochs: int = 1
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None

    def __post_init__(self) -> None:
        """Validate the training settings."""
        if self.mode not in TRAIN_MODES:
            raise ValueError(f"train.mode must be one of {TRAIN_MODES}, got {self.mode!r}")
        if (self.iterations is None) == (self.epochs is None):
            raise ValueError(
                "set exactly one of train.iterations and train.epochs, got "
                f"iterations={self.iterations!r} and epochs={self.epochs!r}. Expressing "
                "the budget twice invites the two from drifting apart; a config using "
                "iterations must set 'epochs: null' explicitly."
            )
        if self.iterations is not None and self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.val_every_n_epochs <= 0:
            raise ValueError(
                f"val_every_n_epochs must be positive, got {self.val_every_n_epochs}"
            )
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive or null, got {self.grad_clip}")


@dataclass(frozen=True)
class LogConfig:
    """Logging cadences and diagnostic sizes.

    Attributes:
        tensorboard: Write TensorBoard scalars and images.
        log_every_n_steps: Step-level scalar cadence.
        diagnostics_every_n_epochs: Cadence for the expensive latent diagnostics.
        prior_samples_for_ce: Prior samples per image for the prior-vs-posterior
            reconstruction comparison.
        diversity_samples: Prior samples per image for the diversity measure.
        diversity_images: Size of the fixed, ambiguity-stratified diversity set.
        panel_images: Rows in the qualitative image panel.
        panel_samples: Prior samples shown per panel row.

    Raises:
        ValueError: If a value is out of range.
    """

    tensorboard: bool = True
    log_every_n_steps: int = 20
    diagnostics_every_n_epochs: int = 5
    prior_samples_for_ce: int = 4
    diversity_samples: int = 16
    diversity_images: int = 64
    panel_images: int = 8
    panel_samples: int = 6

    def __post_init__(self) -> None:
        """Validate the logging settings."""
        for name in (
            "log_every_n_steps",
            "diagnostics_every_n_epochs",
            "prior_samples_for_ce",
            "diversity_samples",
            "diversity_images",
            "panel_images",
            "panel_samples",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.diversity_samples < 2:
            raise ValueError("diversity_samples must be at least 2 to form a pair")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpointing policy.

    Attributes:
        save_best: Keep the best checkpoint by ``monitor``.
        save_last: Keep the most recent checkpoint, which is what a resume uses.
        monitor: Metric selecting the best checkpoint. Must be a ``val/`` metric:
            selecting on test would bias every reported number, and one of the public
            reimplementations does exactly that.
        mode: ``"min"`` or ``"max"``.

    Raises:
        ValueError: If ``monitor`` is not a validation metric, or ``mode`` is unknown.
    """

    save_best: bool = True
    save_last: bool = True
    monitor: str = "val/total"
    mode: str = "min"

    def __post_init__(self) -> None:
        """Validate the checkpointing policy."""
        if not self.monitor.startswith("val/"):
            raise ValueError(
                f"monitor must be a val/ metric, got {self.monitor!r}. Selecting "
                "checkpoints on training or test metrics invalidates the comparison."
            )
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")


@dataclass(frozen=True)
class HeadConfig:
    """Selection-head hyperparameters. Used only when ``train.mode == "selection_head"``.

    Attributes:
        train_samples: Prior candidates drawn per image per training step. 8. Sampling
            re-runs only ``f_comb`` with the base under ``no_grad``, so K candidates cost
            K forward passes of a 1x1-conv stack, not K passes of the U-Net.
        eval_samples: Candidates per image at validation. 16, so the numbers sit beside
            the existing Phase 1 table. **Train-K need not equal eval-K**, because the head
            scores each candidate independently -- a genuine advantage of the regression
            formulation over a listwise or pairwise one, and worth a sentence in the report.
        huber_delta: Transition point of the Huber loss. 0.1, chosen at the scale of
            *within-image* score differences, so the differences the head must actually
            discriminate sit in the quadratic region while cross-bucket errors get linear
            treatment. Not MSE: the target distribution is bounded and bucket-1-heavy near
            zero, so squared error over-weights the few high-target bucket-4 images. Not
            L1: its constant gradient near zero jitters at convergence.
        scorer_channels: Width of each stride-2 convolution in the scorer tower.
        mean_centered_targets: **Pre-registered fallback, off by default.** Achievable
            scores are strongly bucket-dependent (ceilings 0.40 to 1.00), so plain
            regression can score well by predicting each image's typical value while
            ignoring the candidate entirely. If validation Spearman sits near zero while
            the Huber loss looks healthy, switch this on: targets are then centered within
            each image, which removes the between-image component the shortcut exploits.
            Recorded here in advance so that taking it later is a planned contingency
            rather than an unexplained mid-project pivot.

    Raises:
        ValueError: If a value is out of range.
    """

    train_samples: int = 8
    eval_samples: int = 16
    huber_delta: float = 0.1
    scorer_channels: tuple[int, ...] = (32, 64, 128)
    mean_centered_targets: bool = False

    def __post_init__(self) -> None:
        """Validate the head settings."""
        for name in ("train_samples", "eval_samples"):
            value = getattr(self, name)
            if value < 2:
                raise ValueError(
                    f"{name} must be at least 2 -- a single candidate gives the head "
                    f"nothing to discriminate between, got {value}"
                )
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {self.huber_delta}")
        if not self.scorer_channels:
            raise ValueError("scorer_channels must not be empty")
        object.__setattr__(self, "scorer_channels", tuple(self.scorer_channels))


@dataclass(frozen=True)
class ExperimentConfig:
    """The full configuration of one run."""

    run: RunConfig = field(default_factory=RunConfig)
    model: ProbUNetConfig = field(default_factory=ProbUNetConfig)
    loss: ElboConfig = field(default_factory=ElboConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    head: HeadConfig = field(default_factory=HeadConfig)

    def __post_init__(self) -> None:
        """Cross-section validation."""
        if self.schedule.name == "piecewise" and self.schedule.values:
            if abs(self.schedule.values[0] - self.optim.lr) > 1e-12:
                raise ValueError(
                    f"schedule.values[0] ({self.schedule.values[0]}) must equal "
                    f"optim.lr ({self.optim.lr}) so the initial learning rate has a "
                    "single source of truth"
                )

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load a configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed configuration.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If the document is not a mapping, or holds unknown keys.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> ExperimentConfig:
        """Build a configuration from a nested mapping.

        Args:
            raw: Nested mapping of section name to settings.
            source: Label used in error messages.

        Returns:
            The parsed configuration.

        Raises:
            ValueError: If a section or key is unknown.
        """
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(
                f"{source}: unknown config section(s) {unknown}; expected "
                f"{sorted(known)}"
            )

        sections: dict[str, Any] = {}
        for name, dataclass_field in known.items():
            provided = raw.get(name, {})
            if provided is None:
                provided = {}
            if not isinstance(provided, dict):
                raise ValueError(f"{source}: section {name!r} must be a mapping")
            sections[name] = _build(dataclass_field.type, provided, f"{source}:{name}")

        # The pairing seed follows the run seed unless it was set explicitly, so one
        # knob controls run randomness. The split seed is separate and frozen.
        if "pairing_seed" not in raw.get("data", {} if raw.get("data") is None else raw["data"]):
            sections["data"] = dataclasses.replace(
                sections["data"], pairing_seed=sections["run"].seed
            )
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as a plain nested dict.

        Returns:
            A YAML/JSON-serializable mapping.
        """
        return {name: _to_plain(getattr(self, name)) for name in (f.name for f in fields(self))}

    def to_yaml(self) -> str:
        """Render the configuration as YAML text."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


def _build(target: Any, provided: dict[str, Any], source: str) -> Any:
    """Instantiate one config dataclass from a mapping, rejecting unknown keys.

    Args:
        target: The dataclass (or its string annotation) to build.
        provided: The mapping of field name to value.
        source: Label used in error messages.

    Returns:
        The constructed dataclass instance.

    Raises:
        ValueError: If ``provided`` holds a key the dataclass does not define.
    """
    resolved = _resolve(target)
    known = {f.name: f for f in fields(resolved)}
    unknown = sorted(set(provided) - set(known))
    if unknown:
        raise ValueError(
            f"{source}: unknown key(s) {unknown}; expected {sorted(known)}"
        )
    kwargs = {
        name: _coerce(known[name].type, value, f"{source}.{name}")
        for name, value in provided.items()
    }
    return resolved(**kwargs)


_CONFIG_TYPES = {
    "AugmentationConfig": AugmentationConfig,
    "RunConfig": RunConfig,
    "ProbUNetConfig": ProbUNetConfig,
    "ElboConfig": ElboConfig,
    "DataConfig": DataConfig,
    "OptimConfig": OptimConfig,
    "ScheduleConfig": ScheduleConfig,
    "TrainConfig": TrainConfig,
    "LogConfig": LogConfig,
    "CheckpointConfig": CheckpointConfig,
    "HeadConfig": HeadConfig,
}


def _resolve(target: Any) -> Any:
    """Resolve a possibly-stringified dataclass annotation to the class itself.

    ``from __future__ import annotations`` turns field types into strings, so the
    section classes are looked up by name.

    Args:
        target: A dataclass or its name.

    Returns:
        The dataclass.

    Raises:
        TypeError: If the annotation is not a known config dataclass.
    """
    if is_dataclass(target):
        return target
    if isinstance(target, str) and target in _CONFIG_TYPES:
        return _CONFIG_TYPES[target]
    raise TypeError(f"not a known config dataclass: {target!r}")


def _coerce(annotation: Any, value: Any, source: str) -> Any:
    """Coerce a YAML scalar into the type a dataclass field expects.

    Handles the cases this project actually uses: nested config dataclasses, ``Path``,
    tuples of floats, and optional ints. Anything else is passed through and validated by
    the dataclass.

    Args:
        annotation: The field's type annotation, possibly a string.
        value: The YAML value.
        source: Label used in error messages.

    Returns:
        The coerced value.

    Raises:
        ValueError: If a Path, tuple or nested section cannot be coerced.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")

    # A nested config section, e.g. data.augmentation. Recursing through _build keeps
    # unknown-key rejection working at every depth: without this the mapping would reach
    # the dataclass as a plain dict and a typo inside it would never be caught.
    if text in _CONFIG_TYPES and isinstance(value, dict):
        return _build(_CONFIG_TYPES[text], value, source)

    if "Path" in text and value is not None:
        return Path(value)
    if "tuple" in text and value is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{source}: expected a list, got {value!r}")
        return tuple(value)
    if not isinstance(annotation, str):
        origin = get_origin(annotation)
        if origin is tuple and value is not None:
            return tuple(value)
        if origin is not None and type(None) in get_args(annotation) and value is None:
            return None
    return value


def _to_plain(value: Any) -> Any:
    """Convert a config value into something YAML can round-trip.

    Args:
        value: A dataclass, Path, tuple or scalar.

    Returns:
        A plain Python equivalent.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value
