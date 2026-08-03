# Deviations from the paper and the reference implementation

One entry per place where this implementation departs from Kohl et al.
(arXiv:1806.05034) or from the authors' released TF1 code. Each records what we do,
what they do, why, and the expected impact. This is report content: the course grade
rests partly on depth of reasoning, and a documented deviation is worth more than an
undocumented coincidence.

**Reference** below means `probunet-reference/probabilistic_unet-master-2/`, the
authors' TF1 + Sonnet code. Note that its shipped `training/prob_unet_config.py` is a
**Cityscapes** configuration (19 classes, 3 input channels, 256x512 patches, 6
down-sampling steps), *not* the paper's LIDC configuration — so "what the reference
does" sometimes means its code path rather than its config values.

---

## 1. Latent parameterization: log-variance instead of log-sigma

| | |
|---|---|
| **Ours** | Head predicts `(mu, logvar)`; `scale = exp(0.5 * logvar)` |
| **Reference** | Head predicts `(mu, log_sigma)`; `scale_diag = exp(log_sigma)` (`model/probabilistic_unet.py:340`) |
| **Paper** | Appendix H.1 says a 1x1 conv predicts `2N` channels; it does not fix which parameterization |

**Why.** CLAUDE.md specifies log-variance explicitly. Both satisfy the real
requirement — the scale is positive by construction, with no clamp — and both give
`scale ≈ 1` at initialization, since the head output starts near zero.

**Expected impact: negligible.** The two differ by a factor of 2 in the gradient
reaching the scale head. Adam normalizes each parameter by its own gradient RMS, so a
constant factor on the gradients to a fixed set of parameters is largely absorbed by
the optimizer; it is not equivalent to a 2x learning-rate change under Adam the way it
would be under plain SGD. Measured KL at initialization is 0.12 per image, consistent
with `scale ≈ 1` under either convention.

**Reversible in one line** in `src/probunet/model/encoder.py` if exact parity is ever
wanted.

---

## 2. Posterior mask encoding: one uncentred channel instead of centred one-hot

| | |
|---|---|
| **Ours** | Posterior input is `cat([image, mask], 1)` — **2 channels**, mask in {0, 1} |
| **Reference** | One-hots the mask to `num_classes` channels and subtracts 0.5, giving **1 + num_classes** channels with values in {-0.5, +0.5} (`model/probabilistic_unet.py:400-410`) |

**Why.** Both differences are absorbable by the first convolution's learnable
parameters rather than representational:

- *Centering* is an initialization-time offset: `w·(mask − 0.5) + b == w·mask + (b − 0.5w)`.
  The two parameterizations span the same function class; only the starting point of
  the bias differs.
- *One-hot* is redundant for a binary mask: with 2 classes the second channel is
  exactly `1 − mask`, so a layer with a bias can already represent any affine function
  of the one-hot pair from the single channel.

**Expected impact: negligible, at initialization only.** Our first posterior conv sees
an uncentred input, which marginally changes the initial activation statistics of that
one layer. Parameter cost of the difference is 288 params (one extra input channel in
one 3x3 conv).

---

## 3. Channel capping: off by default

| | |
|---|---|
| **Ours** | Strict doubling `[32, 64, 128, 256, 512]` over 4 down-sampling steps. `max_channels` is a config flag, `None` by default |
| **Reference** | `num_channels = [32, 64, 128, 192, 192, 192, 192]` — capped at **6x base = 192** from the 4th scale, over 6 down-sampling steps (`training/prob_unet_config.py:104-105`) |
| **Paper** | Appendix H.1 specifies base 32 doubled per down-sampling step, with no cap |

**Why.** The faithful baseline follows the paper. Capping is a memory optimization in
the released code, and that released config is for Cityscapes at 256x512 — a
resolution where capping matters far more than at our 128x128.

**Expected impact: large on capacity, by design.** See the parameter table below. The
cap is available for the 6 GB RTX 2060 if activations do not fit; results produced with
it are not comparable to uncapped results and must be labelled as a separate
configuration.

**Note on the cap value.** "Capping after the 3rd scale" is ambiguous. Our
`max_channels=128` clamps to the 3rd scale's width; the reference's own value is
`6 x base_channels = 192`. Neither is the paper. If a capped run is needed for
comparability with the reference's *style*, use `max_channels=192`.

### Parameter counts (analytic; verified against the built model in `tests/test_model.py`)

| component | ours (paper) | ours, cap 192 | ours, cap 128 | reference as released |
|---|---|---|---|---|
| U-Net encoder | 7,855,296 | 2,361,408 | 1,365,696 | 4,353,792 |
| U-Net decoder | 3,918,240 | 2,222,304 | 1,411,104 | 4,877,664 |
| **U-Net** | **11,773,536** | 4,583,712 | 2,776,800 | 9,231,456 |
| prior net | 7,861,452 | 2,363,724 | 1,367,244 | 4,356,108 |
| posterior net | 7,861,740 | 2,364,012 | 1,367,532 | 4,361,580 |
| **latent nets total** | **15,723,192** | 4,727,736 | 2,734,776 | 8,717,688 |
| f_comb | 2,370 | 2,370 | 2,370 | 2,931 |
| **total** | **27,499,098** | 9,313,818 | 5,513,946 | 17,952,075 |
| latent / U-Net ratio | 1.335 | 1.031 | 0.985 | 0.944 |
| prior / U-Net encoder | 1.0010 | 1.0010 | 1.0011 | 1.0005 |

The `prior / U-Net encoder` ratio near 1.000 in **every** column is the structural
confirmation of Appendix H.1: the latent nets are the encoder path plus only a 1x1
head. In the reference this is explicit in the constructor — `self._prior` and
`self._posterior` are both `AxisAlignedConvGaussian(num_channels=num_channels,
num_convs_per_block=num_convs_per_block, ...)`, the *same* arguments the U-Net gets
(`model/probabilistic_unet.py:366-383`).

The `latent / U-Net` ratio differs (1.335 for us vs 0.944 for the reference) purely
because of the cap, not because of any structural difference: uncapped, our encoder
(7.86 M) is 2.0x our decoder (3.92 M), because the deep scales double in width. With
the reference's cap the deep scales stop growing and its decoder (4.88 M) slightly
*exceeds* its encoder (4.35 M). Same architecture, different width schedule.

Consequence for the extension: our uncapped configuration gives the latent nets
**more** capacity relative to the U-Net than the reference has, which is the direction
we want — the consensus-selection head depends on the latent space being expressive
enough to encode distinct segmentation variants.

---

## 4. Bilinear up-sampling: `align_corners=False`

| | |
|---|---|
| **Ours** | `F.interpolate(..., mode="bilinear", align_corners=False)`, config flag |
| **Reference** | `tf.image.resize_images(..., method=BILINEAR, align_corners=True)` (`model/probabilistic_unet.py:358-359`) |
| **Paper** | Specifies bilinear interpolation; silent on alignment |

**Why.** `align_corners=False` is the modern default in both PyTorch and TensorFlow 2
and avoids a half-pixel misalignment that shifts feature maps slightly relative to the
skip connections they are concatenated with.

**Expected impact: small.** At worst a sub-pixel spatial offset in the decoder, which
the following 3x3 convolutions can compensate for. Flagged because it is exactly the
kind of difference that makes two "identical" reimplementations disagree by a fraction
of a point.

---

## 5. Split granularity: series-level, not patient-level

| | |
|---|---|
| **Ours** | Grouped by DICOM `series_uid`, stratified by ambiguity, 60/20/20 |
| **Paper** | Does not state the LIDC split protocol |

**Why.** No patient ID survived the preprocessing of the public pickle; `series_uid`
is the only grouping identifier present. See `data/splits/SPLIT_NOTES.md` for the full
analysis and two accepted consequences (uneven series density and lesion size across
splits).

**Expected impact: small residual leakage risk.** LIDC-IDRI has ~1010 patients across
~1018 series, so a handful of patients with two scans could in principle straddle
splits. Far better than patch-level splitting, which would put adjacent slices of the
same nodule on both sides of the boundary.

---

## 6. Training budget and learning-rate schedule

| | |
|---|---|
| **Ours** | 100 epochs = 28,300 steps at batch 32 = **0.91 M sample presentations**. Constant lr 1e-4 |
| **Paper** | 240,000 iterations at batch 32 (Appendix H.1) = **7.68 M sample presentations**. lr decays **1e-4 → 1e-6 in five steps** |

Our budget is **11.8%** of the paper's. Because our batch size matches the paper's,
matching its iteration count would mean 240,000 steps = **848 epochs** ≈ 29 h on the
MacBook's MPS backend.

**Why the budget.** Three reasons, in order of weight:

1. The paper trained with heavy augmentation; we have none in the baseline phase. With
   27.5 M parameters on 9,056 patches — whose effective diversity is lower still, since
   train averages 24.4 near-duplicate slices per series — validation loss will plateau
   well before the paper's budget. Past that point extra epochs buy overfitting, not
   generalization, so the sensible stopping criterion is *where val stops improving*,
   not a step count borrowed from a differently-regularized run.
2. Phases 2 and 3 need several flag-isolated runs before 20.08.2026.
3. Resume support makes the choice non-binding: if validation is still improving at
   epoch 100, continue from `last.pt`.

**Why a constant learning rate.** The paper's five-step decay to 1e-6 is tuned for a
240,000-iteration run. Compressed into ~12% of that budget it would spend most of
training at a learning rate chosen for a regime we never reach, and it would obscure
exactly the thing we want to read off the curve: where validation plateaus. A constant
rate gives that read cleanly.

`piecewise` is available and its milestones are **fractions of total steps**, not
absolute step counts, so a schedule keeps its shape at any budget:

```yaml
schedule:
  name: piecewise
  milestones: [0.2, 0.4, 0.6, 0.8]
  values: [1.0e-4, 5.0e-5, 1.0e-5, 5.0e-6, 1.0e-6]
```

**Expected impact: moderate on final numbers, small on conclusions.** A decayed run
would likely end at a slightly better validation loss. Since the primary comparison is
internal — baseline vs modernized vs extension under identical budget and schedule —
the absolute offset is held constant across every comparison that carries a claim.

**Note on where the wrong numbers came from.** An earlier draft of this file used batch
10 and a three-boundary decay. Both came from the authors' released
`prob_unet_config.py`, which is the **Cityscapes** configuration (H.2), not the LIDC one
(H.1). Recorded here because it is the same trap as the channel schedule in entry 3:
the released config is not the paper's LIDC setup.

---

## 7. IoU is foreground-only, not class-averaged

| | |
|---|---|
| **Ours** | A single **foreground** IoU per mask pair, with two empty masks defined as IoU 1.0 |
| **Reference** | Per-class IoU, then `nanmean` over the class axis (`evaluation/eval_cityscapes.py`, via `metrics_from_conf_matrix`) |
| **Paper** | `d(x, y) = 1 - IoU(x, y)`, speaking of "masks of the lesion", with an explicit rule that both-empty gives `d = 0` |

**Why this matters numerically.** The two routes agree on the both-empty case and
diverge sharply everywhere else. For an empty prediction against a 150-pixel lesion:

| convention | IoU | `d = 1 - IoU` |
|---|---|---|
| foreground-only (ours) | 0.000 | **1.000** |
| class-averaged (reference, binary case) | (0.991 + 0.000)/2 = 0.495 | **0.505** |

Class averaging drags in a background IoU of ~0.99, because background is ~99% of every
128×128 crop. It is a far more forgiving metric and would roughly halve every reported
distance.

**Why foreground-only.** Three reasons, strongest first:

1. **The paper's own both-empty rule is evidence for it.** Under class averaging that
   case needs no rule at all: `metrics_from_conf_matrix` returns `NaN` for a class absent
   from both masks, and `nanmean([1.0, NaN]) = 1.0` gives `d = 0` automatically. The rule
   is only *necessary* when a single foreground IoU has to resolve 0/0. The paper stating
   it explicitly implies the single-IoU reading.
2. The paper describes the quantity as the overlap of "masks of the lesion", not a
   class-averaged score.
3. Follow-up work on this same preprocessed LIDC data reports foreground IoU, and
   CLAUDE.md names those papers as the external anchor. Class averaging would match
   neither them nor any number in the paper — the paper reports **no** LIDC GED value.

**Expected impact: large on absolute values, none on internal comparisons.** Every GED
and IoU figure would be roughly halved under class averaging. Since the primary
comparison is internal — baseline vs modernized vs extension under one convention — and
the external anchor uses foreground IoU, this is the choice that makes our numbers mean
something. A `--iou-mode class_averaged` switch was considered and deliberately **not**
added: two coexisting conventions is how a report ends up quoting one number under the
other's interpretation.

Verified consequence, from an actual evaluation run: the all-empty predictor reaches
Dice **0.75** on bucket 1 (three of four graders empty, 33% of the data) and GED 0.846
overall. Those degenerate baselines are reported alongside every model number precisely
because the metric is this sensitive to the convention.

---

## Non-deviations worth recording

Things that *look* like they could differ but were checked and match:

- **U-Net returns features, not logits.** The reference's `VGG_Decoder._build` returns
  `lower_res_features` and its `num_classes` argument is unused; `f_comb` produces the
  logits. Ours does the same.
- **f_comb hidden width = 32.** The reference passes `num_channels=num_channels[0]` to
  `Conv1x1Decoder` (`model/probabilistic_unet.py:372`), i.e. base channels, with
  `num_1x1_convs=3`. Ours matches, reading the width from `base_channels`.
- **Initialization.** `he_normal()` weights and `truncated_normal_initializer(stddev=0.001)`
  biases, on both. We truncate at ±2σ to match TensorFlow's semantics, since torch's
  `trunc_normal_` bounds are absolute and its defaults would apply no truncation at
  σ=1e-3.
- **Loss reduction.** CE summed over pixels and averaged over batch, KL summed over
  latent dims and averaged over batch, `beta = 1.0`. Matches the reference exactly; see
  `src/probunet/losses/elbo.py` for why the alternative silently redefines beta.
- **Weight decay.** The reference applies L2 as `1e-5 · Σ‖w‖²/2` over weights *and*
  biases, which `torch.optim.Adam(weight_decay=1e-5)` reproduces exactly. So plain
  `Adam`, not `AdamW`.
- **Down-sampling** is average pooling with a 2x2 kernel and stride 2, on both.
