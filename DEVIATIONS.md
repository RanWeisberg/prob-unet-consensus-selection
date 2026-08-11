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

The same caveat applies with more force to augmentation: the repository's **only**
augmentation code lives in `data/cityscapes/data_loader.py`, and it contains **no LIDC
data loader at all**. Entries 8 and 12 are where that matters most.

Phase 1 is a **faithful reproduction**: the architecture, the augmentation, the
240k-iteration budget and the five-step LR decay all follow Appendix H.1. The entries
below are what could not be matched exactly, plus the values the paper leaves unspecified.

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

## 6. Learning-rate decay: the shape is the paper's, the intermediate values are ours

The budget and the schedule *shape* are no longer deviations. Phase 1 trains for the
paper's **240,000 iterations at batch 32** and decays **1e-4 → 1e-6 in five steps**, as
Appendix H.1 states. What remains undocumented by the paper is narrower but real.

| | |
|---|---|
| **Ours** | Six levels on a geometric ladder (ratio `10^(-2/5) = 0.3981` per step), equal-length plateaus at fractions `k/6` |
| **Paper** | "an initial learning rate of 1e−4 that is lowered to 1e−6 in 5 steps" — endpoints and step count only |
| **Reference** | `values [1e-4, 0.5e-4, 1e-5, 0.5e-6]`, `boundaries [80000, 160000, 240000]` (`training/prob_unet_config.py:112-115`) — a **Cityscapes** schedule: four levels, not five steps, and it ends *below* 1e-6 |

```yaml
milestones: [0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333]
values: [1.0e-4, 3.9810717e-5, 1.5848932e-5, 6.3095734e-6, 2.5118864e-6, 1.0e-6]
```

**Two judgment calls, both report content.**

1. *"in 5 steps" = five decay events, hence six levels.* The alternative reading — five
   distinct values, four decays — is defensible, and lands on a rounder 1e-5 midpoint. We
   read "lowered … in 5 steps" as five lowering operations.
2. *Geometric spacing, equal plateaus.* The paper fixes neither. A geometric ladder is the
   only choice that introduces no arbitrary round numbers, and equal plateaus follow the
   reference's own habit of evenly spaced boundaries. Note the reference's last boundary
   (240,000) equals its total step count and so never fires — a detail worth not copying.

**Milestones are fractions of total steps, not absolute step counts**, so the schedule
keeps its shape at any budget instead of hardcoding boundaries tuned for one.

### Budget rounding

The loop's unit is an epoch, so an iteration budget must land on a whole number of them.
9,056 train patches at batch 32 give **283 steps/epoch**, and 240,000 / 283 = 848.06:

| rounding | epochs | steps | error |
|---|---|---|---|
| **nearest (ours)** | **848** | **239,984** | **−16 (−0.007%)** |
| ceiling | 849 | 240,267 | +267 (+0.11%) |

We round to nearest, which is 17× closer. The realized 239,984 is what the milestones are
taken as fractions of, so the schedule spans exactly the run performed. The budget is
configured as `train.iterations: 240000` and the epoch count is **derived**, so changing
the split or the batch size cannot silently change the compute.

**Expected impact: none on conclusions.** Every comparison that carries a claim
(baseline vs ablation vs modernized vs extension) uses the identical budget and schedule.

**Note on where the wrong numbers came from.** An earlier draft of this file used batch
10 and a three-boundary decay. Both came from the authors' released
`prob_unet_config.py`, which is the **Cityscapes** configuration (H.2), not the LIDC one
(H.1). Recorded here because it is the same trap as the channel schedule in entry 3, and
the augmentation parameters in entry 12: the released config is not the paper's LIDC setup.

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

## 8. Augmentation: the paper specifies types, not one single value

Appendix H.1 in full: *"We apply augmentations to the image tiles (180×180 pixels size):
random elastic deformation, rotation, shearing, scaling and a randomly translated crop
that results in a tile size of 128×128 pixels."* That sentence names five transforms and
**zero numbers**. Every magnitude below therefore has a source that is *not* the paper,
and this table is the honest accounting of which is which.

| setting | value | source |
|---|---|---|
| transform types | elastic, rotation, shear, scale, translated crop | **Paper H.1** |
| augment train split only | — | **Paper** ("During training") + **reference** (`do_aug=True` train, `False` val) |
| augment *after* grader pairing | — | **Paper H.1** ("image-grader pairs are drawn randomly. We apply augmentations…") |
| tile size for the transform | `pad_to_px: 180` | **Paper H.1** |
| `rotation_degrees` | 22.5 | **Reference code only** — Cityscapes `angle_x = ±π/8`; rotation is resolution-independent, so it transfers |
| `scale_range` | (0.8, 1.2) | **Reference code only** — Cityscapes `scale`; likewise resolution-independent |
| ranges include the identity, applied p=1 | — | **Reference mechanism** (uniform ranges spanning zero, no separate per-sample probability) |
| crop offset uniform over `[0, 52]²` | — | **Reference mechanism** (`rand_crop_dist = patch_size/2`) |
| `shear` | 0.1 | **OURS** — absent from the paper *and* from `batchgenerators`, which has no shear parameter at all |
| `elastic_alpha_px` / `elastic_sigma_px` | 5.0 / 10.0 | **OURS** — see below |
| `max_redraws` | 3 | **OURS** — see entry 9 |
| **no mirroring** | — | **Paper H.1** (absent) |
| **no gamma / intensity** | — | **Paper H.1** (absent), and the paper says Cityscapes *"additionally* impose random color augmentations" |

**Why the elastic parameters could not be transferred.** The reference uses
`alpha=(0., 800.)`, `sigma=(25., 35.)` (`training/prob_unet_config.py:49-50`), and
`batchgenerators` normalizes its displacement field by an **L2 norm that scales with the
array size**. Those numbers are therefore meaningless at a different resolution — they
are tuned for 256×512. We re-parameterize elastic strength as a **peak displacement in
pixels**, normalizing by peak absolute value, which is interpretable, testable, and
resolution-explicit. Strength is drawn from `(0, alpha)` so the range spans the identity,
mirroring the reference's own mechanism for making transforms stochastic.

**Why mirroring and gamma are excluded.** Both appear in the authors' released code — but
only in `data/cityscapes/data_loader.py`, and **that repository contains no LIDC loader at
all**. The paper's word "additionally" for Cityscapes colour augmentation settles it:
intensity augmentation is a Cityscapes-only addition. Including either would make our
Phase 1 *harsher* than the paper's while wearing the paper's name. Related detail not
copied: the reference applies `GammaTransform` **outside** its `if do_aug:` branch
(`data/cityscapes/data_loader.py:317`), so it augments its validation set too. Ours never
transforms val or test in any configuration.

**Interpolation.** Image bilinear (`order=1`), mask nearest (`order=0`). The reference
leaves `batchgenerators`' default `order_data=3` (cubic) for the image. Bilinear is both
what the mask/image pairing needs and safer: cubic rings outside `[0, 1]` and would break
the dataset's range invariant, which `LidcArrays.validate` enforces.

**Expected impact: significant and intended.** This is regularization the paper had and
our earlier draft lacked. The direction to watch is `diag/gap_total`, the val−train gap.

---

## 9. Tiny lesions can vanish under nearest-neighbour resampling

| | |
|---|---|
| **Ours** | A transform that empties a **non-empty** mask is redrawn up to 3 times; failing that, the sample is returned untransformed. The rate is logged |
| **Reference / paper** | No such guard is described or implemented |

**Why.** Masks are resampled with `order=0`, and the smallest non-empty mask in this
dataset is **one pixel**. A one-pixel region can simply receive no output pixel and
disappear. That converts a non-empty target into an empty one, which is invisible in the
loss curve but corrupts the ambiguity buckets — and those buckets are what the extension
slices its results by. Silence was not an acceptable failure mode, so the event is both
guarded and counted (`train/aug_lesion_lost_fraction`, `train/aug_redraw_rate`).

**Measured on the real train split** (3 epochs, 27,168 augmented samples, the shipped
baseline settings):

| quantity | value |
|---|---|
| lesion lost, **with** the guard | **0 / 27,168 = 0.000000** |
| lesion lost, without the guard | 6 / 16,815 non-empty = 0.036% |
| redraws triggered | 0.00022 per sample |
| median mask area | 71 px before → 70 px after |

So the guard is cheap (it fires on ~1 sample in 4,500, far too rarely to distort the
transform distribution) and it takes lesion loss to exactly zero. The measurement also
confirms `elastic_alpha_px: 5.0` and `scale_range: (0.8, 1.2)` are safe for this dataset's
lesion sizes: no size bucket, including 1–4 px lesions (n=239), lost a single mask.

---

## 10. Initialization: He-normal, where the paper says orthogonal

| | |
|---|---|
| **Ours** | `he_normal()` weights; biases from a truncated normal with σ=0.001 |
| **Reference** | `he_normal()` weights; same bias initialization |
| **Paper** | Appendix H.1: *"All weights of all models are initialized with **orthogonal initialization** having the gain (multiplicative factor) set to 1"* |

**Why this is listed now.** It was previously filed under "Non-deviations worth
recording" on the grounds that our code matches the released code. That is true, but it
made the wrong comparison: the *paper* specifies orthogonal initialization and the
released code does not implement it. Once Phase 1 is a faithful reproduction, matching the
reference is not a defence for diverging from the paper.

**Why it is not changed.** Deliberate scope decision, not an oversight. Changing
initialization changes every run and would have to land before the long run starts;
CLAUDE.md's rule is to raise such a call rather than fold it in silently. The bias
initialization does match H.1 exactly.

**Expected impact: small but genuinely unknown.** Orthogonal and He-normal initialization
have similar variance scaling for these layer shapes, so the difference should wash out
early in a 240k-iteration run. Unlike the other entries here, this one is a divergence
from the paper we could close and chose not to — flagged prominently for that reason.

---

## 11. Augmentation randomness is derived, not global

| | |
|---|---|
| **Ours** | Every augmentation draw comes from `np.random.default_rng([seed, epoch, position])` |
| **Reference** | `MultiThreadedAugmenter` with per-worker seeds and global `np.random` inside the transforms |

**Why.** Three properties fall out of deriving the generator instead of consuming a global
one, and all three matter for a run measured in days:

1. **Resume-safe with nothing stored.** The transform for a sample is a pure function of
   `(seed, epoch, position)`, and the epoch is restored from the checkpoint, so a resumed
   run reproduces the transforms it would have applied uninterrupted. No augmentation RNG
   state needs to be checkpointed.
2. **Worker-count independent.** The augmentation is identical at `num_workers=0` and
   `num_workers=8`, so a throughput change cannot alter the data.
3. **Decoupled from the model's sampling.** The latent `z` is drawn from the global torch
   RNG. If augmentation shared it, adding or changing an augmentation parameter would
   silently shift the entire `z` sequence, and two runs meant to differ in one variable
   would differ in two.

**Expected impact: none on results, large on trustworthiness.** A test asserts that
augmenting does not advance numpy's global state.

---

## 12. Pre-cropped 128×128 data: the outer ring is mirrored tissue, not real CT

**This is the one substantive augmentation deviation.** The paper augments a *180×180*
tile and crops to 128×128; our preprocessed source is *already* 128×128, so there is no
180×180 tile to crop from.

| | |
|---|---|
| **Ours** | Reflect-pad 128→180, sample and apply the transform in the padded frame, then take a randomly translated 128 crop back out |
| **Paper** | Transforms a real 180×180 CT tile, then takes the randomly translated 128 crop |

**Why not just apply the paper's magnitudes to the 128 tile.** Because the 52-pixel margin
is what makes those magnitudes mild, and without it the *same numbers* describe a *harsher*
augmentation:

| transform | artifact-free up to | zero-fill on a bare 128 tile |
|---|---|---|
| rotation ±22.5° | `180 / (cos+sin) = 137.8 px` | **13.97%** of the frame |
| scale 0.8 | `180 × 0.8 = 144.0 px` | **36.50%** of the frame |
| both at once | `110.2 px` | not covered even at 180 |

Both bounds exceed 128, so at the paper's tile size its maximum rotation and its maximum
shrink each produce **zero** border fill. Applying 22.5° and 0.8 to a bare 128 frame while
calling it faithful would have been wrong twice over: harsher than the paper, and wearing
the paper's numbers to say so. (The closed form for the rotation figure is
`1 − 2/(1+cosθ+sinθ)`.)

**What reflect-padding costs.** Two things, both accepted deliberately:

1. **The outer ring is mirrored tissue rather than real anatomy.** Image *and* mask are
   reflected through the same coordinates, so a mirrored lesion carries a mirrored label —
   the pair stays self-consistent and nothing is mislabelled. But mirrored CT is not the
   real neighbouring anatomy the paper's 180×180 tile contained.
2. **At `scale < 1`, reflected copies of the lesion enter the frame.** Measured at
   **2.05%** of non-empty training samples (area growing >1.5×). These duplicates are
   anatomically implausible in a way real 180×180 context was not. Pinned by a test so it
   is documented behaviour rather than a surprise.

Note the third row of the table: even the paper's own 180 margin does not cover maximum
rotation combined with scale 0.8 (110.2 < 128), so residual border fill in that corner is
something **the paper had too**. Ours is mirrored tissue where theirs was zeros.

**What is recovered.** The paper's *randomly translated crop* is reproduced rather than
abandoned, and its rotation and scale magnitudes become usable honestly. An earlier plan
for this project recorded "cannot reproduce the random crop" plus "constant-0 border
wedges" as two separate deviations; reconstructing the margin replaces both with this one.

**Implementation note.** The padding is never materialized. It is folded into the
source-coordinate arithmetic and realized by sampling the original array with scipy's
`mode="mirror"` (equivalent to `numpy.pad(mode="reflect")`), which keeps the whole
operation to a single interpolation per array.

---

## 13. Selection-head training runs without augmentation

**Phase 3 only.** `configs/extension.yaml` sets `data.augmentation.enabled: false`, where
every Phase 1 and Phase 2 config has it on. Phases 1 and 2 are untouched.

The paper has nothing to say here — it has no selection head — so this is a deviation from
*our own* Phase 1 pipeline rather than from Appendix H.1, recorded because a reader
diffing the configs will see augmentation on in two arms and off in the third.

**Why.**

1. **The base is frozen.** Augmentation exists in Phase 1 to regularize a 27.5M-parameter
   generative model. The head is ~150k parameters on a frozen backbone, and it sees
   **freshly resampled candidates every epoch** — 8 new prior draws per image per pass —
   so its effective dataset is already far larger than the patch count. The head's data is
   not scarce, which is the only condition under which augmentation would earn its cost.
2. **There is no augmentation at inference.** The head is deployed on untransformed
   images. Training it on elastically deformed ones hands the frozen base a domain shift
   it was never adapted to, for no compensating benefit.
3. **The risk is asymmetric.** Making augmentation carry four masks under one shared
   sampled transform means editing the augmentation path that Phase 1 and Phase 2 both
   depend on. A regression there would silently damage the comparability of two completed
   phases. Not worth it for a benefit already argued away by (1) and (2).

**Consequence for the data pipeline, and why it costs nothing.** The head needs all four
grader masks per image, which is the **eval-mode** dataset shape. With augmentation off,
that existing shape is used directly for training — there is no third dataset mode and no
new code path. This is also enforced from the other direction: constructing an augmented
eval dataset already **raises** (`tests/test_data.py::test_enabling_augmentation_on_an_eval_dataset_is_an_error`),
so the combination this entry rules out is one the code refuses to build anyway.

**Report framing.** State it as a deliberate scope decision with the frozen base as the
reason, not as an omission. The honest caveat is that the head is therefore trained on
~9,056 distinct images with resampled candidates rather than on an augmented image
distribution; if the head overfits, this is the first thing to revisit.

---

## 14. The head's headline number is the metric its checkpoint was selected on

`configs/extension.yaml` monitors `val/selected_consensus_dice` and keeps the best
checkpoint by it across 30 epochs. **That same metric is then the reported result.** The
number is therefore the maximum of 30 draws of a noisy quantity, not an unbiased estimate of
it -- classic selection-on-validation optimism.

**Why it is set up this way, and why that is still right.** Monitoring the deliverable was a
deliberate Stage 4 choice: a proxy metric can be gamed by a constant predictor, whereas the
selected-sample score cannot (a constant score degenerates to a fixed arbitrary pick and
lands near random). The alternative -- selecting on the Huber loss -- would have been worse,
because a low loss is compatible with zero ranking ability (FINDINGS 4.5). So the setup
stands; the *caveat* is what must be reported.

**Size of the effect -- MEASURED, no longer estimated.**

| quantity | value |
|---|---|
| plateau mean, epochs 10-29 | 0.4897 |
| reported figure (`best.pt`, epoch 28) | 0.4940 |
| **optimism** | **+0.0043** |
| as a fraction of the 0.2895 headroom | **1.5%** |

So the reported number sits **at the top of the noise band**, and the bias is 1.5% of
headroom -- real, small, and worth one sentence rather than a caveat paragraph. It does not
explain why 0.4940 was far above the pre-registered 0.30-0.36 band; that turned out to be a
miscalibrated band rather than an inflated result (FINDINGS 4.5).

**The estimate was CONSERVATIVE -- the test split settled it.** The +0.0043 figure above is
the optimism *estimated* from the validation curve (best minus plateau mean). The **measured**
val-to-test movement is **0.0012** (0.4940 → 0.4928), roughly a third of it. So selecting the
checkpoint on validation cost less than the within-run curve suggested it might, and the
validation figure was a closer estimate of held-out performance than this entry originally
implied.

Report the **test** figure (0.4928) as the headline and this entry as the reason validation
numbers are development record rather than results. The estimate erring on the pessimistic
side is the right direction for a caveat to err in.

**Budget was sufficient.** The monitor **plateaus by epoch 10** and thereafter oscillates in
a band of roughly 0.008, so the 30-epoch budget was not the binding constraint.
`extension.yaml` pre-registered the reading for exactly this outcome -- "if it plateaus by
15, that is a legitimate finding about how little capacity this task needs". It plateaued by
10, and `best.pt` came from epoch 28, i.e. from within the plateau rather than from a still-
rising curve.

**The clean figure, and it must be run exactly once.** The **test** split (3019 patches) has
never been touched by checkpoint selection, so the test-split
`selected_consensus_dice` is the unbiased number. Run it **once**, at the end, and report it
as the headline with the validation figure beside it:

```
python scripts/consensus_headroom.py \
    --head-checkpoint runs/selection-head/checkpoints/best.pt --split test \
    --out results/consensus_selection_test.json
```

CLAUDE.md's rule holds: test is touched once. Do not iterate against it, and do not re-run
it after a change to the head -- that would convert it into a second validation split and
lose the only clean number the project has.

---

## 15. Diagnostic panel indices are array rows, but `panel_batch` wants global indices

**A real bug, found by the Colab demo config, and the first configuration in the project
ever to train on a subset export.** Not a deviation from the paper — a deviation between
what a function documents and what its only caller passes it.

`Trainer._log_panel` calls `data.lidc.panel_batch(self.data.arrays,
self.diagnostic_sets.panel)`. `panel_batch` documents its argument as *"patch indices in
the full dataset's numbering"* and resolves it through `LidcArrays.resolve_indices`. But
`diagnostic_sets.panel` comes from `build_diagnostic_sets(data.datasets["val"], ...)`,
whose indices are **rows of the loaded arrays**.

On the full dataset those two numberings are the same and `resolve_indices` is the
identity, so the mismatch was invisible for the whole of Phases 1–3. On a **subset export**
they are different numberings, `resolve_indices` maps rows through `source_index` as though
they were global, and the run dies:

```
KeyError: 'patches [262, 264, 273] are not in this subset export'
```

**Why it is latent rather than rare.** `resolve_indices` returns its input unchanged when
`source_index is None`. That is a correct and useful behaviour, and it is exactly what hid
the bug: the identity path made a wrong argument indistinguishable from a right one on
every configuration anyone had run. This is the same class as the shadowed
`_selection_head_step` and the shape-only assertions of FINDINGS 2.13 — a check that cannot
fail on the data you have is not a check.

**Status: REPORTED, NOT FIXED.** `configs/colab_demo.yaml` sets `log.tensorboard: false`,
which avoids it — `_log_panel` runs only `if self.writer` — at the cost of the TensorBoard
writer and its panel image. `run_diagnostics` itself still runs and still logs every
scalar. The one-line fix is written out in that config:

```python
panel = self.diagnostic_sets.panel
if self.data.arrays.is_subset:
    panel = self.data.arrays.source_index[panel]
panel_images, panel_masks = panel_batch(self.data.arrays, panel)
```

It is a **no-op on every existing run** (`resolve_indices` is already the identity when
`source_index is None`), so it cannot move a reported number. It is unapplied only because
it changes shared training code that Phase 1, Phase 2 and the head all run through, and
that is not a change a demo config gets to make unilaterally. Apply it and flip
`log.tensorboard` back to `true`; nothing else moves.

**Impact on reported results: none.** Every reported run trained on the full dataset, where
the two numberings coincide.

---

## 16. `diagnostic_indices.json` from a subset run holds subset rows under a global schema

The unfixed sibling of entry 15, same root cause, recorded so it is not rediscovered.

`Trainer.__init__` writes `save_diagnostic_sets(self.diagnostic_sets, run_dir /
"diagnostic_indices.json")`. Those indices are **rows of whatever arrays the run loaded**.
`scripts/export_subset.py` reads that same file with `--indices` and treats its contents as
**full-dataset** indices (`arrays.images[rows]` against the full `.npz`).

On a full-dataset run the two readings agree, which is why the file has always been
correct. A run on a subset export writes subset rows into a file whose one consumer will
read them as global — and, unlike entry 15, this fails **silently**: the indices are legal
integers that address *some* patch, so a panel exported from them would be a panel of the
wrong patches rather than a `KeyError`.

**Status: UNFIXED, and currently harmless.** The only subset-trained run in the project is
the Tier 2 Colab demo, whose `diagnostic_indices.json` nothing consumes and whose run
directory the notebook deletes. It becomes a real hazard the moment anything else trains on
a subset and someone points `export_subset.py --indices` at the result.

The proper fix is to make the numbering explicit rather than positional — write the file
with the global indices (mapping through `source_index` when the arrays are a subset) and
record which numbering it used, so a reader and `export_subset.py` cannot disagree about
what the integers mean. Deliberately left for a change that can be made and tested against
the full pipeline rather than folded into the packaging work that found it.

---

## Non-deviations worth recording

Things that *look* like they could differ but were checked and match:

- **U-Net returns features, not logits.** The reference's `VGG_Decoder._build` returns
  `lower_res_features` and its `num_classes` argument is unused; `f_comb` produces the
  logits. Ours does the same.
- **f_comb hidden width = 32.** The reference passes `num_channels=num_channels[0]` to
  `Conv1x1Decoder` (`model/probabilistic_unet.py:372`), i.e. base channels, with
  `num_1x1_convs=3`. Ours matches, reading the width from `base_channels`.
- **Bias initialization.** `truncated_normal_initializer(stddev=0.001)` on both, and
  Appendix H.1 specifies the same. We truncate at ±2σ to match TensorFlow's semantics,
  since torch's `trunc_normal_` bounds are absolute and its defaults would apply no
  truncation at σ=1e-3. (**Weight** initialization matches the reference but *not* the
  paper — moved out of this section to entry 10, because "matches the released code" is
  not the same claim as "matches the paper".)
- **Loss reduction.** CE summed over pixels and averaged over batch, KL summed over
  latent dims and averaged over batch, `beta = 1.0`. Matches the reference exactly; see
  `src/probunet/losses/elbo.py` for why the alternative silently redefines beta.
- **Weight decay.** The reference applies L2 as `1e-5 · Σ‖w‖²/2` over weights *and*
  biases, which `torch.optim.Adam(weight_decay=1e-5)` reproduces exactly. So plain
  `Adam`, not `AdamW`.
- **Down-sampling** is average pooling with a 2x2 kernel and stride 2, on both.
