# Split notes and known limitations

Companion to `split.json`, which is a frozen artifact and deliberately carries no
narrative text. The split itself must never change; this file records what it costs.

Split: grouped by `series_uid`, stratified over the number of non-empty grader masks
per patch, seed **1806**, target 60/20/20 by patch count.

## What the split achieves

| split | series | patches | patch ratio | patches/series |
|---|---|---|---|---|
| train | 371 | 9,056 | 0.5999 | 24.41 |
| val | 252 | 3,021 | 0.2001 | 11.99 |
| test | 252 | 3,019 | 0.2000 | 11.98 |
| **total** | **875** | **15,096** | | 17.25 |

Ambiguity balance, the quantity the stratification optimizes — fraction of each split
by non-empty grader count:

| split | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| train | 0.0000 | 0.3288 | 0.1825 | 0.1739 | 0.3147 | 2.4745 |
| val | 0.0000 | 0.3287 | 0.1827 | 0.1741 | 0.3145 | 2.4743 |
| test | 0.0000 | 0.3286 | 0.1825 | 0.1739 | 0.3150 | 2.4753 |
| spread | 0.0000 | 0.0003 | 0.0002 | 0.0002 | 0.0005 | **0.0010** |

Grader shape agreement (mean pairwise IoU among non-empty masks), reported but **not**
stratified on: train 0.6279, val 0.6313, test 0.6345 — spread 0.0066.

## Limitation 1 — series density is uneven across splits

**Train holds 42.4% of the series (371 of 875) but 60.0% of the patches**: 24.41
patches per series against 11.99 for val and 11.98 for test.

*Mechanism.* The assignment sorts series by descending patch count and gives each to
the split with the largest relative shortfall. Train's 60% target means it has the
largest shortfall early, so it absorbs the biggest series first. Series size is the
number of annotated slices in a scan, so the largest series land disproportionately in
train by construction, not by chance.

*Consequence.* Train's effective sample diversity is lower than its patch count
suggests: more of its patches are adjacent slices of the same nodule, which are
near-duplicates. 9,056 training patches do not represent 9,056 independent examples.
This does not leak across the train/test boundary — grouping by `series_uid` prevents
that — but it does mean train and test are not exchangeable samples.

## Limitation 2 — lesion size differs across splits

Foreground area of non-empty masks, as a fraction of the 128×128 crop:

| split | non-empty masks | mean | median | mean px |
|---|---|---|---|---|
| train | 22,409 | 0.00942 | 0.00433 | 154.4 |
| val | 7,475 | 0.00855 | 0.00360 | 140.0 |
| test | 7,473 | 0.00889 | 0.00360 | 145.7 |

**Train's median lesion area is 20.3% larger than test's** (0.00433 vs 0.00360).

*Consequence.* Absolute Dice and IoU are not directly comparable across splits: small
objects depress overlap metrics sharply, so test numbers may read slightly *worse*
than train numbers for reasons that have nothing to do with generalization. When
reading a train-vs-test gap in the report, this is a confound to name explicitly
before reaching for "overfitting". The smallest non-empty mask in the dataset is a
single pixel.

## Why these are accepted rather than fixed

Both follow from insisting on whole-series groups while also hitting the patch ratio.
Correcting either one would require trading away the ambiguity stratification, which
matters more: the model's entire purpose is to represent grader disagreement, so
splits that differ in *how ambiguous* their cases are would make GED, oracle Dice and
the consensus-head comparison incoherent. Size and density imbalance only affect the
absolute scale of overlap metrics.

Decision: **accept and document.** The primary comparison in the report is internal
(baseline vs modernized vs extension) under this identical split, so these confounds
are held constant across every comparison that carries a claim.

## Reproducing

```bash
python -m probunet.data.splits            # refuses to overwrite an existing split
python scratch/data_stats.py              # the per-split tables above
```

`tests/test_splits.py::test_committed_split_file_is_reproducible` regenerates the
split from the recorded seed and asserts it matches this file byte for byte, so the
committed split cannot drift from the algorithm.