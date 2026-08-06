# Probabilistic U-Net with a post-hoc consensus selection head

A PyTorch reimplementation of the **Probabilistic U-Net** (Kohl et al., *A
Probabilistic U-Net for Segmentation of Ambiguous Images*, NeurIPS 2018,
[arXiv:1806.05034](https://arxiv.org/abs/1806.05034)), extended with a post-hoc
**consensus selection head** that scores sampled masks by how well they agree with
the set of expert graders, so a single mask can be chosen at inference time without
ground truth.

Course project for *Medical Images Processing with Deep Learning (336033)*.
See `CLAUDE.md` for the full project specification.

## Install

Python 3.12.

```bash
pip install -e ".[dev]"
pytest
```

## Data pipeline

The preprocessed LIDC-IDRI data ships as a Python pickle. `pickle.load` executes
arbitrary code, so the pickle is read **exactly once**, by an isolated script under
`scratch/` that uses a restricted unpickler (numpy and plain containers only; every
other class is replaced by an inert stand-in, so `pydicom` is never imported). The
package itself only ever reads the resulting `.npz`.

```bash
python scratch/inspect_data.py            # audit + structural report
python scratch/convert_data.py            # -> data/processed/lidc.npz + lidc.json
python -m probunet.data.splits            # -> data/splits/split.json
```

## Train and evaluate

```bash
python scripts/train.py --config configs/smoke.yaml       # verify the loop, <1 min
python scripts/train.py --config configs/baseline.yaml    # the real run
python scripts/evaluate.py --checkpoint runs/baseline/checkpoints/best.pt --split val
```

`--split` is required and has no default: development happens on `val`, and `test` is
evaluated once for the final report numbers. Metrics are GED at 1/4/8/16 samples,
oracle / random / Hungarian-matched single-sample quality, and two degenerate baselines
(all-empty predictor, emptiest-sample selection) — all reported aggregate and per
ambiguity bucket, because an all-empty predictor scores Dice 0.75 on the 33% of patches
where three of four graders are empty.

15,096 patches of 128x128, four independent grader masks each, drawn from 875 CT
series. The split is grouped by `series_uid` so no series spans two splits, and
stratified over the number of non-empty grader masks per patch so the splits are
comparable in ambiguity. It is generated **once**, seeded, and committed to the
repository; `load_split()` never regenerates. Known limitations of the split —
uneven series density and lesion size across splits — are recorded in
`data/splits/SPLIT_NOTES.md`.

Training pairs each image with **one randomly chosen** grader mask, redrawn every
epoch and reproducible from the run seed; evaluation keeps all four masks. Empty
masks are never filtered: they are a grader's judgment that no lesion is present.
No normalization is applied — the images are already in [0, 1].

## Train and evaluate

Three variants, **one model implementation**, selected by config:

```bash
python scripts/train.py --config configs/smoke.yaml        # verify the loop, <1 min
python scripts/train.py --config configs/baseline.yaml     # phase 1
python scripts/train.py --config configs/modernized.yaml   # phase 2
python scripts/train.py --config configs/extension.yaml \
    --base-checkpoint runs/modernized/checkpoints/best.pt  # phase 3 (scaffold)

python scripts/evaluate.py --checkpoint runs/baseline/checkpoints/best.pt --split val
python scripts/compare.py --split val \
    --checkpoint baseline=runs/baseline/checkpoints/best.pt \
    --checkpoint modernized=runs/modernized/checkpoints/best.pt
```

`configs/modernized.yaml` is live: it sets `model.latent_covariance: full`, a
full-covariance latent Gaussian parameterized by its Cholesky factor, and that one line is
its only difference from `baseline.yaml` besides the run name. Diff the configs — the only
lines that differ between variants are the flags under test.

`configs/extension.yaml` is still a scaffold: it loads and freezes its base model, logs the
freeze record, and then raises, because the head is phase 3. Its **scoring target is
decided** — a candidate mask is scored by soft Dice against the *soft consensus*
`c = mean of the four grader masks`, so each pixel of `c` says what fraction of graders
included it, and the head regresses onto that score. Consensus is the **selection target
only**: GED and every other distribution metric keep using the four separate grader masks,
because collapsing them into one average would discard exactly the grader spread phase 1
measures. Note that soft-consensus scores are bounded well below 1 by construction — a
perfect mask on an image where only one of four graders saw a lesion scores 0.40, not 1.0 —
so they are reported against the per-bucket ceiling, never against 1.0.

`--split` is required and has no default: development happens on `val`, and `test` is
evaluated once for the final numbers. Metrics are GED at 1/4/8/16 samples, oracle /
random / Hungarian-matched single-sample quality, and two degenerate baselines
(all-empty predictor, emptiest-sample selection) — reported aggregate and per ambiguity
bucket, because an all-empty predictor scores Dice 0.75 on the 33% of patches where
three of four graders are empty.

Device selection is automatic (cuda → mps → cpu) and logged at startup. Seeds do not
reproduce across backends, so every run in one comparison must come from the same
device; checkpoints and results both record the device, torch version and git SHA.

## Where everything lives

The distinction that trips people up on a fresh clone: **training output is ignored,
results are tracked.**

| path | git | contents |
|---|---|---|
| `src/probunet/` | tracked | the package — all real logic |
| `configs/` | tracked | the three variant configs + smoke |
| `scripts/` | tracked | CLI entry points |
| `tests/` | tracked | pytest suite |
| `results/` | **tracked** | evaluation + comparison JSON/CSV — what the notebook reads |
| `data/splits/split.json` | **tracked** | the frozen split (+ `SPLIT_NOTES.md`) |
| `data/processed/lidc.json` | **tracked** | conversion provenance |
| `data/processed/lidc_subset.npz` | **tracked** | ~2 MB, panel patches for the notebook |
| `notebooks/` | tracked | thin narrative layer, no logic |
| `DEVIATIONS.md` | tracked | every departure from the paper/reference |
| `runs/`, `experiments/` | **ignored** | checkpoints (~315 MB each), TensorBoard events |
| `data/raw/` | **ignored** | the source pickle (3.2 GiB) |
| `data/processed/lidc.npz` | **ignored** | the full converted dataset (~450 MB) |
| `scratch/` | **ignored** | one-off scripts, never imported |

`src/probunet/paths.py` is the single source of truth for that table, and
`tests/test_paths.py` asserts it against `git check-ignore` — so a stray `.gitignore`
edit cannot silently make `results/` unreachable from Colab.

Large artifacts are shared out of band:

```bash
python scripts/export_weights.py runs/baseline/checkpoints/best.pt   # ~315 MB -> ~105 MB
python scripts/export_subset.py                                      # -> lidc_subset.npz
```

A weights-only export drops the optimizer state (Adam keeps two moment buffers per
parameter) but keeps the config, epoch and git SHA, so it stays traceable. Full
checkpoints remain the authoritative resumable artifact.

## The notebook

`notebooks/submission.ipynb` runs **on CPU, without training and without the full
dataset**: it needs only the repo (tracked `comparison.json` + `lidc_subset.npz`) plus
three weights-only files. Three training runs exceed a Colab session, and a different
device would produce numbers that are not comparable with the reported ones.
