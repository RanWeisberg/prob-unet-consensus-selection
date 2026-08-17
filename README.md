# A Probabilistic U-Net with a post-hoc consensus-selection head

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RanWeisberg/prob-unet-consensus-selection/blob/main/submission.ipynb)

A PyTorch reimplementation of the Probabilistic U-Net [[1]](#references), trained on
LIDC-IDRI [[3]](#references) and extended in two independent directions: a
**full-covariance latent Gaussian** in place of the paper's axis-aligned one, and a
**post-hoc consensus-selection head** that scores sampled masks by how well they agree with
the four expert graders, so one mask can be chosen at inference time without ground truth.
Both extensions are one config flag away from the reproduction — there is a single model
implementation, selected by config, so every comparison changes exactly one setting.

Course project for *Medical Images Processing with Deep Learning* (336033), by Chenxi Liu
and Ran Weisberg. The 6-page report carries the argumentation; this repository carries the
implementation and every number the report cites.

## The notebook

`submission.ipynb` is the guided tour of the method and its results, and it needs nothing
installed: Colab ships torch, numpy, matplotlib and scipy, and sections 0–6 render recorded
predictions and published tables from files tracked in this repository. Section 7 trains an
undersized model from scratch on a 288-patch subset — the part that executes inside a
session — and section 8 runs the test suite.

It opens two ways. Upload `submission.ipynb` on its own, or follow the badge above, and the
first cell finds no repository beside it and clones this one before doing anything else;
unzip the whole archive into the session instead and that cell finds the files already
there, uses them, and clones nothing.

## Results at a glance

Every figure below is on the held-out test split (3,019 patches), which was evaluated once
after all decisions were frozen. The files named in [Where the numbers live](#where-the-numbers-live)
are the record; nothing here is recomputed.

| finding | measurement |
|---|---|
| **Full covariance improves GED²@16 by 9.0%** (0.3244 → 0.2951), matching the figure published for the same change on near-identical data [[4]](#references) | but by the opposite mechanism to the one we pre-registered: the distribution *sharpened* rather than broadened, and two-thirds of the fidelity gain was paid back in lost sample diversity |
| **The selection head beats a size-only control by +0.0571 soft-consensus Dice**, about two-thirds of the headroom that control leaves | while leaving every distribution metric bit-identical, since the base is frozen |
| **The two extensions do not compose** | full covariance raised the floor without raising the ceiling, shrinking selectable headroom by 15.8%, so the 9% GED gain never reaches a selected output |

Soft-consensus Dice is bounded by its own construction — the aggregate test ceiling is
0.645, not 1.0 — so each row is read against the oracle rather than against 1. One seed per
arm, so these are patterns replicated across two splits, not significance claims.

## Install

Python 3.12 and torch ≥ 2.4 (`torch.amp.GradScaler` does not exist before 2.4; every
reported run used 2.11). Needed to run the pipeline below, not to read the notebook.

```bash
pip install -e ".[dev]"
pytest
```

### What to expect from that `pytest`

The provenance tests in `tests/test_paths.py` shell out to `git check-ignore` and
`git ls-files`, and skip themselves when there is no repository to ask. Opening the
notebook through the Colab badge produces a real clone, so they run there; running the
suite inside an extracted `.zip` has no `.git`, so they skip. Skips in that setting are
expected, not failures.

Tests marked `version_sensitive` pin bitwise loss and latent values and therefore depend on
the exact NumPy and torch build. Deselect them with `-m 'not version_sensitive'` wherever
the build is not controlled; section 8 of the notebook does this.

On Windows, importing `torch` may additionally need the Microsoft Visual C++ 2015-2022
x64 Redistributable.

## Dataset

**LIDC-IDRI is not included here.** The images come from the
[TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri) [[3]](#references);
this project uses the preprocessed release distributed with the public reimplementation
[stefanknegt/Probabilistic-Unet-Pytorch](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) [[9]](#references),
which is the version the follow-up literature trains on: `data_lidc.pickle`, 3.4 GB,
sha256 `327025e97c296a9e02841bb7e9968521147039e9c5474ba3b214c1f8056c177e`, holding 15,096
crops of 128×128 with four independent grader masks each, drawn from 875 CT series. Place it
at `data/raw/data_lidc.pickle` and run, in order:

```bash
python scratch/inspect_data.py      # audit the pickle without importing its classes
python scratch/convert_data.py      # -> data/processed/lidc.npz + lidc.json provenance
python -m probunet.data.splits      # -> data/splits/split.json, seed 1806
```

`pickle.load` executes arbitrary code, so the pickle is read exactly once by an isolated
script under `scratch/` with a restricted unpickler; the package itself only ever reads the
`.npz`. The split is grouped by `series_uid`, so no CT series spans two splits, and
stratified over the number of non-empty grader masks; it is generated once with **seed
1806**, committed as `data/splits/split.json`, and `load_split()` never regenerates it. The
committed split is what the reported runs used, so step three reproduces rather than
replaces it. The two conversion scripts are the only part of `scratch/` that ships; the
rest of that directory is one-off analysis and is not part of the installable package.

Up to three of an image's four masks may be empty. Empty masks are signal, not noise, and
are never filtered; the number of non-empty graders is the image's **ambiguity bucket**, and
every result is broken down by it.

## Reproducing the pipeline

```bash
# 1. train: Phase 1 reproduction, Phase 2 (one flag), Phase 3 head on a frozen base
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/modernized.yaml
python scripts/train.py --config configs/extension.yaml \
    --base-checkpoint runs/baseline/checkpoints/best.pt

# 2. evaluate one checkpoint: GED at 1/4/8/16 samples, oracle / random /
#    Hungarian-matched single-sample quality, aggregate and per ambiguity bucket
python scripts/evaluate.py --checkpoint runs/baseline/checkpoints/best.pt --split test

# 3. compare arms under identical data, split, seed and budget
python scripts/compare.py --split test \
    --checkpoint baseline-short=runs/baseline-short/checkpoints/best.pt \
    --checkpoint modernized-short=runs/modernized-short/checkpoints/best.pt \
    --json results/comparison_test.json --csv results/comparison_test.csv

# 4. the selection table: head, size-prior control, random, oracle and ceiling per bucket
python scripts/consensus_headroom.py --split test \
    --head-checkpoint runs/selection-head/checkpoints/best.pt \
    --out results/consensus_selection_test.json

# 5. export what the notebook renders
python scripts/export_showcase.py      # -> data/processed/showcase.npz
python scripts/make_colab_subset.py    # -> lidc_colab_demo.npz + colab_demo_split.json
```

`--split` has no default anywhere: development happens on `val`, and `test` is evaluated
once, at the end.

## Repository layout

| path | contents |
|---|---|
| `src/probunet/` | the package — model, data, training, evaluation, extension, notebook helpers |
| `scripts/` | command-line entry points, one per stage above |
| `configs/` | one YAML per variant; a diff between two is the record of what that comparison changed |
| `tests/` | pytest suite, CPU-only, skipping cleanly without the full dataset |
| `results/` | tracked evaluation and comparison JSON/CSV — every reported number |
| `submission.ipynb` | the notebook, at the root; a narrative layer whose logic lives in the package |
| `data/splits/` | the frozen split and its notes |
| `data/processed/` | tracked exports (`showcase.npz`, `lidc_colab_demo.npz`); the 450 MB dataset is not committed |
| `runs/` | checkpoints and TensorBoard events — gitignored, never needed to read a result |
| `scratch/` | one-off conversion and analysis scripts — gitignored, never imported by the package |

## Where the numbers live

| claim | file |
|---|---|
| Phase 1 GED table and single-sample quality | `results/evaluation_test_baseline.json` |
| Phase 2 matched pair, aggregate and per bucket | `results/comparison_test.json`, `.csv` |
| each Phase 2 arm on its own | `results/evaluation_test_baseline-short.json`, `evaluation_test_modernized-short.json` |
| Phase 2's refuted mechanism — effective latent rank, validation | `results/latent_geometry_baseline_short.json`, `latent_geometry_modernized_short.json` |
| the selection head on each frozen base | `results/consensus_selection_test.json`, `consensus_selection_modernized_test.json` |
| distribution metrics unchanged by the head | `results/evaluation_test_selection-head.json` against `evaluation_test_baseline.json` |
| the headroom and ceiling pass that preceded the head | `results/consensus_headroom_baseline.json`, `consensus_ceilings_val.json` |

Validation counterparts (`*_val.json`) are the development record and are labelled as such
wherever they appear. Every file carries its own provenance: checkpoint, epoch, device,
torch version, sampling seed and the git revision of both the run and the evaluation.

## Hardware

Full-length training ran on a single RTX 3070 Ti (8 GiB). The Phase 1 budget is the paper's
240,000 iterations at batch 32 and takes days, so it does not fit in a hosted session;
section 7 of the notebook is the path that executes in one. Device selection is automatic
(cuda → mps → cpu) and logged at startup. Seeds do not reproduce across backends, so every
run in a single comparison comes from the same device, and both checkpoints and results
records name it.

## Further reading

- `DEVIATIONS.md` — every departure from the paper and the reference implementations, with
  what it costs.
- `configs/` — the flags under test; a two-line diff is the whole of Phase 2.
- `data/splits/SPLIT_NOTES.md` — known limitations of the split.

## Citing this work

```bibtex
@misc{liu_weisberg_2026_probunet_consensus,
  author       = {Liu, Chenxi and Weisberg, Ran},
  title        = {Full-Covariance Latents and a Consensus-Selection Head
                  for the Probabilistic U-Net},
  year         = {2026},
  note         = {Course project, Medical Images Processing with Deep Learning (336033)},
  howpublished = {\url{https://github.com/RanWeisberg/prob-unet-consensus-selection}}
}
```