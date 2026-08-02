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

## Layout

```
src/probunet/          the package (all real logic lives here)
scratch/               isolated one-off scripts, gitignored, never imported
tests/                 pytest suite
data/raw/              the source pickle (gitignored)
data/processed/        converted .npz (gitignored) + lidc.json schema sidecar
data/splits/split.json the fixed split (tracked)
```