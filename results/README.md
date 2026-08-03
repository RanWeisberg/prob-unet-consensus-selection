# results/

> **The `comparison.json` and `comparison.csv` currently committed here are
> PLACEHOLDERS from a 2-epoch smoke run, not results.** They exist so that
> `notebooks/submission.ipynb` runs on a fresh clone before any real training has
> happened. The `note` field in the JSON says so too. Regenerate them from real
> baseline / modernized / extension runs before writing the report.

Small, **tracked** evaluation summaries. This directory is the counterpart to `runs/`:

| directory | git | contents |
|---|---|---|
| `runs/`, `experiments/` | **ignored** | checkpoints (~330 MB each), TensorBoard events |
| `results/` | **tracked** | evaluation and comparison JSON/CSV (a few KB) |

The split exists because the notebook and the report read *results*, never checkpoints.
If summaries were written into `runs/` they would be gitignored and a fresh clone — a
teammate's, a grader's, or a Colab session's — would have nothing to plot.

Produced by:

```bash
python scripts/evaluate.py --checkpoint runs/baseline/checkpoints/best.pt --split val
python scripts/compare.py --split val \
    --checkpoint baseline=runs/baseline/checkpoints/best.pt \
    --checkpoint modernized=runs/modernized/checkpoints/best.pt
```

`comparison.json` and `comparison.csv` hold every metric — aggregate and per ambiguity
bucket — for every variant plus the degenerate baselines. `notebooks/submission.ipynb`
reads `comparison.json`.

Each file records its own provenance: checkpoint path, checkpoint epoch, the git
revision of both the training run and the evaluation, the device, and the sampling seed.
Numbers from different devices are not comparable, so the device is part of the record.
