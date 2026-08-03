"""The consensus-selection head: Phase 3, not implemented yet.

**Scaffold only.** This package will hold the head that scores each sampled mask by how
well it agrees with the set of expert graders, so one sample can be chosen at inference
time without ground truth -- a learned surrogate for the oracle (best-of-N), which needs
ground truth and is therefore unusable in practice.

What is already in place elsewhere, so that this package stays small:

* :class:`probunet.variants.SegmentationVariant` -- the interface the head's variant will
  satisfy. It returns an index from ``select``; the plain model returns None.
* :func:`probunet.training.freeze.freeze_module` -- the freeze contract, asserted and
  logged. The head is trained on a **frozen** base model: it must not alter the
  generative model, the prior/posterior dynamics or the GED behaviour, since the claim
  is *distribution metrics unchanged, single-sample quality improved*.
* ``train.mode = "selection_head"`` in :class:`probunet.training.config.TrainConfig`,
  which requires ``--base-checkpoint``, loads it, freezes it, and then raises because the
  head does not exist.
* :func:`probunet.evaluation.metrics.selected_sample_dice` -- how a selected sample is
  scored, already used by the emptiest-sample baseline the head must beat.

Two design points recorded now because they shape what goes here:

* The scoring target is **still open** (mean IoU across graders, count of graders above a
  threshold, median, or minimum). It must stay configurable;
  :data:`probunet.evaluation.metrics.AGGREGATIONS` is where the options live.
* Training the head on posterior samples alone would fail: posterior samples are almost
  always good, so the head would never see a bad mask and would learn to output a
  constant. It must train on **prior** samples too.
"""
