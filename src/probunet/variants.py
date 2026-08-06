"""The interface every model variant satisfies.

Three variants are planned and they are **one model implementation**, not three:

* **baseline** -- the faithful reimplementation.
* **modernized** -- the same model with post-2018 improvements behind config flags.
* **extension** -- a consensus-selection head wrapping a *frozen* base model.

Baseline and modernized differ only by :class:`~probunet.model.prob_unet.ProbUNetConfig`
flags, so there is nothing to duplicate; duplicating them into per-variant packages
would let the copies drift and would invalidate the claim that a comparison isolates one
change.

The protocol has two methods because that is the entire difference between the variants
at evaluation time:

* :meth:`SegmentationVariant.sample` -- draw ``n`` segmentations for a batch of images.
* :meth:`SegmentationVariant.select` -- choose one of them **without ground truth**, or
  return None if the variant cannot. Only the extension returns an index.

Evaluation therefore needs exactly one conditional -- "did ``select`` return something?"
-- rather than per-variant branching scattered through the metric code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from probunet.extension.head import SelectionHead
from probunet.model.prob_unet import ProbUNet
from probunet.training.diagnostics import logits_to_mask, reparameterize


@runtime_checkable
class SegmentationVariant(Protocol):
    """A model that can draw segmentation samples and optionally select one.

    Shapes are batched throughout: the architecture's efficiency argument is that one
    U-Net pass serves a whole batch and every extra sample re-runs only ``f_comb``, and
    an unbatched interface would throw that away. A single image is just ``B = 1``.
    """

    name: str

    def sample(self, image: Tensor, n_samples: int) -> Tensor:
        """Draw ``n_samples`` hard masks per image.

        Args:
            image: Image batch of shape ``(B, C, H, W)``.
            n_samples: Samples per image.

        Returns:
            A uint8 mask tensor of shape ``(B, n_samples, H, W)``.
        """
        ...

    def select(self, samples: Tensor, image: Tensor) -> Tensor | None:
        """Choose one sample per image, without access to ground truth.

        Args:
            samples: Masks of shape ``(B, n_samples, H, W)``.
            image: The images the samples came from, shape ``(B, C, H, W)``.

        Returns:
            Chosen indices of shape ``(B,)``, or None if this variant does not select.
        """
        ...


class ProbUNetVariant:
    """Adapter exposing a :class:`ProbUNet` through :class:`SegmentationVariant`.

    Serves both the baseline and the modernized variant: they are the same class with
    different config flags. :meth:`select` returns None, because a plain Probabilistic
    U-Net has no principled way to pick one of its samples -- which is the gap the
    extension exists to fill.
    """

    def __init__(
        self,
        model: ProbUNet,
        name: str = "probunet",
        generator: torch.Generator | None = None,
    ) -> None:
        """Wrap a model.

        Args:
            model: The model, already on the target device.
            name: Label used in reports.
            generator: Optional CPU generator for reproducible sampling noise.
        """
        self.model = model
        self.name = name
        self.generator = generator

    @torch.no_grad()
    def sample(self, image: Tensor, n_samples: int) -> Tensor:
        """Draw samples from the prior, re-running only ``f_comb`` per sample.

        Args:
            image: Image batch of shape ``(B, C, H, W)``.
            n_samples: Samples per image.

        Returns:
            A uint8 mask tensor of shape ``(B, n_samples, H, W)``.

        Raises:
            ValueError: If ``n_samples`` is not positive.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        encoded = self.model.encode(image)
        return torch.stack(
            [
                logits_to_mask(
                    self.model.reconstruct(encoded, reparameterize(encoded.prior, self.generator))
                )
                for _ in range(n_samples)
            ],
            dim=1,
        )

    def select(self, samples: Tensor, image: Tensor) -> Tensor | None:
        """Return None: this variant does not select a single sample.

        Args:
            samples: Unused.
            image: Unused.

        Returns:
            None.
        """
        return None


class SelectionHeadVariant(ProbUNetVariant):
    """The Phase 3 extension: a frozen base plus a head that picks one sample.

    The **only** variant whose :meth:`select` returns an index, which is what gives
    evaluation exactly one variant-dependent branch instead of per-variant conditionals.

    Sampling is inherited unchanged from :class:`ProbUNetVariant` and runs on the frozen
    base, so **attaching the head cannot move any distribution metric**. That is not merely
    intended: with the same generator seed this variant's samples are bit-identical to the
    plain variant's, and ``test_ged_is_bit_identical_with_and_without_the_head`` asserts
    the GED numbers agree exactly. "Distribution metrics unchanged, single-sample quality
    improved" is the claim the extension makes, and its first half is a property of this
    class rather than a hope about training.
    """

    def __init__(
        self,
        head: SelectionHead,
        name: str = "extension",
        generator: torch.Generator | None = None,
        by_area: bool = False,
    ) -> None:
        """Wrap a trained head.

        Args:
            head: The trained :class:`~probunet.extension.head.SelectionHead`.
            name: Label used in reports.
            generator: Optional CPU generator for reproducible sampling noise.
            by_area: Select with the **size-prior control** instead of the real scorer.
                Lets the control be evaluated through exactly the same path as the head, so
                the two columns of the results table differ only in which scorer chose.
        """
        super().__init__(head.base, name=name, generator=generator)
        self.head = head
        self.by_area = by_area

    @torch.no_grad()
    def select(self, samples: Tensor, image: Tensor) -> Tensor:
        """Choose one sample per image, **without ground truth**.

        Args:
            samples: Candidate masks of shape ``(B, n, H, W)``.
            image: The images the samples came from, shape ``(B, C, H, W)``.

        Returns:
            Chosen indices of shape ``(B,)``.
        """
        if self.by_area:
            # The control sees the candidate's area and nothing else -- not even the image.
            return self.head.select_by_area(samples)
        return self.head.select(self.head.encode_base(image), samples)
