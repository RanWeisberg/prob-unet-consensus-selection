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
