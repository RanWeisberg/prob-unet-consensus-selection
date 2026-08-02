"""Probabilistic U-Net reimplementation with a post-hoc consensus selection head.

A PyTorch reimplementation of Kohl et al., *A Probabilistic U-Net for Segmentation
of Ambiguous Images* (NeurIPS 2018, arXiv:1806.05034), plus a consensus selection
head that scores sampled masks against the set of expert graders.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"