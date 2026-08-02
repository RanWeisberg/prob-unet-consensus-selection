"""Dataset loading, splitting and augmentation for the preprocessed LIDC-IDRI data.

Import submodules directly, e.g. ``from probunet.data.splits import load_split``.
Nothing is re-exported here on purpose: an eager re-export makes
``python -m probunet.data.splits`` emit a double-import RuntimeWarning.
"""