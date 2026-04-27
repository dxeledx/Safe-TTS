from __future__ import annotations

from typing import Sequence

import numpy as np


def reorder_proba_columns(proba: np.ndarray, model_classes: Sequence[str], class_order: Sequence[str]) -> np.ndarray:
    """Reorder `predict_proba` outputs to match a desired `class_order`."""

    model_classes = list(model_classes)
    indices = []
    for c in class_order:
        if c not in model_classes:
            raise ValueError(f"Class '{c}' not found in model classes {model_classes}.")
        indices.append(model_classes.index(c))
    return np.asarray(proba)[:, indices]

