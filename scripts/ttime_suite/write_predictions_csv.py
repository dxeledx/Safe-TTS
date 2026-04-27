from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_class_order(path: Path) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError(f"Invalid class_order JSON at {path}")
    return [str(x) for x in obj]


def _ints_to_labels(y_int: np.ndarray, *, class_order: Sequence[str]) -> np.ndarray:
    order = [str(c) for c in class_order]
    return np.asarray([order[int(i)] for i in np.asarray(y_int, dtype=int).reshape(-1)], dtype=object)


def build_predictions_df(
    *,
    method: str,
    subject: int,
    y_true_int: np.ndarray,
    proba: np.ndarray,
    class_order: Sequence[str],
    trial: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build SAFE_TTA-compatible per-trial prediction table.

    Output columns follow existing SAFE_TTA prediction exports:
      method, subject, trial, y_true, y_pred, proba_<class1>, ...
    """

    y_true_int = np.asarray(y_true_int, dtype=int).reshape(-1)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2D (n_trials,n_classes); got {proba.shape}")
    if proba.shape[0] != y_true_int.shape[0]:
        raise ValueError("y_true_int and proba must have same n_trials")
    n_classes = int(proba.shape[1])
    if n_classes != len(class_order):
        raise ValueError(f"proba has {n_classes} classes but class_order has {len(class_order)}")

    y_true = _ints_to_labels(y_true_int, class_order=class_order)
    y_pred_int = np.argmax(proba, axis=1).astype(int)
    y_pred = _ints_to_labels(y_pred_int, class_order=class_order)

    if trial is None:
        trial = np.arange(int(len(y_true_int)), dtype=int)
    trial = np.asarray(trial, dtype=int).reshape(-1)
    if trial.shape[0] != y_true_int.shape[0]:
        raise ValueError("trial must have same n_trials")

    df = pd.DataFrame(
        {
            "method": str(method),
            "subject": int(subject),
            "trial": trial,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    for i, c in enumerate(list(class_order)):
        df[f"proba_{c}"] = proba[:, int(i)]
    return df


def write_predictions_csv(
    *,
    out_csv: Path,
    method: str,
    subject: int,
    y_true_int: np.ndarray,
    proba: np.ndarray,
    class_order: Sequence[str],
    trial: np.ndarray | None = None,
) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = build_predictions_df(
        method=method,
        subject=subject,
        y_true_int=y_true_int,
        proba=proba,
        class_order=class_order,
        trial=trial,
    )
    df.to_csv(out_csv, index=False)

