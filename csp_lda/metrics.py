from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    class_order: Sequence[str],
    average: str = "macro",
) -> Dict[str, float]:
    """Compute common classification metrics for EEG MI experiments."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(class_order),
        average=average,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=list(class_order))

    n_classes = len(class_order)
    if n_classes == 2:
        # For binary, sklearn expects either:
        # - y_true as 1D {0,1} and y_score as (n_samples,), or
        # - y_true as (n_samples,1) and y_score as (n_samples,)
        # Use the 2nd class in `class_order` as positive label.
        y_true_bin = (y_true == class_order[1]).astype(int)
        auc = roc_auc_score(y_true_bin, y_proba[:, 1])
    else:
        # Multiclass AUC (macro OVR).
        y_true_bin = label_binarize(y_true, classes=list(class_order))
        auc = roc_auc_score(
            y_true_bin,
            y_proba,
            average=average,
            multi_class="ovr",
        )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "kappa": float(kappa),
    }


def summarize_results(results_df: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    """Return mean/std/min/max summary for selected metric columns."""

    summary = {
        "mean": results_df[metric_columns].mean(numeric_only=True),
        "std": results_df[metric_columns].std(numeric_only=True),
        "min": results_df[metric_columns].min(numeric_only=True),
        "max": results_df[metric_columns].max(numeric_only=True),
    }
    return pd.DataFrame(summary)

