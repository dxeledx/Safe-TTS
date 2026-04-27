from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.metrics import ConfusionMatrixDisplay


def plot_csp_patterns(
    csp: CSP,
    info,
    *,
    output_path: Path,
    title: str = "CSP Patterns",
    dpi: int = 300,
) -> None:
    fig = csp.plot_patterns(info, components=range(csp.n_components), show=False)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=list(labels),
        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(
    results_df: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    output_path: Path,
    title: str = "LOSO Metrics (mean ± std)",
    dpi: int = 300,
) -> None:
    means = results_df[list(metric_columns)].mean(numeric_only=True).to_numpy()
    stds = results_df[list(metric_columns)].std(numeric_only=True).to_numpy()

    x = np.arange(len(metric_columns))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metric_columns), rotation=0)
    # Some metrics (e.g., kappa) can be negative in edge cases.
    ymin = float(min(0.0, np.min(means - stds)))
    ax.set_ylim(ymin, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison_bar(
    results_by_method: Dict[str, pd.DataFrame],
    *,
    metric: str,
    output_path: Path,
    title: str = "Model comparison by subject",
    dpi: int = 300,
) -> None:
    methods = list(results_by_method.keys())
    if not methods:
        raise ValueError("results_by_method is empty.")

    # Determine subject order from the first method.
    first_df = results_by_method[methods[0]].sort_values("subject")
    subjects = first_df["subject"].to_numpy()

    x = np.arange(len(subjects))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, method in enumerate(methods):
        df = results_by_method[method].set_index("subject").loc[subjects]
        values = df[metric].to_numpy()
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects])
    ax.set_xlabel("Subject")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_class_marginal_trajectory(
    p_bars: np.ndarray,
    *,
    class_order: Sequence[str],
    output_path: Path,
    x: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    title: str = "Class-marginal trajectory",
    dpi: int = 300,
) -> None:
    p_bars = np.asarray(p_bars, dtype=np.float64)
    if p_bars.ndim != 2:
        raise ValueError(f"Expected p_bars shape (n_steps,n_classes); got {p_bars.shape}.")
    n_steps, n_classes = p_bars.shape
    if len(class_order) != n_classes:
        raise ValueError("class_order length mismatch with p_bars.")

    if x is None:
        x = np.arange(n_steps)
    x = np.asarray(x)
    if x.shape[0] != n_steps:
        raise ValueError("x length mismatch with p_bars.")

    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name in enumerate(class_order):
        ax.plot(x, p_bars[:, k], marker="o", linewidth=1.5, markersize=3, label=str(name))

    if prior is not None:
        prior = np.asarray(prior, dtype=np.float64).reshape(-1)
        if prior.shape[0] != n_classes:
            raise ValueError("prior length mismatch with p_bars.")
        for k, name in enumerate(class_order):
            ax.hlines(
                prior[k],
                xmin=float(np.min(x)),
                xmax=float(np.max(x)),
                colors="k",
                linestyles="dashed",
                linewidth=0.8,
                alpha=0.35,
            )

    ax.set_xlabel("Candidate order / iteration")
    ax.set_ylabel("p̄ (mean predicted probability)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=min(4, n_classes), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_objective_vs_accuracy_scatter(
    objectives: np.ndarray,
    accuracies: np.ndarray,
    *,
    output_path: Path,
    title: str = "Objective vs accuracy",
    dpi: int = 300,
) -> None:
    objectives = np.asarray(objectives, dtype=np.float64).reshape(-1)
    accuracies = np.asarray(accuracies, dtype=np.float64).reshape(-1)
    if objectives.shape[0] != accuracies.shape[0]:
        raise ValueError("objectives/accuracies length mismatch.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(objectives, accuracies, s=20, alpha=0.75)
    ax.set_xlabel("Unlabeled objective (lower is better)")
    ax.set_ylabel("True accuracy (higher is better)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
