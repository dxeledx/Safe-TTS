from __future__ import annotations

from datetime import date
import importlib.metadata as im
from pathlib import Path
import subprocess
from typing import Sequence

import pandas as pd

from .config import ExperimentConfig
from .metrics import summarize_results


def today_yyyymmdd() -> str:
    return date.today().strftime("%Y%m%d")


def _pkg_version(name: str) -> str:
    try:
        return im.version(name)
    except im.PackageNotFoundError:
        return "not-installed"

def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def write_results_txt(
    results_df: pd.DataFrame,
    *,
    config: ExperimentConfig,
    output_path: Path,
    metric_columns: Sequence[str],
    overall_metrics: dict | None = None,
    protocol_name: str = "LOSO",
    command_line: str | None = None,
) -> None:
    summary_df = summarize_results(results_df, metric_columns=metric_columns)

    lines = []
    lines.append(f"Date: {today_yyyymmdd()}")
    commit = _git_commit()
    if commit:
        lines.append(f"Git commit: {commit}")
    if command_line:
        lines.append(f"Command: {command_line}")
    lines.append("")
    lines.append("=== Experiment Config ===")
    lines.append(f"Dataset: {config.dataset}")
    lines.append(_format_preprocessing(config))
    lines.append(f"Model: CSP(n_components={config.model.csp_n_components}) + LDA(default)")
    lines.append(f"Metrics average: {config.metrics_average}")
    lines.append("")
    lines.append("=== Package Versions ===")
    lines.append(f"moabb: {_pkg_version('moabb')}")
    lines.append(f"braindecode: {_pkg_version('braindecode')}")
    lines.append(f"mne: {_pkg_version('mne')}")
    lines.append(f"scikit-learn: {_pkg_version('scikit-learn')}")
    lines.append("")

    lines.append(f"=== Per-Subject ({protocol_name}) Results ===")
    lines.append(results_df.to_string(index=False))
    lines.append("")

    lines.append("=== Summary (across subjects) ===")
    lines.append(summary_df.to_string())
    lines.append("")

    if overall_metrics is not None:
        lines.append("=== Overall (all test trials concatenated) ===")
        for k in metric_columns:
            if k in overall_metrics:
                lines.append(f"{k}: {overall_metrics[k]:.6f}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_results_txt_multi(
    results_by_method: dict[str, pd.DataFrame],
    *,
    config: ExperimentConfig,
    output_path: Path,
    metric_columns: Sequence[str],
    overall_metrics_by_method: dict[str, dict[str, float]] | None = None,
    method_details_by_method: dict[str, str] | None = None,
    protocol_name: str = "LOSO",
    command_line: str | None = None,
) -> None:
    lines = []
    lines.append(f"Date: {today_yyyymmdd()}")
    commit = _git_commit()
    if commit:
        lines.append(f"Git commit: {commit}")
    if command_line:
        lines.append(f"Command: {command_line}")
    lines.append("")
    lines.append("=== Experiment Config ===")
    lines.append(f"Dataset: {config.dataset}")
    lines.append(_format_preprocessing(config))
    lines.append(f"Model: CSP(n_components={config.model.csp_n_components}) + LDA(default)")
    lines.append(f"Metrics average: {config.metrics_average}")
    lines.append("")
    lines.append("=== Package Versions ===")
    lines.append(f"moabb: {_pkg_version('moabb')}")
    lines.append(f"braindecode: {_pkg_version('braindecode')}")
    lines.append(f"mne: {_pkg_version('mne')}")
    lines.append(f"scikit-learn: {_pkg_version('scikit-learn')}")
    lines.append("")

    if method_details_by_method:
        lines.append("=== Methods ===")
        for name in sorted(method_details_by_method.keys()):
            lines.append(f"{name}: {method_details_by_method[name]}")
        lines.append("")

    for method_name in sorted(results_by_method.keys()):
        df = results_by_method[method_name]
        lines.append(f"=== Method: {method_name} ===")
        lines.append(df.to_string(index=False))
        lines.append("")
        lines.append(f"Summary (across subjects, {protocol_name}):")
        lines.append(summarize_results(df, metric_columns=metric_columns).to_string())
        lines.append("")
        if overall_metrics_by_method is not None and method_name in overall_metrics_by_method:
            lines.append("Overall (all test trials concatenated):")
            overall = overall_metrics_by_method[method_name]
            for k in metric_columns:
                if k in overall:
                    lines.append(f"{k}: {overall[k]:.6f}")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_preprocessing(config: ExperimentConfig) -> str:
    sessions = list(getattr(config.preprocessing, "sessions", [])) or "ALL"
    preprocess = getattr(config.preprocessing, "preprocess", "moabb")
    car = bool(getattr(config.preprocessing, "car", False))
    base = (
        "Preprocessing: "
        f"mode={preprocess}, "
        f"bandpass {config.preprocessing.fmin}-{config.preprocessing.fmax} Hz, "
        f"resample {config.preprocessing.resample} Hz, "
        f"epoch tmin={config.preprocessing.tmin}s, tmax={config.preprocessing.tmax}s, "
        f"events={list(config.preprocessing.events)}, "
        f"sessions={sessions}"
    )
    if car:
        base += ", CAR=True"
    if preprocess == "paper_fir":
        base += (
            f", FIR(order={getattr(config.preprocessing, 'paper_fir_order', 50)}, "
            f"window={getattr(config.preprocessing, 'paper_fir_window', 'hamming')}, causal=True)"
        )
    return base
