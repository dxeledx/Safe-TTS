from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, out_no_ext: Path) -> None:
    out_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_no_ext.with_suffix(".pdf"), bbox_inches="tight")


def _short_method_name(m: str) -> str:
    mapping = {
        "ea-stack-multi-safe-csp-lda": "Ours",
        "ea-stack-multi-safe-csp-lda_strict_delta": "Ours (Δ-feats)",
        "ea-stack-multi-safe-csp-lda_strict_delta_minpred002": "Ours (Δ-feats + minPred)",
        "ea-csp-lda": "EA",
        "csp-lda": "CSP-LDA",
        "fbcsp-lda": "FBCSP-LDA",
        "ea-fbcsp-lda": "EA-FBCSP",
        "riemann-mdm": "Riemann-MDM",
        "lea-csp-lda": "LEA-CSP",
        "lea-rot-csp-lda": "LEA-ROT-CSP",
        # Backward-compatible aliases.
        "rpa-csp-lda": "LEA-CSP",
        "rpa-mdm": "RPA-MDM",
        "rpa-ts-lr": "RPA-TS-LR",
        "ts-lr": "TS-LR",
        "tsa-csp-lda": "LEA-ROT-CSP",
    }
    return mapping.get(m, m)


def _load_main_table(fig_dir: Path) -> pd.DataFrame:
    p = Path(fig_dir) / "main_table.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    need = {"method", "mean_accuracy", "std_accuracy", "worst_accuracy", "mean_kappa", "std_kappa"}
    missing = need.difference(df.columns)
    if missing:
        raise RuntimeError(f"main_table.csv missing columns: {sorted(missing)}")
    return df


def _load_per_subject(fig_dir: Path) -> pd.DataFrame:
    p = Path(fig_dir) / "per_subject_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "subject" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "subject"})
        else:
            raise RuntimeError("per_subject_metrics.csv must contain 'subject' or 'Unnamed: 0' column")
    df["subject"] = df["subject"].astype(int)
    return df.sort_values("subject")


def _bar_mean_with_std(
    df: pd.DataFrame,
    *,
    metric: str,
    std_col: str,
    out_no_ext: Path,
    title: str,
    ours: str,
    baseline: str,
) -> None:
    order = df.sort_values(metric, ascending=True)["method"].astype(str).to_list()
    df = df.set_index("method").loc[order].reset_index()

    vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    std = pd.to_numeric(df[std_col], errors="coerce").to_numpy(dtype=float)
    methods = df["method"].astype(str).to_numpy()

    colors = np.array(["#9A9A9A"] * len(methods), dtype=object)
    colors[methods == baseline] = "#4C72B0"
    colors[methods == ours] = "#DD8452"

    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.barh(y, vals, xerr=std, color=colors, alpha=0.95, ecolor="k", capsize=3, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    # Annotate values at bar ends.
    x_max = float(np.nanmax(vals + std)) if np.isfinite(vals).any() else 1.0
    pad = 0.01 * x_max
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(float(v) + pad, float(yi), f"{v:.3f}", va="center", fontsize=9, alpha=0.9)

    fig.tight_layout()
    _save_fig(fig, out_no_ext)
    plt.close(fig)


def _scatter_mean_vs_worst(
    df: pd.DataFrame,
    *,
    out_no_ext: Path,
    title: str,
    ours: str,
    baseline: str,
) -> None:
    x = pd.to_numeric(df["mean_accuracy"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["worst_accuracy"], errors="coerce").to_numpy(dtype=float)
    methods = df["method"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for xi, yi, m in zip(x, y, methods):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        if m == ours:
            ax.scatter([xi], [yi], s=120, marker="*", color="#DD8452", edgecolor="k", linewidth=0.6, zorder=3)
        elif m == baseline:
            ax.scatter([xi], [yi], s=70, marker="s", color="#4C72B0", edgecolor="k", linewidth=0.4, zorder=3)
        else:
            ax.scatter([xi], [yi], s=55, marker="o", color="#9A9A9A", edgecolor="k", linewidth=0.2, zorder=2)
        ax.text(float(xi) + 0.002, float(yi) + 0.002, _short_method_name(m), fontsize=9, alpha=0.9)

    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Worst-subject accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_no_ext)
    plt.close(fig)


def _bar_neg_transfer(
    df: pd.DataFrame,
    *,
    out_no_ext: Path,
    title: str,
    ours: str,
    baseline: str,
) -> None:
    if "neg_transfer_rate_vs_baseline" not in df.columns:
        raise RuntimeError("main_table.csv missing 'neg_transfer_rate_vs_baseline'")

    df = df.copy()
    df["neg_transfer_rate_vs_baseline"] = pd.to_numeric(df["neg_transfer_rate_vs_baseline"], errors="coerce")

    order = df.sort_values("mean_accuracy", ascending=True)["method"].astype(str).to_list()
    df = df.set_index("method").loc[order].reset_index()

    methods = df["method"].astype(str).to_numpy()
    vals = df["neg_transfer_rate_vs_baseline"].to_numpy(dtype=float)

    colors = np.array(["#9A9A9A"] * len(methods), dtype=object)
    colors[methods == baseline] = "#4C72B0"
    colors[methods == ours] = "#DD8452"

    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.barh(y, vals, color=colors, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Negative transfer rate vs baseline")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(float(v) + 0.02, float(yi), f"{v:.3f}", va="center", fontsize=9, alpha=0.9)

    fig.tight_layout()
    _save_fig(fig, out_no_ext)
    plt.close(fig)


def _per_subject_scatter(
    df_subj: pd.DataFrame,
    *,
    base_col: str,
    ours_col: str,
    out_no_ext: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    subj = df_subj["subject"].astype(int).to_numpy()
    x = pd.to_numeric(df_subj[base_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_subj[ours_col], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    lo = float(np.nanmin(np.r_[x, y]))
    hi = float(np.nanmax(np.r_[x, y]))
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="k", linewidth=1, alpha=0.35)
    ax.scatter(x, y, s=55, color="#DD8452", edgecolor="k", linewidth=0.25, alpha=0.9)
    for si, xi, yi in zip(subj, x, y):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(float(xi) + 0.002, float(yi) + 0.002, str(int(si)), fontsize=9, alpha=0.85)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_no_ext)
    plt.close(fig)


def _per_subject_delta_bar(
    df_subj: pd.DataFrame,
    *,
    base_col: str,
    ours_col: str,
    out_no_ext: Path,
    title: str,
    ylabel: str,
) -> None:
    subj = df_subj["subject"].astype(int).to_numpy()
    base = pd.to_numeric(df_subj[base_col], errors="coerce").to_numpy(dtype=float)
    ours = pd.to_numeric(df_subj[ours_col], errors="coerce").to_numpy(dtype=float)
    delta = ours - base

    colors = np.where(delta > 0, "#DD8452", np.where(delta < 0, "#4C72B0", "#9A9A9A"))

    x = np.arange(len(subj))
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.bar(x, delta, color=colors, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in subj])
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_no_ext)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate paper-ready comparison figures from a main_table.csv + per_subject_metrics.csv folder."
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        required=True,
        help="Folder containing main_table.csv and per_subject_metrics.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: --fig-dir)")
    ap.add_argument("--prefix", type=str, default="loso4_strong_baselines_v1")
    ap.add_argument("--ours", type=str, default="ea-stack-multi-safe-csp-lda")
    ap.add_argument("--baseline", type=str, default="ea-csp-lda")
    ap.add_argument(
        "--task-label",
        type=str,
        default="LOSO-4class",
        help="Label used in figure titles (e.g., 'BNCI2014_001 LOSO-4class').",
    )
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else fig_dir
    _ensure_dir(out_dir)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    main_table = _load_main_table(fig_dir)
    subj = _load_per_subject(fig_dir)

    ours = str(args.ours)
    baseline = str(args.baseline)
    prefix = str(args.prefix)
    task = str(args.task_label)

    _bar_mean_with_std(
        main_table,
        metric="mean_accuracy",
        std_col="std_accuracy",
        out_no_ext=out_dir / f"{prefix}_bar_mean_accuracy",
        title=f"{task}: mean accuracy (±std across subjects)",
        ours=ours,
        baseline=baseline,
    )
    _bar_mean_with_std(
        main_table,
        metric="mean_kappa",
        std_col="std_kappa",
        out_no_ext=out_dir / f"{prefix}_bar_mean_kappa",
        title=f"{task}: mean kappa (±std across subjects)",
        ours=ours,
        baseline=baseline,
    )
    _scatter_mean_vs_worst(
        main_table,
        out_no_ext=out_dir / f"{prefix}_scatter_mean_vs_worst",
        title=f"{task}: mean vs worst-subject accuracy",
        ours=ours,
        baseline=baseline,
    )
    _bar_neg_transfer(
        main_table,
        out_no_ext=out_dir / f"{prefix}_bar_neg_transfer_rate",
        title=f"{task}: negative transfer rate vs EA (lower is better)",
        ours=ours,
        baseline=baseline,
    )

    base_acc = f"acc_{baseline}"
    ours_acc = f"acc_{ours}"
    if base_acc not in subj.columns or ours_acc not in subj.columns:
        raise RuntimeError(f"per_subject_metrics.csv missing columns: {base_acc} / {ours_acc}")
    delta_acc = (subj[ours_acc].astype(float) - subj[base_acc].astype(float)).to_numpy()
    n_neg = int((delta_acc < 0).sum())
    n_pos = int((delta_acc > 0).sum())
    n_zero = int((np.isclose(delta_acc, 0)).sum())
    mean_delta = float(np.mean(delta_acc))

    _per_subject_scatter(
        subj,
        base_col=base_acc,
        ours_col=ours_acc,
        out_no_ext=out_dir / f"{prefix}_scatter_subject_acc_{baseline}_vs_{ours}",
        title=f"Per-subject accuracy: {baseline} vs {ours} (Δmean={mean_delta:+.3f}, +{n_pos}/0{n_zero}/-{n_neg})",
        xlabel=f"{baseline} accuracy",
        ylabel=f"{ours} accuracy",
    )
    _per_subject_delta_bar(
        subj,
        base_col=base_acc,
        ours_col=ours_acc,
        out_no_ext=out_dir / f"{prefix}_bar_subject_delta_acc_{ours}_minus_{baseline}",
        title=f"Per-subject Δaccuracy: {ours} − {baseline} (Δmean={mean_delta:+.3f})",
        ylabel="Δaccuracy",
    )

    base_k = f"kappa_{baseline}"
    ours_k = f"kappa_{ours}"
    if base_k in subj.columns and ours_k in subj.columns:
        delta_k = (subj[ours_k].astype(float) - subj[base_k].astype(float)).to_numpy()
        n_neg_k = int((delta_k < 0).sum())
        n_pos_k = int((delta_k > 0).sum())
        n_zero_k = int((np.isclose(delta_k, 0)).sum())
        mean_delta_k = float(np.mean(delta_k))

        _per_subject_scatter(
            subj,
            base_col=base_k,
            ours_col=ours_k,
            out_no_ext=out_dir / f"{prefix}_scatter_subject_kappa_{baseline}_vs_{ours}",
            title=f"Per-subject kappa: {baseline} vs {ours} (Δmean={mean_delta_k:+.3f}, +{n_pos_k}/0{n_zero_k}/-{n_neg_k})",
            xlabel=f"{baseline} kappa",
            ylabel=f"{ours} kappa",
        )
        _per_subject_delta_bar(
            subj,
            base_col=base_k,
            ours_col=ours_k,
            out_no_ext=out_dir / f"{prefix}_bar_subject_delta_kappa_{ours}_minus_{baseline}",
            title=f"Per-subject Δkappa: {ours} − {baseline} (Δmean={mean_delta_k:+.3f})",
            ylabel="Δkappa",
        )


if __name__ == "__main__":
    main()
