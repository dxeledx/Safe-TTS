from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _load_candidates(run_dir: Path, method: str) -> pd.DataFrame:
    diag_root = Path(run_dir) / "diagnostics" / method
    cand_paths = sorted(diag_root.glob("subject_*/candidates.csv"))
    if not cand_paths:
        raise RuntimeError(f"No candidates.csv under {diag_root}")

    parts = []
    for p in cand_paths:
        m = re.search(r"subject_(\d+)", str(p))
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p)
        df.insert(0, "subject", subj)
        parts.append(df)
    if not parts:
        raise RuntimeError(f"No readable candidates.csv under {diag_root}")
    return pd.concat(parts, axis=0, ignore_index=True)


def _per_subject_summary(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, df in cand.groupby("subject"):
        df = df.copy()
        id_rows = df[df["kind"] == "identity"]
        if id_rows.empty:
            continue
        id_acc = float(id_rows["accuracy"].iloc[0])
        oracle_acc = float(df["accuracy"].astype(float).max())

        sel_rows = df[df.get("is_selected", 0).astype(int) == 1]
        if sel_rows.empty:
            sel_rows = id_rows
        sel = sel_rows.iloc[0]
        sel_acc = float(sel["accuracy"])
        sel_family = str(sel.get("cand_family", "ea"))

        rows.append(
            {
                "subject": int(subj),
                "id_acc": id_acc,
                "sel_acc": sel_acc,
                "oracle_acc": oracle_acc,
                "delta": sel_acc - id_acc,
                "oracle_gap": oracle_acc - sel_acc,
                "sel_family": sel_family,
            }
        )
    return pd.DataFrame(rows).sort_values("subject")


def _plot_delta(summary: pd.DataFrame, *, out: Path, title: str) -> None:
    fam_colors = {
        "ea": "#9A9A9A",
        "rpa": "#4C72B0",
        "tsa": "#8172B3",
        "chan": "#DD8452",
        "fbcsp": "#C44E52",
        "other": "#55A868",
    }
    subjects = summary["subject"].astype(int).to_numpy()
    delta = summary["delta"].astype(float).to_numpy()
    fam = summary["sel_family"].astype(str).str.lower().to_numpy()
    colors = [fam_colors.get(f, fam_colors["other"]) for f in fam]

    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.bar(x, delta, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in subjects])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Δacc (selected − EA)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_oracle_gap(summary: pd.DataFrame, *, out: Path, title: str) -> None:
    subjects = summary["subject"].astype(int).to_numpy()
    gap = summary["oracle_gap"].astype(float).to_numpy()
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.bar(x, gap, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in subjects])
    ax.set_xlabel("Subject")
    ax.set_ylabel("oracle_acc − selected_acc")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_headroom(summary: pd.DataFrame, *, out: Path, title: str) -> None:
    subjects = summary["subject"].astype(int).to_numpy()
    id_acc = summary["id_acc"].astype(float).to_numpy()
    sel_acc = summary["sel_acc"].astype(float).to_numpy()
    oracle_acc = summary["oracle_acc"].astype(float).to_numpy()

    headroom = oracle_acc - id_acc
    eaten = sel_acc - id_acc
    eaten = np.clip(eaten, 0.0, None)

    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, headroom, color="#4C72B0", alpha=0.35, label="headroom (oracle − EA)")
    ax.bar(x, eaten, color="#DD8452", alpha=0.9, label="eaten (selected − EA)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in subjects])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Δacc")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_true(
    cand: pd.DataFrame,
    *,
    pred_col: str,
    out: Path,
    title: str,
    xlabel: str,
) -> None:
    # Compute true improvement per candidate relative to identity in the same subject.
    df = cand.copy()
    df["accuracy"] = df["accuracy"].astype(float)
    id_acc = (
        df[df["kind"] == "identity"]
        .groupby("subject")["accuracy"]
        .first()
        .rename("id_acc")
        .reset_index()
    )
    df = df.merge(id_acc, on="subject", how="left")
    df["true_improve"] = df["accuracy"] - df["id_acc"]
    df[pred_col] = pd.to_numeric(df.get(pred_col, np.nan), errors="coerce")

    x = df[pred_col].to_numpy(dtype=float)
    y = df["true_improve"].to_numpy(dtype=float)
    fam = df.get("cand_family", "other").astype(str).str.lower().to_numpy()

    fam_colors = {
        "ea": "#9A9A9A",
        "rpa": "#4C72B0",
        "tsa": "#8172B3",
        "chan": "#DD8452",
        "fbcsp": "#C44E52",
        "other": "#55A868",
    }
    colors = [fam_colors.get(f, fam_colors["other"]) for f in fam]

    rho = _spearman(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.4)
    ax.axvline(0.0, color="k", linewidth=1, alpha=0.25)
    ax.scatter(x, y, s=18, alpha=0.7, c=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("True Δacc vs EA")
    ax.set_title(f"{title} (Spearman={rho:.3f})")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot candidate-set diagnostics from diagnostics/*/subject_XX/candidates.csv.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = _load_candidates(Path(args.run_dir), method=str(args.method))
    summ = _per_subject_summary(cand)

    def _has_finite(col: str) -> bool:
        vals = pd.to_numeric(cand.get(col, np.nan), errors="coerce").to_numpy(dtype=float)
        return bool(np.isfinite(vals).any())

    _plot_delta(
        summ,
        out=out_dir / f"{args.prefix}_delta.png",
        title=f"{args.prefix}: Δacc (selected − EA)",
    )
    _plot_oracle_gap(
        summ,
        out=out_dir / f"{args.prefix}_oracle_gap.png",
        title=f"{args.prefix}: oracle gap (oracle − selected)",
    )
    _plot_headroom(
        summ,
        out=out_dir / f"{args.prefix}_headroom.png",
        title=f"{args.prefix}: headroom vs eaten",
    )

    if "ridge_pred_improve" in cand.columns and _has_finite("ridge_pred_improve"):
        _plot_pred_vs_true(
            cand,
            pred_col="ridge_pred_improve",
            out=out_dir / f"{args.prefix}_ridge_vs_true.png",
            title=f"{args.prefix}: ridge_pred_improve vs true Δacc",
            xlabel="ridge_pred_improve",
        )
    if "guard_p_pos" in cand.columns and _has_finite("guard_p_pos"):
        _plot_pred_vs_true(
            cand,
            pred_col="guard_p_pos",
            out=out_dir / f"{args.prefix}_guard_vs_true.png",
            title=f"{args.prefix}: guard_p_pos vs true Δacc",
            xlabel="guard_p_pos",
        )
    if "bandit_score" in cand.columns and _has_finite("bandit_score"):
        _plot_pred_vs_true(
            cand,
            pred_col="bandit_score",
            out=out_dir / f"{args.prefix}_bandit_vs_true.png",
            title=f"{args.prefix}: bandit_score vs true Δacc",
            xlabel="bandit_score",
        )
    if "probe_mixup_hard_best" in cand.columns and _has_finite("probe_mixup_hard_best"):
        cand2 = cand.copy()
        cand2["neg_probe_mixup_hard_best"] = -pd.to_numeric(cand2["probe_mixup_hard_best"], errors="coerce")
        _plot_pred_vs_true(
            cand2,
            pred_col="neg_probe_mixup_hard_best",
            out=out_dir / f"{args.prefix}_neg_probe_hard_vs_true.png",
            title=f"{args.prefix}: -probe_mixup_hard_best vs true Δacc",
            xlabel="-probe_mixup_hard_best",
        )
    if "evidence_nll_best" in cand.columns and _has_finite("evidence_nll_best"):
        cand2 = cand.copy()
        cand2["neg_evidence_nll_best"] = -pd.to_numeric(cand2["evidence_nll_best"], errors="coerce")
        _plot_pred_vs_true(
            cand2,
            pred_col="neg_evidence_nll_best",
            out=out_dir / f"{args.prefix}_neg_evidence_vs_true.png",
            title=f"{args.prefix}: -evidence_nll_best vs true Δacc",
            xlabel="-evidence_nll_best",
        )
    if "iwcv_ucb" in cand.columns and _has_finite("iwcv_ucb"):
        cand2 = cand.copy()
        cand2["neg_iwcv_ucb"] = -pd.to_numeric(cand2["iwcv_ucb"], errors="coerce")
        _plot_pred_vs_true(
            cand2,
            pred_col="neg_iwcv_ucb",
            out=out_dir / f"{args.prefix}_neg_iwcv_ucb_vs_true.png",
            title=f"{args.prefix}: -iwcv_ucb vs true Δacc",
            xlabel="-iwcv_ucb",
        )
    if "iwcv_nll" in cand.columns and _has_finite("iwcv_nll"):
        cand2 = cand.copy()
        cand2["neg_iwcv_nll"] = -pd.to_numeric(cand2["iwcv_nll"], errors="coerce")
        _plot_pred_vs_true(
            cand2,
            pred_col="neg_iwcv_nll",
            out=out_dir / f"{args.prefix}_neg_iwcv_nll_vs_true.png",
            title=f"{args.prefix}: -iwcv_nll vs true Δacc",
            xlabel="-iwcv_nll",
        )


if __name__ == "__main__":
    main()
