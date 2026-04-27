from __future__ import annotations

"""Analyze whether certificate/guard scores contain actionable information.

Inputs
------
This script consumes the diagnostics emitted by:
  scripts/offline_safe_tta_multi_select_crc_from_predictions.py

Specifically, it expects:
  <run_dir>/diagnostics/<method_name>/subject_XX/candidates.csv

The candidates.csv is assumed to contain, at minimum:
  - subject, kind in {identity,candidate}, cand_key, cand_family
  - accuracy
  - ridge_pred_improve (certificate prediction)
  - guard_p_pos (guard probability)
  - accept (candidate passes guard+threshold for this subject under the calibrated policy)

Outputs
-------
In out_dir:
  - cert_pred_vs_true_scatter.png
  - guard_roc.png
  - guard_reliability.png
  - metrics.csv
Optionally (when --tag is provided):
  - accepted_vs_rejected_scatter_<tag>.png
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ece_binary(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 0.0, 1.0)
    y = y[mask]
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(in_bin):
            continue
        w = float(np.mean(in_bin))
        acc = float(np.mean(y[in_bin]))
        conf = float(np.mean(p[in_bin]))
        ece += w * abs(acc - conf)
    return float(ece)


def _load_candidates(run_dir: Path, method: str) -> pd.DataFrame:
    diag_root = Path(run_dir) / "diagnostics" / str(method)
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
        if "subject" not in df.columns:
            df.insert(0, "subject", subj)
        parts.append(df)
    if not parts:
        raise RuntimeError(f"No readable candidates.csv under {diag_root}")
    return pd.concat(parts, axis=0, ignore_index=True)


def _prepare_candidate_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kind"] = df["kind"].astype(str)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["ridge_pred_improve"] = pd.to_numeric(df.get("ridge_pred_improve", np.nan), errors="coerce")
    df["guard_p_pos"] = pd.to_numeric(df.get("guard_p_pos", np.nan), errors="coerce")
    df["accept"] = pd.to_numeric(df.get("accept", 0), errors="coerce").fillna(0).astype(int)

    id_acc = df[df["kind"] == "identity"].groupby("subject")["accuracy"].first().rename("id_acc")
    df = df.join(id_acc, on="subject")
    df["true_improve"] = df["accuracy"] - df["id_acc"]
    return df[df["kind"] == "candidate"].copy()


def _plot_cert_scatter(cand: pd.DataFrame, *, out_png: Path, title: str) -> tuple[float, float]:
    x = cand["ridge_pred_improve"].to_numpy(dtype=float)
    y = cand["true_improve"].to_numpy(dtype=float)
    rho = _spearman(x, y)
    r = _pearson(x, y)

    colors = np.where(y >= 0.0, "#4C72B0", "#C44E52")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    ax.axvline(0.0, color="k", linewidth=1, alpha=0.2)
    ax.scatter(x, y, s=22, alpha=0.75, c=colors)
    ax.set_xlabel("predicted Δ (certificate)")
    ax.set_ylabel("true Δ (accuracy gain)")
    ax.set_title(f"{title}\\nPearson={r:.3f}, Spearman={rho:.3f}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r, rho


def _plot_guard_roc(p: np.ndarray, y: np.ndarray, *, out_png: Path, title: str) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 0.0, 1.0)
    y = y[mask]
    if y.size < 2 or len(np.unique(y)) < 2:
        auc = float("nan")
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
    else:
        auc = float(roc_auc_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2.2, label=(f"AUC={auc:.3f}" if np.isfinite(auc) else "AUC=nan"))
    ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.25)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return auc


def _plot_guard_reliability(p: np.ndarray, y: np.ndarray, *, out_png: Path, title: str, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 0.0, 1.0)
    y = y[mask]

    ece = _ece_binary(p, y, n_bins=n_bins)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_n = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(in_bin):
            continue
        bin_acc.append(float(np.mean(y[in_bin])))
        bin_conf.append(float(np.mean(p[in_bin])))
        bin_n.append(int(np.sum(in_bin)))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.35, label="ideal")
    if bin_acc:
        ax.plot(bin_conf, bin_acc, marker="o", color="#4C72B0", label="empirical")
        for xc, yc, n in zip(bin_conf, bin_acc, bin_n, strict=False):
            ax.text(float(xc), float(yc), f"n={int(n)}", fontsize=8, alpha=0.8)
    ax.set_xlabel("predicted p(pos)")
    ax.set_ylabel("empirical P(pos)")
    ax.set_title(f"{title}\\nECE={ece:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ece


def _plot_accept_scatter(cand: pd.DataFrame, *, out_png: Path, title: str) -> None:
    # candidate-level scatter colored by candidate acceptance under the calibrated policy
    x = cand["ridge_pred_improve"].to_numpy(dtype=float)
    y = cand["true_improve"].to_numpy(dtype=float)
    a = cand["accept"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    ax.axvline(0.0, color="k", linewidth=1, alpha=0.2)
    ax.scatter(x[a == 0], y[a == 0], s=20, alpha=0.55, color="#9A9A9A", label="rejected candidates")
    ax.scatter(x[a == 1], y[a == 1], s=28, alpha=0.85, color="#DD8452", label="accepted candidates")
    ax.set_xlabel("predicted Δ (certificate)")
    ax.set_ylabel("true Δ (accuracy gain)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze certificate/guard diagnostics for alignment-selection action set.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, required=True, help="method-name used in diagnostics/<method>/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="", help="Optional tag for accepted_vs_rejected scatter filename.")
    ap.add_argument("--no-main", action="store_true", help="Only write accepted_vs_rejected scatter (skip main plots/metrics).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_candidates(Path(args.run_dir), method=str(args.method))
    cand = _prepare_candidate_level(df)
    if cand.empty:
        raise RuntimeError("No candidate rows found in candidates.csv")

    # Main plots/metrics (candidate-level; largely alpha-invariant)
    if not bool(args.no_main):
        r, rho = _plot_cert_scatter(
            cand,
            out_png=out_dir / "cert_pred_vs_true_scatter.png",
            title="Certificate predicted Δ vs true Δ (candidate-level)",
        )

        y_pos = (cand["true_improve"].to_numpy(dtype=float) > 0.0).astype(int)
        p_pos = cand["guard_p_pos"].to_numpy(dtype=float)
        auc = _plot_guard_roc(
            p_pos,
            y_pos,
            out_png=out_dir / "guard_roc.png",
            title="Guard ROC: predict (true Δ > 0)",
        )
        ece = _plot_guard_reliability(
            p_pos,
            y_pos,
            out_png=out_dir / "guard_reliability.png",
            title="Guard reliability: predicted p(pos) vs empirical P(pos)",
        )
        brier = float(np.mean((np.clip(p_pos, 0.0, 1.0) - y_pos) ** 2))

        # Top-1 hit (per-subject)
        hits = []
        for subj, sdf in cand.groupby("subject"):
            sdf = sdf.copy()
            i_pred = int(np.nanargmax(sdf["ridge_pred_improve"].to_numpy(dtype=float)))
            i_true = int(np.nanargmax(sdf["true_improve"].to_numpy(dtype=float)))
            hits.append(int(str(sdf.iloc[i_pred]["cand_key"]) == str(sdf.iloc[i_true]["cand_key"])))
        top1 = float(np.mean(hits)) if hits else float("nan")

        metrics = pd.DataFrame(
            [
                {
                    "n_subjects": int(cand["subject"].nunique()),
                    "n_candidates": int(len(cand)),
                    "cert_pearson": float(r),
                    "cert_spearman": float(rho),
                    "cert_top1_hit_rate": float(top1),
                    "guard_auc": float(auc),
                    "guard_ece": float(ece),
                    "guard_brier": float(brier),
                }
            ]
        )
        metrics.to_csv(out_dir / "metrics.csv", index=False)

    # Optional: alpha-dependent accept scatter
    tag = str(args.tag).strip()
    if tag:
        safe = re.sub(r"[^0-9a-zA-Z_.-]+", "_", tag)
        _plot_accept_scatter(
            cand,
            out_png=out_dir / f"accepted_vs_rejected_scatter_{safe}.png",
            title=f"Accepted vs rejected candidates ({safe})",
        )

    print(f"Read run: {Path(args.run_dir)}")
    print(f"Wrote dir: {out_dir}")


if __name__ == "__main__":
    main()
