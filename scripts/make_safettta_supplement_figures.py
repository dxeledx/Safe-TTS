from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 file match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_method_comparison(run_dir: Path) -> pd.DataFrame:
    p = _find_single(run_dir, "*_method_comparison.csv")
    return pd.read_csv(p)


def _extract_method_table(results_txt: Path, *, method: str) -> pd.DataFrame:
    txt = results_txt.read_text(encoding="utf-8")
    pat = re.compile(rf"=== Method: {re.escape(method)} ===\n(.*?)\n\nSummary", re.S)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Method block not found: {method} in {results_txt}")
    block = m.group(1).rstrip()
    return pd.read_fwf(io.StringIO(block))


def _approx_wall_time_seconds(run_dir: Path) -> float:
    times = []
    for p in Path(run_dir).rglob("*"):
        if not p.is_file():
            continue
        try:
            times.append(p.stat().st_mtime)
        except FileNotFoundError:
            continue
    if len(times) < 2:
        return float("nan")
    return float(max(times) - min(times))


def _load_bnci_candidates(run_dir: Path, *, method: str) -> pd.DataFrame:
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


def _ece_binary(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    if p.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
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


def _plot_reliability(
    p: np.ndarray,
    y: np.ndarray,
    *,
    out_png: Path,
    out_pdf: Path,
    title: str,
    n_bins: int = 10,
) -> None:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 0.0, 1.0)
    y = y[mask].astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_acc = []
    bin_conf = []
    bin_n = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(in_bin):
            continue
        bin_centers.append(float((lo + hi) / 2.0))
        bin_acc.append(float(np.mean(y[in_bin])))
        bin_conf.append(float(np.mean(p[in_bin])))
        bin_n.append(int(np.sum(in_bin)))

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.4, label="ideal")
    if bin_centers:
        ax.plot(bin_conf, bin_acc, marker="o", color="#4C72B0", label="empirical")
        for xc, yc, n in zip(bin_conf, bin_acc, bin_n):
            ax.text(float(xc), float(yc), f"n={int(n)}", fontsize=8, alpha=0.8)
    ax.set_xlabel("Predicted p(pos)")
    ax.set_ylabel("Empirical P(pos)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(
    p: np.ndarray,
    y: np.ndarray,
    *,
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 0.0, 1.0)
    y = y[mask].astype(int)
    if y.size < 2 or len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return auc


def _plot_accept_rate_bar(df: pd.DataFrame, *, out_png: Path, out_pdf: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(df))
    ax.bar(x, df["accept_rate"].astype(float).to_numpy(), color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"].astype(str).to_numpy())
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accept-rate")
    ax.set_title(title)
    for i, v in enumerate(df["accept_rate"].astype(float).to_numpy()):
        ax.text(i, float(v) + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_family_counts(counts: pd.Series, *, out_png: Path, out_pdf: Path, title: str) -> None:
    fam_order = ["ea", "rpa", "tsa", "chan", "fbcsp", "other"]
    counts = counts.copy()
    counts.index = counts.index.astype(str).str.lower()
    for f in fam_order:
        if f not in counts.index:
            counts.loc[f] = 0
    counts = counts.loc[fam_order]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(counts))
    ax.bar(x, counts.to_numpy(dtype=float), color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.to_numpy())
    ax.set_ylabel("#subjects")
    ax.set_title(title)
    for i, v in enumerate(counts.to_numpy(dtype=float)):
        ax.text(i, float(v) + 0.05, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_hgd_pred_disagree_scatter(
    df: pd.DataFrame,
    *,
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    x = df["pred_disagree"].astype(float).to_numpy()
    y = df["delta_cand_vs_ea"].astype(float).to_numpy()
    sel = df["selected_cand"].astype(bool).to_numpy()
    tau = float(np.nanmedian(df["tau"].astype(float).to_numpy()))
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    ax.axvline(tau, color="#C44E52", linewidth=1.5, alpha=0.8, label=f"tau (median)={tau:.3f}")
    ax.scatter(x[~sel], y[~sel], s=45, color="#9A9A9A", alpha=0.8, label="rejected (fallback EA)")
    ax.scatter(x[sel], y[sel], s=55, color="#DD8452", alpha=0.85, label="accepted (use candidate)")
    for xi, yi, si in zip(x, y, df["subject"].astype(int).to_numpy()):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(float(xi), float(yi), str(int(si)), fontsize=8, alpha=0.75)
    ax.set_xlabel("pred_disagree (lower is safer)")
    ax.set_ylabel("True Δacc (candidate − EA)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_hgd_failure_cases(
    df: pd.DataFrame,
    *,
    out_png: Path,
    out_pdf: Path,
    title: str,
    subjects: list[int],
) -> None:
    df = df.set_index("subject", drop=False)
    fig, axes = plt.subplots(1, len(subjects), figsize=(6.6, 3.4), sharey=True)
    if len(subjects) == 1:
        axes = [axes]
    for ax, s in zip(axes, subjects):
        row = df.loc[int(s)]
        vals = [float(row["acc_ea"]), float(row["acc_cand"]), float(row["acc_selected"])]
        labels = ["EA", "cand", "selected"]
        colors = ["#9A9A9A", "#C44E52", "#4C72B0"]
        ax.bar(np.arange(3), vals, color=colors, alpha=0.9)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(labels)
        ax.set_title(f"S{s}")
        tau = float(row["tau"])
        pdv = float(row["pred_disagree"])
        delta = float(row["delta_cand_vs_ea"])
        ax.text(
            0.02,
            0.98,
            f"pred_disagree={pdv:.3f}\ntau={tau:.3f}\nΔcand={delta:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SAFE‑TTA supplement tables/figures (cost, selection behavior, calibration, failure cases) from existing outputs."
    )
    ap.add_argument("--bnci-run", type=Path, required=True)
    ap.add_argument("--bnci-method", type=str, default="ea-stack-multi-safe-csp-lda")
    ap.add_argument("--bnci-base", type=str, default="ea-csp-lda")
    ap.add_argument("--hgd-run", type=Path, required=True)
    ap.add_argument("--physio-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_tables = out_root / "tables"
    out_figs = out_root / "figures"
    _ensure_dir(out_tables)
    _ensure_dir(out_figs)

    # ===== Dataset-level accept-rate + candidate counts + wall-time (approx) =====
    rows = []

    # BNCI (stack multi safe; has candidates.csv + results.txt)
    bnci_mc = _read_method_comparison(args.bnci_run)
    bnci_row = bnci_mc[bnci_mc["method"].astype(str) == str(args.bnci_method)].iloc[0].to_dict()
    bnci_n = int(bnci_row.get("n_subjects", 0))
    bnci_accept = float(bnci_row.get("accept_rate", float("nan")))
    bnci_delta = float(bnci_row.get("mean_delta_vs_ea", float("nan")))
    bnci_neg = float(bnci_row.get("neg_transfer_rate_vs_ea", float("nan")))
    bnci_cand = _load_bnci_candidates(args.bnci_run, method=str(args.bnci_method))
    bnci_k = float(bnci_cand.groupby("subject").size().mean())
    rows.append(
        {
            "dataset": "BNCI2014_001",
            "n_subjects": bnci_n,
            "accept_rate": bnci_accept,
            "mean_delta_vs_ea": bnci_delta,
            "neg_transfer_rate_vs_ea": bnci_neg,
            "n_candidates_mean": bnci_k,
            "selector": "ridge+guard (LOSO-calibrated) + safe fallback",
            "online_or_offline": "online",
        }
    )

    # HGD (pred_disagree gate)
    hgd_mc = _read_method_comparison(args.hgd_run)
    hgd_safe = hgd_mc[hgd_mc["method"].astype(str) == "ea-fbcsp-pred-disagree-safe"].iloc[0].to_dict()
    hgd_n = int(hgd_safe.get("n_subjects", 0))
    hgd_accept = float(hgd_safe.get("accept_rate", float("nan")))
    hgd_delta = float(hgd_safe.get("mean_delta_vs_ea", float("nan")))
    hgd_neg = float(hgd_safe.get("neg_transfer_rate_vs_ea", float("nan")))
    rows.append(
        {
            "dataset": "Schirrmeister2017",
            "n_subjects": hgd_n,
            "accept_rate": hgd_accept,
            "mean_delta_vs_ea": hgd_delta,
            "neg_transfer_rate_vs_ea": hgd_neg,
            "n_candidates_mean": 2.0,
            "selector": "pred_disagree gate (tau LOSO-calibrated) + safe fallback",
            "online_or_offline": "offline-built (from predictions)",
        }
    )

    # Physio (offline guard selector EA vs LEA)
    phys_mc = _read_method_comparison(args.physio_run)
    phys_safe = phys_mc[phys_mc["method"].astype(str) == "safe-tta-offline-ea-vs-lea"].iloc[0].to_dict()
    phys_n = int(phys_safe.get("n_subjects", 0))
    phys_accept = float(phys_safe.get("accept_rate", float("nan")))
    phys_delta = float(phys_safe.get("meanΔacc_vs_ea-csp-lda", phys_safe.get("mean_delta_vs_ea", float("nan"))))
    phys_neg = float(phys_safe.get("neg_transfer_vs_ea-csp-lda", phys_safe.get("neg_transfer_rate_vs_ea", float("nan"))))
    rows.append(
        {
            "dataset": "PhysionetMI",
            "n_subjects": phys_n,
            "accept_rate": phys_accept,
            "mean_delta_vs_ea": phys_delta,
            "neg_transfer_rate_vs_ea": phys_neg,
            "n_candidates_mean": 2.0,
            "selector": "ridge+guard (LOSO-calibrated) + safe fallback",
            "online_or_offline": "offline",
        }
    )

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_tables / "selection_summary.csv", index=False)
    (out_tables / "selection_summary.md").write_text(df_summary.to_markdown(index=False) + "\n", encoding="utf-8")

    _plot_accept_rate_bar(
        df_summary[["dataset", "accept_rate"]].copy(),
        out_png=out_figs / "accept_rate_by_dataset.png",
        out_pdf=out_figs / "accept_rate_by_dataset.pdf",
        title="SAFE‑TTA accept-rate (non-anchor action selected)",
    )

    # ===== BNCI selection behavior: selected families + rejected(pre->EA) =====
    bnci_sel = (
        bnci_cand[bnci_cand.get("is_selected", 0).astype(int) == 1]
        .copy()
        .sort_values(["subject", "idx"])
    )
    bnci_sel["cand_family"] = bnci_sel.get("cand_family", "ea").astype(str).str.lower()
    _plot_family_counts(
        bnci_sel["cand_family"].value_counts(),
        out_png=out_figs / "bnci_selected_family_counts.png",
        out_pdf=out_figs / "bnci_selected_family_counts.pdf",
        title="BNCI2014_001: selected family counts (SAFE‑TTA)",
    )

    bnci_results = _find_single(args.bnci_run, "*_results.txt")
    df_bnci_method = _extract_method_table(bnci_results, method=str(args.bnci_method))
    # Subjects where a non-EA pre-choice gets rejected back to EA.
    pre = df_bnci_method.get("stack_multi_pre_family", pd.Series(["ea"] * len(df_bnci_method))).astype(str).str.lower()
    fin = df_bnci_method.get("stack_multi_family", pd.Series(["ea"] * len(df_bnci_method))).astype(str).str.lower()
    rejected_pre = pre[(fin == "ea") & (pre != "ea")]
    if not rejected_pre.empty:
        _plot_family_counts(
            rejected_pre.value_counts(),
            out_png=out_figs / "bnci_rejected_pre_family_counts.png",
            out_pdf=out_figs / "bnci_rejected_pre_family_counts.pdf",
            title="BNCI2014_001: families rejected by safety gates (pre→EA)",
        )

    block_cols = [
        ("stack_multi_min_pred_blocked", "stack_multi_min_pred_block_reason"),
        ("stack_multi_fbcsp_blocked", "stack_multi_fbcsp_block_reason"),
        ("stack_multi_tsa_blocked", "stack_multi_tsa_block_reason"),
    ]
    block_rows = []
    for blocked_col, reason_col in block_cols:
        if blocked_col in df_bnci_method.columns:
            blocked = df_bnci_method[blocked_col].fillna(0).astype(int).to_numpy()
            if reason_col in df_bnci_method.columns:
                reasons = df_bnci_method[reason_col].fillna("").astype(str).to_numpy()
            else:
                reasons = np.array([""] * len(df_bnci_method), dtype=str)
            for b, r in zip(blocked, reasons):
                if int(b) > 0:
                    block_rows.append({"gate": blocked_col, "reason": str(r).strip()})
    df_blocks = pd.DataFrame(block_rows)
    if not df_blocks.empty:
        df_block_counts = df_blocks.groupby(["gate", "reason"]).size().reset_index(name="count").sort_values(
            ["gate", "count"], ascending=[True, False]
        )
        df_block_counts.to_csv(out_tables / "bnci_gate_block_reasons.csv", index=False)
        (out_tables / "bnci_gate_block_reasons.md").write_text(
            df_block_counts.to_markdown(index=False) + "\n", encoding="utf-8"
        )

    # ===== Calibration: guard reliability (BNCI candidate-level; Physio subject-level) =====
    # BNCI candidate-level calibration: predict whether candidate improves over identity.
    bnci_id = (
        bnci_cand[bnci_cand["kind"].astype(str) == "identity"]
        .groupby("subject")["accuracy"]
        .first()
        .astype(float)
        .rename("id_acc")
    )
    bnci_non_id = bnci_cand[bnci_cand["kind"].astype(str) != "identity"].copy()
    bnci_non_id["accuracy"] = bnci_non_id["accuracy"].astype(float)
    bnci_non_id = bnci_non_id.join(bnci_id, on="subject")
    bnci_non_id["y_pos"] = (bnci_non_id["accuracy"] > bnci_non_id["id_acc"]).astype(int)
    bnci_p = pd.to_numeric(bnci_non_id.get("guard_p_pos", np.nan), errors="coerce").to_numpy(dtype=float)
    bnci_y = bnci_non_id["y_pos"].to_numpy(dtype=int)
    bnci_auc = _plot_roc(
        bnci_p,
        bnci_y,
        out_png=out_figs / "bnci_guard_roc.png",
        out_pdf=out_figs / "bnci_guard_roc.pdf",
        title="BNCI2014_001: guard ROC (candidate improves over EA)",
    )
    bnci_ece = _ece_binary(bnci_p, bnci_y, n_bins=5)
    _plot_reliability(
        bnci_p,
        bnci_y,
        out_png=out_figs / "bnci_guard_reliability.png",
        out_pdf=out_figs / "bnci_guard_reliability.pdf",
        title=f"BNCI2014_001: guard reliability (ECE={bnci_ece:.3f})",
        n_bins=5,
    )

    # Physio subject-level calibration: guard_p_pos predicts whether LEA improves over EA.
    phys_sel = pd.read_csv(_find_single(args.physio_run, "*_per_subject_selection.csv"))
    phys_y = (phys_sel["delta_cand_vs_ea"].astype(float) > 0.0).astype(int).to_numpy()
    phys_p = phys_sel["guard_p_pos"].astype(float).to_numpy()
    phys_auc = _plot_roc(
        phys_p,
        phys_y,
        out_png=out_figs / "physio_guard_roc.png",
        out_pdf=out_figs / "physio_guard_roc.pdf",
        title="PhysionetMI: guard ROC (LEA improves over EA)",
    )
    phys_ece = _ece_binary(phys_p, phys_y, n_bins=10)
    _plot_reliability(
        phys_p,
        phys_y,
        out_png=out_figs / "physio_guard_reliability.png",
        out_pdf=out_figs / "physio_guard_reliability.pdf",
        title=f"PhysionetMI: guard reliability (ECE={phys_ece:.3f})",
        n_bins=10,
    )

    # HGD: certificate score AUC (pred_disagree; lower is better => use -pred_disagree).
    hgd_gate = pd.read_csv(_find_single(args.hgd_run, "*_pred_disagree_gate_per_subject.csv"))
    hgd_y = (hgd_gate["delta_cand_vs_ea"].astype(float) > 0.0).astype(int).to_numpy()
    hgd_score = (-hgd_gate["pred_disagree"].astype(float)).to_numpy()
    hgd_auc = float("nan")
    if len(np.unique(hgd_y)) >= 2:
        hgd_auc = float(roc_auc_score(hgd_y, hgd_score))

    # Calibration summary table
    calib_rows = [
        {"dataset": "BNCI2014_001", "metric": "guard_auc", "value": bnci_auc},
        {"dataset": "BNCI2014_001", "metric": "guard_ece", "value": bnci_ece},
        {"dataset": "PhysionetMI", "metric": "guard_auc", "value": phys_auc},
        {"dataset": "PhysionetMI", "metric": "guard_ece", "value": phys_ece},
        {"dataset": "Schirrmeister2017", "metric": "pred_disagree_auc", "value": hgd_auc},
    ]
    df_calib = pd.DataFrame(calib_rows)
    df_calib.to_csv(out_tables / "certificate_calibration_metrics.csv", index=False)
    (out_tables / "certificate_calibration_metrics.md").write_text(df_calib.to_markdown(index=False) + "\n", encoding="utf-8")

    # ===== HGD failure cases =====
    _plot_hgd_pred_disagree_scatter(
        hgd_gate,
        out_png=out_figs / "hgd_pred_disagree_vs_delta.png",
        out_pdf=out_figs / "hgd_pred_disagree_vs_delta.pdf",
        title="Schirrmeister2017: pred_disagree vs true Δacc (EA↔EA-FBCSP)",
    )
    hgd_neg = hgd_gate[(hgd_gate["selected_cand"].astype(bool) == False) & (hgd_gate["delta_cand_vs_ea"].astype(float) < 0.0)].copy()
    hgd_neg = hgd_neg.sort_values("delta_cand_vs_ea", ascending=True)
    case_subjects = [int(s) for s in hgd_neg["subject"].head(2).astype(int).to_list()] if not hgd_neg.empty else []
    if case_subjects:
        _plot_hgd_failure_cases(
            hgd_gate,
            out_png=out_figs / "hgd_failure_cases.png",
            out_pdf=out_figs / "hgd_failure_cases.pdf",
            title="Schirrmeister2017: failure cases avoided by SAFE‑TTA gate",
            subjects=case_subjects,
        )
        df_case = hgd_gate[hgd_gate["subject"].astype(int).isin(case_subjects)].copy()
        df_case = df_case[
            [
                "subject",
                "pred_disagree",
                "tau",
                "acc_ea",
                "acc_cand",
                "acc_selected",
                "delta_cand_vs_ea",
                "selected_cand",
            ]
        ].sort_values("delta_cand_vs_ea")
        df_case.to_csv(out_tables / "hgd_failure_cases.csv", index=False)
        (out_tables / "hgd_failure_cases.md").write_text(df_case.to_markdown(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
