from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .alignment import (
    apply_spatial_transform,
    blend_with_identity,
    class_cov_diff,
    orthogonal_align_symmetric,
    sorted_eigh,
)
from .model import TrainedModel
from .localmix import knn_channel_neighbors_from_names
from .proba import reorder_proba_columns as _reorder_proba_columns

def _write_zo_diagnostics(
    zo_diag: dict,
    *,
    out_dir: Path,
    tag: str,
    subject: int,
    model: TrainedModel,
    z_t: np.ndarray,
    y_true: np.ndarray,
    class_order: Sequence[str],
) -> None:
    """Write per-subject EA-ZO/OEA-ZO diagnostics (analysis-only; uses labels)."""

    out_dir = Path(out_dir)
    tag = tag or "zo"
    diag_dir = out_dir / "diagnostics" / str(tag) / f"subject_{int(subject):02d}"
    diag_dir.mkdir(parents=True, exist_ok=True)

    records = list(zo_diag.get("records", []))
    if not records:
        return

    # Candidate evaluation on the labeled target fold (analysis-only).
    rows = []
    for idx, rec in enumerate(records):
        Q = np.asarray(rec.get("Q"), dtype=np.float64)
        X = apply_spatial_transform(Q, z_t)
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        row = {
            "idx": int(idx),
            "kind": str(rec.get("kind", "")),
            "iter": int(rec.get("iter", -1)),
            "order": int(rec.get("order", idx)),
            "objective": float(rec.get("objective", np.nan)),
            "score": float(rec.get("score", np.nan)),
            "objective_base": float(rec.get("objective_base", np.nan)),
            "evidence_nll_best": float(rec.get("evidence_nll_best", np.nan)),
            "evidence_nll_full": float(rec.get("evidence_nll_full", np.nan)),
            "iwcv_nll": float(rec.get("iwcv_nll", np.nan)),
            "iwcv_eff_n": float(rec.get("iwcv_eff_n", np.nan)),
            "iwcv_ucb": float(rec.get("iwcv_ucb", np.nan)),
            "iwcv_var": float(rec.get("iwcv_var", np.nan)),
            "iwcv_se": float(rec.get("iwcv_se", np.nan)),
            "dev_nll": float(rec.get("dev_nll", np.nan)),
            "dev_eta": float(rec.get("dev_eta", np.nan)),
            "dev_mean_w": float(rec.get("dev_mean_w", np.nan)),
            "dev_eff_n": float(rec.get("dev_eff_n", np.nan)),
            "probe_mixup_best": float(rec.get("probe_mixup_best", np.nan)),
            "probe_mixup_full": float(rec.get("probe_mixup_full", np.nan)),
            "probe_mixup_pairs_best": int(rec.get("probe_mixup_pairs_best", -1)),
            "probe_mixup_pairs_full": int(rec.get("probe_mixup_pairs_full", -1)),
            "probe_mixup_keep_best": int(rec.get("probe_mixup_keep_best", -1)),
            "probe_mixup_keep_full": int(rec.get("probe_mixup_keep_full", -1)),
            "probe_mixup_frac_intra_best": float(rec.get("probe_mixup_frac_intra_best", np.nan)),
            "probe_mixup_frac_intra_full": float(rec.get("probe_mixup_frac_intra_full", np.nan)),
            "probe_mixup_hard_best": float(rec.get("probe_mixup_hard_best", np.nan)),
            "probe_mixup_hard_full": float(rec.get("probe_mixup_hard_full", np.nan)),
            "probe_mixup_hard_pairs_best": int(rec.get("probe_mixup_hard_pairs_best", -1)),
            "probe_mixup_hard_pairs_full": int(rec.get("probe_mixup_hard_pairs_full", -1)),
            "probe_mixup_hard_keep_best": int(rec.get("probe_mixup_hard_keep_best", -1)),
            "probe_mixup_hard_keep_full": int(rec.get("probe_mixup_hard_keep_full", -1)),
            "probe_mixup_hard_frac_intra_best": float(rec.get("probe_mixup_hard_frac_intra_best", np.nan)),
            "probe_mixup_hard_frac_intra_full": float(rec.get("probe_mixup_hard_frac_intra_full", np.nan)),
            "ridge_pred_improve": float(rec.get("ridge_pred_improve", np.nan)),
            "guard_p_pos": float(rec.get("guard_p_pos", np.nan)),
            "pen_marginal": float(rec.get("pen_marginal", np.nan)),
            "pen_trust": float(rec.get("pen_trust", np.nan)),
            "pen_l2": float(rec.get("pen_l2", np.nan)),
            "drift_best": float(rec.get("drift_best", np.nan)),
            "drift_full": float(rec.get("drift_full", np.nan)),
            "mean_entropy": float(rec.get("mean_entropy", np.nan)),
            "mean_confidence": float(rec.get("mean_confidence", np.nan)),
            "entropy_bar": float(rec.get("entropy_bar", np.nan)),
            "n_keep": int(rec.get("n_keep", -1)),
            "n_best_total": int(rec.get("n_best_total", -1)),
            "n_full_total": int(rec.get("n_full_total", -1)),
            "accuracy": float(acc),
        }
        p_bar = np.asarray(rec.get("p_bar_full", []), dtype=np.float64).reshape(-1)
        for k, name in enumerate(class_order):
            row[f"pbar_{name}"] = float(p_bar[k]) if k < p_bar.shape[0] else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["order", "idx"])
    df.to_csv(diag_dir / "candidates.csv", index=False)

    obj = df["objective"].to_numpy(dtype=np.float64)
    acc = df["accuracy"].to_numpy(dtype=np.float64)
    pearson = float(np.corrcoef(obj, acc)[0, 1]) if obj.size >= 2 else float("nan")
    # Spearman via ranks (no ties handling needed for a quick diagnostic).
    obj_r = obj.argsort().argsort().astype(np.float64)
    acc_r = acc.argsort().argsort().astype(np.float64)
    spearman = float(np.corrcoef(obj_r, acc_r)[0, 1]) if obj.size >= 2 else float("nan")

    ev = df["evidence_nll_best"].to_numpy(dtype=np.float64)
    pearson_ev = float("nan")
    spearman_ev = float("nan")
    if ev.size >= 2 and np.isfinite(ev).any():
        pearson_ev = float(np.corrcoef(ev, acc)[0, 1])
        ev_r = ev.argsort().argsort().astype(np.float64)
        spearman_ev = float(np.corrcoef(ev_r, acc_r)[0, 1])

    iw = df["iwcv_nll"].to_numpy(dtype=np.float64)
    pearson_iw = float("nan")
    spearman_iw = float("nan")
    if iw.size >= 2 and np.isfinite(iw).any():
        pearson_iw = float(np.corrcoef(iw, acc)[0, 1])
        iw_r = iw.argsort().argsort().astype(np.float64)
        spearman_iw = float(np.corrcoef(iw_r, acc_r)[0, 1])

    dev = df["dev_nll"].to_numpy(dtype=np.float64)
    pearson_dev = float("nan")
    spearman_dev = float("nan")
    if dev.size >= 2 and np.isfinite(dev).any():
        pearson_dev = float(np.corrcoef(dev, acc)[0, 1])
        dev_r = dev.argsort().argsort().astype(np.float64)
        spearman_dev = float(np.corrcoef(dev_r, acc_r)[0, 1])

    pm = df["probe_mixup_best"].to_numpy(dtype=np.float64)
    pearson_pm = float("nan")
    spearman_pm = float("nan")
    if pm.size >= 2 and np.isfinite(pm).any():
        pearson_pm = float(np.corrcoef(pm, acc)[0, 1])
        pm_r = pm.argsort().argsort().astype(np.float64)
        spearman_pm = float(np.corrcoef(pm_r, acc_r)[0, 1])

    pm_h = df["probe_mixup_hard_best"].to_numpy(dtype=np.float64)
    pearson_pmh = float("nan")
    spearman_pmh = float("nan")
    if pm_h.size >= 2 and np.isfinite(pm_h).any():
        pearson_pmh = float(np.corrcoef(pm_h, acc)[0, 1])
        pmh_r = pm_h.argsort().argsort().astype(np.float64)
        spearman_pmh = float(np.corrcoef(pmh_r, acc_r)[0, 1])

    prior = zo_diag.get("marginal_prior")
    prior_arr = None if prior is None else np.asarray(prior, dtype=np.float64).reshape(-1)

    # Plots
    from .plots import plot_class_marginal_trajectory, plot_objective_vs_accuracy_scatter

    p_cols = [c for c in df.columns if c.startswith("pbar_")]
    p_bars = df[p_cols].to_numpy(dtype=np.float64)
    x = df["order"].to_numpy(dtype=int)
    plot_class_marginal_trajectory(
        p_bars,
        class_order=class_order,
        x=x,
        prior=prior_arr,
        output_path=diag_dir / "pbar_trajectory.png",
        title=f"Subject {subject} — p̄ trajectory ({tag})",
    )
    plot_objective_vs_accuracy_scatter(
        obj,
        acc,
        output_path=diag_dir / "objective_vs_accuracy.png",
        title=f"Subject {subject} — objective vs acc (pearson={pearson:.3f}, spearman={spearman:.3f})",
    )
    if np.isfinite(ev).any():
        plot_objective_vs_accuracy_scatter(
            ev,
            acc,
            output_path=diag_dir / "evidence_vs_accuracy.png",
            title=f"Subject {subject} — evidence(-log p) vs acc (pearson={pearson_ev:.3f}, spearman={spearman_ev:.3f})",
        )
    if np.isfinite(iw).any():
        plot_objective_vs_accuracy_scatter(
            iw,
            acc,
            output_path=diag_dir / "iwcv_nll_vs_accuracy.png",
            title=f"Subject {subject} — IWCV-NLL vs acc (pearson={pearson_iw:.3f}, spearman={spearman_iw:.3f})",
        )
    if np.isfinite(dev).any():
        plot_objective_vs_accuracy_scatter(
            dev,
            acc,
            output_path=diag_dir / "dev_nll_vs_accuracy.png",
            title=f"Subject {subject} — DEV-NLL vs acc (pearson={pearson_dev:.3f}, spearman={spearman_dev:.3f})",
        )

    iw_ucb = df["iwcv_ucb"].to_numpy(dtype=np.float64)
    pearson_iw_ucb = float("nan")
    spearman_iw_ucb = float("nan")
    if iw_ucb.size >= 2 and np.isfinite(iw_ucb).any():
        pearson_iw_ucb = float(np.corrcoef(iw_ucb, acc)[0, 1])
        iucb_r = iw_ucb.argsort().argsort().astype(np.float64)
        spearman_iw_ucb = float(np.corrcoef(iucb_r, acc_r)[0, 1])
        plot_objective_vs_accuracy_scatter(
            iw_ucb,
            acc,
            output_path=diag_dir / "iwcv_ucb_vs_accuracy.png",
            title=f"Subject {subject} — IWCV-UCB vs acc (pearson={pearson_iw_ucb:.3f}, spearman={spearman_iw_ucb:.3f})",
        )
    if np.isfinite(pm).any():
        plot_objective_vs_accuracy_scatter(
            pm,
            acc,
            output_path=diag_dir / "probe_mixup_vs_accuracy.png",
            title=f"Subject {subject} — probe_mixup vs acc (pearson={pearson_pm:.3f}, spearman={spearman_pm:.3f})",
        )
    if np.isfinite(pm_h).any():
        plot_objective_vs_accuracy_scatter(
            pm_h,
            acc,
            output_path=diag_dir / "probe_mixup_hard_vs_accuracy.png",
            title=f"Subject {subject} — probe_mixup_hard vs acc (pearson={pearson_pmh:.3f}, spearman={spearman_pmh:.3f})",
        )

    ridge_pred = df["ridge_pred_improve"].to_numpy(dtype=np.float64)
    pearson_ridge = float("nan")
    spearman_ridge = float("nan")
    if ridge_pred.size >= 2 and np.isfinite(ridge_pred).any():
        pearson_ridge = float(np.corrcoef(ridge_pred, acc)[0, 1])
        rp_r = ridge_pred.argsort().argsort().astype(np.float64)
        spearman_ridge = float(np.corrcoef(rp_r, acc_r)[0, 1])
        plot_objective_vs_accuracy_scatter(
            ridge_pred,
            acc,
            output_path=diag_dir / "ridge_pred_improve_vs_accuracy.png",
            title=f"Subject {subject} — ridge_pred vs acc (pearson={pearson_ridge:.3f}, spearman={spearman_ridge:.3f})",
        )

    guard_p = df["guard_p_pos"].to_numpy(dtype=np.float64)
    pearson_guard = float("nan")
    spearman_guard = float("nan")
    if guard_p.size >= 2 and np.isfinite(guard_p).any():
        pearson_guard = float(np.corrcoef(guard_p, acc)[0, 1])
        gp_r = guard_p.argsort().argsort().astype(np.float64)
        spearman_guard = float(np.corrcoef(gp_r, acc_r)[0, 1])
        plot_objective_vs_accuracy_scatter(
            guard_p,
            acc,
            output_path=diag_dir / "guard_p_pos_vs_accuracy.png",
            title=f"Subject {subject} — guard_p_pos vs acc (pearson={pearson_guard:.3f}, spearman={spearman_guard:.3f})",
        )
    if "score" in df.columns and np.isfinite(df["score"].to_numpy()).any():
        score = df["score"].to_numpy(dtype=np.float64)
        pearson_s = float(np.corrcoef(score, acc)[0, 1]) if score.size >= 2 else float("nan")
        score_r = score.argsort().argsort().astype(np.float64)
        spearman_s = float(np.corrcoef(score_r, acc_r)[0, 1]) if score.size >= 2 else float("nan")
        plot_objective_vs_accuracy_scatter(
            score,
            acc,
            output_path=diag_dir / "score_vs_accuracy.png",
            title=f"Subject {subject} — score vs acc (pearson={pearson_s:.3f}, spearman={spearman_s:.3f})",
        )

    # Small text summary
    best_by_evidence = -1
    if np.isfinite(ev).any():
        try:
            best_by_evidence = int(df.loc[df["evidence_nll_best"].idxmin(), "idx"])
        except Exception:
            best_by_evidence = -1
    best_by_iwcv = -1
    if np.isfinite(iw).any():
        try:
            best_by_iwcv = int(df.loc[df["iwcv_nll"].idxmin(), "idx"])
        except Exception:
            best_by_iwcv = -1
    best_by_iwcv_ucb = -1
    if np.isfinite(iw_ucb).any():
        try:
            best_by_iwcv_ucb = int(df.loc[df["iwcv_ucb"].idxmin(), "idx"])
        except Exception:
            best_by_iwcv_ucb = -1
    best_by_dev = -1
    if np.isfinite(dev).any():
        try:
            best_by_dev = int(df.loc[df["dev_nll"].idxmin(), "idx"])
        except Exception:
            best_by_dev = -1
    best_by_probe = -1
    if np.isfinite(pm).any():
        try:
            best_by_probe = int(df.loc[df["probe_mixup_best"].idxmin(), "idx"])
        except Exception:
            best_by_probe = -1
    best_by_probe_hard = -1
    if np.isfinite(pm_h).any():
        try:
            best_by_probe_hard = int(df.loc[df["probe_mixup_hard_best"].idxmin(), "idx"])
        except Exception:
            best_by_probe_hard = -1
    best_by_ridge = -1
    if np.isfinite(ridge_pred).any():
        try:
            best_by_ridge = int(df.loc[df["ridge_pred_improve"].idxmax(), "idx"])
        except Exception:
            best_by_ridge = -1
    best_by_guard = -1
    if np.isfinite(guard_p).any():
        try:
            best_by_guard = int(df.loc[df["guard_p_pos"].idxmax(), "idx"])
        except Exception:
            best_by_guard = -1
    lines = [
        f"tag: {tag}",
        f"subject: {subject}",
        f"n_candidates: {len(df)}",
        f"pearson(objective, accuracy): {pearson:.6f}",
        f"spearman(objective, accuracy): {spearman:.6f}",
        f"pearson(evidence, accuracy): {pearson_ev:.6f}",
        f"spearman(evidence, accuracy): {spearman_ev:.6f}",
        f"pearson(iwcv_nll, accuracy): {pearson_iw:.6f}",
        f"spearman(iwcv_nll, accuracy): {spearman_iw:.6f}",
        f"pearson(iwcv_ucb, accuracy): {pearson_iw_ucb:.6f}",
        f"spearman(iwcv_ucb, accuracy): {spearman_iw_ucb:.6f}",
        f"pearson(dev_nll, accuracy): {pearson_dev:.6f}",
        f"spearman(dev_nll, accuracy): {spearman_dev:.6f}",
        f"pearson(probe_mixup, accuracy): {pearson_pm:.6f}",
        f"spearman(probe_mixup, accuracy): {spearman_pm:.6f}",
        f"pearson(probe_mixup_hard, accuracy): {pearson_pmh:.6f}",
        f"spearman(probe_mixup_hard, accuracy): {spearman_pmh:.6f}",
        f"pearson(ridge_pred_improve, accuracy): {pearson_ridge:.6f}",
        f"spearman(ridge_pred_improve, accuracy): {spearman_ridge:.6f}",
        f"pearson(guard_p_pos, accuracy): {pearson_guard:.6f}",
        f"spearman(guard_p_pos, accuracy): {spearman_guard:.6f}",
        f"best_by_objective: idx={int(df.loc[df['objective'].idxmin(), 'idx'])}",
        f"best_by_evidence: idx={best_by_evidence}",
        f"best_by_iwcv_nll: idx={best_by_iwcv}",
        f"best_by_iwcv_ucb: idx={best_by_iwcv_ucb}",
        f"best_by_dev_nll: idx={best_by_dev}",
        f"best_by_probe_mixup: idx={best_by_probe}",
        f"best_by_probe_mixup_hard: idx={best_by_probe_hard}",
        f"best_by_ridge_pred_improve: idx={best_by_ridge}",
        f"best_by_guard_p_pos: idx={best_by_guard}",
        f"best_by_accuracy: idx={int(df.loc[df['accuracy'].idxmax(), 'idx'])}",
    ]
    if "score" in df.columns and np.isfinite(df["score"].to_numpy()).any():
        lines.append(f"best_by_score: idx={int(df.loc[df['score'].idxmin(), 'idx'])}")
    if prior_arr is not None:
        lines.append("marginal_prior: " + ", ".join([f"{x:.4f}" for x in prior_arr.tolist()]))
    (diag_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_pseudo_indices(
    *,
    y_pseudo: np.ndarray,
    proba: np.ndarray,
    class_order: Sequence[str],
    confidence: float,
    topk_per_class: int,
    balance: bool,
) -> np.ndarray:
    """Select trial indices to use for pseudo-label covariance estimation.

    This is a simple stabilization layer for OEA(TTA-Q) to reduce the impact of noisy pseudo labels.
    """

    y_pseudo = np.asarray(y_pseudo)
    proba = np.asarray(proba, dtype=np.float64)
    class_order = [str(c) for c in class_order]
    n_classes = len(class_order)
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")
    if proba.ndim != 2 or proba.shape[1] != n_classes:
        raise ValueError(f"Expected proba shape (n_samples,{n_classes}); got {proba.shape}.")

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    try:
        pred_idx = np.fromiter((class_to_idx[str(c)] for c in y_pseudo), dtype=int, count=len(y_pseudo))
    except KeyError as e:
        raise ValueError(f"Pseudo label contains unknown class '{e.args[0]}'.") from e

    conf = proba[np.arange(len(y_pseudo)), pred_idx]
    keep = conf >= float(confidence)
    if not np.any(keep):
        return np.array([], dtype=int)

    idx_by_class: Dict[str, np.ndarray] = {}
    for c in class_order:
        idx_by_class[c] = np.where(keep & (y_pseudo == c))[0]
    nonempty = [c for c in class_order if idx_by_class[c].size > 0]
    if len(nonempty) < 2:
        return np.array([], dtype=int)

    if int(topk_per_class) > 0:
        k = int(topk_per_class)
        for c in nonempty:
            idx = idx_by_class[c]
            ci = class_to_idx[c]
            idx = idx[np.argsort(proba[idx, ci])[::-1][:k]]
            idx_by_class[c] = idx
        nonempty = [c for c in class_order if idx_by_class[c].size > 0]
        if len(nonempty) < 2:
            return np.array([], dtype=int)

    if balance:
        k = int(min(idx_by_class[c].size for c in nonempty))
        for c in nonempty:
            idx_by_class[c] = idx_by_class[c][:k]

    out = np.concatenate([idx_by_class[c] for c in nonempty], axis=0)
    return np.asarray(out, dtype=int)


def _soft_class_cov_diff(
    X: np.ndarray,
    *,
    proba: np.ndarray,
    class_order: Sequence[str],
    eps: float,
    shrinkage: float,
) -> np.ndarray:
    """Soft pseudo-label covariance signature using class probabilities as weights.

    For each trial i, compute Ci = Xi Xi^T. Then:
      Σ_c = sum_i w_{i,c} Ci / sum_i w_{i,c}

    - Binary: return Δ = Σ_1 - Σ_0.
    - Multiclass: return between-class scatter D = Σ_k π_k (Σ_k - Σ̄)(Σ_k - Σ̄),
      where π_k is the (soft) class mass and Σ̄ = Σ_k π_k Σ_k.
    """

    X = np.asarray(X, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    class_order = [str(c) for c in class_order]
    n_classes = len(class_order)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n_trials,n_channels,n_times); got {X.shape}.")
    if proba.ndim != 2 or proba.shape[0] != X.shape[0] or proba.shape[1] != n_classes:
        raise ValueError(f"Expected proba shape (n_trials,{n_classes}); got {proba.shape}.")

    w = np.clip(proba, 0.0, 1.0)
    row_sum = np.sum(w, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-12)
    w = w / row_sum
    w_sum = np.sum(w, axis=0)  # (n_classes,)
    if float(np.min(w_sum)) <= 0.0:
        raise ValueError("Soft pseudo-label weights degenerate (some class mass sums to zero).")

    n_trials, n_channels, _ = X.shape
    # Trial covariances Ci = Xi Xi^T.
    cov_trials = np.einsum("nct,ndt->ncd", X, X, optimize=True)
    covs = np.einsum("nk,ncd->kcd", w, cov_trials, optimize=True)
    covs = covs / w_sum[:, None, None]
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

    if shrinkage > 0.0:
        alpha = float(shrinkage)
        eye = np.eye(n_channels, dtype=np.float64)
        traces = np.trace(covs, axis1=1, axis2=2) / float(n_channels)
        covs = (1.0 - alpha) * covs + alpha * traces[:, None, None] * eye[None, :, :]

    if n_classes == 2:
        diff = covs[1] - covs[0]
        diff = 0.5 * (diff + diff.T)
        jitter = float(eps) * float(np.max(np.abs(np.diag(diff))) + 1.0)
        diff = diff + jitter * np.eye(n_channels, dtype=np.float64)
        return diff

    pi = w_sum / float(np.sum(w_sum))
    sw = np.einsum("k,kcd->cd", pi, covs, optimize=True)
    sw = 0.5 * (sw + sw.T)
    delta = covs - sw[None, :, :]
    sb = np.einsum("k,kce,ked->cd", pi, delta, delta, optimize=True)
    sb = 0.5 * (sb + sb.T)

    eigvals, eigvecs = np.linalg.eigh(sw)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Deterministic signs for stability.
    for i in range(eigvecs.shape[1]):
        col = eigvecs[:, i]
        j = int(np.argmax(np.abs(col)))
        if col[j] < 0:
            eigvecs[:, i] = -col
    floor = float(eps) * float(np.max(eigvals)) if np.max(eigvals) > 0 else float(eps)
    eigvals = np.maximum(eigvals, floor)
    sw_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    sig = sw_inv_sqrt @ sb @ sw_inv_sqrt
    sig = 0.5 * (sig + sig.T)
    jitter = float(eps) * float(np.max(np.abs(np.diag(sig))) + 1.0)
    sig = sig + jitter * np.eye(n_channels, dtype=np.float64)
    return sig


def _optimize_qt_oea_zo(
    *,
    z_t: np.ndarray,
    model: TrainedModel,
    class_order: Sequence[str],
    d_ref: np.ndarray,
    lda_evidence: dict | None,
    channel_names: Sequence[str] | None = None,
    eps: float,
    shrinkage: float,
    pseudo_mode: str,
    warm_start: str,
    warm_iters: int,
    q_blend: float,
    objective: str,
    transform: str = "orthogonal",
    localmix_neighbors: int = 4,
    localmix_self_bias: float = 3.0,
    infomax_lambda: float,
    reliable_metric: str,
    reliable_threshold: float,
    reliable_alpha: float,
    trust_lambda: float,
    trust_q0: str,
    marginal_mode: str,
    marginal_beta: float,
    marginal_tau: float,
    marginal_prior: np.ndarray | None,
    bilevel_iters: int,
    bilevel_temp: float,
    bilevel_step: float,
    bilevel_coverage_target: float,
    bilevel_coverage_power: float,
    drift_mode: str,
    drift_gamma: float,
    drift_delta: float,
    min_improvement: float,
    holdout_fraction: float,
    fallback_min_marginal_entropy: float,
    iters: int,
    lr: float,
    mu: float,
    n_rotations: int,
    seed: int,
    l2: float,
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Zero-order optimize a target transform on channel space (Q or A) via a low-dim parameterization.

    This implements a practical "optimistic selection" variant for the target subject:
    freeze the trained classifier and update only Q_t using unlabeled target data.
    """

    z_t = np.asarray(z_t, dtype=np.float64)
    n_trials, n_channels, _n_times = z_t.shape
    rng = np.random.RandomState(int(seed))
    class_order = [str(c) for c in class_order]
    n_classes = len(class_order)
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")

    if pseudo_mode not in {"hard", "soft"}:
        raise ValueError("pseudo_mode must be one of: 'hard', 'soft'")
    if warm_start not in {"none", "delta"}:
        raise ValueError("warm_start must be one of: 'none', 'delta'")
    if not (0.0 <= float(holdout_fraction) < 1.0):
        raise ValueError("holdout_fraction must be in [0,1).")
    if float(fallback_min_marginal_entropy) < 0.0:
        raise ValueError("fallback_min_marginal_entropy must be >= 0.")
    if reliable_metric not in {"none", "confidence", "entropy"}:
        raise ValueError("reliable_metric must be one of: 'none', 'confidence', 'entropy'")
    if float(reliable_alpha) <= 0.0:
        raise ValueError("reliable_alpha must be > 0.")
    if reliable_metric == "confidence" and not (0.0 <= float(reliable_threshold) <= 1.0):
        raise ValueError("reliable_threshold must be in [0,1] when metric='confidence'.")
    if reliable_metric == "entropy" and float(reliable_threshold) < 0.0:
        raise ValueError("reliable_threshold must be >= 0 when metric='entropy'.")
    if float(trust_lambda) < 0.0:
        raise ValueError("trust_lambda must be >= 0.")
    if trust_q0 not in {"identity", "delta"}:
        raise ValueError("trust_q0 must be one of: 'identity', 'delta'.")
    if drift_mode not in {"none", "penalty", "hard"}:
        raise ValueError("drift_mode must be one of: 'none', 'penalty', 'hard'.")
    if float(drift_gamma) < 0.0:
        raise ValueError("drift_gamma must be >= 0.")
    if float(drift_delta) < 0.0:
        raise ValueError("drift_delta must be >= 0.")
    if marginal_mode not in {"none", "l2_uniform", "kl_uniform", "hinge_uniform", "hard_min", "kl_prior"}:
        raise ValueError(
            "marginal_mode must be one of: "
            "'none', 'l2_uniform', 'kl_uniform', 'hinge_uniform', 'hard_min', 'kl_prior'."
        )
    if float(marginal_beta) < 0.0:
        raise ValueError("marginal_beta must be >= 0.")
    if not (0.0 <= float(marginal_tau) <= 1.0):
        raise ValueError("marginal_tau must be in [0,1].")
    if float(min_improvement) < 0.0:
        raise ValueError("min_improvement must be >= 0.")
    if int(bilevel_iters) < 0:
        raise ValueError("bilevel_iters must be >= 0.")
    if float(bilevel_temp) <= 0.0:
        raise ValueError("bilevel_temp must be > 0.")
    if float(bilevel_step) < 0.0:
        raise ValueError("bilevel_step must be >= 0.")
    if not (0.0 < float(bilevel_coverage_target) <= 1.0):
        raise ValueError("bilevel_coverage_target must be in (0,1].")
    if float(bilevel_coverage_power) < 0.0:
        raise ValueError("bilevel_coverage_power must be >= 0.")

    transform = str(transform)
    if transform not in {"orthogonal", "rot_scale", "local_mix", "local_mix_then_ea", "local_affine_then_ea"}:
        raise ValueError(
            "transform must be one of: "
            "'orthogonal', 'rot_scale', 'local_mix', 'local_mix_then_ea', 'local_affine_then_ea'"
        )

    # Optional KL(π || p̄) prior (π fixed during optimization).
    marginal_prior_vec: np.ndarray | None = None
    if marginal_mode == "kl_prior":
        if marginal_prior is None:
            raise ValueError("marginal_prior must be provided when marginal_mode='kl_prior'.")
        marginal_prior_vec = np.asarray(marginal_prior, dtype=np.float64).reshape(-1)
        if marginal_prior_vec.shape[0] != int(n_classes):
            raise ValueError(
                f"marginal_prior length mismatch: expected {n_classes}, got {marginal_prior_vec.shape[0]}."
            )
        marginal_prior_vec = np.clip(marginal_prior_vec, 1e-12, 1.0)
        marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

    do_diag = bool(return_diagnostics)
    diag_records: List[dict] = []

    # Unlabeled holdout split: use one subset to update (SPSA gradient estimation) and
    # the other subset to select the best iterate (reduces overfitting to the same trials).
    if float(holdout_fraction) > 0.0 and int(n_trials) > 1:
        perm = rng.permutation(int(n_trials))
        n_hold = int(round(float(holdout_fraction) * float(n_trials)))
        n_hold = max(1, min(int(n_trials) - 1, n_hold))
        idx_best = perm[:n_hold]
        idx_opt = perm[n_hold:]
        z_opt = z_t[idx_opt]
        z_best = z_t[idx_best]
    else:
        z_opt = z_t
        z_best = z_t

    csp = model.csp
    lda = model.pipeline.named_steps["lda"]
    projector = model.pipeline.named_steps.get("proj", None)
    F = np.asarray(csp.filters_[: int(csp.n_components)], dtype=np.float64)

    # Determine whether CSP uses log(power).
    use_log = True if (getattr(csp, "log", None) is None) else bool(getattr(csp, "log"))

    pairs: List[tuple[int, int]] = []
    rot_dim = 0
    max_abs_log_scale = 2.0  # exp(±2) ~= [0.135, 7.39]

    raw_cov: np.ndarray | None = None
    q_anchor_override: np.ndarray | None = None

    if transform in {"local_mix", "local_mix_then_ea", "local_affine_then_ea"}:
        if warm_start != "none":
            raise ValueError(
                "warm_start must be 'none' when transform is "
                "'local_mix', 'local_mix_then_ea', or 'local_affine_then_ea'."
            )
        if trust_q0 != "identity":
            raise ValueError(
                "trust_q0 must be 'identity' when transform is "
                "'local_mix', 'local_mix_then_ea', or 'local_affine_then_ea'."
            )
        if float(localmix_self_bias) < 0.0:
            raise ValueError("localmix_self_bias must be >= 0.")

        if transform in {"local_mix_then_ea", "local_affine_then_ea"}:
            # Precompute raw covariance for EA-after-A whitening:
            # cov(A X) = A cov(X) Aᵀ, so per candidate we only need an eigendecomposition.
            raw_cov = np.zeros((int(n_channels), int(n_channels)), dtype=np.float64)
            for i in range(int(n_trials)):
                xi = z_t[i]
                raw_cov += xi @ xi.T
            raw_cov /= float(n_trials)
            raw_cov = 0.5 * (raw_cov + raw_cov.T)

            # Anchor transform: EA whitening in raw space (A=I).
            cov0 = raw_cov.copy()
            if float(shrinkage) > 0.0:
                alpha = float(shrinkage)
                cov0 = (1.0 - alpha) * cov0 + alpha * (np.trace(cov0) / float(n_channels)) * np.eye(
                    int(n_channels), dtype=np.float64
                )
            evals0, evecs0 = sorted_eigh(cov0)
            floor0 = float(eps) * float(np.max(evals0)) if float(np.max(evals0)) > 0.0 else float(eps)
            evals0 = np.maximum(evals0, floor0)
            q_anchor_override = evecs0 @ np.diag(1.0 / np.sqrt(evals0)) @ evecs0.T

        k = int(localmix_neighbors)
        if k < 0:
            raise ValueError("localmix_neighbors must be >= 0.")
        k = min(k, int(n_channels) - 1)

        if channel_names is not None and len(list(channel_names)) == int(n_channels):
            neighbors = knn_channel_neighbors_from_names(list(channel_names), k=int(k))
        else:
            neighbors = [[j for j in range(int(n_channels)) if j != int(i)][: int(k)] for i in range(int(n_channels))]

        allowed: list[list[int]] = []
        for i in range(int(n_channels)):
            idx = [int(i)] + [int(j) for j in neighbors[int(i)] if int(j) != int(i)]
            seen: set[int] = set()
            uniq: list[int] = []
            for j in idx:
                if int(j) in seen:
                    continue
                seen.add(int(j))
                uniq.append(int(j))
            allowed.append(uniq)

        dims = np.asarray([len(a) for a in allowed], dtype=int)
        if np.any(dims < 1):
            raise RuntimeError("Invalid local_mix neighborhood: empty support.")
        offsets = np.concatenate(([0], np.cumsum(dims))).astype(int)

        theta = np.zeros(int(offsets[-1]), dtype=np.float64)
        best_theta = theta.copy()
        best_obj = float("inf")

        # Initialization close to identity:
        # - local_mix: softmax over logits with positive self-bias and negative neighbor bias.
        # - local_affine_then_ea: signed weights with L2-normalization and positive self-bias.
        base_logits: list[np.ndarray] = []
        base_weights: list[np.ndarray] = []
        for a in allowed:
            bl = np.full(len(a), -float(localmix_self_bias), dtype=np.float64)
            bl[0] = float(localmix_self_bias)
            base_logits.append(bl)

            bw = np.zeros(len(a), dtype=np.float64)
            bw[0] = float(localmix_self_bias)
            base_weights.append(bw)

        def _softmax(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float64)
            v = v - float(np.max(v))
            v = np.clip(v, -50.0, 50.0)
            e = np.exp(v)
            return e / float(np.sum(e))

        def _build_transform(theta_vec: np.ndarray) -> np.ndarray:
            theta_vec = np.asarray(theta_vec, dtype=np.float64)
            if theta_vec.shape[0] != int(theta.shape[0]):
                raise ValueError("theta_vec length mismatch for local_mix.")

            A = np.zeros((int(n_channels), int(n_channels)), dtype=np.float64)
            for i in range(int(n_channels)):
                start = int(offsets[int(i)])
                end = int(offsets[int(i) + 1])
                if transform in {"local_mix", "local_mix_then_ea"}:
                    logits = base_logits[int(i)] + theta_vec[start:end]
                    w = _softmax(logits)
                    A[int(i), np.asarray(allowed[int(i)], dtype=int)] = w
                else:
                    # Signed local linear mixing (row-wise), normalized to unit L2 norm.
                    w_raw = base_weights[int(i)] + theta_vec[start:end]
                    denom = float(np.linalg.norm(w_raw)) + 1e-12
                    w = w_raw / denom
                    A[int(i), np.asarray(allowed[int(i)], dtype=int)] = w

            if float(q_blend) < 1.0:
                A = (1.0 - float(q_blend)) * np.eye(int(n_channels), dtype=np.float64) + float(q_blend) * A

            if transform in {"local_mix_then_ea", "local_affine_then_ea"}:
                if raw_cov is None:
                    raise RuntimeError("raw_cov missing for local_mix_then_ea/local_affine_then_ea.")
                cov_a = A @ raw_cov @ A.T
                cov_a = 0.5 * (cov_a + cov_a.T)
                if float(shrinkage) > 0.0:
                    alpha = float(shrinkage)
                    cov_a = (1.0 - alpha) * cov_a + alpha * (
                        np.trace(cov_a) / float(n_channels)
                    ) * np.eye(int(n_channels), dtype=np.float64)
                evals, evecs = sorted_eigh(cov_a)
                floor = float(eps) * float(np.max(evals)) if float(np.max(evals)) > 0.0 else float(eps)
                evals = np.maximum(evals, floor)
                whitening = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
                return whitening @ A

            return A

    else:
        # Random set of (i,j) planes; fixed per fold for reproducibility.
        pairs = _sample_givens_pairs(n_channels=n_channels, n_rotations=int(n_rotations), rng=rng)
        rot_dim = int(len(pairs))
        scale_dim = int(n_channels) if transform == "rot_scale" else 0
        theta = np.zeros(rot_dim + scale_dim, dtype=np.float64)
        best_theta = theta.copy()
        best_obj = float("inf")

    def _maybe_project(feats: np.ndarray) -> np.ndarray:
        """Apply an optional feature-space projector between CSP and LDA.

        This supports pipelines like: align -> CSP -> proj -> LDA.
        """

        if projector is None:
            return feats
        return np.asarray(projector.transform(feats), dtype=np.float64)

    if transform in {"orthogonal", "rot_scale"}:
        def _build_transform(theta_vec: np.ndarray) -> np.ndarray:
            theta_vec = np.asarray(theta_vec, dtype=np.float64)
            phi_vec = theta_vec[:rot_dim]
            Q = _build_q_from_givens(n_channels=n_channels, pairs=pairs, angles=phi_vec)
            if float(q_blend) < 1.0:
                Q = blend_with_identity(Q, float(q_blend))
            if transform == "rot_scale":
                log_s = np.clip(theta_vec[rot_dim:], -max_abs_log_scale, max_abs_log_scale)
                scales = np.exp(log_s).reshape(-1, 1)
                return scales * Q
            return Q

    def _proba_from_theta(theta_vec: np.ndarray, z_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = _build_transform(theta_vec)
        FQ = F @ A
        Y = np.einsum("kc,nct->nkt", FQ, z_data, optimize=True)
        power = np.mean(Y * Y, axis=2)
        power = np.maximum(power, 1e-20)
        feats = np.log(power) if use_log else power
        feats = _maybe_project(feats)
        proba = lda.predict_proba(feats)
        return _reorder_proba_columns(proba, lda.classes_, list(class_order)), A, feats

    def _proba_from_Q(Q: np.ndarray, z_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = np.asarray(Q, dtype=np.float64)
        if Q.shape != (int(n_channels), int(n_channels)):
            raise ValueError(f"Expected transform shape ({n_channels},{n_channels}); got {Q.shape}.")
        FQ = F @ Q
        Y = np.einsum("kc,nct->nkt", FQ, z_data, optimize=True)
        power = np.mean(Y * Y, axis=2)
        power = np.maximum(power, 1e-20)
        feats = np.log(power) if use_log else power
        feats = _maybe_project(feats)
        proba = lda.predict_proba(feats)
        return _reorder_proba_columns(proba, lda.classes_, list(class_order)), feats

    # Anchor predictions at EA (Q=I). Used for drift guard/certificate.
    q_anchor = (
        np.asarray(q_anchor_override, dtype=np.float64)
        if q_anchor_override is not None
        else np.eye(int(n_channels), dtype=np.float64)
    )
    proba_anchor_best, feats_anchor_best = _proba_from_Q(q_anchor, z_best)
    proba_anchor_full, feats_anchor_full = _proba_from_Q(q_anchor, z_t)

    def _kl_drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Per-sample KL(p0 || p1)."""

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        p0 = np.clip(p0, 1e-12, 1.0)
        p1 = np.clip(p1, 1e-12, 1.0)
        p0 = p0 / np.sum(p0, axis=1, keepdims=True)
        p1 = p1 / np.sum(p1, axis=1, keepdims=True)
        return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

    def _kl_drift(p0: np.ndarray, p1: np.ndarray) -> float:
        """Mean KL(p0 || p1) across samples."""

        return float(np.mean(_kl_drift_vec(p0, p1)))

    def _score_with_drift(obj: float, drift: float) -> float:
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            return float("inf")
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            return float(obj) + float(drift_gamma) * float(drift)
        return float(obj)

    def _maybe_select_keep(proba: np.ndarray) -> np.ndarray:
        """Optionally select a reliable subset based on confidence/top-k/balance settings.

        When the user provides any of {pseudo_confidence, pseudo_topk_per_class, pseudo_balance},
        we reuse the pseudo selection logic to keep only confident trials. This is used to
        stabilize *all* ZO objectives (entropy/infomax/confidence) and not only pseudo_ce.
        """

        if float(pseudo_confidence) <= 0.0 and int(pseudo_topk_per_class) == 0 and not bool(pseudo_balance):
            return np.arange(proba.shape[0], dtype=int)
        pred_idx = np.argmax(proba, axis=1)
        classes_arr = np.asarray(class_order, dtype=object)
        y_pseudo = classes_arr[pred_idx]
        return _select_pseudo_indices(
            y_pseudo=y_pseudo,
            proba=proba,
            class_order=class_order,
            confidence=float(pseudo_confidence),
            topk_per_class=int(pseudo_topk_per_class),
            balance=bool(pseudo_balance),
        )

    q0 = np.asarray(q_anchor, dtype=np.float64)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    objective_name = str(objective)
    is_bilevel = objective_name.endswith("_bilevel")
    objective_core = objective_name[: -len("_bilevel")] if is_bilevel else objective_name
    if objective_core not in {"entropy", "infomax", "confidence", "pseudo_ce", "lda_nll"}:
        raise ValueError(
            "objective must be one of: "
            "'entropy', 'infomax', 'confidence', 'pseudo_ce', 'lda_nll', or bilevel variants ending with '_bilevel'."
        )

    def _row_entropy(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        return -np.sum(p * np.log(p), axis=1)

    def _solve_inner_wq(*, p: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Lower-level solver: continuous weights w and soft labels q for a fixed Q.

        This is an explicit (iterative) solver that:
        - assigns continuous reliability weights w_i in [0,1] (based on confidence/entropy),
        - computes soft pseudo-labels q_i via class scaling to match a prior π (weighted),
        - returns q̄ = weighted mean(q).
        """

        p = np.asarray(p, dtype=np.float64)
        n = int(p.shape[0])
        if n == 0:
            raise ValueError("Empty p in bilevel solver.")
        prior = np.asarray(prior, dtype=np.float64).reshape(-1)
        prior = np.clip(prior, 1e-12, 1.0)
        prior = prior / float(np.sum(prior))

        temp = float(bilevel_temp)
        it_n = int(bilevel_iters)
        step0 = float(bilevel_step)
        cov_target = float(bilevel_coverage_target)
        cov_pow = float(bilevel_coverage_power)

        a = np.ones(p.shape[1], dtype=np.float64)
        p_adj = np.clip(p, 1e-12, 1.0) ** (1.0 / temp)
        q = p_adj / np.sum(p_adj, axis=1, keepdims=True)
        w = np.ones(n, dtype=np.float64)

        for _ in range(max(1, it_n)):
            q_unn = p_adj * a.reshape(1, -1)
            q_unn = np.clip(q_unn, 1e-12, 1.0)
            q = q_unn / np.sum(q_unn, axis=1, keepdims=True)

            if reliable_metric == "confidence":
                conf = np.max(q, axis=1)
                w = _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
            elif reliable_metric == "entropy":
                ent = _row_entropy(q)
                w = _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
            else:
                w = np.ones(n, dtype=np.float64)

            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                w = np.ones(n, dtype=np.float64)
                w_sum = float(n)

            q_bar = np.sum(w.reshape(-1, 1) * q, axis=0) / w_sum
            q_bar = np.clip(q_bar, 1e-12, 1.0)
            q_bar = q_bar / float(np.sum(q_bar))

            coverage = float(w_sum) / float(n)
            cov_scale = min(1.0, max(0.0, coverage / cov_target))
            if cov_pow != 1.0:
                cov_scale = float(cov_scale) ** float(cov_pow)
            step = float(step0) * float(cov_scale)
            if step > 0.0:
                ratio = prior / q_bar
                ratio = np.clip(ratio, 1e-6, 1e6)
                a = a * (ratio**step)
                a = np.clip(a, 1e-6, 1e6)
                a = a / float(np.exp(np.mean(np.log(a))))

        q_unn = p_adj * a.reshape(1, -1)
        q_unn = np.clip(q_unn, 1e-12, 1.0)
        q = q_unn / np.sum(q_unn, axis=1, keepdims=True)
        w_sum = float(np.sum(w))
        q_bar = np.sum(w.reshape(-1, 1) * q, axis=0) / max(1e-12, w_sum)
        q_bar = np.clip(q_bar, 1e-12, 1.0)
        q_bar = q_bar / float(np.sum(q_bar))

        eff_n = 0.0
        denom = float(np.sum(w * w))
        if denom > 1e-12:
            eff_n = float(w_sum * w_sum) / denom

        stats = {
            "coverage": float(w_sum) / float(n),
            "w_sum": float(w_sum),
            "eff_n": float(eff_n),
            "a": a.copy(),
        }
        return w, q, q_bar, stats

    def _objective_from_proba(
        *, proba: np.ndarray, feats: np.ndarray | None, Q: np.ndarray, theta_vec: np.ndarray | None
    ) -> float:
        proba = np.asarray(proba, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        if objective_core in {"entropy", "infomax", "confidence", "lda_nll"}:
            keep = _maybe_select_keep(proba)
            if keep.size == 0:
                return 1e6
            proba = proba[keep]
            if feats is not None:
                feats = np.asarray(feats, dtype=np.float64)[keep]

        # Normalize to a valid distribution for entropy computations.
        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)

        if objective_core == "lda_nll":
            if feats is None:
                return 1e6
            if lda_evidence is None:
                raise ValueError("lda_evidence must be provided when objective='lda_nll'.")
            mu_e = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
            priors_e = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
            cov_inv_e = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
            logdet_e = float(lda_evidence.get("logdet", 0.0))
            if mu_e.ndim != 2:
                return 1e6
            if cov_inv_e.ndim != 2:
                return 1e6
            if priors_e.shape[0] != int(mu_e.shape[0]):
                return 1e6
            if feats.shape[1] != int(mu_e.shape[1]):
                return 1e6
            if cov_inv_e.shape != (int(mu_e.shape[1]), int(mu_e.shape[1])):
                return 1e6

            f = np.asarray(feats, dtype=np.float64)
            diff = f[:, None, :] - mu_e[None, :, :]
            qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv_e, diff, optimize=True)
            log_norm = float(mu_e.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet_e)
            log_gauss = -0.5 * (log_norm + qf)
            log_pr = np.log(np.clip(priors_e, 1e-12, 1.0)).reshape(1, -1)
            log_joint = log_pr + log_gauss
            m = np.max(log_joint, axis=1, keepdims=True)
            log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

            # Optional reliability weighting w_i based on posterior entropy/confidence.
            w = np.ones(p.shape[0], dtype=np.float64)
            if reliable_metric != "none":
                ent = _row_entropy(p)
                conf = np.max(p, axis=1)
                if reliable_metric == "confidence":
                    w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
                else:
                    w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                return 1e6
            val = float(-np.sum(w * log_p) / w_sum)
        elif objective_core == "pseudo_ce":
            pred_idx = np.argmax(p, axis=1)
            classes_arr = np.asarray(class_order, dtype=object)
            y_pseudo = classes_arr[pred_idx]
            keep = _select_pseudo_indices(
                y_pseudo=y_pseudo,
                proba=p,
                class_order=class_order,
                confidence=float(pseudo_confidence),
                topk_per_class=int(pseudo_topk_per_class),
                balance=bool(pseudo_balance),
            )
            if keep.size == 0:
                return 1e6
            pred_idx_k = np.argmax(p[keep], axis=1)
            conf_k = p[keep, pred_idx_k]
            conf_k = np.clip(conf_k, 1e-12, 1.0)
            nll = -np.log(conf_k)
            val = float(np.mean(conf_k * nll))
        else:
            if is_bilevel:
                if objective_core not in {"entropy", "infomax"}:
                    return 1e6
                prior = (
                    np.ones(int(n_classes), dtype=np.float64) / float(n_classes)
                    if marginal_prior_vec is None
                    else np.asarray(marginal_prior_vec, dtype=np.float64)
                )
                w, q, q_bar, stats = _solve_inner_wq(p=p, prior=prior)
                w_sum = float(stats["w_sum"])
                if w_sum <= 1e-12:
                    return 1e6

                ent_q = _row_entropy(q)
                base = float(np.sum(w * ent_q) / w_sum)
                if objective_core == "infomax":
                    ent_bar = -float(np.sum(q_bar * np.log(q_bar)))
                    base = float(base) - float(infomax_lambda) * float(ent_bar)
                val = float(base)

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    if marginal_mode == "hard_min":
                        if float(np.min(q_bar)) < tau:
                            return 1e6
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((q_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(q_bar)))
                        elif marginal_mode == "kl_prior":
                            if marginal_prior_vec is None:
                                return 1e6
                            pen = float(-np.sum(marginal_prior_vec * np.log(q_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - q_bar) ** 2))
                        val = float(val) + float(marginal_beta) * float(pen)
            else:
                ent = _row_entropy(p)
                conf = np.max(p, axis=1)

                w = np.ones(p.shape[0], dtype=np.float64)
                if reliable_metric != "none":
                    if reliable_metric == "confidence":
                        w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
                    else:
                        w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))

                w_sum = float(np.sum(w))
                if w_sum <= 1e-12:
                    return 1e6

                p_bar = np.mean(p, axis=0)
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))

                if objective_core == "entropy":
                    val = float(np.sum(w * ent) / w_sum)
                elif objective_core == "confidence":
                    val = float(np.sum(w * (1.0 - conf)) / w_sum)
                else:
                    ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
                    val = float(np.sum(w * ent) / w_sum) - float(infomax_lambda) * ent_bar

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    if marginal_mode == "hard_min":
                        if float(np.min(p_bar)) < tau:
                            return 1e6
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((p_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(p_bar)))
                        elif marginal_mode == "kl_prior":
                            if marginal_prior_vec is None:
                                return 1e6
                            pen = float(-np.sum(marginal_prior_vec * np.log(p_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - p_bar) ** 2))
                        val = float(val) + float(marginal_beta) * float(pen)

        if float(trust_lambda) > 0.0:
            val += float(trust_lambda) * float(np.mean((Q - q0) ** 2))
        if l2 > 0.0 and theta_vec is not None:
            val += float(l2) * float(np.mean(theta_vec * theta_vec))
        return float(val)

    def _objective_details_from_proba(
        *, proba: np.ndarray, feats: np.ndarray | None, Q: np.ndarray, theta_vec: np.ndarray | None
    ) -> tuple[float, dict]:
        proba = np.asarray(proba, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        details: dict = {}

        keep = np.arange(proba.shape[0], dtype=int)
        if objective_core in {"entropy", "infomax", "confidence", "lda_nll"}:
            keep = _maybe_select_keep(proba)
            if keep.size == 0:
                return 1e6, {"n_keep": 0}
            proba = proba[keep]
            if feats is not None:
                feats = np.asarray(feats, dtype=np.float64)[keep]
        details["n_keep"] = int(keep.size)

        # Normalize to a valid distribution for entropy computations.
        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        details["n_samples"] = int(p.shape[0])

        if objective_core == "lda_nll":
            if feats is None:
                return 1e6, {"n_keep": int(keep.size)}
            if lda_evidence is None:
                raise ValueError("lda_evidence must be provided when objective='lda_nll'.")
            mu_e = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
            priors_e = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
            cov_inv_e = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
            logdet_e = float(lda_evidence.get("logdet", 0.0))

            f = np.asarray(feats, dtype=np.float64)
            diff = f[:, None, :] - mu_e[None, :, :]
            qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv_e, diff, optimize=True)
            log_norm = float(mu_e.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet_e)
            log_gauss = -0.5 * (log_norm + qf)
            log_pr = np.log(np.clip(priors_e, 1e-12, 1.0)).reshape(1, -1)
            log_joint = log_pr + log_gauss
            m = np.max(log_joint, axis=1, keepdims=True)
            log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

            ent_p = _row_entropy(p)
            conf = np.max(p, axis=1)
            w = np.ones(p.shape[0], dtype=np.float64)
            if reliable_metric != "none":
                if reliable_metric == "confidence":
                    w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
                else:
                    w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent_p))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                return 1e6, {"n_keep": int(keep.size)}
            base = float(-np.sum(w * log_p) / w_sum)
            details["mean_entropy"] = float(np.sum(w * ent_p) / w_sum)
            details["mean_confidence"] = float(np.sum(w * conf) / w_sum)
            details["objective_base"] = float(base)
            val = float(base)

            p_bar = np.mean(p, axis=0)
            p_bar = np.clip(p_bar, 1e-12, 1.0)
            p_bar = p_bar / float(np.sum(p_bar))
            ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
            details["entropy_bar"] = float(ent_bar)

            if marginal_mode != "none":
                tau = float(marginal_tau)
                pen = 0.0
                if marginal_mode == "hard_min":
                    if float(np.min(p_bar)) < tau:
                        return 1e6, details
                elif marginal_mode == "kl_prior":
                    if marginal_prior_vec is None:
                        return 1e6, details
                    pen = float(-np.sum(marginal_prior_vec * np.log(p_bar)))
                elif float(marginal_beta) > 0.0:
                    if marginal_mode == "l2_uniform":
                        u = 1.0 / float(n_classes)
                        pen = float(np.mean((p_bar - u) ** 2))
                    elif marginal_mode == "kl_uniform":
                        pen = float(-np.mean(np.log(p_bar)))
                    else:
                        pen = float(np.mean(np.maximum(0.0, tau - p_bar) ** 2))
                if float(marginal_beta) > 0.0 and marginal_mode != "hard_min":
                    val = float(val) + float(marginal_beta) * float(pen)
                details["pen_marginal"] = float(pen)
        elif objective_core == "pseudo_ce":
            # pseudo_ce: hard pseudo labels + optional filtering
            pred_idx = np.argmax(p, axis=1)
            classes_arr = np.asarray(class_order, dtype=object)
            y_pseudo = classes_arr[pred_idx]
            keep = _select_pseudo_indices(
                y_pseudo=y_pseudo,
                proba=p,
                class_order=class_order,
                confidence=float(pseudo_confidence),
                topk_per_class=int(pseudo_topk_per_class),
                balance=bool(pseudo_balance),
            )
            if keep.size == 0:
                return 1e6, {"n_keep": 0}
            pred_idx_k = np.argmax(p[keep], axis=1)
            conf_k = p[keep, pred_idx_k]
            conf_k = np.clip(conf_k, 1e-12, 1.0)
            nll = -np.log(conf_k)
            # Weight by confidence (encourages self-consistent high-confidence predictions).
            base = float(np.mean(conf_k * nll))
            details["objective_base"] = base
            val = float(base)
        else:
            ent_p = _row_entropy(p)
            conf_p = np.max(p, axis=1)
            details["mean_entropy"] = float(np.mean(ent_p))
            details["mean_confidence"] = float(np.mean(conf_p))

            if is_bilevel:
                if objective_core not in {"entropy", "infomax"}:
                    return 1e6, details
                prior = (
                    np.ones(int(n_classes), dtype=np.float64) / float(n_classes)
                    if marginal_prior_vec is None
                    else np.asarray(marginal_prior_vec, dtype=np.float64)
                )
                w, q, q_bar, stats = _solve_inner_wq(p=p, prior=prior)
                w_sum = float(stats["w_sum"])
                if w_sum <= 1e-12:
                    return 1e6, details
                details["coverage"] = float(stats["coverage"])
                details["eff_n"] = float(stats["eff_n"])
                details["q_bar"] = np.asarray(q_bar, dtype=np.float64).copy()

                ent_q = _row_entropy(q)
                details["mean_entropy_q"] = float(np.mean(ent_q))
                base = float(np.sum(w * ent_q) / w_sum)
                if objective_core == "infomax":
                    ent_bar = -float(np.sum(q_bar * np.log(q_bar)))
                    details["entropy_bar"] = float(ent_bar)
                    base = float(base) - float(infomax_lambda) * float(ent_bar)
                details["objective_base"] = float(base)
                val = float(base)

                # For reference: unweighted marginal of p.
                p_bar = np.mean(p, axis=0)
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))
                details["p_bar"] = p_bar.copy()

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    pen = 0.0
                    if marginal_mode == "hard_min":
                        if float(np.min(q_bar)) < tau:
                            return 1e6, details
                    elif marginal_mode == "kl_prior":
                        if marginal_prior_vec is None:
                            return 1e6, details
                        pen = float(-np.sum(marginal_prior_vec * np.log(q_bar)))
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((q_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(q_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - q_bar) ** 2))
                    if float(marginal_beta) > 0.0 and marginal_mode != "hard_min":
                        val = float(val) + float(marginal_beta) * float(pen)
                    details["pen_marginal"] = float(pen)
            else:
                w = np.ones(p.shape[0], dtype=np.float64)
                if reliable_metric != "none":
                    if reliable_metric == "confidence":
                        w = w * _sigmoid(float(reliable_alpha) * (conf_p - float(reliable_threshold)))
                    else:
                        w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent_p))

                w_sum = float(np.sum(w))
                if w_sum <= 1e-12:
                    return 1e6, {"n_keep": int(p.shape[0])}

                p_bar = np.mean(p, axis=0)
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))
                details["p_bar"] = p_bar.copy()

                if objective_core == "entropy":
                    base = float(np.sum(w * ent_p) / w_sum)
                    details["objective_base"] = base
                    val = float(base)
                elif objective_core == "confidence":
                    base = float(np.sum(w * (1.0 - conf_p)) / w_sum)
                    details["objective_base"] = base
                    val = float(base)
                else:
                    ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
                    base = float(np.sum(w * ent_p) / w_sum) - float(infomax_lambda) * ent_bar
                    details["entropy_bar"] = float(ent_bar)
                    details["objective_base"] = base
                    val = float(base)

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    pen = 0.0
                    if marginal_mode == "hard_min":
                        if float(np.min(p_bar)) < tau:
                            return 1e6, details
                    elif marginal_mode == "kl_prior":
                        if marginal_prior_vec is None:
                            return 1e6, details
                        pen = float(-np.sum(marginal_prior_vec * np.log(p_bar)))
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((p_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(p_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - p_bar) ** 2))
                    if float(marginal_beta) > 0.0 and marginal_mode != "hard_min":
                        val = float(val) + float(marginal_beta) * float(pen)
                    details["pen_marginal"] = float(pen)

        pen_trust = 0.0
        if float(trust_lambda) > 0.0:
            pen_trust = float(trust_lambda) * float(np.mean((Q - q0) ** 2))
            val += pen_trust
        details["pen_trust"] = float(pen_trust)
        pen_l2 = 0.0
        if l2 > 0.0 and theta_vec is not None:
            pen_l2 = float(l2) * float(np.mean(theta_vec * theta_vec))
            val += pen_l2
        details["pen_l2"] = float(pen_l2)
        details["objective"] = float(val)
        return float(val), details

    def eval_theta(theta_vec: np.ndarray, z_data: np.ndarray) -> float:
        proba, Q, feats = _proba_from_theta(theta_vec, z_data)
        return _objective_from_proba(proba=proba, feats=feats, Q=Q, theta_vec=theta_vec)

    def _evidence_nll_from_outputs(*, proba: np.ndarray, feats: np.ndarray) -> float:
        """Compute -log p(z) under the frozen CSP+LDA Gaussian mixture (if available)."""

        if lda_evidence is None:
            return float("nan")

        proba = np.asarray(proba, dtype=np.float64)
        feats = np.asarray(feats, dtype=np.float64)

        keep = _maybe_select_keep(proba)
        if keep.size == 0:
            return float("nan")
        proba = proba[keep]
        feats = feats[keep]

        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)

        mu_e = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
        priors_e = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
        cov_inv_e = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
        logdet_e = float(lda_evidence.get("logdet", 0.0))
        if mu_e.ndim != 2:
            return float("nan")
        if feats.shape[1] != int(mu_e.shape[1]):
            return float("nan")

        diff = feats[:, None, :] - mu_e[None, :, :]
        qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv_e, diff, optimize=True)
        log_norm = float(mu_e.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet_e)
        log_gauss = -0.5 * (log_norm + qf)
        log_pr = np.log(np.clip(priors_e, 1e-12, 1.0)).reshape(1, -1)
        log_joint = log_pr + log_gauss
        m = np.max(log_joint, axis=1, keepdims=True)
        log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

        w = np.ones(p.shape[0], dtype=np.float64)
        if reliable_metric != "none":
            ent = _row_entropy(p)
            conf = np.max(p, axis=1)
            if reliable_metric == "confidence":
                w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
            else:
                w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return float("nan")
        return float(-np.sum(w * log_p) / w_sum)

    def _probe_mixup_from_outputs(
        *,
        proba: np.ndarray,
        feats: np.ndarray,
        seed_local: int,
        n_pairs: int = 200,
        lam: float = 0.5,
        mode: str = "soft",
        beta_alpha: float = 0.0,
    ) -> tuple[float, dict]:
        """MixUp-style probe score on CSP feature space (label-free).

        Lower is better. Uses pseudo labels from `proba` after reliable filtering.
        """

        proba = np.asarray(proba, dtype=np.float64)
        feats = np.asarray(feats, dtype=np.float64)
        keep = _maybe_select_keep(proba)
        if keep.size == 0:
            return float("nan"), {"n_keep": 0, "n_pairs": 0}

        p = np.asarray(proba[keep], dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        f = np.asarray(feats[keep], dtype=np.float64)

        y_idx = np.argmax(p, axis=1).astype(int)
        classes = np.unique(y_idx).tolist()
        idx_by_class = {c: np.flatnonzero(y_idx == c) for c in classes}
        classes_intra = [c for c in classes if int(idx_by_class[c].size) >= 2]
        has_inter = len(classes) >= 2
        if not classes_intra and not has_inter:
            return float("nan"), {"n_keep": int(keep.size), "n_pairs": 0}

        rng_probe = np.random.RandomState(int(seed_local))
        n_pairs = int(max(0, n_pairs))
        if n_pairs == 0:
            return float("nan"), {"n_keep": int(keep.size), "n_pairs": 0}
        n_pairs = min(n_pairs, 10_000)

        mode = str(mode)
        if mode not in {"soft", "hard_major"}:
            raise ValueError("probe_mixup mode must be 'soft' or 'hard_major'.")
        beta_alpha = float(beta_alpha)
        if beta_alpha < 0.0:
            raise ValueError("beta_alpha must be >= 0.")

        i_list: list[int] = []
        j_list: list[int] = []
        ki_list: list[int] = []
        kj_list: list[int] = []
        lam_list: list[float] = []
        same_list: list[bool] = []

        # Fallback to all indices for sampling.
        for _ in range(n_pairs):
            use_intra = bool(classes_intra) and (not has_inter or rng_probe.rand() < 0.5)
            if use_intra:
                c = int(rng_probe.choice(classes_intra))
                idxs = idx_by_class[c]
                a, b = rng_probe.choice(idxs, size=2, replace=False).tolist()
                lam_val = float(lam)
                if beta_alpha > 0.0:
                    lam_val = float(rng_probe.beta(beta_alpha, beta_alpha))
                    lam_val = float(np.clip(lam_val, 1e-6, 1.0 - 1e-6))
                i_list.append(int(a))
                j_list.append(int(b))
                ki_list.append(int(c))
                kj_list.append(int(c))
                lam_list.append(float(lam_val))
                same_list.append(True)
            else:
                # Inter-class: sample two different classes.
                if not has_inter:
                    continue
                c1, c2 = rng_probe.choice(classes, size=2, replace=False).tolist()
                a = int(rng_probe.choice(idx_by_class[int(c1)]))
                b = int(rng_probe.choice(idx_by_class[int(c2)]))
                lam_val = float(lam)
                if beta_alpha > 0.0:
                    lam_val = float(rng_probe.beta(beta_alpha, beta_alpha))
                    lam_val = float(np.clip(lam_val, 1e-6, 1.0 - 1e-6))
                # MixVal-style: when λ>0.5 use the hard pseudo label of the dominant sample.
                # We implement this by folding λ to [0.5,1] and swapping (i,j) when needed.
                if mode == "hard_major" and lam_val < 0.5:
                    a, b = b, a
                    c1, c2 = c2, c1
                    lam_val = 1.0 - lam_val
                i_list.append(int(a))
                j_list.append(int(b))
                ki_list.append(int(c1))
                kj_list.append(int(c2))
                lam_list.append(float(lam_val))
                same_list.append(False)

        if not i_list:
            return float("nan"), {"n_keep": int(keep.size), "n_pairs": 0}

        i_arr = np.asarray(i_list, dtype=int)
        j_arr = np.asarray(j_list, dtype=int)
        lam_arr = np.asarray(lam_list, dtype=np.float64)
        ki_arr = np.asarray(ki_list, dtype=int)
        kj_arr = np.asarray(kj_list, dtype=int)
        same_arr = np.asarray(same_list, dtype=bool)

        f_mix = lam_arr.reshape(-1, 1) * f[i_arr] + (1.0 - lam_arr).reshape(-1, 1) * f[j_arr]
        proba_mix = lda.predict_proba(f_mix)
        proba_mix = _reorder_proba_columns(proba_mix, lda.classes_, list(class_order))
        proba_mix = np.clip(proba_mix, 1e-12, 1.0)
        proba_mix = proba_mix / np.sum(proba_mix, axis=1, keepdims=True)
        logp = np.log(proba_mix)

        lp_i = logp[np.arange(logp.shape[0]), ki_arr]
        lp_j = logp[np.arange(logp.shape[0]), kj_arr]
        if mode == "hard_major":
            # After folding λ>=0.5 and (i,j) swap, class ki is always the dominant pseudo label.
            ce = -lp_i
        else:
            ce = np.where(same_arr, -lp_i, -(lam_arr * lp_i + (1.0 - lam_arr) * lp_j))
        score = float(np.mean(ce))
        stats = {
            "n_keep": int(keep.size),
            "n_pairs": int(ce.size),
            "frac_intra": float(np.mean(same_arr)) if same_arr.size else float("nan"),
        }
        return float(score), stats

    def _record_candidate(*, kind: str, iter_idx: int, theta_vec: np.ndarray | None, Q: np.ndarray) -> None:
        if not do_diag:
            return
        proba_best, feats_best = _proba_from_Q(Q, z_best)
        obj, details = _objective_details_from_proba(proba=proba_best, feats=feats_best, Q=Q, theta_vec=theta_vec)
        drift_best_vec = _kl_drift_vec(proba_anchor_best, proba_best)
        drift_best = float(np.mean(drift_best_vec))
        proba_full, feats_full = _proba_from_Q(Q, z_t)
        drift_full_vec = _kl_drift_vec(proba_anchor_full, proba_full)
        drift_full = float(np.mean(drift_full_vec))
        p_bar_full = np.mean(np.clip(proba_full, 1e-12, 1.0), axis=0)
        p_bar_full = p_bar_full / float(np.sum(p_bar_full))
        evidence_nll_best = _evidence_nll_from_outputs(proba=proba_best, feats=feats_best)
        evidence_nll_full = _evidence_nll_from_outputs(proba=proba_full, feats=feats_full)
        probe_mixup_best, probe_stats_best = _probe_mixup_from_outputs(
            proba=proba_best,
            feats=feats_best,
            seed_local=int(seed) + 10007 * (int(len(diag_records)) + 1),
        )
        probe_mixup_hard_best, probe_stats_hard_best = _probe_mixup_from_outputs(
            proba=proba_best,
            feats=feats_best,
            seed_local=int(seed) + 10007 * (int(len(diag_records)) + 1),
            mode="hard_major",
            beta_alpha=0.4,
        )
        probe_mixup_full, probe_stats_full = _probe_mixup_from_outputs(
            proba=proba_full,
            feats=feats_full,
            seed_local=int(seed) + 10007 * (int(len(diag_records)) + 1) + 13,
        )
        probe_mixup_hard_full, probe_stats_hard_full = _probe_mixup_from_outputs(
            proba=proba_full,
            feats=feats_full,
            seed_local=int(seed) + 10007 * (int(len(diag_records)) + 1) + 13,
            mode="hard_major",
            beta_alpha=0.4,
        )
        rec = {
            "kind": str(kind),
            "iter": int(iter_idx),
            "order": int(len(diag_records)),
            "objective": float(obj),
            "score": float(_score_with_drift(float(obj), float(drift_best))),
            "objective_base": float(details.get("objective_base", np.nan)),
            "pen_marginal": float(details.get("pen_marginal", 0.0)),
            "pen_trust": float(details.get("pen_trust", 0.0)),
            "pen_l2": float(details.get("pen_l2", 0.0)),
            "mean_entropy": float(details.get("mean_entropy", np.nan)),
            "mean_entropy_q": float(details.get("mean_entropy_q", np.nan)),
            "mean_confidence": float(details.get("mean_confidence", np.nan)),
            "entropy_bar": float(details.get("entropy_bar", np.nan)),
            "n_keep": int(details.get("n_keep", -1)),
            "n_best_total": int(z_best.shape[0]),
            "n_full_total": int(z_t.shape[0]),
            "evidence_nll_best": float(evidence_nll_best),
            "evidence_nll_full": float(evidence_nll_full),
            "probe_mixup_best": float(probe_mixup_best),
            "probe_mixup_full": float(probe_mixup_full),
            "probe_mixup_pairs_best": int(probe_stats_best.get("n_pairs", 0)),
            "probe_mixup_pairs_full": int(probe_stats_full.get("n_pairs", 0)),
            "probe_mixup_keep_best": int(probe_stats_best.get("n_keep", 0)),
            "probe_mixup_keep_full": int(probe_stats_full.get("n_keep", 0)),
            "probe_mixup_frac_intra_best": float(probe_stats_best.get("frac_intra", np.nan)),
            "probe_mixup_frac_intra_full": float(probe_stats_full.get("frac_intra", np.nan)),
            "probe_mixup_hard_best": float(probe_mixup_hard_best),
            "probe_mixup_hard_full": float(probe_mixup_hard_full),
            "probe_mixup_hard_pairs_best": int(probe_stats_hard_best.get("n_pairs", 0)),
            "probe_mixup_hard_pairs_full": int(probe_stats_hard_full.get("n_pairs", 0)),
            "probe_mixup_hard_keep_best": int(probe_stats_hard_best.get("n_keep", 0)),
            "probe_mixup_hard_keep_full": int(probe_stats_hard_full.get("n_keep", 0)),
            "probe_mixup_hard_frac_intra_best": float(probe_stats_hard_best.get("frac_intra", np.nan)),
            "probe_mixup_hard_frac_intra_full": float(probe_stats_hard_full.get("frac_intra", np.nan)),
            "drift_best": float(drift_best),
            "drift_best_std": float(np.std(drift_best_vec)),
            "drift_best_q50": float(np.quantile(drift_best_vec, 0.50)),
            "drift_best_q90": float(np.quantile(drift_best_vec, 0.90)),
            "drift_best_q95": float(np.quantile(drift_best_vec, 0.95)),
            "drift_best_max": float(np.max(drift_best_vec)),
            "drift_best_tail_frac": float(np.mean(drift_best_vec > float(drift_delta))) if float(drift_delta) > 0.0 else 0.0,
            "drift_full": float(drift_full),
            "drift_full_std": float(np.std(drift_full_vec)),
            "drift_full_q50": float(np.quantile(drift_full_vec, 0.50)),
            "drift_full_q90": float(np.quantile(drift_full_vec, 0.90)),
            "drift_full_q95": float(np.quantile(drift_full_vec, 0.95)),
            "drift_full_max": float(np.max(drift_full_vec)),
            "drift_full_tail_frac": float(np.mean(drift_full_vec > float(drift_delta))) if float(drift_delta) > 0.0 else 0.0,
            "coverage": float(details.get("coverage", np.nan)),
            "eff_n": float(details.get("eff_n", np.nan)),
            "p_bar_full": p_bar_full.astype(np.float64),
            "q_bar": details.get("q_bar", None),
            "transform": str(transform),
            "Q": np.asarray(Q, dtype=np.float64),
        }
        if transform == "rot_scale" and theta_vec is not None and int(theta_vec.shape[0]) > rot_dim:
            log_s = np.asarray(theta_vec[rot_dim:], dtype=np.float64)
            rec["log_s_mean_abs"] = float(np.mean(np.abs(log_s)))
            rec["log_s_max_abs"] = float(np.max(np.abs(log_s)))
        diag_records.append(rec)

    # Optional warm start: build Q_Δ from pseudo-label Δ-alignment and approximate it with our Givens pairs.
    q_delta: np.ndarray | None = None
    if (warm_start == "delta" or trust_q0 == "delta") and int(warm_iters) > 0:
        q_cur = np.eye(int(n_channels), dtype=np.float64)
        for _ in range(int(warm_iters)):
            X_cur = apply_spatial_transform(q_cur, z_t)
            proba = model.predict_proba(X_cur)
            proba = _reorder_proba_columns(proba, model.classes_, list(class_order))
            try:
                if pseudo_mode == "soft":
                    d_t = _soft_class_cov_diff(
                        z_t,
                        proba=proba,
                        class_order=class_order,
                        eps=float(eps),
                        shrinkage=float(shrinkage),
                    )
                else:
                    y_pseudo = np.asarray(model.predict(X_cur))
                    keep = _select_pseudo_indices(
                        y_pseudo=y_pseudo,
                        proba=proba,
                        class_order=class_order,
                        confidence=float(pseudo_confidence),
                        topk_per_class=int(pseudo_topk_per_class),
                        balance=bool(pseudo_balance),
                    )
                    if keep.size == 0:
                        q_delta = None
                        break
                    d_t = class_cov_diff(
                        z_t[keep],
                        y_pseudo[keep],
                        class_order=class_order,
                        eps=float(eps),
                        shrinkage=float(shrinkage),
                    )
                q_cur = orthogonal_align_symmetric(d_t, d_ref)
                q_delta = q_cur
            except ValueError:
                q_delta = None
                break

        if trust_q0 == "delta" and q_delta is not None:
            q0 = blend_with_identity(q_delta, float(q_blend))

        if q_delta is not None:
            # Greedy 2D Procrustes per plane to get a good rotation initialization (angles on our Givens pairs).
            q_work = np.eye(int(n_channels), dtype=np.float64)
            phi_init = np.zeros(rot_dim, dtype=np.float64)
            for k, (i, j) in enumerate(pairs):
                a = q_work[:, i]
                b = q_work[:, j]
                ti = q_delta[:, i]
                tj = q_delta[:, j]
                m11 = float(np.dot(a, ti))
                m12 = float(np.dot(a, tj))
                m21 = float(np.dot(b, ti))
                m22 = float(np.dot(b, tj))
                angle = float(np.arctan2(m21 - m12, m11 + m22))
                phi_init[k] = angle
                # Apply the rotation to q_work columns i,j (right multiplication).
                c = float(np.cos(angle))
                s = float(np.sin(angle))
                col_i = q_work[:, i].copy()
                col_j = q_work[:, j].copy()
                q_work[:, i] = c * col_i + s * col_j
                q_work[:, j] = -s * col_i + c * col_j
            if warm_start == "delta":
                theta[:rot_dim] = phi_init.copy()

    # Candidate set on holdout (Step A): always include anchor (EA) and, if available, Q_delta.
    best_Q_override: np.ndarray | None = None
    theta_id = np.zeros_like(best_theta)
    proba_id = proba_anchor_best
    obj_id = _objective_from_proba(
        proba=proba_id,
        feats=feats_anchor_best,
        Q=q_anchor,
        theta_vec=theta_id,
    )
    score_id = _score_with_drift(float(obj_id), 0.0)
    _record_candidate(kind="identity", iter_idx=-1, theta_vec=theta_id, Q=q_anchor)
    best_theta = theta_id.copy()
    best_obj = float(obj_id)
    best_score = float(score_id)

    if q_delta is not None:
        q_delta_b = blend_with_identity(q_delta, float(q_blend))
        proba_qd, feats_qd = _proba_from_Q(q_delta_b, z_best)
        obj_qd = _objective_from_proba(proba=proba_qd, feats=feats_qd, Q=q_delta_b, theta_vec=None)
        drift_qd = _kl_drift(proba_anchor_best, proba_qd)
        score_qd = _score_with_drift(float(obj_qd), float(drift_qd))
        _record_candidate(kind="q_delta", iter_idx=-2, theta_vec=None, Q=q_delta_b)
        if score_qd < best_score:
            best_obj = float(obj_qd)
            best_score = float(score_qd)
            best_Q_override = q_delta_b

    # If we warm-started (rotation angles), compare that initial point too.
    if np.any(theta != 0.0):
        proba_init, Q_init, feats_init = _proba_from_theta(theta, z_best)
        obj_init = _objective_from_proba(proba=proba_init, feats=feats_init, Q=Q_init, theta_vec=theta)
        drift_init = _kl_drift(proba_anchor_best, proba_init)
        score_init = _score_with_drift(float(obj_init), float(drift_init))
        if do_diag:
            q_init = _build_transform(theta)
            _record_candidate(kind="warm_init", iter_idx=0, theta_vec=theta.copy(), Q=q_init)
        if score_init < best_score:
            best_obj = float(obj_init)
            best_score = float(score_init)
            best_theta = theta.copy()
            best_Q_override = None

    # SPSA / two-point random-direction estimator
    for t in range(int(iters)):
        u = rng.choice([-1.0, 1.0], size=theta.shape[0]).astype(np.float64)
        theta_plus = theta + float(mu) * u
        theta_minus = theta - float(mu) * u
        f_plus = eval_theta(theta_plus, z_opt)
        f_minus = eval_theta(theta_minus, z_opt)
        g = (f_plus - f_minus) / (2.0 * float(mu)) * u
        step = float(lr) / np.sqrt(float(t) + 1.0)
        theta = theta - step * g

        # Track best iterate (not the +/- perturbations used only for gradient estimation).
        proba_tmp, Q_tmp, feats_tmp = _proba_from_theta(theta, z_best)
        f_theta = _objective_from_proba(proba=proba_tmp, feats=feats_tmp, Q=Q_tmp, theta_vec=theta)
        drift_tmp = _kl_drift(proba_anchor_best, proba_tmp)
        score_tmp = _score_with_drift(float(f_theta), float(drift_tmp))
        if do_diag:
            _record_candidate(kind="iter", iter_idx=int(t + 1), theta_vec=theta.copy(), Q=Q_tmp)
        if score_tmp < best_score:
            best_obj = float(f_theta)
            best_score = float(score_tmp)
            best_theta = theta.copy()
            best_Q_override = None

    # Optional safety: require a minimum improvement over identity (otherwise keep identity).
    if float(min_improvement) > 0.0 and (float(score_id) - float(best_score)) < float(min_improvement):
        best_theta = theta_id.copy()
        best_obj = float(obj_id)
        best_score = float(score_id)
        best_Q_override = None

    if best_Q_override is not None:
        Q = best_Q_override
    else:
        Q = _build_transform(best_theta)

    # Safety fallback: if target predictions collapse to a single class, fall back to a safer Q.
    if float(fallback_min_marginal_entropy) > 0.0:
        proba_best, _feats_best = _proba_from_Q(Q, z_t)
        p_bar = np.mean(np.clip(proba_best, 1e-12, 1.0), axis=0)
        p_bar = p_bar / float(np.sum(p_bar))
        ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
        if ent_bar < float(fallback_min_marginal_entropy):
            # Compare against identity (EA) and optional Q_delta.
            candidates: List[np.ndarray] = [np.eye(int(n_channels), dtype=np.float64)]
            if q_delta is not None:
                candidates.append(blend_with_identity(q_delta, float(q_blend)))

            best_ent = -1.0
            best_q = candidates[0]
            for q_cand in candidates:
                FQ = F @ q_cand
                Y = np.einsum("kc,nct->nkt", FQ, z_t, optimize=True)
                power = np.mean(Y * Y, axis=2)
                power = np.maximum(power, 1e-20)
                feats = np.log(power) if use_log else power
                proba_cand = lda.predict_proba(feats)
                proba_cand = _reorder_proba_columns(proba_cand, lda.classes_, list(class_order))
                p_bar_c = np.mean(np.clip(proba_cand, 1e-12, 1.0), axis=0)
                p_bar_c = p_bar_c / float(np.sum(p_bar_c))
                ent_c = -float(np.sum(p_bar_c * np.log(p_bar_c)))
                if ent_c > best_ent:
                    best_ent = ent_c
                    best_q = q_cand
            Q = best_q

    if do_diag:
        diag = {
            "records": diag_records,
            "class_order": list(class_order),
            "transform": str(transform),
            "marginal_mode": str(marginal_mode),
            "marginal_prior": None if marginal_prior_vec is None else marginal_prior_vec.astype(np.float64),
            "drift_mode": str(drift_mode),
            "drift_gamma": float(drift_gamma),
            "drift_delta": float(drift_delta),
        }
        return Q, diag
    return Q


def _sample_givens_pairs(
    *, n_channels: int, n_rotations: int, rng: np.random.RandomState
) -> List[tuple[int, int]]:
    if n_rotations < 1:
        raise ValueError("n_rotations must be >= 1.")
    all_pairs: List[tuple[int, int]] = []
    for i in range(int(n_channels)):
        for j in range(i + 1, int(n_channels)):
            all_pairs.append((i, j))
    rng.shuffle(all_pairs)
    n_rotations = min(int(n_rotations), len(all_pairs))
    return all_pairs[:n_rotations]


def _apply_givens_right(mat: np.ndarray, *, pairs: List[tuple[int, int]], angles: np.ndarray) -> np.ndarray:
    """Return mat @ Q(angles) where Q is a product of Givens rotations (right-multiplication)."""

    out = np.asarray(mat, dtype=np.float64).copy()
    angles = np.asarray(angles, dtype=np.float64)
    if len(pairs) != angles.shape[0]:
        raise ValueError("pairs and angles length mismatch.")

    for (i, j), theta in zip(pairs, angles):
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        col_i = out[:, i].copy()
        col_j = out[:, j].copy()
        out[:, i] = c * col_i + s * col_j
        out[:, j] = -s * col_i + c * col_j
    return out


def _build_q_from_givens(
    *, n_channels: int, pairs: List[tuple[int, int]], angles: np.ndarray
) -> np.ndarray:
    Q = np.eye(int(n_channels), dtype=np.float64)
    Q = _apply_givens_right(Q, pairs=pairs, angles=angles)
    return Q
