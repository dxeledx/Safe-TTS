#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csp_lda.certificate import candidate_features_from_record, train_logistic_guard, train_ridge_certificate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Offline SAFE-TTA selection from saved LOSO prediction CSVs.\n\n"
            "This script performs per-subject LOSO-style calibration (train on other subjects only) to select\n"
            "between an EA anchor and one candidate method using only unlabeled target prediction statistics.\n"
            "It is designed for large datasets (e.g., PhysionetMI) where re-running heavy stacked calibration in a\n"
            "single process can hit OOM.\n"
        )
    )
    p.add_argument("--ea-preds", type=Path, required=True, help="Path to EA predictions_all_methods.csv.")
    p.add_argument("--cand-preds", type=Path, required=True, help="Path to candidate predictions_all_methods.csv.")
    p.add_argument(
        "--cand-family",
        type=str,
        default="rpa",
        help="Candidate family name used in feature one-hots (default: rpa).",
    )
    p.add_argument(
        "--guard-threshold",
        type=float,
        default=0.5,
        help="Accept candidate if guard_p_pos >= threshold (default: 0.5).",
    )
    p.add_argument(
        "--anchor-guard-delta",
        type=float,
        default=0.05,
        help="Additional constraint: guard_p_pos(cand) >= guard_p_pos(anchor) + delta (default: 0.05).",
    )
    p.add_argument(
        "--min-pred-improve",
        type=float,
        default=0.02,
        help="Accept candidate only if predicted improvement >= eps (default: 0.02).",
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regression alpha (default: 1.0).",
    )
    p.add_argument(
        "--guard-c",
        type=float,
        default=1.0,
        help="Logistic guard C (default: 1.0).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (will write predictions_all_methods.csv and method_comparison.csv).",
    )
    p.add_argument(
        "--method-name",
        type=str,
        default="offline-safe-tta",
        help="Method name to write in output CSVs (default: offline-safe-tta).",
    )
    p.add_argument(
        "--date-prefix",
        type=str,
        default="",
        help="File prefix for outputs (default: today's YYYYMMDD).",
    )
    return p.parse_args()


def _row_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p0 = np.clip(p0, 1e-12, 1.0)
    p1 = np.clip(p1, 1e-12, 1.0)
    p0 = p0 / np.sum(p0, axis=1, keepdims=True)
    p1 = p1 / np.sum(p1, axis=1, keepdims=True)
    return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)


def _record_from_proba(
    *,
    p_id: np.ndarray,
    p_c: np.ndarray,
    y_pred_id: np.ndarray,
    y_pred_c: np.ndarray,
    cand_family: str,
    kind: str,
    drift_delta: float = 0.0,
) -> dict:
    p_c = np.asarray(p_c, dtype=np.float64)
    p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
    p_bar = p_bar / float(np.sum(p_bar))
    ent = _row_entropy(p_c)
    ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))
    mean_conf = float(np.mean(np.max(np.clip(p_c, 1e-12, 1.0), axis=1)))

    d = _drift_vec(p_id, p_c)
    y_pred_id = np.asarray(y_pred_id, dtype=object).reshape(-1)
    y_pred_c = np.asarray(y_pred_c, dtype=object).reshape(-1)
    pred_disagree = float(np.mean(y_pred_id != y_pred_c)) if y_pred_id.shape == y_pred_c.shape else 0.0

    rec = {
        "kind": str(kind),
        "cand_family": str(cand_family).strip().lower(),
        "cand_rank": 0.0,
        "cand_lambda": 0.0,
        # Base (label-free) stats
        "objective_base": float(np.mean(ent)),
        "pen_marginal": 0.0,
        "mean_entropy": float(np.mean(ent)),
        "entropy_bar": float(ent_bar),
        "mean_confidence": float(mean_conf),
        "pred_disagree": float(pred_disagree),
        "drift_best": float(np.mean(d)),
        "drift_best_std": float(np.std(d)),
        "drift_best_q90": float(np.quantile(d, 0.90)),
        "drift_best_q95": float(np.quantile(d, 0.95)),
        "drift_best_max": float(np.max(d)),
        "drift_best_tail_frac": float(np.mean(d > float(drift_delta))) if float(drift_delta) > 0.0 else 0.0,
        "p_bar_full": p_bar.astype(np.float64),
        "q_bar": np.zeros_like(p_bar, dtype=np.float64),
    }
    # Back-compat keys used by some selectors.
    rec["objective"] = float(rec["objective_base"])
    rec["score"] = float(rec["objective_base"])
    return rec


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = str(args.date_prefix).strip() or datetime.now().strftime("%Y%m%d")

    ea = pd.read_csv(Path(args.ea_preds))
    cand = pd.read_csv(Path(args.cand_preds))

    required = {"subject", "trial", "y_true", "y_pred"}
    if not required.issubset(ea.columns):
        raise ValueError(f"EA CSV missing columns {sorted(required - set(ea.columns))}.")
    if not required.issubset(cand.columns):
        raise ValueError(f"Candidate CSV missing columns {sorted(required - set(cand.columns))}.")

    proba_cols = [c for c in ea.columns if str(c).startswith("proba_")]
    if not proba_cols:
        raise ValueError("EA CSV has no proba_* columns.")
    for c in proba_cols:
        if c not in cand.columns:
            raise ValueError(f"Candidate CSV missing proba column: {c}")

    class_order = [str(c).replace("proba_", "", 1) for c in proba_cols]
    n_classes = int(len(class_order))
    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")

    # Merge and validate alignment.
    m = ea.merge(cand, on=["subject", "trial", "y_true"], suffixes=("_ea", "_cand"))
    if m.empty:
        raise RuntimeError("Merged dataframe is empty; check that subject/trial/y_true keys align.")

    subjects = sorted({int(s) for s in m["subject"].unique().tolist()})

    # Pre-slice per subject for speed.
    by_subject: dict[int, pd.DataFrame] = {int(s): g.copy() for s, g in m.groupby("subject", sort=True)}

    rows_all: list[dict] = []
    chosen_y_pred: list[pd.Series] = []
    chosen_proba: list[np.ndarray] = []

    per_subject_summary_rows: list[dict] = []

    for t in subjects:
        train_subjects = [s for s in subjects if s != int(t)]
        if len(train_subjects) < 2:
            raise RuntimeError("Need at least 3 subjects for LOSO-style calibration.")

        # Build training set: one row per training subject (candidate only).
        X_rows: list[np.ndarray] = []
        y_improve: list[float] = []
        y_guard: list[int] = []
        feat_names: tuple[str, ...] | None = None

        for s in train_subjects:
            df_s = by_subject[int(s)]
            y_true = df_s["y_true"].to_numpy(object)

            p_id = df_s[[f"{c}_ea" for c in proba_cols]].to_numpy(np.float64)
            p_c = df_s[[f"{c}_cand" for c in proba_cols]].to_numpy(np.float64)
            y_pred_id = df_s["y_pred_ea"].to_numpy(object)
            y_pred_c = df_s["y_pred_cand"].to_numpy(object)

            acc_id = float(np.mean(y_pred_id == y_true))
            acc_c = float(np.mean(y_pred_c == y_true))
            improve = float(acc_c - acc_id)

            rec_id = _record_from_proba(
                p_id=p_id,
                p_c=p_id,
                y_pred_id=y_pred_id,
                y_pred_c=y_pred_id,
                cand_family="ea",
                kind="identity",
            )
            rec_c = _record_from_proba(
                p_id=p_id,
                p_c=p_c,
                y_pred_id=y_pred_id,
                y_pred_c=y_pred_c,
                cand_family=str(args.cand_family),
                kind="candidate",
            )

            x, names = candidate_features_from_record(rec_c, n_classes=n_classes, include_pbar=True)
            x0, names0 = candidate_features_from_record(rec_id, n_classes=n_classes, include_pbar=True)
            if names != names0:
                raise RuntimeError("Feature name mismatch between anchor and candidate.")
            x_delta = np.asarray(x - x0, dtype=np.float64).reshape(-1)
            names_delta = tuple([f"delta_{n}" for n in names])

            if feat_names is None:
                feat_names = names_delta

            X_rows.append(x_delta)
            y_improve.append(improve)
            y_guard.append(1 if improve > 0.0 else 0)

        X_tr = np.vstack(X_rows).astype(np.float64, copy=False)
        y_tr = np.asarray(y_improve, dtype=np.float64)
        yb_tr = np.asarray(y_guard, dtype=int)
        if feat_names is None:
            raise RuntimeError("No training rows.")

        cert = train_ridge_certificate(X_tr, y_tr, feature_names=feat_names, alpha=float(args.ridge_alpha))
        guard = train_logistic_guard(X_tr, yb_tr, feature_names=feat_names, c=float(args.guard_c))

        # Test subject features.
        df_t = by_subject[int(t)]
        y_true_t = df_t["y_true"].to_numpy(object)
        p_id_t = df_t[[f"{c}_ea" for c in proba_cols]].to_numpy(np.float64)
        p_c_t = df_t[[f"{c}_cand" for c in proba_cols]].to_numpy(np.float64)
        y_pred_id_t = df_t["y_pred_ea"].to_numpy(object)
        y_pred_c_t = df_t["y_pred_cand"].to_numpy(object)

        rec_id_t = _record_from_proba(
            p_id=p_id_t,
            p_c=p_id_t,
            y_pred_id=y_pred_id_t,
            y_pred_c=y_pred_id_t,
            cand_family="ea",
            kind="identity",
        )
        rec_c_t = _record_from_proba(
            p_id=p_id_t,
            p_c=p_c_t,
            y_pred_id=y_pred_id_t,
            y_pred_c=y_pred_c_t,
            cand_family=str(args.cand_family),
            kind="candidate",
        )

        x_t, names_t = candidate_features_from_record(rec_c_t, n_classes=n_classes, include_pbar=True)
        x0_t, names0_t = candidate_features_from_record(rec_id_t, n_classes=n_classes, include_pbar=True)
        if names_t != names0_t:
            raise RuntimeError("Anchor/candidate feature name mismatch on test subject.")
        x_delta_t = np.asarray(x_t - x0_t, dtype=np.float64).reshape(1, -1)

        pred_improve = float(cert.predict_accuracy(x_delta_t)[0])
        p_pos = float(guard.predict_pos_proba(x_delta_t)[0])
        p_pos_anchor = float(guard.predict_pos_proba(np.zeros_like(x_delta_t))[0])

        accept = (
            (p_pos >= float(args.guard_threshold))
            and (p_pos >= p_pos_anchor + float(args.anchor_guard_delta))
            and (pred_improve >= float(args.min_pred_improve))
        )

        if accept:
            y_pred_sel = df_t["y_pred_cand"].astype(str)
            proba_sel = p_c_t
        else:
            y_pred_sel = df_t["y_pred_ea"].astype(str)
            proba_sel = p_id_t

        acc_ea = float(np.mean(y_pred_id_t == y_true_t))
        acc_c = float(np.mean(y_pred_c_t == y_true_t))
        acc_sel = float(np.mean(y_pred_sel.to_numpy(object) == y_true_t))

        per_subject_summary_rows.append(
            {
                "subject": int(t),
                "accept": int(bool(accept)),
                "guard_p_pos": float(p_pos),
                "guard_p_pos_anchor": float(p_pos_anchor),
                "ridge_pred_improve": float(pred_improve),
                "acc_ea": float(acc_ea),
                "acc_cand": float(acc_c),
                "acc_selected": float(acc_sel),
                "delta_cand_vs_ea": float(acc_c - acc_ea),
                "delta_selected_vs_ea": float(acc_sel - acc_ea),
            }
        )

        df_out = pd.DataFrame(
            {
                "method": str(args.method_name),
                "subject": df_t["subject"].astype(int),
                "trial": df_t["trial"].astype(int),
                "y_true": df_t["y_true"].astype(str),
                "y_pred": y_pred_sel.astype(str),
            }
        )
        for i, c in enumerate(class_order):
            df_out[f"proba_{c}"] = proba_sel[:, int(i)]
        rows_all.append(df_out)

    pred_all = pd.concat(rows_all, axis=0, ignore_index=True).sort_values(["subject", "trial"])
    pred_all.to_csv(out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False)

    per_subj = pd.DataFrame(per_subject_summary_rows).sort_values("subject")
    per_subj.to_csv(out_dir / f"{date_prefix}_per_subject_selection.csv", index=False)

    # Summary table (same columns as other method_comparison tables).
    mean_acc = float(per_subj["acc_selected"].mean())
    worst_acc = float(per_subj["acc_selected"].min())
    mean_delta = float(per_subj["delta_selected_vs_ea"].mean())
    neg_transfer = float((per_subj["delta_selected_vs_ea"] < 0.0).mean())
    accept_rate = float(per_subj["accept"].mean())

    comp = pd.DataFrame(
        [
            {
                "method": str(args.method_name),
                "n_subjects": int(per_subj.shape[0]),
                "mean_accuracy": float(mean_acc),
                "worst_accuracy": float(worst_acc),
                "meanΔacc_vs_ea-csp-lda": float(mean_delta),
                "neg_transfer_vs_ea-csp-lda": float(neg_transfer),
                "accept_rate": float(accept_rate),
            }
        ]
    )
    comp.to_csv(out_dir / f"{date_prefix}_method_comparison.csv", index=False)


if __name__ == "__main__":
    main()
