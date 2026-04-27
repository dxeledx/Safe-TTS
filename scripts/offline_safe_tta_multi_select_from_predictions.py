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


class _ConstantGuard:
    def __init__(self, *, p_pos: float, feature_names: tuple[str, ...]) -> None:
        self._p_pos = float(p_pos)
        self.feature_names = tuple(feature_names)

    def predict_pos_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        n = int(features.shape[0]) if features.ndim == 2 else 1
        return np.full((n,), fill_value=self._p_pos, dtype=np.float64)


def _parse_csv_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_method_family_map(raw: str) -> dict[str, str]:
    raw = str(raw).strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --candidate-family-map entry: {part!r} (expected method=family)")
        m, fam = part.split("=", 1)
        m = m.strip()
        fam = fam.strip()
        if not m or not fam:
            raise ValueError(f"Invalid --candidate-family-map entry: {part!r} (empty method/family)")
        out[m] = fam
    return out


def _infer_family(method: str) -> str:
    m = str(method).strip().lower()
    if m in {"ea", "ea-csp-lda"} or m.startswith("ea-"):
        return "ea"
    if "lea" in m or m.startswith("rpa") or "logea" in m:
        return "rpa"
    if "tsa" in m or "rot" in m:
        return "tsa"
    if "fbcsp" in m:
        return "fbcsp"
    if "mdm" in m:
        return "mdm"
    if "ts-" in m or "tangent" in m:
        # Not a perfect match (LR vs SVC), but keeps the family one-hot informative.
        return "ts_svc"
    return "other"


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Offline SAFE-TTA multi-armed selection from a merged predictions_all_methods.csv.\n\n"
            "For each test subject, trains a ridge certificate + logistic guard on the remaining subjects only\n"
            "(LOSO-style calibration), then selects among multiple candidate methods using only unlabeled\n"
            "target prediction statistics at selection time.\n"
        )
    )
    p.add_argument("--preds", type=Path, required=True, help="Merged predictions_all_methods.csv (multiple methods).")
    p.add_argument("--anchor-method", type=str, default="ea-csp-lda", help="Anchor method (default: ea-csp-lda).")
    p.add_argument(
        "--candidate-methods",
        type=str,
        default="ALL",
        help="Comma-separated candidate methods, or ALL to use all methods except anchor.",
    )
    p.add_argument(
        "--candidate-family-map",
        type=str,
        default="",
        help="Optional mapping 'method=family,method=family'. Families are used for feature one-hots.",
    )
    p.add_argument("--guard-threshold", type=float, default=0.5)
    p.add_argument("--anchor-guard-delta", type=float, default=0.05)
    p.add_argument("--min-pred-improve", type=float, default=0.02)
    p.add_argument(
        "--guard-gray-margin",
        type=float,
        default=0.02,
        help="Symmetric gray-zone margin for guard labels; rows with |Δacc| <= margin are ignored by the guard.",
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--guard-c", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method-name", type=str, default="safe-tta-offline-multi")
    p.add_argument("--date-prefix", type=str, default="", help="Output file prefix (default: YYYYMMDD).")
    p.add_argument("--no-diagnostics", action="store_true", help="Do not write per-subject candidates.csv.")
    return p.parse_args()


def _guard_label_from_improve(*, improve: float, margin: float) -> int | None:
    margin = max(0.0, float(margin))
    improve = float(improve)
    if margin <= 0.0:
        return 1 if improve > 0.0 else 0
    if improve > margin:
        return 1
    if improve < -margin:
        return 0
    return None


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = str(args.date_prefix).strip() or datetime.now().strftime("%Y%m%d")

    df = pd.read_csv(Path(args.preds))
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"Preds CSV missing columns {sorted(required - set(df.columns))}.")

    proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
    if not proba_cols:
        raise ValueError("Preds CSV has no proba_* columns.")
    class_order = [str(c).replace("proba_", "", 1) for c in proba_cols]
    n_classes = int(len(class_order))
    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")

    anchor_method = str(args.anchor_method).strip()
    all_methods = sorted({str(m) for m in df["method"].unique().tolist()})
    if anchor_method not in all_methods:
        raise ValueError(f"Anchor method {anchor_method!r} not found in preds. Methods: {all_methods}")

    cand_methods_raw = str(args.candidate_methods).strip()
    if cand_methods_raw.upper() == "ALL":
        cand_methods = [m for m in all_methods if m != anchor_method]
    else:
        cand_methods = [m for m in _parse_csv_list(cand_methods_raw) if m != anchor_method]
    if not cand_methods:
        raise ValueError("No candidate methods specified (or only anchor given).")
    if float(args.guard_gray_margin) < 0.0:
        raise ValueError("--guard-gray-margin must be >= 0.")
    missing = [m for m in cand_methods if m not in all_methods]
    if missing:
        raise ValueError(f"Candidate methods missing in preds: {missing}. Available: {all_methods}")

    fam_map = _parse_method_family_map(args.candidate_family_map)
    cand_families: dict[str, str] = {m: fam_map.get(m, _infer_family(m)) for m in cand_methods}

    # Slice per method; require full alignment (same subject/trial/y_true keys).
    df_anchor = df[df["method"].astype(str) == anchor_method].copy()
    if df_anchor.empty:
        raise RuntimeError("Anchor dataframe is empty after filtering.")

    df_anchor = df_anchor.sort_values(["subject", "trial"]).reset_index(drop=True)
    subjects_anchor = sorted({int(s) for s in df_anchor["subject"].unique().tolist()})
    key_anchor = df_anchor[["subject", "trial", "y_true"]].copy()

    df_cands: dict[str, pd.DataFrame] = {}
    common_subjects = set(subjects_anchor)
    for m in cand_methods:
        dm = df[df["method"].astype(str) == str(m)].copy()
        if dm.empty:
            raise RuntimeError(f"Candidate dataframe is empty for method={m}")
        dm = dm.sort_values(["subject", "trial"]).reset_index(drop=True)
        # Check global key alignment to anchor.
        if not dm[["subject", "trial", "y_true"]].equals(key_anchor):
            raise RuntimeError(
                f"Key mismatch between anchor and candidate method={m}. "
                "Ensure the runs were merged from the same trials/protocol."
            )
        df_cands[m] = dm
        common_subjects &= set(int(s) for s in dm["subject"].unique().tolist())

    subjects = sorted(common_subjects)
    if len(subjects) < 3:
        raise RuntimeError(f"Need at least 3 common subjects, got {len(subjects)}: {subjects}")

    # Pre-slice per subject for speed.
    by_subject_anchor: dict[int, pd.DataFrame] = {int(s): g.copy() for s, g in df_anchor.groupby("subject", sort=True)}
    by_subject_cands: dict[str, dict[int, pd.DataFrame]] = {
        m: {int(s): g.copy() for s, g in dm.groupby("subject", sort=True)} for m, dm in df_cands.items()
    }

    diagnostics_root = out_dir / "diagnostics" / str(args.method_name)
    if not bool(args.no_diagnostics):
        diagnostics_root.mkdir(parents=True, exist_ok=True)

    rows_all: list[pd.DataFrame] = []
    per_subject_summary_rows: list[dict] = []

    for t in subjects:
        train_subjects = [s for s in subjects if s != int(t)]
        if len(train_subjects) < 2:
            raise RuntimeError("Need at least 3 subjects for LOSO-style calibration.")

        # Build training set: one row per (training subject, candidate method).
        X_rows: list[np.ndarray] = []
        y_improve: list[float] = []
        X_guard_rows: list[np.ndarray] = []
        y_guard: list[int] = []
        feat_names: tuple[str, ...] | None = None

        for s in train_subjects:
            df_s_id = by_subject_anchor[int(s)]
            y_true_s = df_s_id["y_true"].to_numpy(object)
            p_id = df_s_id[proba_cols].to_numpy(np.float64)
            y_pred_id = df_s_id["y_pred"].to_numpy(object)
            acc_id = float(np.mean(y_pred_id == y_true_s))

            rec_id = _record_from_proba(
                p_id=p_id,
                p_c=p_id,
                y_pred_id=y_pred_id,
                y_pred_c=y_pred_id,
                cand_family="ea",
                kind="identity",
            )
            x0, names0 = candidate_features_from_record(rec_id, n_classes=n_classes, include_pbar=True)

            for m in cand_methods:
                df_s_c = by_subject_cands[m][int(s)]
                y_pred_c = df_s_c["y_pred"].to_numpy(object)
                p_c = df_s_c[proba_cols].to_numpy(np.float64)
                acc_c = float(np.mean(y_pred_c == y_true_s))
                improve = float(acc_c - acc_id)

                rec_c = _record_from_proba(
                    p_id=p_id,
                    p_c=p_c,
                    y_pred_id=y_pred_id,
                    y_pred_c=y_pred_c,
                    cand_family=str(cand_families[m]),
                    kind="candidate",
                )
                x, names = candidate_features_from_record(rec_c, n_classes=n_classes, include_pbar=True)
                if names != names0:
                    raise RuntimeError("Feature name mismatch between anchor and candidate features.")
                x_delta = np.asarray(x - x0, dtype=np.float64).reshape(-1)
                names_delta = tuple([f"delta_{n}" for n in names])
                if feat_names is None:
                    feat_names = names_delta

                X_rows.append(x_delta)
                y_improve.append(improve)
                guard_label = _guard_label_from_improve(improve=float(improve), margin=float(args.guard_gray_margin))
                if guard_label is not None:
                    X_guard_rows.append(np.asarray(x_delta, dtype=np.float64).reshape(-1))
                    y_guard.append(int(guard_label))

        if feat_names is None:
            raise RuntimeError("No training rows.")
        X_tr = np.vstack(X_rows).astype(np.float64, copy=False)
        y_tr = np.asarray(y_improve, dtype=np.float64)
        X_guard = np.vstack(X_guard_rows).astype(np.float64, copy=False) if X_guard_rows else np.zeros((0, X_tr.shape[1]), dtype=np.float64)
        yb_tr = np.asarray(y_guard, dtype=int)

        cert = train_ridge_certificate(X_tr, y_tr, feature_names=feat_names, alpha=float(args.ridge_alpha))
        try:
            if X_guard.shape[0] <= 0 or yb_tr.size <= 0:
                raise RuntimeError("No non-gray rows available for guard training.")
            guard = train_logistic_guard(X_guard, yb_tr, feature_names=feat_names, c=float(args.guard_c))
        except Exception:
            guard = _ConstantGuard(p_pos=float(np.mean(yb_tr)) if yb_tr.size else 0.5, feature_names=feat_names)

        # Test subject arrays.
        df_t_id = by_subject_anchor[int(t)]
        y_true_t = df_t_id["y_true"].to_numpy(object)
        y_pred_id_t = df_t_id["y_pred"].to_numpy(object)
        p_id_t = df_t_id[proba_cols].to_numpy(np.float64)
        acc_id_t = float(np.mean(y_pred_id_t == y_true_t))

        rec_id_t = _record_from_proba(
            p_id=p_id_t,
            p_c=p_id_t,
            y_pred_id=y_pred_id_t,
            y_pred_c=y_pred_id_t,
            cand_family="ea",
            kind="identity",
        )
        x0_t, names0_t = candidate_features_from_record(rec_id_t, n_classes=n_classes, include_pbar=True)
        x0_delta = np.zeros((1, x0_t.shape[0]), dtype=np.float64)
        p_pos_anchor = float(guard.predict_pos_proba(x0_delta)[0])

        cand_rows = []
        # Identity row (EA).
        cand_rows.append(
            {
                "idx": 0,
                "is_selected": 0,
                "kind": "identity",
                "cand_family": "ea",
                "cand_key": str(anchor_method),
                "ridge_pred_improve": 0.0,
                "guard_p_pos": float(p_pos_anchor),
                "accept": 0,
                "accuracy": float(acc_id_t),
            }
            | {k: v for k, v in rec_id_t.items() if k not in {"kind", "cand_family"}}
        )

        # Evaluate each candidate.
        cand_eval = []
        for j, m in enumerate(cand_methods, start=1):
            df_t_c = by_subject_cands[m][int(t)]
            y_pred_c_t = df_t_c["y_pred"].to_numpy(object)
            p_c_t = df_t_c[proba_cols].to_numpy(np.float64)
            acc_c_t = float(np.mean(y_pred_c_t == y_true_t))

            rec_c_t = _record_from_proba(
                p_id=p_id_t,
                p_c=p_c_t,
                y_pred_id=y_pred_id_t,
                y_pred_c=y_pred_c_t,
                cand_family=str(cand_families[m]),
                kind="candidate",
            )
            x_t, names_t = candidate_features_from_record(rec_c_t, n_classes=n_classes, include_pbar=True)
            if names_t != names0_t:
                raise RuntimeError("Anchor/candidate feature name mismatch on test subject.")
            x_delta_t = np.asarray(x_t - x0_t, dtype=np.float64).reshape(1, -1)

            pred_improve = float(cert.predict_accuracy(x_delta_t)[0])
            p_pos = float(guard.predict_pos_proba(x_delta_t)[0])

            accept = (
                (p_pos >= float(args.guard_threshold))
                and (p_pos >= p_pos_anchor + float(args.anchor_guard_delta))
                and (pred_improve >= float(args.min_pred_improve))
            )

            row = (
                {
                    "idx": int(j),
                    "is_selected": 0,
                    "kind": "candidate",
                    "cand_family": str(cand_families[m]),
                    "cand_key": str(m),
                    "ridge_pred_improve": float(pred_improve),
                    "guard_p_pos": float(p_pos),
                    "accept": int(bool(accept)),
                    "accuracy": float(acc_c_t),
                }
                | {k: v for k, v in rec_c_t.items() if k not in {"kind", "cand_family"}}
            )
            cand_rows.append(row)
            cand_eval.append(
                {
                    "method": str(m),
                    "family": str(cand_families[m]),
                    "pred_improve": float(pred_improve),
                    "p_pos": float(p_pos),
                    "accept": bool(accept),
                    "acc": float(acc_c_t),
                }
            )

        # Select: best predicted improvement among accepted candidates; else EA.
        best_pre = max(cand_eval, key=lambda r: float(r["pred_improve"])) if cand_eval else None
        accepted = [r for r in cand_eval if bool(r["accept"])]
        best_acc = max([acc_id_t] + [float(r["acc"]) for r in cand_eval]) if cand_eval else acc_id_t
        oracle_acc = float(best_acc)

        selected_method = str(anchor_method)
        selected_family = "ea"
        selected_pred_improve = 0.0
        selected_p_pos = float(p_pos_anchor)
        accept_any = False

        if accepted:
            best = max(accepted, key=lambda r: float(r["pred_improve"]))
            selected_method = str(best["method"])
            selected_family = str(best["family"])
            selected_pred_improve = float(best["pred_improve"])
            selected_p_pos = float(best["p_pos"])
            accept_any = True

        # Mark selected in candidates table.
        for r in cand_rows:
            if r["kind"] == "identity" and selected_method == anchor_method:
                r["is_selected"] = 1
            if r["kind"] == "candidate" and str(r["cand_key"]) == selected_method:
                r["is_selected"] = 1

        # Save candidates.csv (diagnostics).
        if not bool(args.no_diagnostics):
            subj_dir = diagnostics_root / f"subject_{int(t):02d}"
            subj_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cand_rows).to_csv(subj_dir / "candidates.csv", index=False)

        # Selected predictions (per trial).
        if selected_method == anchor_method:
            df_sel = df_t_id.copy()
        else:
            df_sel = by_subject_cands[selected_method][int(t)].copy()
        df_out = pd.DataFrame(
            {
                "method": str(args.method_name),
                "subject": df_sel["subject"].astype(int),
                "trial": df_sel["trial"].astype(int),
                "y_true": df_sel["y_true"].astype(str),
                "y_pred": df_sel["y_pred"].astype(str),
            }
        )
        for c in proba_cols:
            df_out[c] = df_sel[c].astype(float)
        rows_all.append(df_out)

        acc_sel_t = float(np.mean(df_sel["y_pred"].to_numpy(object) == y_true_t))
        oracle_gap = float(oracle_acc - acc_sel_t)
        headroom = float(oracle_acc - acc_id_t)

        per_subject_summary_rows.append(
            {
                "subject": int(t),
                "selected_method": str(selected_method),
                "selected_family": str(selected_family),
                "accept": int(bool(accept_any)),
                "guard_p_pos_selected": float(selected_p_pos),
                "guard_p_pos_anchor": float(p_pos_anchor),
                "ridge_pred_improve_selected": float(selected_pred_improve),
                "pre_best_method": str(best_pre["method"]) if best_pre is not None else "",
                "pre_best_pred_improve": float(best_pre["pred_improve"]) if best_pre is not None else float("nan"),
                "acc_anchor": float(acc_id_t),
                "acc_selected": float(acc_sel_t),
                "oracle_acc": float(oracle_acc),
                "headroom": float(headroom),
                "oracle_gap": float(oracle_gap),
                "delta_selected_vs_anchor": float(acc_sel_t - acc_id_t),
            }
        )

    pred_all = pd.concat(rows_all, axis=0, ignore_index=True).sort_values(["subject", "trial"])
    pred_all.to_csv(out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False)

    per_subj = pd.DataFrame(per_subject_summary_rows).sort_values("subject")
    per_subj.to_csv(out_dir / f"{date_prefix}_per_subject_selection.csv", index=False)

    mean_acc = float(per_subj["acc_selected"].mean())
    worst_acc = float(per_subj["acc_selected"].min())
    mean_delta = float(per_subj["delta_selected_vs_anchor"].mean())
    neg_transfer = float((per_subj["delta_selected_vs_anchor"] < 0.0).mean())
    accept_rate = float(per_subj["accept"].mean())
    mean_oracle = float(per_subj["oracle_acc"].mean())
    mean_headroom = float(per_subj["headroom"].mean())
    mean_oracle_gap = float(per_subj["oracle_gap"].mean())

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
                "oracle_mean_accuracy": float(mean_oracle),
                "headroom_mean": float(mean_headroom),
                "oracle_gap_mean": float(mean_oracle_gap),
                "n_candidates": int(len(cand_methods) + 1),
            }
        ]
    )
    comp.to_csv(out_dir / f"{date_prefix}_method_comparison.csv", index=False)


if __name__ == "__main__":
    main()
