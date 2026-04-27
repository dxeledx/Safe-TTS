from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepGates:
    guard_threshold: float
    probe_hard_worsen: float
    global_min_pred_improve: float
    fbcsp_guard_threshold: float
    fbcsp_min_pred_improve: float
    fbcsp_drift_delta: float
    tsa_guard_threshold: float
    tsa_min_pred_improve: float
    tsa_drift_delta: float


def _rank_desc_min(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.asarray([], dtype=np.float64)
    v = np.where(np.isfinite(v), v, -np.inf)
    order = np.argsort(-v, kind="mergesort")
    ranks_sorted = (np.arange(v.size, dtype=np.float64) + 1.0).copy()
    sv = v[order]
    start = 0
    while start < sv.size:
        end = start + 1
        while end < sv.size and sv[end] == sv[start]:
            end += 1
        ranks_sorted[start:end] = ranks_sorted[start]
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _apply_family_gates(df: pd.DataFrame, *, gates: SweepGates) -> pd.DataFrame:
    df = df.copy()

    def _apply_one(
        sub: pd.DataFrame,
        *,
        family_guard_thr: float,
        min_pred: float,
        drift_delta: float,
    ) -> pd.DataFrame:
        if float(family_guard_thr) >= 0.0:
            thr = max(float(gates.guard_threshold), float(family_guard_thr))
            sub = sub[np.isfinite(sub["guard_p_pos"].to_numpy()) & (sub["guard_p_pos"].astype(float) >= float(thr))]
        if float(min_pred) > 0.0:
            sub = sub[np.isfinite(sub["ridge_pred_improve"].to_numpy()) & (sub["ridge_pred_improve"].astype(float) >= float(min_pred))]
        if float(drift_delta) > 0.0:
            sub = sub[np.isfinite(sub["drift_best"].to_numpy()) & (sub["drift_best"].astype(float) <= float(drift_delta))]
        return sub

    fam = df["cand_family"].astype(str).str.lower()
    keep_other = ~fam.isin(["fbcsp", "tsa"])
    out = [df[keep_other]]
    out.append(
        _apply_one(
            df[fam == "fbcsp"],
            family_guard_thr=float(gates.fbcsp_guard_threshold),
            min_pred=float(gates.fbcsp_min_pred_improve),
            drift_delta=float(gates.fbcsp_drift_delta),
        )
    )
    out.append(
        _apply_one(
            df[fam == "tsa"],
            family_guard_thr=float(gates.tsa_guard_threshold),
            min_pred=float(gates.tsa_min_pred_improve),
            drift_delta=float(gates.tsa_drift_delta),
        )
    )
    out_df = pd.concat(out, axis=0)
    return out_df.loc[~out_df.index.duplicated(keep="first")]


def _select_one_subject(
    cand_df: pd.DataFrame,
    *,
    anchor_guard_delta: float,
    gates: SweepGates,
) -> dict:
    df = cand_df.copy()
    df["kind"] = df["kind"].astype(str)
    df["cand_family"] = df["cand_family"].astype(str).str.lower()

    anchors = df[df["kind"] == "identity"]
    if anchors.empty:
        raise RuntimeError("No identity record found in candidates.csv (expected EA anchor).")
    anchor = anchors.iloc[0]

    anchor_p_pos = _safe_float(anchor.get("guard_p_pos", float("nan")))
    anchor_probe_hard = _safe_float(anchor.get("probe_mixup_hard_best", float("nan")))
    anchor_acc = _safe_float(anchor.get("accuracy", float("nan")))
    # Oracle over the *available candidate set* (analysis only; uses labels in candidates.csv).
    # Always compute this, even if gates reject everything.
    oracle_acc_full = float(cand_df["accuracy"].astype(float).max())

    cand = df[df["kind"] != "identity"].copy()

    # Base guard threshold (identity is never filtered; candidates must pass).
    cand = cand[np.isfinite(cand["guard_p_pos"].to_numpy()) & (cand["guard_p_pos"].astype(float) >= float(gates.guard_threshold))]

    # EA-anchor relative guard delta gate.
    if np.isfinite(anchor_p_pos) and float(anchor_guard_delta) > 0.0:
        thr = float(anchor_p_pos) + float(anchor_guard_delta)
        cand = cand[cand["guard_p_pos"].astype(float) >= float(thr)]

    # EA-anchor relative probe_hard gate (MixVal-style hard-major probes; smaller is better).
    if float(gates.probe_hard_worsen) > -1.0 and np.isfinite(anchor_probe_hard):
        thr_probe = float(anchor_probe_hard) + float(gates.probe_hard_worsen)
        cand = cand[np.isfinite(cand["probe_mixup_hard_best"].to_numpy()) & (cand["probe_mixup_hard_best"].astype(float) <= float(thr_probe))]

    # Family-specific gates (FBCSP/TSA).
    cand = _apply_family_gates(cand, gates=gates)

    # Global min_pred_improve gate (applies to all non-identity candidates).
    if float(gates.global_min_pred_improve) > 0.0:
        cand = cand[np.isfinite(cand["ridge_pred_improve"].to_numpy()) & (cand["ridge_pred_improve"].astype(float) >= float(gates.global_min_pred_improve))]

    # Safe fallback.
    if cand.empty:
        return {
            "accept": 0,
            "selected_family": "ea",
            "selected_acc": float(anchor_acc),
            "anchor_acc": float(anchor_acc),
            "delta_acc": 0.0,
            "oracle_acc": float(oracle_acc_full),
        }

    # Borda (ridge_pred_improve + probe_improve).
    pred = cand["ridge_pred_improve"].astype(float).to_numpy()
    if np.isfinite(anchor_probe_hard):
        probe_imp = float(anchor_probe_hard) - cand["probe_mixup_hard_best"].astype(float).to_numpy()
    else:
        probe_imp = np.full_like(pred, np.nan, dtype=np.float64)

    ranks_pred = _rank_desc_min(pred)
    ranks_probe = _rank_desc_min(probe_imp)
    score = ranks_pred + ranks_probe  # smaller is better
    best_i = int(np.argmin(score))

    best_pred = float(pred[best_i])
    best_probe = float(probe_imp[best_i]) if np.isfinite(probe_imp[best_i]) else -float("inf")
    best = cand.iloc[best_i]

    # Safety fallback: if both signals are non-positive, abstain (EA).
    if float(max(best_pred, best_probe)) <= 0.0 and np.isfinite(anchor_acc):
        return {
            "accept": 0,
            "selected_family": "ea",
            "selected_acc": float(anchor_acc),
            "anchor_acc": float(anchor_acc),
            "delta_acc": 0.0,
            "oracle_acc": float(oracle_acc_full),
        }

    sel_acc = _safe_float(best.get("accuracy", float("nan")))
    fam = str(best.get("cand_family", "other"))

    return {
        "accept": 1,
        "selected_family": fam,
        "selected_acc": float(sel_acc),
        "anchor_acc": float(anchor_acc),
        "delta_acc": float(sel_acc) - float(anchor_acc),
        "oracle_acc": float(oracle_acc_full),
    }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_sweep(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

    x = df["global_min_pred_improve"].astype(float).to_numpy()

    axes[0].plot(x, df["mean_delta_vs_ea"].astype(float).to_numpy(), marker="o")
    axes[0].axhline(0.0, color="k", linewidth=1, alpha=0.4)
    axes[0].set_ylabel("Mean Δacc vs EA")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(x, df["accept_rate"].astype(float).to_numpy(), marker="o", color="#DD8452")
    axes[1].set_ylabel("Accept rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(x, df["neg_transfer_rate_vs_ea"].astype(float).to_numpy(), marker="o", color="#C44E52")
    axes[2].set_ylabel("Neg-transfer rate")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, linestyle="--", alpha=0.3)

    axes[3].plot(x, df["oracle_mean_gap"].astype(float).to_numpy(), marker="o", color="#4C72B0")
    axes[3].set_ylabel("Oracle gap")
    axes[3].set_xlabel("global_min_pred_improve")
    axes[3].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline global_min_pred_improve sweep from saved candidates.csv (no reruns)."
    )
    ap.add_argument(
        "--diagnostics-dir",
        type=Path,
        required=True,
        help="Path like outputs/.../diagnostics/ea-stack-multi-safe-csp-lda containing subject_*/candidates.csv.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--min-preds",
        type=str,
        default="0,0.005,0.01,0.02,0.03,0.05",
        help="Comma-separated global_min_pred_improve sweep values.",
    )
    ap.add_argument("--anchor-guard-delta", type=float, default=0.05)
    ap.add_argument("--guard-threshold", type=float, default=0.5)
    ap.add_argument("--probe-hard-worsen", type=float, default=-0.01)
    ap.add_argument("--fbcsp-guard-threshold", type=float, default=0.95)
    ap.add_argument("--fbcsp-min-pred-improve", type=float, default=0.05)
    ap.add_argument("--fbcsp-drift-delta", type=float, default=0.15)
    ap.add_argument("--tsa-guard-threshold", type=float, default=0.95)
    ap.add_argument("--tsa-min-pred-improve", type=float, default=0.05)
    ap.add_argument("--tsa-drift-delta", type=float, default=0.15)
    ap.add_argument("--title", type=str, default="BNCI2014_001 4-class LOSO: global min_pred sweep (offline)")
    ap.add_argument("--prefix", type=str, default="bnci_min_pred_sweep")
    args = ap.parse_args()

    diag = Path(args.diagnostics_dir)
    subj_dirs = sorted([p for p in diag.glob("subject_*") if p.is_dir()])
    if not subj_dirs:
        raise RuntimeError(f"No subject_* folders found under {diag}")

    cand_by_subject: dict[int, pd.DataFrame] = {}
    for sd in subj_dirs:
        cand_path = sd / "candidates.csv"
        if not cand_path.exists():
            continue
        subject = int(sd.name.split("_")[-1])
        cand_by_subject[subject] = pd.read_csv(cand_path)

    if not cand_by_subject:
        raise RuntimeError(f"No candidates.csv found under {diag}")

    min_preds = [float(x.strip()) for x in str(args.min_preds).split(",") if x.strip()]

    per_subject_rows: list[dict] = []
    sweep_rows: list[dict] = []

    subjects = sorted(cand_by_subject.keys())
    for mp in min_preds:
        gates = SweepGates(
            guard_threshold=float(args.guard_threshold),
            probe_hard_worsen=float(args.probe_hard_worsen),
            global_min_pred_improve=float(mp),
            fbcsp_guard_threshold=float(args.fbcsp_guard_threshold),
            fbcsp_min_pred_improve=float(args.fbcsp_min_pred_improve),
            fbcsp_drift_delta=float(args.fbcsp_drift_delta),
            tsa_guard_threshold=float(args.tsa_guard_threshold),
            tsa_min_pred_improve=float(args.tsa_min_pred_improve),
            tsa_drift_delta=float(args.tsa_drift_delta),
        )

        selected_acc = []
        anchor_acc = []
        accept = []
        delta_acc = []
        oracle_acc = []
        fams: list[str] = []

        for s in subjects:
            out = _select_one_subject(
                cand_by_subject[s],
                anchor_guard_delta=float(args.anchor_guard_delta),
                gates=gates,
            )
            per_subject_rows.append({"subject": int(s), "global_min_pred_improve": float(mp), **out})
            selected_acc.append(float(out["selected_acc"]))
            anchor_acc.append(float(out["anchor_acc"]))
            accept.append(int(out["accept"]))
            delta_acc.append(float(out["delta_acc"]))
            oracle_acc.append(float(out["oracle_acc"]))
            fams.append(str(out["selected_family"]))

        delta_arr = np.asarray(delta_acc, dtype=np.float64)
        sel_arr = np.asarray(selected_acc, dtype=np.float64)
        oracle_arr = np.asarray(oracle_acc, dtype=np.float64)

        sweep_rows.append(
            {
                "global_min_pred_improve": float(mp),
                "n_subjects": int(len(subjects)),
                "mean_accuracy": float(np.mean(sel_arr)),
                "worst_accuracy": float(np.min(sel_arr)),
                "accept_rate": float(np.mean(accept)),
                "mean_delta_vs_ea": float(np.mean(delta_arr)),
                "neg_transfer_rate_vs_ea": float(np.mean(delta_arr < 0.0)),
                "oracle_mean_accuracy": float(np.mean(oracle_arr)),
                "oracle_mean_gap": float(np.mean(oracle_arr - sel_arr)),
                "selected_family_counts": ";".join(
                    [f"{k}:{v}" for k, v in pd.Series(fams).value_counts().sort_index().to_dict().items()]
                ),
            }
        )

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    df_sweep = pd.DataFrame(sweep_rows).sort_values("global_min_pred_improve").reset_index(drop=True)
    df_subject = (
        pd.DataFrame(per_subject_rows)
        .sort_values(["global_min_pred_improve", "subject"])
        .reset_index(drop=True)
    )
    df_sweep.to_csv(out_dir / f"{args.prefix}_sweep_metrics.csv", index=False)
    df_subject.to_csv(out_dir / f"{args.prefix}_per_subject.csv", index=False)

    _plot_sweep(
        df_sweep,
        out_path=out_dir / f"{args.prefix}_risk_reward.png",
        title=str(args.title),
    )


if __name__ == "__main__":
    main()
