#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateResult:
    subject: int
    tau: float
    pred_disagree: float
    acc_ea: float
    acc_cand: float
    acc_selected: float
    selected_cand: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a merged 'run directory' for an EA-anchor pred_disagree gate, with optional "
            "LOSO-style calibration of tau using only training subjects (per target subject)."
        )
    )
    p.add_argument("--ea-run-dir", type=Path, required=True, help="Run dir containing *_predictions_all_methods.csv for EA.")
    p.add_argument(
        "--cand-run-dir",
        type=Path,
        required=True,
        help="Run dir containing *_predictions_all_methods.csv for candidate (e.g., ea-fbcsp-lda).",
    )
    p.add_argument("--ea-method", type=str, default="ea-csp-lda", help="Method name in EA run.")
    p.add_argument("--cand-method", type=str, default="ea-fbcsp-lda", help="Method name in candidate run.")
    p.add_argument(
        "--out-run-dir",
        type=Path,
        required=True,
        help="Output directory that will mimic an outputs/* run folder (writes *_predictions_all_methods.csv, *_method_comparison.csv, *_results.txt).",
    )
    p.add_argument("--date-prefix", type=str, required=True, help="YYYYMMDD prefix for output filenames.")
    p.add_argument(
        "--mode",
        type=str,
        choices=["fixed_tau", "calib_loso_safe0"],
        default="calib_loso_safe0",
        help=(
            "fixed_tau: use a single tau for all subjects. "
            "calib_loso_safe0: per-subject tau chosen using only other subjects, maximizing train mean acc "
            "subject to 0 neg-transfer on the training subjects."
        ),
    )
    p.add_argument("--tau", type=float, default=0.37, help="Tau for mode=fixed_tau.")
    p.add_argument("--sweep-max", type=float, default=0.6, help="Max tau considered for calibration grid (inclusive).")
    p.add_argument("--sweep-steps", type=int, default=61, help="Number of tau grid points in [0, sweep_max].")
    p.add_argument(
        "--selected-method-name",
        type=str,
        default="ea-fbcsp-pred-disagree-safe",
        help="Method name used for the gated selection output.",
    )
    return p.parse_args()


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 file match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _load_preds(run_dir: Path, method: str) -> pd.DataFrame:
    pred_path = _find_single(run_dir, "*_predictions_all_methods.csv")
    df = pd.read_csv(pred_path)
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {pred_path}: {sorted(missing)}")
    df = df[df["method"].astype(str) == str(method)].copy()
    if df.empty:
        raise RuntimeError(f"No rows for method={method} in {pred_path}")
    # Keep required + proba_* if present.
    keep_cols = [c for c in df.columns if c in required or c.startswith("proba_")]
    df = df[keep_cols].copy()
    df["subject"] = df["subject"].astype(int)
    df["trial"] = df["trial"].astype(int)
    return df.sort_values(["subject", "trial"]).reset_index(drop=True)


def _pred_disagree_by_subject(ea: pd.DataFrame, cand: pd.DataFrame) -> pd.Series:
    m = ea.merge(cand, on=["subject", "trial", "y_true"], suffixes=("_ea", "_cand"))
    if m.empty:
        raise RuntimeError("Empty merge; ensure both runs use identical dataset/events/sessions/preprocess.")
    return (m["y_pred_ea"].astype(str) != m["y_pred_cand"].astype(str)).groupby(m["subject"]).mean()


def _acc_by_subject(df: pd.DataFrame) -> pd.Series:
    correct = (df["y_pred"].astype(str) == df["y_true"].astype(str)).astype(float)
    return correct.groupby(df["subject"]).mean()


def _apply_gate(
    *,
    ea: pd.DataFrame,
    cand: pd.DataFrame,
    taus: dict[int, float],
    selected_method_name: str,
) -> tuple[pd.DataFrame, list[GateResult]]:
    m = ea.merge(cand, on=["subject", "trial", "y_true"], suffixes=("_ea", "_cand"))
    if m.empty:
        raise RuntimeError("Empty merge; ensure both runs use identical dataset/events/sessions/preprocess.")

    # Per-subject pred_disagree and accuracies.
    m["disagree_trial"] = (m["y_pred_ea"].astype(str) != m["y_pred_cand"].astype(str)).astype(float)
    pred_disagree = m.groupby("subject")["disagree_trial"].mean()
    acc_ea = (m["y_pred_ea"].astype(str) == m["y_true"].astype(str)).groupby(m["subject"]).mean()
    acc_cand = (m["y_pred_cand"].astype(str) == m["y_true"].astype(str)).groupby(m["subject"]).mean()

    # Decide selection per subject.
    sel_mask = pd.Series(False, index=pred_disagree.index, dtype=bool)
    for s in pred_disagree.index.tolist():
        tau = float(taus[int(s)])
        sel_mask.loc[s] = bool(float(pred_disagree.loc[s]) <= tau)

    # Build selected predictions (row-wise based on subject decision).
    out = pd.DataFrame(
        {
            "method": selected_method_name,
            "subject": m["subject"].astype(int),
            "trial": m["trial"].astype(int),
            "y_true": m["y_true"].astype(str),
        }
    )
    use_cand = sel_mask.reindex(out["subject"]).to_numpy(dtype=bool)
    out["y_pred"] = np.where(use_cand, m["y_pred_cand"].astype(str), m["y_pred_ea"].astype(str))

    # Carry probabilities from the chosen model when present in both sources.
    proba_cols = [c for c in m.columns if c.startswith("proba_") and c.endswith("_ea")]
    for c_ea in proba_cols:
        c_base = c_ea[: -len("_ea")]
        c_cand = c_base + "_cand"
        if c_cand not in m.columns:
            continue
        out[c_base] = np.where(use_cand, m[c_cand].to_numpy(dtype=float), m[c_ea].to_numpy(dtype=float))

    results: list[GateResult] = []
    for s in pred_disagree.index.tolist():
        tau = float(taus[int(s)])
        selected = bool(sel_mask.loc[s])
        acc_sel = float(acc_cand.loc[s] if selected else acc_ea.loc[s])
        results.append(
            GateResult(
                subject=int(s),
                tau=tau,
                pred_disagree=float(pred_disagree.loc[s]),
                acc_ea=float(acc_ea.loc[s]),
                acc_cand=float(acc_cand.loc[s]),
                acc_selected=acc_sel,
                selected_cand=selected,
            )
        )
    return out.sort_values(["subject", "trial"]).reset_index(drop=True), results


def _choose_tau_calib_safe0(
    *,
    subjects: list[int],
    pred_disagree: pd.Series,
    acc_ea: pd.Series,
    acc_cand: pd.Series,
    sweep_max: float,
    sweep_steps: int,
) -> dict[int, float]:
    taus_grid = np.linspace(0.0, float(sweep_max), int(sweep_steps))
    out: dict[int, float] = {}
    for test_s in subjects:
        train = [s for s in subjects if s != test_s]
        if not train:
            out[int(test_s)] = 0.0
            continue

        best_key: tuple[float, float] | None = None
        best_tau: float | None = None
        for tau in taus_grid:
            mask = pred_disagree.loc[train] <= float(tau)
            acc_sel = acc_ea.loc[train].copy()
            acc_sel[mask] = acc_cand.loc[train][mask]
            delta = acc_sel - acc_ea.loc[train]
            neg_rate = float((delta < 0.0).mean())
            if neg_rate > 0.0:
                continue
            key = (float(acc_sel.mean()), float(mask.mean()))
            if best_key is None or key > best_key:
                best_key = key
                best_tau = float(tau)

        out[int(test_s)] = float(best_tau if best_tau is not None else 0.0)
    return out


def _write_results_txt(
    *,
    out_dir: Path,
    date_prefix: str,
    args: argparse.Namespace,
    summary: dict,
    sources: dict,
) -> None:
    lines = []
    lines.append(f"Date: {date_prefix}")
    lines.append("Git commit: (offline merged from existing runs)")
    lines.append("Command:")
    lines.append(
        "  "
        + " ".join(
            [
                "scripts/build_pred_disagree_calib_run.py",
                f"--ea-run-dir {sources['ea_run_dir']}",
                f"--cand-run-dir {sources['cand_run_dir']}",
                f"--ea-method {sources['ea_method']}",
                f"--cand-method {sources['cand_method']}",
                f"--mode {args.mode}",
                f"--tau {float(args.tau)}",
                f"--sweep-max {float(args.sweep_max)}",
                f"--sweep-steps {int(args.sweep_steps)}",
                f"--selected-method-name {sources['selected_method_name']}",
            ]
        )
    )
    lines.append("")
    lines.append("=== Selection rule ===")
    lines.append("EA-anchor pred_disagree gate between EA and candidate:")
    lines.append("  pred_disagree = mean_i[ argmax p_EA(x_i) != argmax p_cand(x_i) ]")
    lines.append("")
    if str(args.mode) == "fixed_tau":
        lines.append(f"Mode: fixed_tau (tau={float(args.tau):.6f})")
    else:
        lines.append(
            "Mode: calib_loso_safe0 (for each test subject, choose tau on training subjects "
            "to maximize train mean accuracy subject to 0 neg-transfer on training subjects)"
        )
    lines.append("")
    lines.append("=== Summary (across subjects) ===")
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    (out_dir / f"{date_prefix}_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = _load_preds(Path(args.ea_run_dir), method=str(args.ea_method))
    cand = _load_preds(Path(args.cand_run_dir), method=str(args.cand_method))

    # Compute per-subject stats to calibrate taus.
    pred_disagree = _pred_disagree_by_subject(ea, cand).sort_index()
    acc_ea = _acc_by_subject(ea).sort_index()
    acc_cand = _acc_by_subject(cand).sort_index()
    subjects = [int(s) for s in pred_disagree.index.tolist()]

    if str(args.mode) == "fixed_tau":
        taus = {int(s): float(args.tau) for s in subjects}
    else:
        taus = _choose_tau_calib_safe0(
            subjects=subjects,
            pred_disagree=pred_disagree,
            acc_ea=acc_ea,
            acc_cand=acc_cand,
            sweep_max=float(args.sweep_max),
            sweep_steps=int(args.sweep_steps),
        )

    selected_df, gate_rows = _apply_gate(
        ea=ea,
        cand=cand,
        taus=taus,
        selected_method_name=str(args.selected_method_name),
    )

    # Write merged predictions file with 3 methods.
    merged = pd.concat(
        [
            ea.assign(method=str(args.ea_method)),
            cand.assign(method=str(args.cand_method)),
            selected_df,
        ],
        axis=0,
        ignore_index=True,
    )
    merged = merged.sort_values(["method", "subject", "trial"]).reset_index(drop=True)
    date = str(args.date_prefix)
    merged.to_csv(out_dir / f"{date}_predictions_all_methods.csv", index=False)

    # Per-subject table.
    per_subject = pd.DataFrame([r.__dict__ for r in gate_rows]).sort_values("subject")
    per_subject["delta_cand_vs_ea"] = per_subject["acc_cand"] - per_subject["acc_ea"]
    per_subject["delta_selected_vs_ea"] = per_subject["acc_selected"] - per_subject["acc_ea"]
    per_subject.to_csv(out_dir / f"{date}_pred_disagree_gate_per_subject.csv", index=False)

    # Summary + method comparison.
    mean_ea = float(per_subject["acc_ea"].mean())
    mean_cand = float(per_subject["acc_cand"].mean())
    mean_sel = float(per_subject["acc_selected"].mean())
    delta_sel = float(per_subject["delta_selected_vs_ea"].mean())
    neg_rate = float((per_subject["delta_selected_vs_ea"] < 0.0).mean())
    accept_rate = float(per_subject["selected_cand"].mean())
    worst_ea = float(per_subject["acc_ea"].min())
    worst_sel = float(per_subject["acc_selected"].min())
    tau_mean = float(per_subject["tau"].mean())
    tau_min = float(per_subject["tau"].min())
    tau_max = float(per_subject["tau"].max())

    summary = {
        "n_subjects": int(per_subject.shape[0]),
        "mean_acc_ea": mean_ea,
        "mean_acc_cand": mean_cand,
        "mean_acc_selected": mean_sel,
        "mean_delta_selected_vs_ea": delta_sel,
        "neg_transfer_rate_selected_vs_ea": neg_rate,
        "accept_rate": accept_rate,
        "worst_acc_ea": worst_ea,
        "worst_acc_selected": worst_sel,
        "tau_mean": tau_mean,
        "tau_min": tau_min,
        "tau_max": tau_max,
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{date}_pred_disagree_gate_summary.csv", index=False)

    # Minimal method comparison for compatibility with registry scripts.
    rows = [
        {
            "method": str(args.ea_method),
            "n_subjects": int(per_subject.shape[0]),
            "mean_accuracy": mean_ea,
            "worst_accuracy": worst_ea,
            "accept_rate": float("nan"),
        },
        {
            "method": str(args.cand_method),
            "n_subjects": int(per_subject.shape[0]),
            "mean_accuracy": mean_cand,
            "worst_accuracy": float(per_subject["acc_cand"].min()),
            "mean_delta_vs_ea": float((per_subject["acc_cand"] - per_subject["acc_ea"]).mean()),
            "neg_transfer_rate_vs_ea": float((per_subject["acc_cand"] - per_subject["acc_ea"] < 0.0).mean()),
            "accept_rate": 1.0,
        },
        {
            "method": str(args.selected_method_name),
            "n_subjects": int(per_subject.shape[0]),
            "mean_accuracy": mean_sel,
            "worst_accuracy": worst_sel,
            "mean_delta_vs_ea": delta_sel,
            "neg_transfer_rate_vs_ea": neg_rate,
            "accept_rate": accept_rate,
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / f"{date}_method_comparison.csv", index=False)

    # Provenance.
    prov = [
        "Built from:",
        f"- EA run: {str(Path(args.ea_run_dir))} (method={str(args.ea_method)})",
        f"- Candidate run: {str(Path(args.cand_run_dir))} (method={str(args.cand_method)})",
        f"- Gate mode: {str(args.mode)}",
    ]
    (out_dir / f"{date}_MERGED_FROM.txt").write_text("\n".join(prov) + "\n", encoding="utf-8")

    _write_results_txt(
        out_dir=out_dir,
        date_prefix=date,
        args=args,
        summary=summary,
        sources={
            "ea_run_dir": str(Path(args.ea_run_dir)),
            "cand_run_dir": str(Path(args.cand_run_dir)),
            "ea_method": str(args.ea_method),
            "cand_method": str(args.cand_method),
            "selected_method_name": str(args.selected_method_name),
        },
    )

    print("Wrote:", out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
