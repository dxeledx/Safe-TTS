from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 file match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_selected_acc(run_dir: Path, method: str) -> pd.Series:
    pred_path = _find_single(run_dir, "*_predictions_all_methods.csv")
    df = pd.read_csv(pred_path)
    df = df[df["method"].astype(str) == str(method)].copy()
    if df.empty:
        raise RuntimeError(f"No predictions for method={method} in {pred_path}")
    df["correct"] = (df["y_true"].astype(str) == df["y_pred"].astype(str)).astype(float)
    return df.groupby("subject", sort=True)["correct"].mean()


def _candidate_stats(*, run_dir: Path, method: str) -> dict:
    diag_root = Path(run_dir) / "diagnostics" / str(method)
    cand_paths = sorted(diag_root.glob("subject_*/candidates.csv"))
    if not cand_paths:
        return {}

    sel_acc = _load_selected_acc(run_dir, method=method)

    per_subj = []
    for cand_path in cand_paths:
        subj = int(Path(cand_path).parent.name.split("_")[-1])
        df = pd.read_csv(cand_path)
        acc = df["accuracy"].astype(float).to_numpy()

        id_df = df[df["kind"].astype(str) == "identity"]
        id_acc = float(id_df["accuracy"].iloc[0]) if len(id_df) else float("nan")
        oracle_acc = float(np.nanmax(acc))
        sel = float(sel_acc.get(subj, float("nan")))

        pmh = df["probe_mixup_hard_best"].astype(float).to_numpy() if "probe_mixup_hard_best" in df.columns else None
        rp = df["ridge_pred_improve"].astype(float).to_numpy() if "ridge_pred_improve" in df.columns else None
        gp = df["guard_p_pos"].astype(float).to_numpy() if "guard_p_pos" in df.columns else None

        rho_pmh = float("nan")
        if pmh is not None:
            rho_pmh = _spearman(-pmh, acc)

        rho_ridge = float("nan")
        if rp is not None:
            rho_ridge = _spearman(rp, acc)

        rho_guard = float("nan")
        if gp is not None:
            rho_guard = _spearman(gp, acc)

        nan_rate_pmh = float("nan")
        if pmh is not None:
            nan_rate_pmh = float(np.mean(~np.isfinite(pmh)))

        per_subj.append(
            {
                "subject": subj,
                "id_acc": id_acc,
                "sel_acc": sel,
                "oracle_acc": oracle_acc,
                "rho_probe_hard": rho_pmh,
                "rho_ridge": rho_ridge,
                "rho_guard": rho_guard,
                "nan_rate_probe_hard": nan_rate_pmh,
                "n_candidates": int(len(df)),
            }
        )

    t = pd.DataFrame(per_subj).sort_values("subject")
    id_mean = float(np.nanmean(t["id_acc"].to_numpy(dtype=float)))
    sel_mean = float(np.nanmean(t["sel_acc"].to_numpy(dtype=float)))
    oracle_mean = float(np.nanmean(t["oracle_acc"].to_numpy(dtype=float)))
    headroom = oracle_mean - id_mean
    eaten = sel_mean - id_mean
    frac = float("nan") if not np.isfinite(headroom) or abs(headroom) < 1e-12 else float(eaten / headroom)

    return {
        "id_mean": id_mean,
        "sel_mean": sel_mean,
        "oracle_mean": oracle_mean,
        "gap_sel_mean": float(np.nanmean((t["oracle_acc"] - t["sel_acc"]).to_numpy(dtype=float))),
        "headroom_mean": headroom,
        "eaten_mean": eaten,
        "frac_headroom_eaten": frac,
        "rho_probe_hard_mean": float(np.nanmean(t["rho_probe_hard"].to_numpy(dtype=float))),
        "rho_ridge_mean": float(np.nanmean(t["rho_ridge"].to_numpy(dtype=float))),
        "rho_guard_mean": float(np.nanmean(t["rho_guard"].to_numpy(dtype=float))),
        "nan_rate_probe_hard_mean": float(np.nanmean(t["nan_rate_probe_hard"].to_numpy(dtype=float))),
        "n_candidates_mean": float(np.nanmean(t["n_candidates"].to_numpy(dtype=float))),
    }


def _method_comparison_stats(*, run_dir: Path, method: str) -> dict:
    mc_path = _find_single(run_dir, "*_method_comparison.csv")
    df = pd.read_csv(mc_path)
    df = df[df["method"].astype(str) == str(method)].copy()
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    out = {"method_comparison_path": str(mc_path)}
    for k in [
        "mean_accuracy",
        "worst_accuracy",
        "mean_delta_vs_ea",
        "neg_transfer_rate_vs_ea",
        "accept_rate",
        "guard_improve_spearman",
        "cert_improve_spearman",
        "guard_train_auc_mean",
        "guard_train_spearman_mean",
    ]:
        v = row.get(k, float("nan"))
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize certificate reliability + headroom from existing LOSO runs.")
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--method", type=str, default="ea-stack-multi-safe-csp-lda")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        row = {"run_dir": str(run_dir)}
        row.update(_method_comparison_stats(run_dir=run_dir, method=args.method))
        row.update(_candidate_stats(run_dir=run_dir, method=args.method))
        rows.append(row)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        md = out.to_markdown(index=False)
        args.out_md.write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

