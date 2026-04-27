from __future__ import annotations

"""
Analyze DeepTransferEEG suite predictions (SAFE_TTA format) and produce a candidate-arm report.

Input
-----
predictions_all_methods.csv:
  method,subject,trial,y_true,y_pred,proba_*

Outputs (to --out-dir)
----------------------
- per_subject_acc.csv         (rows=subjects, cols=methods)
- method_summary.csv          (mean, worst, neg-transfer vs baseline, etc.)
- method_mean_acc_bar.png
- delta_vs_baseline_scatter.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze TTA suite predictions and make summary figures.")
    p.add_argument("--preds", type=Path, required=True, help="predictions_all_methods.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--baseline-method", type=str, default="eegnet_ea")
    return p.parse_args()


def _accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    # subject x method accuracy
    subj = df["subject"].astype(int)
    method = df["method"].astype(str)
    correct = (df["y_true"].astype(str) == df["y_pred"].astype(str)).astype(int)
    tab = (
        pd.DataFrame({"subject": subj, "method": method, "correct": correct})
        .groupby(["subject", "method"], sort=True)["correct"]
        .mean()
        .unstack("method")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return tab


def _method_summary(*, acc: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if baseline not in acc.columns:
        raise RuntimeError(f"baseline method {baseline!r} not found in columns: {list(acc.columns)}")

    base = acc[baseline]
    rows = []
    for m in acc.columns:
        a = acc[m]
        delta = a - base
        rows.append(
            {
                "method": m,
                "mean_acc": float(a.mean()),
                "worst_subject_acc": float(a.min()),
                "neg_transfer_rate_vs_baseline": float((delta < 0).mean()),
                "mean_delta_vs_baseline": float(delta.mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["method"])
    return out


def _plot_mean_bar(*, summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.sort_values("mean_acc", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(df["method"], df["mean_acc"] * 100.0, color="#4C78A8")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_scatter(*, acc: pd.DataFrame, baseline: str, out_path: Path) -> None:
    base = acc[baseline]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for m in acc.columns:
        if m == baseline:
            continue
        delta = acc[m] - base
        ax.scatter(base * 100.0, delta * 100.0, s=28, alpha=0.75, label=m)
    ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.6)
    ax.set_xlabel(f"Baseline acc (%) [{baseline}]")
    ax.set_ylabel("Delta acc vs baseline (pp)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    preds = Path(args.preds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds)
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    acc = _accuracy_table(df)
    acc.to_csv(out_dir / "per_subject_acc.csv", index=True)

    summary = _method_summary(acc=acc, baseline=str(args.baseline_method))
    summary.to_csv(out_dir / "method_summary.csv", index=False)

    _plot_mean_bar(summary=summary, out_path=out_dir / "method_mean_acc_bar.png")
    _plot_delta_scatter(acc=acc, baseline=str(args.baseline_method), out_path=out_dir / "delta_vs_baseline_scatter.png")

    print(f"[done] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

