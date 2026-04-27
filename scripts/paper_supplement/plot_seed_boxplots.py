#!/usr/bin/env python3
from __future__ import annotations

"""
Make paper-friendly seed-robustness boxplots from a seed_sensitivity_table.csv.

Expected columns (examples in docs/paper_assets/safe_tta_cand2e_two_dataset_v1/data/physionetmi/*):
- calib_seed
- mean_accuracy_after_review
- cond_rate_after_review   (empirical conditional harm rate among accepted, post-review)
- chosen_review_budget_frac (q*)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot seed robustness boxplots (mean acc / harm rate / q*).")
    p.add_argument("--in-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _boxplot(values: np.ndarray, *, out_png: Path, ylabel: str, title: str, dpi: int) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    ax.boxplot(values, vert=True, widths=0.5, showmeans=True)
    # jittered points
    if values.size > 0:
        rng = np.random.default_rng(0)
        x = 1.0 + 0.06 * rng.normal(size=int(values.size))
        ax.scatter(x, values, s=32, alpha=0.85, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "iqr_low": float("nan"), "iqr_high": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size >= 2 else 0.0,
        "median": float(np.median(v)),
        "iqr_low": float(np.quantile(v, 0.25)),
        "iqr_high": float(np.quantile(v, 0.75)),
    }


def main() -> int:
    args = parse_args()
    df = pd.read_csv(Path(args.in_csv))

    required = {"mean_accuracy_after_review", "cond_rate_after_review", "chosen_review_budget_frac"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{args.in_csv}: missing columns: {missing}. Have: {list(df.columns)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_acc = pd.to_numeric(df["mean_accuracy_after_review"], errors="coerce").to_numpy(dtype=float)
    harm_rate = pd.to_numeric(df["cond_rate_after_review"], errors="coerce").to_numpy(dtype=float)
    q_star = pd.to_numeric(df["chosen_review_budget_frac"], errors="coerce").to_numpy(dtype=float)

    _boxplot(
        mean_acc,
        out_png=out_dir / "boxplot_mean_acc.png",
        ylabel="mean accuracy (after review)",
        title="Seed robustness: mean accuracy",
        dpi=int(args.dpi),
    )
    _boxplot(
        harm_rate,
        out_png=out_dir / "boxplot_neg_transfer.png",
        ylabel="empirical harm rate (cond, after review)",
        title="Seed robustness: harm rate",
        dpi=int(args.dpi),
    )
    _boxplot(
        q_star,
        out_png=out_dir / "boxplot_q_star.png",
        ylabel="q* (review budget fraction)",
        title="Seed robustness: q*",
        dpi=int(args.dpi),
    )

    rows = []
    for name, arr in [
        ("mean_accuracy_after_review", mean_acc),
        ("cond_rate_after_review", harm_rate),
        ("chosen_review_budget_frac", q_star),
    ]:
        s = _stats(arr)
        rows.append({"metric": name} | s)
    pd.DataFrame(rows).to_csv(out_dir / "seed_summary_stats.csv", index=False)

    print(f"[done] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

