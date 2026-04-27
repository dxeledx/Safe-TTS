#!/usr/bin/env python3
from __future__ import annotations

"""
Plot subject-level accept/reject (and reviewed) indicators from SAFE-TTA per_subject_selection.csv.

Input
-----
*_per_subject_selection.csv written by scripts/offline_safe_tta_multi_select_crc_from_predictions.py

Output
------
One PNG figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-subject accept/reject markers.")
    p.add_argument("--per-subject", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, default="")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(Path(args.per_subject))
    if "subject" not in df.columns or "accept" not in df.columns:
        raise RuntimeError(f"{args.per_subject}: requires columns subject,accept. Have: {list(df.columns)}")

    df = df.copy()
    df["subject"] = pd.to_numeric(df["subject"], errors="coerce").fillna(-1).astype(int)
    df = df.sort_values("subject").reset_index(drop=True)

    accept = pd.to_numeric(df["accept"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
    reviewed = (
        pd.to_numeric(df["review_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
        if "review_flag" in df.columns
        else np.zeros_like(accept, dtype=bool)
    )

    n = int(df.shape[0])
    x = np.arange(n, dtype=int)

    # Layout: make very wide plots for large-N datasets (PhysioNetMI=109).
    fig_w = 12.5 if n >= 80 else (9.5 if n >= 40 else 7.0)
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))

    colors = np.where(accept, "#4C72B0", "#BDBDBD")
    ax.bar(x, np.ones_like(x, dtype=float), color=colors, width=1.0, linewidth=0)

    if np.any(reviewed):
        ax.scatter(
            x[reviewed],
            np.full(int(np.sum(reviewed)), 1.03, dtype=float),
            marker="*",
            s=36,
            color="#DD8452",
            edgecolor="white",
            linewidth=0.4,
            label="reviewed",
            zorder=5,
        )

    # Ticks: keep readable.
    if n > 35:
        ticks = np.unique(np.clip(np.linspace(0, n - 1, 7).round().astype(int), 0, n - 1))
        ax.set_xticks(ticks.tolist())
        ax.set_xticklabels(df["subject"].to_numpy()[ticks].tolist(), fontsize=9)
    else:
        ax.set_xticks(x.tolist())
        ax.set_xticklabels(df["subject"].astype(int).tolist(), fontsize=9)

    ax.set_yticks([])
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("subject (sorted)")
    ax.set_title(str(args.title).strip() or "SAFE-TTA accept/reject per subject")
    ax.grid(False)
    if np.any(reviewed):
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

