#!/usr/bin/env python3
from __future__ import annotations

"""
Plot a "risk–benefit" scatter where point size/color encode review budget q*.

Typical SAFE-TTA summary.csv columns (analyze_safe_tta_risk_curves.py):
- accept_rate
- chosen_review_budget_frac (q*)
- cond_clopper_pearson_ucb_after_review_eps0  (post conditional-risk UCB)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot risk–benefit scatter with q* encoding.")
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--x-col", type=str, default="cond_clopper_pearson_ucb_after_review_eps0")
    p.add_argument("--y-col", type=str, default="accept_rate")
    p.add_argument("--q-col", type=str, default="chosen_review_budget_frac")
    p.add_argument("--title", type=str, default="")
    p.add_argument("--annotate", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(Path(args.summary))

    for c in [args.x_col, args.y_col, args.q_col]:
        if c not in df.columns:
            raise RuntimeError(f"{args.summary}: missing column {c!r}. Have: {list(df.columns)}")

    x = pd.to_numeric(df[args.x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[args.y_col], errors="coerce").to_numpy(dtype=float)
    q = pd.to_numeric(df[args.q_col], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(q)
    x, y, q = x[m], y[m], q[m]
    d = df.loc[m].reset_index(drop=True)

    if x.size < 1:
        raise RuntimeError("No finite rows found after filtering.")

    # Encode q* with both size and color for readability in print.
    q_clip = np.clip(q, 0.0, 1.0)
    sizes = 40.0 + 420.0 * q_clip

    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=q_clip,
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_xlabel(args.x_col)
    ax.set_ylabel(args.y_col)
    ax.set_title(str(args.title).strip() or "Risk–benefit (size/color=q*)")
    ax.grid(True, linestyle="--", alpha=0.3)

    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("q* (review budget fraction)")

    if bool(args.annotate) and "variant" in d.columns and x.size <= 40:
        for i, v in enumerate(d["variant"].astype(str).tolist()):
            ax.annotate(v, (float(x[i]), float(y[i])), fontsize=8, alpha=0.8, xytext=(4, 3), textcoords="offset points")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

