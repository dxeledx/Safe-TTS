from __future__ import annotations

"""Plot the minimal achievable one-sided CP upper bound when k=0 for n samples.

This visualizes why small-S datasets (e.g., BNCI with n_train=S-1=8 per split)
can make strict risk bounds vacuous.

Example:
  python3 scripts/plot_safe_tta_small_sample_vacuous.py \
    --out docs/experiments/figures/20260131_cp_ucb_min_v1.png \
    --delta 0.05 \
    --n-max 200 \
    --mark 8,108
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def ucb_min_k0(n: np.ndarray, delta: float) -> np.ndarray:
    # For binomial with k=0 successes (negT events) and one-sided CP upper bound,
    # the smallest possible UCB is: 1 - delta^(1/n)
    n = np.asarray(n, dtype=float)
    return 1.0 - np.power(float(delta), 1.0 / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=200)
    ap.add_argument("--mark", type=str, default="8,108", help="Comma-separated n values to annotate")
    ap.add_argument("--title", type=str, default="Small-sample limit of CP UCB (k=0)")
    args = ap.parse_args()

    n_min = max(1, int(args.n_min))
    n_max = int(args.n_max)
    if n_min > n_max:
        raise RuntimeError(f"--n-min ({n_min}) must be <= --n-max ({n_max})")

    ns = np.arange(n_min, n_max + 1)
    y = ucb_min_k0(ns, float(args.delta))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ns, y, color="#4C72B0", linewidth=2)
    ax.set_xlabel("n (number of subjects)")
    ax.set_ylabel("min possible UCB when k=0")
    ymax = min(1.0, float(np.nanmax(y)) * 1.05 + 0.02)
    ax.set_ylim(0.0, ymax)
    ax.grid(True, linestyle="--", alpha=0.3)
    title = str(args.title).strip()
    if title:
        ax.set_title(f"{title} (delta={args.delta})")

    marks = []
    for part in str(args.mark).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            marks.append(int(part))
        except Exception:
            pass

    for n in marks:
        if n < 1 or n > int(args.n_max):
            continue
        yy = float(ucb_min_k0(np.array([n]), float(args.delta))[0])
        ax.scatter([n], [yy], color="#DD8452", zorder=3)
        ax.annotate(f"n={n}\nUCBmin={yy:.3f}", (n, yy), xytext=(8, 10), textcoords="offset points", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
