from __future__ import annotations

"""Plot conditional negative-transfer risk among accepted subjects from summary.csv.

This script is designed to complement:
  - scripts/analyze_safe_tta_risk_curves.py  (writes summary.csv)

It reads the conditional-risk columns:
  - cond_neg_event_rate_eps{eps}
  - cond_clopper_pearson_ci_low_eps{eps}
  - cond_clopper_pearson_ci_high_eps{eps}
  - cond_clopper_pearson_ucb_eps{eps}
and produces:
  - conditional_risk_curve.png
  - conditional_risk_table.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_palette(name: str) -> tuple[str, str]:
    key = str(name).strip().lower()
    palettes: dict[str, tuple[str, str]] = {
        "default": ("#4C72B0", "#DD8452"),
        "paper": ("#1F77B4", "#D62728"),
        "colorblind": ("#0072B2", "#D55E00"),
    }
    return palettes.get(key, palettes["default"])


def _parse_alpha(variant: str) -> float:
    v = str(variant)
    m = re.search(r"alpha([0-9]*\.?[0-9]+)", v)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"\ba([0-9]*\.?[0-9]+)\b", v)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot conditional risk (neg|acc) curves from summary.csv")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=str, default="0", help="eps value (e.g., '0' or '0.005').")
    ap.add_argument(
        "--scope",
        type=str,
        default="pre",
        choices=["pre", "after_review"],
        help="Which conditional risk to plot: pre-review (default) or after_review (review-proxy final decisions).",
    )
    ap.add_argument(
        "--x-axis",
        type=str,
        default="alpha",
        choices=["alpha", "accept_rate"],
        help="Default alpha parses from the 'variant' column; falls back to accept_rate when missing.",
    )
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--plot-ucb", action="store_true", help="Plot one-sided CP UCB as dashed line.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-size", type=float, default=9.0)
    ap.add_argument("--line-width", type=float, default=2.2)
    ap.add_argument("--risk-ymax", type=float, default=None, help="Optional fixed upper limit for y-axis.")
    ap.add_argument(
        "--palette",
        type=str,
        default="paper",
        choices=["default", "paper", "colorblind"],
        help="Color palette preset.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise RuntimeError(f"Missing --summary: {summary_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": float(args.font_size),
            "axes.labelsize": float(args.font_size),
            "axes.titlesize": float(args.font_size) + 1.0,
            "legend.fontsize": float(args.font_size) - 0.5,
            "xtick.labelsize": float(args.font_size) - 0.5,
            "ytick.labelsize": float(args.font_size) - 0.5,
        }
    )
    color_main, color_aux = _resolve_palette(args.palette)

    df = pd.read_csv(summary_path)
    if "variant" not in df.columns:
        raise RuntimeError("summary.csv missing column 'variant'")

    eps = str(args.eps).strip()
    if not eps:
        raise RuntimeError("--eps cannot be empty")

    if str(args.scope) == "after_review":
        col_y = f"cond_neg_event_rate_after_review_eps{eps}"
        col_lo = f"cond_clopper_pearson_ci_low_after_review_eps{eps}"
        col_hi = f"cond_clopper_pearson_ci_high_after_review_eps{eps}"
        col_ucb = f"cond_clopper_pearson_ucb_after_review_eps{eps}"
        col_k = f"cond_neg_event_count_after_review_eps{eps}"
    else:
        col_y = f"cond_neg_event_rate_eps{eps}"
        col_lo = f"cond_clopper_pearson_ci_low_eps{eps}"
        col_hi = f"cond_clopper_pearson_ci_high_eps{eps}"
        col_ucb = f"cond_clopper_pearson_ucb_eps{eps}"
        col_k = f"neg_event_count_eps{eps}"

    for c in [col_y, col_lo, col_hi, col_ucb]:
        if c not in df.columns:
            raise RuntimeError(f"summary.csv missing required column: {c!r}. Have: {list(df.columns)}")

    # X axis
    if str(args.x_axis) == "alpha":
        alpha = df["variant"].astype(str).apply(_parse_alpha).to_numpy(dtype=float)
        if not np.isfinite(alpha).any():
            x = pd.to_numeric(df["accept_rate"], errors="coerce").to_numpy(dtype=float)
            x_label = "acceptance rate"
        else:
            df = df.assign(alpha=alpha)
            df = df.sort_values("alpha", na_position="last").reset_index(drop=True)
            x = df["alpha"].to_numpy(dtype=float)
            x_label = "risk budget alpha"
    else:
        df = df.sort_values("accept_rate", na_position="last").reset_index(drop=True)
        x = pd.to_numeric(df["accept_rate"], errors="coerce").to_numpy(dtype=float)
        x_label = "acceptance rate"

    y = pd.to_numeric(df[col_y], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df[col_lo], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df[col_hi], errors="coerce").to_numpy(dtype=float)
    ucb = pd.to_numeric(df[col_ucb], errors="coerce").to_numpy(dtype=float)

    accepted_n = pd.to_numeric(df.get("accepted_n", np.nan), errors="coerce").to_numpy(dtype=float)
    accept_rate = pd.to_numeric(df.get("accept_rate", np.nan), errors="coerce").to_numpy(dtype=float)
    k = pd.to_numeric(df.get(col_k, np.nan), errors="coerce").to_numpy(dtype=float) if col_k in df.columns else None

    # Write table even when there are no accepted subjects (all conditional-risk values are NaN).
    out = pd.DataFrame(
        {
            "alpha": df["alpha"] if "alpha" in df.columns else np.nan,
            "accept_rate": df.get("accept_rate", np.nan),
            "accepted_n": df.get("accepted_n", np.nan),
            "k_neg": df.get(col_k, np.nan) if col_k in df.columns else np.nan,
            "cond_risk": df.get(col_y, np.nan),
            "cp95_low": df.get(col_lo, np.nan),
            "cp95_high": df.get(col_hi, np.nan),
            "cp_ucb": df.get(col_ucb, np.nan),
            "chosen_review_budget_frac": df.get("chosen_review_budget_frac", np.nan),
            "chosen_review_budget_n": df.get("chosen_review_budget_n", np.nan),
            "review_infeasible": df.get("review_infeasible", np.nan),
        }
    )
    out.to_csv(out_dir / "conditional_risk_table.csv", index=False)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        # No accepted subjects for any variant => conditional risk undefined.
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        title = str(args.title).strip() if args.title else ""
        if not title:
            title = f"Conditional risk among accepted (eps={eps})"
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("conditional neg-transfer rate")
        ymax = float(args.risk_ymax) if args.risk_ymax is not None else 1.0
        ax.set_ylim(-0.01, ymax)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.text(
            0.5,
            0.5,
            "No accepted subjects; conditional risk is undefined.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=max(8.0, float(args.font_size)),
            alpha=0.9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "conditional_risk_curve.png", dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

        print(f"Read: {summary_path}")
        print(f"Wrote: {out_dir / 'conditional_risk_curve.png'}")
        print(f"Wrote: {out_dir / 'conditional_risk_table.csv'}")
        return

    x0 = x[mask]
    y0 = y[mask]
    lo0 = lo[mask]
    hi0 = hi[mask]
    u0 = ucb[mask]
    n0 = accepted_n[mask]
    ar0 = accept_rate[mask]
    k0 = k[mask] if k is not None else None

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        x0,
        y0,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=float(args.line_width),
        color=color_main,
        label="Pr(neg | acc)",
    )
    # CI band
    ax.fill_between(x0, lo0, hi0, color=color_main, alpha=0.18, label="CP 95% CI")
    if bool(args.plot_ucb):
        ax.plot(
            x0,
            u0,
            linestyle=(0, (4, 2)),
            linewidth=max(1.8, float(args.line_width) - 0.2),
            color=color_aux,
            label="CP one-sided UCB",
        )

    # annotate with accept_n
    for xi, yi, ni, ari in zip(x0, y0, n0, ar0, strict=False):
        if np.isfinite(ni) and np.isfinite(ari):
            ax.annotate(
                f"n={int(ni)}",
                (float(xi), float(yi)),
                xytext=(3, 6),
                textcoords="offset points",
                fontsize=max(7.5, float(args.font_size) - 1.0),
                alpha=0.8,
            )

    title = str(args.title).strip() if args.title else ""
    if not title:
        title = f"Conditional risk among accepted (eps={eps})"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("conditional neg-transfer rate")
    ymax = float(np.nanmax([np.nanmax(hi0), np.nanmax(u0) if bool(args.plot_ucb) else np.nanmax(hi0)]))
    ymax = max(0.02, ymax)
    ymax = min(1.0, ymax * 1.15 + 0.01)
    if args.risk_ymax is not None:
        ymax = float(args.risk_ymax)
    ax.set_ylim(-0.01, ymax)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=max(7.5, float(args.font_size) - 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "conditional_risk_curve.png", dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Read: {summary_path}")
    print(f"Wrote: {out_dir / 'conditional_risk_curve.png'}")
    print(f"Wrote: {out_dir / 'conditional_risk_table.csv'}")


if __name__ == "__main__":
    main()
