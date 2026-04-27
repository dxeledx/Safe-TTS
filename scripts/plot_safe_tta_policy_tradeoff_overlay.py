from __future__ import annotations

"""Overlay tradeoff curves across multiple policies.

Each input must be a summary.csv produced by scripts/analyze_safe_tta_risk_curves.py
with conditional-risk columns enabled.

Outputs (in out_dir):
  - policy_tradeoff_overlay.png
  - policy_comparison_table.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_alpha(variant: str) -> float:
    v = str(variant)
    m = re.search(r"alpha([0-9]*\.?[0-9]+)", v)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return float("nan")


def _parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        if ":" not in it:
            raise ValueError(f"Invalid --input {it!r} (expected LABEL:PATH)")
        label, path = it.split(":", 1)
        label = label.strip()
        path = Path(path.strip())
        if not label:
            raise ValueError(f"Invalid --input {it!r}: empty label")
        if not path.exists():
            raise ValueError(f"Missing path in --input {it!r}: {path}")
        out.append((label, path))
    if not out:
        raise ValueError("No valid --input entries parsed.")
    return out


def _resolve_palette(name: str) -> dict[str, str]:
    key = str(name).strip().lower()
    if key == "colorblind":
        return {
            "full": "#0072B2",
            "pred_only": "#D55E00",
            "guard_only": "#009E73",
            "norisk": "#CC79A7",
        }
    if key == "paper":
        return {
            "full": "#1F77B4",
            "pred_only": "#D62728",
            "guard_only": "#2CA02C",
            "norisk": "#9467BD",
        }
    return {
        "full": "#4C72B0",
        "pred_only": "#C44E52",
        "guard_only": "#55A868",
        "norisk": "#DD8452",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay tradeoff curves across policies.")
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable. Format: LABEL:PATH_TO_summary.csv (e.g., full:.../summary.csv).",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=str, default="0", help="eps suffix used in cond_* columns (default: 0).")
    ap.add_argument(
        "--cond-scope",
        type=str,
        default="pre",
        choices=["pre", "after_review"],
        help="Use conditional risk columns from pre-review decisions (default) or after_review (review-proxy).",
    )
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument(
        "--norisk",
        type=Path,
        default=None,
        help="Optional summary.csv for a single no-risk point (alpha=1), plotted as a star marker.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-size", type=float, default=9.0)
    ap.add_argument("--line-width", type=float, default=2.2)
    ap.add_argument("--risk-ymax", type=float, default=None, help="Optional fixed upper limit for risk subplot.")
    ap.add_argument(
        "--utility-all-col",
        type=str,
        default="mean_delta_all",
        help="Column name for all-subject utility curve (default: mean_delta_all).",
    )
    ap.add_argument(
        "--utility-acc-col",
        type=str,
        default="mean_delta_accepted",
        help="Column name for accepted-subject utility curve (default: mean_delta_accepted).",
    )
    ap.add_argument(
        "--palette",
        type=str,
        default="paper",
        choices=["default", "paper", "colorblind"],
        help="Color palette preset.",
    )
    args = ap.parse_args()

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

    eps = str(args.eps).strip()
    if not eps:
        raise RuntimeError("--eps cannot be empty")

    if str(args.cond_scope) == "after_review":
        col_cond = f"cond_neg_event_rate_after_review_eps{eps}"
        col_lo = f"cond_clopper_pearson_ci_low_after_review_eps{eps}"
        col_hi = f"cond_clopper_pearson_ci_high_after_review_eps{eps}"
    else:
        col_cond = f"cond_neg_event_rate_eps{eps}"
        col_lo = f"cond_clopper_pearson_ci_low_eps{eps}"
        col_hi = f"cond_clopper_pearson_ci_high_eps{eps}"

    inputs = _parse_inputs(list(args.input))

    # Load and concatenate for the comparison table
    table_parts = []

    colors = _resolve_palette(args.palette)

    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.18)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_u = fig.add_subplot(gs[1, 0], sharex=ax_r)

    # Overlay each policy
    for label, path in inputs:
        df = pd.read_csv(path)
        if "variant" not in df.columns:
            raise RuntimeError(f"{path}: missing column 'variant'")
        req_cols = [col_cond, col_lo, col_hi, "accept_rate", str(args.utility_all_col)]
        for c in req_cols:
            if c not in df.columns:
                raise RuntimeError(f"{path}: missing required column {c!r}. Have: {list(df.columns)}")
        if str(args.utility_acc_col) not in df.columns:
            # Keep backward compatibility: allow using only all-subject utility.
            df[str(args.utility_acc_col)] = np.nan

        df = df.assign(alpha=df["variant"].astype(str).apply(_parse_alpha))
        df = df.sort_values("accept_rate", na_position="last").reset_index(drop=True)

        x = pd.to_numeric(df["accept_rate"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[col_cond], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(df[col_lo], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[col_hi], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            x0, y0, lo0, hi0 = x[mask], y[mask], lo[mask], hi[mask]
            color = colors.get(label, None) or None
            ax_r.plot(
                x0,
                y0,
                marker="o",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=float(args.line_width),
                color=color,
                label=label,
            )
            ax_r.fill_between(x0, lo0, hi0, color=color, alpha=0.12)

        # Utility overlay
        y_all = pd.to_numeric(df[str(args.utility_all_col)], errors="coerce").to_numpy(dtype=float)
        y_acc = pd.to_numeric(df[str(args.utility_acc_col)], errors="coerce").to_numpy(dtype=float)
        m2 = np.isfinite(x) & np.isfinite(y_all)
        if np.any(m2):
            ax_u.plot(
                x[m2],
                y_all[m2],
                marker="o",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=float(args.line_width),
                color=(colors.get(label, "#4C72B0")),
                alpha=0.55,
                linestyle="-",
                label=f"{label}: {args.utility_all_col}",
            )
        m3 = np.isfinite(x) & np.isfinite(y_acc)
        if np.any(m3):
            ax_u.plot(
                x[m3],
                y_acc[m3],
                marker="s",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=float(args.line_width),
                color=(colors.get(label, "#4C72B0")),
                alpha=0.95,
                linestyle="-",
                label=f"{label}: {args.utility_acc_col}",
            )

        df_out = df.copy()
        df_out.insert(0, "policy", str(label))
        df_out.insert(1, "summary_csv", str(path))
        table_parts.append(df_out)

    # Optional: no-risk point (star marker)
    if args.norisk is not None:
        df0 = pd.read_csv(Path(args.norisk))
        if df0.shape[0] < 1:
            raise RuntimeError(f"--norisk summary is empty: {args.norisk}")
        r0 = df0.iloc[0]
        x0 = float(r0.get("accept_rate", float("nan")))
        y0 = float(r0.get(col_cond, float("nan")))
        lo0 = float(r0.get(col_lo, float("nan")))
        hi0 = float(r0.get(col_hi, float("nan")))
        if np.isfinite(x0) and np.isfinite(y0):
            ax_r.scatter(
                [x0],
                [y0],
                s=140,
                marker="*",
                color=colors.get("norisk", "#DD8452"),
                edgecolor="k",
                linewidth=0.6,
                zorder=5,
                label="no-risk (alpha=1)",
            )
            if np.isfinite(lo0) and np.isfinite(hi0):
                ax_r.vlines(
                    [x0],
                    [lo0],
                    [hi0],
                    color=colors.get("norisk", "#DD8452"),
                    alpha=0.7,
                    linewidth=max(1.8, float(args.line_width) - 0.2),
                )
        y_all0 = float(r0.get(str(args.utility_all_col), float("nan")))
        y_acc0 = float(r0.get(str(args.utility_acc_col), float("nan")))
        if np.isfinite(x0) and np.isfinite(y_all0):
            ax_u.scatter(
                [x0],
                [y_all0],
                s=110,
                marker="*",
                color=colors.get("norisk", "#DD8452"),
                edgecolor="k",
                linewidth=0.6,
                zorder=5,
            )
        if np.isfinite(x0) and np.isfinite(y_acc0):
            ax_u.scatter(
                [x0],
                [y_acc0],
                s=110,
                marker="*",
                color=colors.get("norisk", "#DD8452"),
                edgecolor="k",
                linewidth=0.6,
                zorder=5,
            )

    # Styling
    title = str(args.title).strip() if args.title else f"Policy overlay (conditional risk, eps={eps})"
    fig.suptitle(title, y=0.995)

    ax_r.set_ylabel("Pr(neg | acc)")
    ax_r.grid(True, linestyle="--", alpha=0.3)
    try:
        risk_ymax = float(np.nanmax(ax_r.get_lines()[0].get_ydata())) if ax_r.get_lines() else 0.2
    except Exception:
        risk_ymax = 0.2
    try:
        for col in ax_r.collections:
            if hasattr(col, "get_paths"):
                for p in col.get_paths():
                    ys = p.vertices[:, 1]
                    if ys.size:
                        risk_ymax = max(risk_ymax, float(np.nanmax(ys)))
    except Exception:
        pass
    risk_ymax = min(1.0, max(0.05, risk_ymax * 1.12 + 0.01))
    if args.risk_ymax is not None:
        risk_ymax = float(args.risk_ymax)
    ax_r.set_ylim(-0.01, risk_ymax)
    ax_r.legend(loc="best", fontsize=max(7.5, float(args.font_size) - 0.5))

    ax_u.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    ax_u.set_xlabel("acceptance rate")
    ax_u.set_ylabel("Δ accuracy (selected − anchor)")
    ax_u.grid(True, linestyle="--", alpha=0.3)
    ax_u.legend(loc="best", fontsize=max(7.0, float(args.font_size) - 1.0), ncol=1)

    fig.tight_layout()
    out_png = out_dir / "policy_tradeoff_overlay.png"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    df_table = pd.concat(table_parts, axis=0, ignore_index=True) if table_parts else pd.DataFrame()
    df_table.to_csv(out_dir / "policy_comparison_table.csv", index=False)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_dir / 'policy_comparison_table.csv'}")


if __name__ == "__main__":
    main()
