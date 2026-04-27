from __future__ import annotations

"""Plot a paper-friendly combined tradeoff figure from analyze_safe_tta_risk_curves.py summary.csv.

Example:
  python3 scripts/plot_safe_tta_tradeoff_from_summary.py \
    --summary docs/experiments/figures/20260129_physionetmi_crc_cp_alpha0.20_topm_sweep_riskcurves_v1/summary.csv \
    --out-dir docs/experiments/figures/20260129_physionetmi_crc_cp_alpha0.20_topm_sweep_riskcurves_v1 \
    --title "PhysioNetMI alpha=0.20 delta=0.05 topm sweep"
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_eps_list(columns: list[str]) -> list[str]:
    # Columns look like: neg_event_rate_eps0.005
    eps = []
    for c in columns:
        m = re.fullmatch(r"neg_event_rate_eps(.+)", str(c))
        if m:
            eps.append(m.group(1))
    # Sort numerically when possible.
    def key(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 1e9

    eps = sorted(set(eps), key=key)
    return eps


def _short_variant(v: str) -> str:
    v = str(v)
    # Common patterns in our sweeps
    v = v.replace("_delta0.05_scopeAll", "")
    v = v.replace("_scopeAll", "")
    v = v.replace("topm", "m")
    v = v.replace("alpha", "a")
    return v


def _pretty_x_label(x_axis: str) -> str:
    x_axis = str(x_axis)
    if x_axis == "accept_rate":
        return "acceptance rate"
    if x_axis == "selection_coverage":
        return "selection coverage"
    return x_axis


def _parse_eps_filter(eps_list: list[str], eps_arg: str | None) -> list[str]:
    if not eps_arg:
        return eps_list

    # Match numerically to tolerate formatting differences (e.g., "0" vs "0.0").
    eps_map: dict[float, str] = {}
    for eps in eps_list:
        try:
            eps_map[float(eps)] = str(eps)
        except Exception:
            pass

    want: list[float] = []
    for part in str(eps_arg).split(","):
        part = part.strip()
        if not part:
            continue
        want.append(float(part))

    picked: list[str] = []
    for w in want:
        if w in eps_map:
            picked.append(eps_map[w])
            continue
        # Fallback: float rounding issues
        found = None
        for k, v in eps_map.items():
            if abs(k - w) <= 1e-12:
                found = v
                break
        if found is None:
            raise RuntimeError(f"--eps requested {w} but available eps are: {sorted(eps_map.keys())}")
        picked.append(found)

    # Unique, preserve order
    out: list[str] = []
    seen = set()
    for e in picked:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _resolve_palette(name: str) -> tuple[list[str], list[str]]:
    key = str(name).strip().lower()
    if key == "colorblind":
        risk_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        util_colors = ["#0072B2", "#D55E00", "#009E73"]
        return risk_colors, util_colors
    if key == "paper":
        risk_colors = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD"]
        util_colors = ["#1F77B4", "#D62728", "#2CA02C"]
        return risk_colors, util_colors
    return (plt.rcParams["axes.prop_cycle"].by_key().get("color", []), ["#4C72B0", "#DD8452", "#55A868"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot combined risk/utility figure from summary.csv")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--x-axis",
        type=str,
        default="accept_rate",
        choices=["accept_rate", "selection_coverage"],
    )
    ap.add_argument(
        "--eps",
        type=str,
        default=None,
        help="Comma-separated eps values to plot (e.g., '0' or '0,0.005'). Default: plot all found.",
    )
    ap.add_argument(
        "--variant-regex",
        type=str,
        default=None,
        help="Only keep rows whose variant matches this regex (re.search).",
    )
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--annotate", action="store_true", help="Annotate points with variant labels")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-size", type=float, default=9.0)
    ap.add_argument("--line-width", type=float, default=2.2)
    ap.add_argument("--risk-ymax", type=float, default=None, help="Optional fixed upper limit for risk subplot.")
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

    df = pd.read_csv(summary_path)
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
    if args.variant_regex:
        pat = re.compile(str(args.variant_regex))
        if "variant" not in df.columns:
            raise RuntimeError("summary missing column 'variant' required for --variant-regex")
        keep = df["variant"].astype(str).apply(lambda v: bool(pat.search(v)))
        df = df.loc[keep].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"--variant-regex matched 0 rows: {args.variant_regex!r}")

    if args.x_axis not in df.columns:
        raise RuntimeError(f"summary missing x-axis column {args.x_axis!r}. Have: {list(df.columns)}")

    eps_list = _extract_eps_list(list(df.columns))
    if not eps_list:
        raise RuntimeError("No neg_event_rate_eps* columns found in summary")
    eps_list = _parse_eps_filter(eps_list, args.eps)

    df = df.sort_values(args.x_axis, na_position="last").reset_index(drop=True)

    x = pd.to_numeric(df[args.x_axis], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x)
    dfp = df.loc[finite].reset_index(drop=True)
    x = pd.to_numeric(dfp[args.x_axis], errors="coerce").to_numpy(dtype=float)

    title_base = str(args.title).strip() if args.title else ""

    # Combined figure: top risk, bottom utility
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.18)

    ax_risk = fig.add_subplot(gs[0, 0])
    ax_util = fig.add_subplot(gs[1, 0], sharex=ax_risk)

    risk_colors, util_colors = _resolve_palette(args.palette)

    # Risk
    all_risk = []
    for i, eps in enumerate(eps_list):
        y = pd.to_numeric(dfp[f"neg_event_rate_eps{eps}"], errors="coerce").to_numpy(dtype=float)
        u = pd.to_numeric(dfp[f"clopper_pearson_ucb_eps{eps}"], errors="coerce").to_numpy(dtype=float)
        all_risk.append(y)
        all_risk.append(u)
        color = risk_colors[i % len(risk_colors)] if risk_colors else None
        ax_risk.plot(
            x,
            y,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=float(args.line_width),
            color=color,
            label=f"risk (eps={eps})",
        )
        ax_risk.plot(
            x,
            u,
            linestyle=(0, (4, 2)),
            linewidth=float(args.line_width),
            color=color,
            label=f"UCB (eps={eps})",
        )

    ax_risk.set_ylabel("neg-transfer event rate")
    try:
        ymax = float(np.nanmax(np.concatenate(all_risk))) if all_risk else 1.0
    except Exception:
        ymax = 1.0
    ymax = max(0.02, ymax)
    ymax = min(1.0, ymax * 1.15 + 0.01)
    if args.risk_ymax is not None:
        ymax = float(args.risk_ymax)
    ax_risk.set_ylim(-0.01, ymax)
    ax_risk.grid(True, linestyle="--", alpha=0.3)
    ax_risk.legend(loc="best", fontsize=max(7.5, float(args.font_size) - 0.5), ncol=2 if len(eps_list) > 1 else 1)

    # Utility
    ax_util.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    util_series = [
        ("mean_delta_all", "mean $\\Delta$ (all)", util_colors[0] if len(util_colors) > 0 else "#4C72B0"),
        ("mean_delta_accepted", "mean $\\Delta$ (accepted)", util_colors[1] if len(util_colors) > 1 else "#DD8452"),
        ("q20_delta_accepted", "20th pct $\\Delta$ (accepted)", util_colors[2] if len(util_colors) > 2 else "#55A868"),
    ]
    all_util = []
    for name, label, color in util_series:
        if name in dfp.columns:
            y = pd.to_numeric(dfp[name], errors="coerce").to_numpy(dtype=float)
            all_util.append(y)
            ax_util.plot(
                x,
                y,
                marker="o",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=float(args.line_width),
                color=color,
                label=label,
            )

    if all_util:
        try:
            ymin = float(np.nanmin(np.concatenate(all_util)))
            ymax = float(np.nanmax(np.concatenate(all_util)))
        except Exception:
            ymin, ymax = -0.01, 0.01
        span = max(1e-9, ymax - ymin)
        pad = max(0.002, 0.15 * span)
        ax_util.set_ylim(ymin - pad, ymax + pad)

    ax_util.set_xlabel(_pretty_x_label(args.x_axis))
    ax_util.set_ylabel("delta accuracy (selected - anchor)")
    ax_util.grid(True, linestyle="--", alpha=0.3)
    ax_util.legend(loc="best", fontsize=max(7.5, float(args.font_size) - 0.5))

    if title_base:
        fig.suptitle(title_base, y=0.995)

    # X limits: keep tight with a small margin.
    try:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        span = max(1e-9, xmax - xmin)
        pad = 0.04 * span + 0.01
        ax_risk.set_xlim(max(-0.02, xmin - pad), min(1.02, xmax + pad))
    except Exception:
        pass

    # Annotation: useful when there are few points (e.g., BNCI alpha sweep)
    if args.annotate or len(dfp) <= 8:
        for i, r in dfp.iterrows():
            lab = _short_variant(r.get("variant", str(i)))
            ax_risk.annotate(
                lab,
                (x[i], 0.0),
                xytext=(3, 6),
                textcoords="offset points",
                fontsize=max(7.5, float(args.font_size) - 1.0),
                alpha=0.9,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tradeoff_curve.png"
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Read: {summary_path}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
