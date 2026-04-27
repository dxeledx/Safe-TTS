from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_inputs(inputs: list[str], glob_pat: str | None) -> pd.DataFrame:
    paths: list[Path] = []
    for it in inputs:
        p = Path(str(it).strip())
        if p.exists():
            paths.append(p)
        else:
            raise RuntimeError(f"Input does not exist: {p}")

    if glob_pat:
        paths.extend(sorted(Path(".").glob(glob_pat)))

    if not paths:
        raise RuntimeError("No input files found. Use --input and/or --glob.")

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        if "method" not in df.columns:
            # treat as a generic row table
            pass
        df = df.copy()
        df.insert(0, "source_csv", str(p))
        rows.append(df)

    if not rows:
        raise RuntimeError("All input files are empty.")
    return pd.concat(rows, axis=0, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot net utility vs review budget/cost.")
    ap.add_argument("--input", action="append", default=[], help="Path to method_comparison.csv (repeatable).")
    ap.add_argument("--glob", type=str, default=None, help="Optional glob for method_comparison.csv files.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title", type=str, default="Review proxy cost curve")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_inputs(list(args.input), args.glob)

    required = ["review_budget_frac", "review_cost", "net_utility", "residual_harm_rate_after_review"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column {c!r}. Have: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[np.isfinite(df["review_budget_frac"]) & np.isfinite(df["review_cost"])].copy()
    if df.empty:
        raise RuntimeError("No finite rows for review_budget_frac/review_cost.")

    table = (
        df.sort_values(["review_cost", "review_budget_frac", "net_utility"], ascending=[True, True, False])
        .drop_duplicates(subset=["review_cost", "review_budget_frac"], keep="first")
        .reset_index(drop=True)
    )
    table.to_csv(out_dir / "review_cost_table.csv", index=False)

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 1, hspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    costs = sorted(table["review_cost"].dropna().unique().tolist())
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])

    for i, c in enumerate(costs):
        dfc = table[np.isclose(table["review_cost"], c)].sort_values("review_budget_frac")
        x = dfc["review_budget_frac"].to_numpy(dtype=float)
        y_net = dfc["net_utility"].to_numpy(dtype=float)
        y_harm = dfc["residual_harm_rate_after_review"].to_numpy(dtype=float)
        color = colors[i % len(colors)]

        ax1.plot(x, y_net, marker="o", linewidth=2.0, color=color, label=f"cost={c:g}")
        ax2.plot(x, y_harm, marker="s", linewidth=2.0, color=color, label=f"cost={c:g}")

    ax1.axhline(0.0, color="k", linewidth=1.0, alpha=0.35)
    ax1.set_ylabel("net utility")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="best")

    ax2.set_xlabel("review budget fraction q")
    ax2.set_ylabel("residual harm rate")
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(str(args.title))
    fig.tight_layout()
    out_png = out_dir / "review_cost_curve.png"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_dir / 'review_cost_table.csv'}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
