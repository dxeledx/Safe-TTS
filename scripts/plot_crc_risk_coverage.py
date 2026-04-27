from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 file match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _infer_alpha(run_dir: Path) -> float:
    m = re.search(r"alpha([0-9.]+)", str(run_dir))
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")


def _load_method_row(run_dir: Path, method: str) -> dict[str, float]:
    comp_path = _find_single(run_dir, "*_method_comparison.csv")
    df = pd.read_csv(comp_path)
    if "method" not in df.columns:
        raise RuntimeError(f"{comp_path} missing 'method' column")
    row = df[df["method"].astype(str) == str(method)]
    if row.empty:
        raise RuntimeError(f"Method {method!r} not found in {comp_path}. Methods: {sorted(df['method'].astype(str).tolist())}")
    r = row.iloc[0]
    out: dict[str, float] = {}
    for k in [
        "mean_accuracy",
        "worst_accuracy",
        "meanΔacc_vs_ea-csp-lda",
        "neg_transfer_vs_ea-csp-lda",
        "accept_rate",
    ]:
        out[k] = float(r.get(k, float("nan")))
    return out


def _load_per_subject_stats(run_dir: Path) -> dict[str, float]:
    try:
        path = _find_single(run_dir, "*_per_subject_selection.csv")
    except Exception:
        return {}
    df = pd.read_csv(path)
    out: dict[str, float] = {}
    if "feasible_empty" in df.columns:
        out["feasible_empty_rate"] = float(pd.to_numeric(df["feasible_empty"], errors="coerce").mean())
    if "acc_anchor" in df.columns:
        out["anchor_worst_acc"] = float(pd.to_numeric(df["acc_anchor"], errors="coerce").min())
        out["anchor_mean_acc"] = float(pd.to_numeric(df["acc_anchor"], errors="coerce").mean())
    if "acc_selected" in df.columns:
        out["selected_worst_acc"] = float(pd.to_numeric(df["acc_selected"], errors="coerce").min())
        out["selected_mean_acc"] = float(pd.to_numeric(df["acc_selected"], errors="coerce").mean())
    if "delta_selected_vs_anchor" in df.columns:
        out["worst_delta"] = float(pd.to_numeric(df["delta_selected_vs_anchor"], errors="coerce").min())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot CRC risk–coverage curves from multiple offline selector run dirs (method_comparison + per_subject)."
    )
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    args = ap.parse_args()

    rows = []
    for d in args.run_dirs:
        run_dir = Path(d)
        alpha = _infer_alpha(run_dir)
        m = _load_method_row(run_dir, method=str(args.method))
        ps = _load_per_subject_stats(run_dir)
        rows.append(
            {
                "alpha": float(alpha),
                "accept_rate": float(m.get("accept_rate", float("nan"))),
                "neg_transfer": float(m.get("neg_transfer_vs_ea-csp-lda", float("nan"))),
                "mean_delta": float(m.get("meanΔacc_vs_ea-csp-lda", float("nan"))),
                "worst_accuracy": float(m.get("worst_accuracy", float("nan"))),
                "mean_accuracy": float(m.get("mean_accuracy", float("nan"))),
                "feasible_empty_rate": float(ps.get("feasible_empty_rate", float("nan"))),
                "anchor_worst_acc": float(ps.get("anchor_worst_acc", float("nan"))),
                "anchor_mean_acc": float(ps.get("anchor_mean_acc", float("nan"))),
                "worst_delta": float(ps.get("worst_delta", float("nan"))),
                "run_dir": str(run_dir),
            }
        )

    df = pd.DataFrame(rows).sort_values("alpha")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.prefix}_risk_coverage_points.csv", index=False)

    # Plot 1: risk vs coverage.
    x = df["accept_rate"].to_numpy(dtype=float)
    y = df["neg_transfer"].to_numpy(dtype=float)
    al = df["alpha"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(x, y, marker="o", color="#4C72B0")
    for xi, yi, ai in zip(x, y, al, strict=False):
        ax.annotate(f"α={ai:g}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("accept_rate (coverage)")
    ax.set_ylabel("neg_transfer_rate vs EA")
    ax.set_title("CRC-CP: risk–coverage curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}_risk_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: mean Δacc vs coverage.
    y2 = df["mean_delta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.4)
    ax.plot(x, y2, marker="o", color="#DD8452")
    for xi, yi, ai in zip(x, y2, al, strict=False):
        ax.annotate(f"α={ai:g}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("accept_rate (coverage)")
    ax.set_ylabel("mean Δacc vs EA")
    ax.set_title("CRC-CP: utility–coverage curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}_delta_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: worst-subject accuracy vs alpha (with anchor baseline line if available).
    worst = df["worst_accuracy"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(al, worst, marker="o", color="#4C72B0", label="worst_accuracy (selected)")
    anchor_worst = df["anchor_worst_acc"].to_numpy(dtype=float)
    if np.isfinite(anchor_worst).any():
        ax.axhline(float(np.nanmin(anchor_worst)), color="k", linestyle="--", alpha=0.4, label="anchor worst (EA)")
    ax.set_xlabel("risk budget α")
    ax.set_ylabel("worst-subject accuracy")
    ax.set_title("CRC-CP: worst-subject vs α")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}_worst_vs_alpha.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: feasible_empty rate vs alpha.
    fe = df["feasible_empty_rate"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(al, fe, marker="o", color="#8172B3")
    ax.set_xlabel("risk budget α")
    ax.set_ylabel("feasible_empty rate")
    ax.set_title("CRC-CP: vacuous-bound abstain rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}_feasible_empty_vs_alpha.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_dir / (args.prefix + '_risk_coverage_points.csv')}")


if __name__ == "__main__":
    main()

