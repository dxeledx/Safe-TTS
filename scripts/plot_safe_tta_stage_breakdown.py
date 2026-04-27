from __future__ import annotations

"""Plot stage/reason breakdown across variants in a sweep.

Reads each variant's newest *per_subject_selection.csv and counts the 'reason'
(or falls back to accept/selected flags) to explain why subjects were not accepted.

Example:
  python3 scripts/plot_safe_tta_stage_breakdown.py \
    --root outputs/20260129/4class/review_physio_s1-109_alpha0.20_topm_sweep \
    --out-dir docs/experiments/figures/20260129_physio_stage_breakdown_v1 \
    --title "PhysioNetMI stage breakdown"
"""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_newest(variant_dir: Path) -> Path:
    cands = sorted(variant_dir.rglob("*per_subject_selection.csv"))
    if not cands:
        raise RuntimeError(f"No per_subject_selection.csv under {variant_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _get_accept(df: pd.DataFrame) -> np.ndarray:
    if "accept" in df.columns:
        v = pd.to_numeric(df["accept"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return (v > 0.5).astype(int)
    if "accepted" in df.columns:
        v = pd.to_numeric(df["accepted"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return (v > 0.5).astype(int)
    return np.zeros(len(df), dtype=int)


def _short_variant(v: str) -> str:
    v = str(v)
    v = v.replace("_delta0.05_scopeAll", "")
    v = v.replace("_scopeAll", "")
    v = v.replace("topm", "m")
    v = v.replace("alpha", "a")
    return v


def _pretty_x_axis(x_axis: str) -> str:
    x_axis = str(x_axis)
    if x_axis == "accept_rate":
        return "acc"
    if x_axis == "selection_coverage":
        return "cov"
    return x_axis


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, default=None, help="Root directory containing variant subdirs.")
    src.add_argument(
        "--stage-csv",
        type=Path,
        default=None,
        help="Existing stage_breakdown.csv produced by this script (replot only).",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument(
        "--x-axis",
        type=str,
        default="accept_rate",
        choices=["accept_rate", "selection_coverage"],
    )
    ap.add_argument(
        "--variant-regex",
        type=str,
        default=None,
        help="Only include variant directories whose name matches this regex (re.search).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage_csv:
        stage_csv = Path(args.stage_csv)
        if not stage_csv.exists():
            raise RuntimeError(f"Missing --stage-csv: {stage_csv}")
        df_sum = pd.read_csv(stage_csv)
        if "variant" not in df_sum.columns:
            raise RuntimeError(f"--stage-csv missing required column 'variant': {stage_csv}")
    else:
        root = Path(args.root)
        vdirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if args.variant_regex:
            pat = re.compile(str(args.variant_regex))
            vdirs = [p for p in vdirs if pat.search(p.name)]
        if not vdirs:
            raise RuntimeError(f"No variant subdirs under {root}")

        rows = []
        for v in vdirs:
            csv_path = _find_newest(v)
            df = pd.read_csv(csv_path)
            n = int(len(df))
            acc = _get_accept(df)

            # selection coverage
            if "selected_by_topm" in df.columns:
                sel = pd.to_numeric(df["selected_by_topm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                selection_coverage = float(sel.sum() / n)
            else:
                selection_coverage = float("nan")

            accept_rate = float(acc.sum() / n)

            # reason breakdown
            if "reason" in df.columns:
                reasons = df["reason"].astype(str)
            else:
                # Fallback: accepted vs not accepted
                reasons = pd.Series(np.where(acc == 1, "accepted", "rejected"))

            counts = reasons.value_counts(dropna=False).to_dict()

            row = {
                "variant": v.name,
                "per_subject_csv": str(csv_path),
                "n_subjects": n,
                "accept_rate": accept_rate,
                "selection_coverage": selection_coverage,
            }
            # Keep a small, stable set of common buckets.
            for key in [
                "accepted",
                "not_in_topm",
                "no_guard_ok",
                "no_pred_ok",
                "feasible_empty",
                "violated",
                "rejected",
            ]:
                row[f"count_{key}"] = int(counts.get(key, 0))

            # Capture any other reasons (rare) into count_other
            known = {"accepted","not_in_topm","no_guard_ok","no_pred_ok","feasible_empty","violated","rejected"}
            other = sum(int(vv) for kk, vv in counts.items() if kk not in known)
            row["count_other"] = int(other)

            rows.append(row)

        df_sum = pd.DataFrame(rows)

    if args.variant_regex:
        pat = re.compile(str(args.variant_regex))
        df_sum = df_sum.loc[df_sum["variant"].astype(str).apply(lambda v: bool(pat.search(v)))].reset_index(drop=True)
        if df_sum.empty:
            raise RuntimeError(f"--variant-regex matched 0 rows: {args.variant_regex!r}")

    df_sum = df_sum.sort_values(args.x_axis, na_position="last").reset_index(drop=True)
    df_sum.to_csv(out_dir / "stage_breakdown.csv", index=False)

    # Plot stacked bars by variant order; x-axis label shows accept_rate/coverage.
    labels = [_short_variant(v) for v in df_sum["variant"].tolist()]
    xs = np.arange(len(labels))

    parts_all = [
        ("accepted", "#55A868"),
        ("not_in_topm", "#8172B2"),
        ("no_guard_ok", "#CCB974"),
        ("no_pred_ok", "#64B5CD"),
        ("violated", "#C44E52"),
        ("feasible_empty", "#8C8C8C"),
        ("other", "#999999"),
    ]

    # Drop categories that are identically zero across all shown variants (keeps legend clean).
    parts = []
    for name, color in parts_all:
        col = "count_other" if name == "other" else f"count_{name}"
        if col not in df_sum.columns:
            continue
        if name != "accepted" and float(pd.to_numeric(df_sum[col], errors="coerce").fillna(0.0).sum()) <= 0.0:
            continue
        parts.append((name, color))

    fig_w = max(7.0, 0.95 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    bottom = np.zeros(len(labels), dtype=float)
    n = df_sum["n_subjects"].to_numpy(dtype=float)

    for name, color in parts:
        col = "count_other" if name == "other" else f"count_{name}"
        vals = df_sum[col].to_numpy(dtype=float)
        frac = np.where(n > 0, vals / n, 0.0)
        ax.bar(xs, frac, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, label=name)
        bottom += frac

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction of subjects")

    xlab = []
    metric = pd.to_numeric(df_sum[args.x_axis], errors="coerce").to_numpy(dtype=float)
    metric_name = _pretty_x_axis(args.x_axis)
    for v, m in zip(labels, metric):
        if np.isfinite(m):
            xlab.append(f"{v}\n{metric_name}={m:.3f}")
        else:
            xlab.append(v)

    ax.set_xticks(xs)
    rotate = 45 if len(xlab) > 4 else 0
    ax.set_xticklabels(xlab, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    title = str(args.title).strip()
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=3 if len(parts) > 4 else 2)

    fig.tight_layout()
    fig.savefig(out_dir / "stage_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
