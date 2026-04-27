#!/usr/bin/env python3
from __future__ import annotations

"""
Build a compact "failure boundary" table across datasets/settings.

This script is intentionally lightweight: it consumes SAFE-TTA run folders that contain:
- *_per_subject_selection.csv
- *_method_comparison.csv

and outputs a single CSV table suitable for Discussion/Appendix.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        if ":" not in it:
            raise ValueError(f"Invalid --input {it!r}. Expected LABEL:RUN_DIR")
        label, run_dir = it.split(":", 1)
        label = label.strip()
        run_dir_p = Path(run_dir.strip())
        if not label:
            raise ValueError(f"Invalid --input {it!r}: empty label")
        if not run_dir_p.exists():
            raise ValueError(f"--input path missing: {run_dir_p}")
        out.append((label, run_dir_p))
    if not out:
        raise ValueError("No valid --input parsed.")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a failure-boundary table from SAFE-TTA run folders.")
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable. Format: LABEL:RUN_DIR (RUN_DIR contains *_per_subject_selection.csv and *_method_comparison.csv).",
    )
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _summarize_one(label: str, run_dir: Path) -> dict[str, object]:
    per_subject_path = _find_single(run_dir, "*_per_subject_selection.csv")
    comp_path = _find_single(run_dir, "*_method_comparison.csv")

    per = pd.read_csv(per_subject_path)
    comp = pd.read_csv(comp_path)
    if comp.shape[0] < 1:
        raise RuntimeError(f"{comp_path} is empty")
    row = comp.iloc[0].to_dict()

    if "acc_anchor" not in per.columns:
        raise RuntimeError(f"{per_subject_path}: missing column acc_anchor")
    if "oracle_acc" not in per.columns:
        raise RuntimeError(f"{per_subject_path}: missing column oracle_acc")

    # Prefer final metrics (after review); fall back to selected metrics.
    if "acc_final" in per.columns:
        acc_safe = pd.to_numeric(per["acc_final"], errors="coerce").to_numpy(dtype=float)
        delta_safe = pd.to_numeric(per.get("delta_final_vs_anchor", np.nan), errors="coerce").to_numpy(dtype=float)
    else:
        acc_safe = pd.to_numeric(per.get("acc_selected", np.nan), errors="coerce").to_numpy(dtype=float)
        delta_safe = pd.to_numeric(per.get("delta_selected_vs_anchor", np.nan), errors="coerce").to_numpy(dtype=float)

    acc_anchor = pd.to_numeric(per["acc_anchor"], errors="coerce").to_numpy(dtype=float)
    acc_oracle = pd.to_numeric(per["oracle_acc"], errors="coerce").to_numpy(dtype=float)

    neg_transfer = float(np.nanmean((delta_safe < 0.0).astype(float)))
    out: dict[str, object] = {
        "label": str(label),
        "run_dir": str(run_dir),
        "n_subjects": int(per.shape[0]),
        "anchor_mean_acc": float(np.nanmean(acc_anchor)),
        "oracle_mean_acc": float(np.nanmean(acc_oracle)),
        "safe_tta_mean_acc": float(np.nanmean(acc_safe)),
        "neg_transfer_rate": float(neg_transfer),
        "review_infeasible": int(row.get("review_infeasible", 0)),
        "chosen_review_budget_frac": float(row.get("chosen_review_budget_frac", float("nan"))),
        "cond_ucb_after_review": float(row.get("cond_ucb_after_review", float("nan"))),
        "cond_rate_after_review": float(row.get("cond_rate_after_review", float("nan"))),
        "accept_rate": float(row.get("accept_rate", float("nan"))),
    }
    return out


def main() -> int:
    args = parse_args()
    inputs = _parse_inputs(list(args.input))
    rows = [_summarize_one(label, run_dir) for label, run_dir in inputs]
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[done] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

