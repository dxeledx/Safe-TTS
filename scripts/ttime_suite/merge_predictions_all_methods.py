from __future__ import annotations

"""
Merge per-method, per-subject predictions CSVs into predictions_all_methods.csv.

This is useful when running sharded suite jobs with `run_suite_loso.py --skip-merge`.

Expected layout:
  <out_dir>/predictions/method=<method>/subject=<subject>.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge DeepTransferEEG suite predictions into predictions_all_methods.csv")
    p.add_argument("--out-dir", type=Path, required=True, help="Suite output dir containing predictions/ subdir")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <out-dir>/predictions_all_methods.csv)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    pred_root = out_dir / "predictions"
    if not pred_root.exists():
        raise RuntimeError(f"Missing predictions dir: {pred_root}")

    dfs: list[pd.DataFrame] = []
    for csv_path in sorted(pred_root.rglob("subject=*.csv")):
        dfs.append(pd.read_csv(csv_path))
    if not dfs:
        raise RuntimeError(f"No per-subject predictions found under: {pred_root}")

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    # Keep it stable for downstream diffs/debug.
    sort_cols = [c for c in ["subject", "method", "trial"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out_csv = Path(args.out_csv) if args.out_csv is not None else (out_dir / "predictions_all_methods.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[done] wrote: {out_csv}  rows={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

