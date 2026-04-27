from __future__ import annotations

"""Verify that paper/main figure directories are reproducible from on-disk artifacts.

Checks:
  1) summary.csv exists and has required columns
  2) each per_subject_csv referenced by summary.csv exists
  3) per_subject_csv contains required columns for downstream analysis

Exit code:
  0 when all checks pass
  1 otherwise
"""

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(p: str) -> Path:
    p = str(p)
    path = Path(p)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _check_summary(summary_csv: Path, *, report: list[str]) -> None:
    df = pd.read_csv(summary_csv)
    required = {"variant", "per_subject_csv"}
    if not required.issubset(df.columns):
        report.append(f"[BAD] {summary_csv}: missing columns {sorted(required - set(df.columns))}")
        return

    for i, row in df.iterrows():
        p = row.get("per_subject_csv", "")
        if not isinstance(p, str) or not p.strip():
            report.append(f"[BAD] {summary_csv}: empty per_subject_csv at row {i}")
            continue
        csv_path = _resolve_path(p)
        if not csv_path.exists():
            report.append(f"[MISSING] {summary_csv}: per_subject_csv not found: {csv_path}")
            continue
        try:
            df_ps = pd.read_csv(csv_path, nrows=5)
        except Exception as e:
            report.append(f"[BAD] {summary_csv}: failed to read {csv_path}: {type(e).__name__}: {e}")
            continue
        needed_cols = {"subject", "accept", "acc_anchor", "acc_selected"}
        if not needed_cols.issubset(df_ps.columns):
            report.append(
                f"[BAD] {summary_csv}: {csv_path} missing cols {sorted(needed_cols - set(df_ps.columns))}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify that docs/experiments/figures/**/summary.csv sources exist.")
    ap.add_argument(
        "--glob",
        type=str,
        default="docs/experiments/figures/**/summary.csv",
        help="Glob to locate summary.csv files (relative to repo root).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output report path (relative to repo root).",
    )
    args = ap.parse_args()

    glob_pat = str(args.glob)
    paths = sorted((ROOT / ".").glob(glob_pat))
    if not paths:
        print(f"No summary.csv matched glob: {glob_pat}", file=sys.stderr)
        raise SystemExit(1)

    report: list[str] = []
    for p in paths:
        _check_summary(Path(p), report=report)

    if args.out is not None:
        out_path = _resolve_path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(report) + ("\n" if report else ""), encoding="utf-8")

    if report:
        print("\n".join(report), file=sys.stderr)
        print(f"FAILED: {len(report)} issue(s) across {len(paths)} summary.csv files", file=sys.stderr)
        raise SystemExit(1)

    print(f"OK: {len(paths)} summary.csv files verified.")


if __name__ == "__main__":
    main()

