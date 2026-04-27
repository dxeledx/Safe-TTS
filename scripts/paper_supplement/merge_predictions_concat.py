#!/usr/bin/env python3
from __future__ import annotations

"""
Merge two SAFE_TTA-style predictions_all_methods.csv files.

Use-case
--------
- base: DeepTransferEEG TTA suite predictions (multiple methods; shared (subject,trial,y_true) keys)
- extra: SAFE-TTA output predictions (typically a single method like safe-tta-cand2e)

The script **strictly validates** that the extra file uses the exact same per-subject
(trial, y_true) key as the base anchor method, then concatenates methods.
"""

import argparse
from pathlib import Path

import pandas as pd


def _find_proba_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("proba_")]
    if not cols:
        raise RuntimeError("No proba_* columns found.")
    return cols


def _load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns: {missing}")
    proba_cols = _find_proba_cols(df)
    keep = ["method", "subject", "trial", "y_true", "y_pred"] + proba_cols
    return df[keep].copy()


def _key(df: pd.DataFrame) -> pd.DataFrame:
    return df[["subject", "trial", "y_true"]].sort_values(["subject", "trial"]).reset_index(drop=True)


def _assert_all_methods_share_key(df: pd.DataFrame, *, ref_key: pd.DataFrame, path: Path) -> None:
    for m in sorted(df["method"].astype(str).unique().tolist()):
        dm = df[df["method"].astype(str) == str(m)].copy()
        if dm.empty:
            continue
        k = _key(dm)
        if not k.equals(ref_key):
            raise RuntimeError(f"{path}: key mismatch between ref and method={m!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge two predictions_all_methods.csv (strict key validation).")
    p.add_argument("--base-preds", type=Path, required=True)
    p.add_argument("--extra-preds", type=Path, required=True)
    p.add_argument("--base-anchor-method", type=str, default="eegnet_ea")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--allow-overwrite-methods",
        action="store_true",
        help="Allow extra methods to overwrite base methods if names collide (default: error).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_preds)
    extra_path = Path(args.extra_preds)
    out_path = Path(args.out)

    base = _load_preds(base_path)
    extra = _load_preds(extra_path)

    anchor = str(args.base_anchor_method).strip()
    base_anchor = base[base["method"].astype(str) == anchor].copy()
    if base_anchor.empty:
        raise RuntimeError(f"--base-anchor-method={anchor!r} not found in {base_path}")
    ref_key = _key(base_anchor)

    _assert_all_methods_share_key(base, ref_key=ref_key, path=base_path)
    _assert_all_methods_share_key(extra, ref_key=ref_key, path=extra_path)

    base_methods = set(base["method"].astype(str).unique().tolist())
    extra_methods = set(extra["method"].astype(str).unique().tolist())
    overlap = sorted(base_methods & extra_methods)
    if overlap and not bool(args.allow_overwrite_methods):
        raise RuntimeError(f"Method name collision between base and extra: {overlap}. Use --allow-overwrite-methods.")

    if overlap:
        base = base[~base["method"].astype(str).isin(overlap)].copy()

    merged = pd.concat([base, extra], axis=0, ignore_index=True)
    merged = merged.sort_values(["method", "subject", "trial"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[done] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

