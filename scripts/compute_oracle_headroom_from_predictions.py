from __future__ import annotations

"""Compute oracle-with-fallback headroom from a predictions_all_methods.csv file.

This is a lightweight diagnostic used to check whether an action set has enough
"room to select" (i.e., candidates are sometimes better than the anchor).

Outputs:
  - prints a one-line summary to stdout
  - optionally writes a per-subject CSV (acc_anchor, acc_best, delta, headroom)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_csv_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute oracle headroom from predictions_all_methods.csv")
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--anchor-method", type=str, required=True)
    ap.add_argument(
        "--candidate-methods",
        type=str,
        default="ALL",
        help="Comma-separated candidate methods, or ALL to use all methods except anchor.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Optional per-subject CSV output path.")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise SystemExit(f"Missing --preds: {preds_path}")

    df = pd.read_csv(preds_path)
    required = {"method", "subject", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{preds_path}: missing columns {sorted(required - set(df.columns))}")

    anchor = str(args.anchor_method).strip()
    all_methods = sorted({str(m) for m in df["method"].astype(str).unique().tolist()})
    if anchor not in all_methods:
        raise SystemExit(f"Anchor method {anchor!r} not in preds. Methods: {all_methods}")

    cand_raw = str(args.candidate_methods).strip()
    if cand_raw.upper() == "ALL":
        cands = [m for m in all_methods if m != anchor]
    else:
        cands = [m for m in _parse_csv_list(cand_raw) if m != anchor]
    if not cands:
        raise SystemExit("No candidates selected.")
    missing = [m for m in cands if m not in all_methods]
    if missing:
        raise SystemExit(f"Candidate methods missing in preds: {missing}. Available: {all_methods}")

    df = df.copy()
    df["correct"] = (df["y_pred"].astype(str) == df["y_true"].astype(str)).astype(int)

    acc = df.groupby(["subject", "method"], as_index=True)["correct"].mean().unstack("method")
    acc = acc.sort_index()

    if anchor not in acc.columns:
        raise SystemExit(f"Anchor {anchor!r} missing in per-subject accuracy table.")
    for m in cands:
        if m not in acc.columns:
            raise SystemExit(f"Candidate {m!r} missing in per-subject accuracy table.")

    acc_anchor = acc[anchor].astype(float)
    acc_cands = acc[cands].astype(float)

    acc_best = acc_cands.max(axis=1)
    best_method = acc_cands.idxmax(axis=1).astype(str)
    delta_best = (acc_best - acc_anchor).astype(float)
    headroom = delta_best.clip(lower=0.0)
    oracle_fallback = (acc_anchor + headroom).astype(float)

    n = int(len(acc_anchor))
    headroom_mean = float(headroom.mean()) if n > 0 else float("nan")
    coverage = float(np.mean(delta_best > 1e-12)) if n > 0 else float("nan")
    mean_anchor = float(acc_anchor.mean()) if n > 0 else float("nan")
    mean_oracle = float(oracle_fallback.mean()) if n > 0 else float("nan")

    print(
        "oracle_with_fallback:"
        f" n={n}"
        f" mean_anchor={mean_anchor:.4f}"
        f" mean_oracle={mean_oracle:.4f}"
        f" mean_headroom={headroom_mean:.4f}"
        f" pr_headroom_gt0={coverage:.3f}"
    )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(
            {
                "subject": acc_anchor.index.astype(int),
                "acc_anchor": acc_anchor.to_numpy(dtype=float),
                "acc_best": acc_best.to_numpy(dtype=float),
                "best_method": best_method.to_numpy(dtype=object),
                "delta_best": delta_best.to_numpy(dtype=float),
                "headroom": headroom.to_numpy(dtype=float),
                "oracle_fallback_acc": oracle_fallback.to_numpy(dtype=float),
            }
        )
        out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

