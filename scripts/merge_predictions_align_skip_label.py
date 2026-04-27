#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_csv_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _infer_date_prefix(path: Path) -> str:
    name = path.name
    parts = name.split("_", 1)
    if parts and len(parts[0]) == 8 and parts[0].isdigit():
        return parts[0]
    return datetime.now().strftime("%Y%m%d")


def _align_keep_indices(*, y_ref: list[str], y_cand: list[str], skip_label: str, subject: int) -> list[int]:
    """Return indices into y_cand to keep such that y_cand[keep] == y_ref.

    Assumes y_cand is y_ref with extra `skip_label` tokens inserted (order preserved).
    """
    keep: list[int] = []
    j = 0
    for lab in y_ref:
        while j < len(y_cand) and y_cand[j] != lab:
            if y_cand[j] == skip_label:
                j += 1
                continue
            raise RuntimeError(
                f"Subject {subject}: cannot align sequences at ref_lab={lab!r}, cand_lab={y_cand[j]!r} (j={j})."
            )
        if j >= len(y_cand):
            raise RuntimeError(f"Subject {subject}: candidate sequence ended early while matching {lab!r}.")
        keep.append(j)
        j += 1

    tail = y_cand[j:]
    if any(t != skip_label for t in tail):
        raise RuntimeError(f"Subject {subject}: non-{skip_label} labels remain in candidate tail: {sorted(set(tail))}")
    return keep


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Merge two predictions_all_methods.csv files by aligning per-subject trial keys.\n\n"
            "This tool supports a common mismatch pattern: the candidate run includes additional trials with a single "
            "`skip_label` (e.g. extra 'rest' trials), while the reference/anchor run does not.\n"
            "We align each subject by greedily skipping `skip_label` entries in the candidate y_true sequence until "
            "it matches the reference y_true sequence, then drop the skipped rows for all candidate methods and "
            "reassign (trial,y_true) to the reference keys.\n\n"
            "Outputs a run-style folder containing per-method *_predictions.csv, *_predictions_all_methods.csv, "
            "*_method_comparison.csv, and *_MERGED_FROM.txt."
        )
    )
    p.add_argument("--base-preds", type=Path, required=True, help="Reference predictions_all_methods.csv (anchor keys).")
    p.add_argument("--cand-preds", type=Path, required=True, help="Candidate predictions_all_methods.csv (extra trials).")
    p.add_argument("--anchor-method", type=str, required=True, help="Anchor method name in --base-preds.")
    p.add_argument(
        "--cand-methods",
        type=str,
        required=True,
        help="Comma-separated candidate methods to import from --cand-preds (must share the same keys).",
    )
    p.add_argument(
        "--skip-label",
        type=str,
        default="rest",
        help="Label allowed to be skipped in candidate y_true sequence during alignment (default: rest).",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output run directory.")
    p.add_argument("--date-prefix", type=str, default="", help="Output filename prefix (default: infer from base).")
    p.add_argument("--keep-base-anchor-only", action="store_true", help="Only keep the anchor method from base.")
    args = p.parse_args()

    cand_methods = _parse_csv_list(args.cand_methods)
    if not cand_methods:
        raise SystemExit("--cand-methods is empty.")

    df_base = pd.read_csv(Path(args.base_preds))
    df_cand = pd.read_csv(Path(args.cand_preds))

    required = {"method", "subject", "trial", "y_true", "y_pred"}
    for name, df in [("base", df_base), ("cand", df_cand)]:
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{name} preds missing columns {sorted(missing)}.")

    proba_cols = [c for c in df_base.columns if str(c).startswith("proba_")]
    if not proba_cols:
        raise RuntimeError("No proba_* columns found in base preds.")
    proba_cols_c = [c for c in df_cand.columns if str(c).startswith("proba_")]
    if set(proba_cols_c) != set(proba_cols):
        raise RuntimeError("proba_* columns mismatch between base and candidate preds.")

    keep_cols = ["method", "subject", "trial", "y_true", "y_pred"] + proba_cols
    df_base = df_base[keep_cols].copy()
    df_cand = df_cand[keep_cols].copy()

    anchor_method = str(args.anchor_method).strip()
    df_ref = df_base[df_base["method"].astype(str) == anchor_method].copy()
    if df_ref.empty:
        raise RuntimeError(f"Anchor method {anchor_method!r} not found in base preds.")
    df_ref = df_ref.sort_values(["subject", "trial"]).reset_index(drop=True)

    # Base consistency: all methods in base share anchor keys.
    key_ref = df_ref[["subject", "trial", "y_true"]].copy()
    for m in sorted(df_base["method"].astype(str).unique().tolist()):
        dm = df_base[df_base["method"].astype(str) == m].sort_values(["subject", "trial"]).reset_index(drop=True)
        if not dm[["subject", "trial", "y_true"]].equals(key_ref):
            raise RuntimeError(f"Base run key mismatch between anchor and method={m}.")

    # Candidate consistency: all candidate methods share keys.
    key_cand0 = None
    for m in cand_methods:
        dm = df_cand[df_cand["method"].astype(str) == m].copy()
        if dm.empty:
            raise RuntimeError(f"Candidate method {m!r} missing in candidate preds.")
        dm = dm.sort_values(["subject", "trial"]).reset_index(drop=True)
        k = dm[["subject", "trial", "y_true"]].copy()
        if key_cand0 is None:
            key_cand0 = k
        elif not k.equals(key_cand0):
            raise RuntimeError(f"Candidate run key mismatch between methods (first mismatch at method={m}).")

    subjects = sorted(int(s) for s in df_ref["subject"].unique().tolist())
    skip_label = str(args.skip_label).strip()
    if not skip_label:
        raise SystemExit("--skip-label must be non-empty.")

    aligned_parts: list[pd.DataFrame] = []
    skip_rows: list[dict[str, int]] = []

    for subj in subjects:
        ref_s = df_ref[df_ref["subject"].astype(int) == int(subj)].sort_values("trial").reset_index(drop=True)
        y_ref = ref_s["y_true"].astype(str).tolist()

        cand_s0 = df_cand[
            (df_cand["method"].astype(str) == cand_methods[0]) & (df_cand["subject"].astype(int) == int(subj))
        ].sort_values("trial")
        cand_s0 = cand_s0.reset_index(drop=True)
        y_cand = cand_s0["y_true"].astype(str).tolist()

        keep_idx = _align_keep_indices(y_ref=y_ref, y_cand=y_cand, skip_label=skip_label, subject=int(subj))
        n_skip = int(len(y_cand) - len(y_ref))
        skip_rows.append({"subject": int(subj), "n_ref": len(y_ref), "n_cand": len(y_cand), "n_skip": n_skip})

        new_trials = ref_s["trial"].to_numpy(int)
        new_y_true = ref_s["y_true"].to_numpy(object)

        for m in cand_methods:
            dm = df_cand[
                (df_cand["method"].astype(str) == m) & (df_cand["subject"].astype(int) == int(subj))
            ].sort_values("trial")
            dm = dm.reset_index(drop=True)
            if dm.shape[0] != len(y_cand):
                raise RuntimeError(f"Subject {subj}: method={m} row count mismatch vs cand key.")
            dm_keep = dm.iloc[keep_idx].copy()
            dm_keep["trial"] = new_trials
            dm_keep["y_true"] = new_y_true
            aligned_parts.append(dm_keep)

    df_anchor = df_ref.copy()
    if not bool(args.keep_base_anchor_only):
        df_anchor = df_base.copy()
    merged = pd.concat([df_anchor] + aligned_parts, axis=0, ignore_index=True)

    # Validate final key equality across all methods.
    key_anchor = df_anchor[df_anchor["method"].astype(str) == anchor_method]
    key_anchor = key_anchor[["subject", "trial", "y_true"]].sort_values(["subject", "trial"]).reset_index(drop=True)
    for m in sorted(merged["method"].astype(str).unique().tolist()):
        dm = merged[merged["method"].astype(str) == m].sort_values(["subject", "trial"]).reset_index(drop=True)
        if not dm[["subject", "trial", "y_true"]].equals(key_anchor):
            raise RuntimeError(f"Merged key mismatch: method={m}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = str(args.date_prefix).strip() or _infer_date_prefix(Path(args.base_preds))

    for m in sorted(merged["method"].astype(str).unique().tolist()):
        dm = merged[merged["method"].astype(str) == m].sort_values(["subject", "trial"]).reset_index(drop=True)
        (out_dir / f"{date_prefix}_{m}_predictions.csv").write_text(
            dm.drop(columns=["method"]).to_csv(index=False), encoding="utf-8"
        )

    merged.sort_values(["method", "subject", "trial"]).to_csv(out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False)

    merged2 = merged.copy()
    merged2["correct"] = (merged2["y_true"].astype(str) == merged2["y_pred"].astype(str)).astype(float)
    acc = merged2.groupby(["method", "subject"], sort=True)["correct"].mean().reset_index()
    rows = []
    for m in sorted(acc["method"].astype(str).unique().tolist()):
        g = acc[acc["method"].astype(str) == m]
        rows.append(
            {
                "method": str(m),
                "n_subjects": int(g.shape[0]),
                "mean_accuracy": float(g["correct"].mean()),
                "worst_accuracy": float(g["correct"].min()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{date_prefix}_method_comparison.csv", index=False)

    pd.DataFrame(skip_rows).to_csv(out_dir / f"{date_prefix}_ALIGN_SKIP_SUMMARY.csv", index=False)

    prov_lines = [
        "Merged from:",
        f"- base: {Path(args.base_preds)}",
        f"- cand: {Path(args.cand_preds)}",
        "",
        "Alignment rule:",
        f"- Per subject, align candidate y_true sequence to anchor's by greedily skipping extra '{skip_label}' labels in candidate.",
        "- Drop skipped rows from all candidate methods and reassign (trial,y_true) to match the anchor keys.",
        "",
        f"Anchor method: {anchor_method}",
        f"Candidate methods: {', '.join(cand_methods)}",
    ]
    (out_dir / f"{date_prefix}_MERGED_FROM.txt").write_text("\n".join(prov_lines) + "\n", encoding="utf-8")

    skip_df = pd.DataFrame(skip_rows)
    print("[OK] wrote", out_dir)
    print("methods:", sorted(merged["method"].astype(str).unique().tolist()))
    print("skip n_skip unique:", sorted(skip_df["n_skip"].unique().tolist()))


if __name__ == "__main__":
    main()

