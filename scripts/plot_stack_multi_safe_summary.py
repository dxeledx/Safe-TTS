from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_method_table(results_txt: Path, *, method: str) -> pd.DataFrame:
    txt = results_txt.read_text(encoding="utf-8")
    pat = re.compile(rf"=== Method: {re.escape(method)} ===\n(.*?)\n\nSummary", re.S)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Method block not found: {method} in {results_txt}")
    # Keep the leading padding on the header line; pd.read_fwf relies on fixed-width alignment.
    block = m.group(1).rstrip()
    # NOTE: use fixed-width parsing. The printed table contains optional string columns
    # (e.g., *_block_reason) that may be blank for some rows; whitespace-splitting would
    # shift columns and silently corrupt downstream plots.
    return pd.read_fwf(io.StringIO(block))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_delta(
    df_method: pd.DataFrame,
    df_base: pd.DataFrame,
    *,
    out: Path,
    title: str,
) -> None:
    base = df_base.set_index("subject")["accuracy"].astype(float)
    method = df_method.set_index("subject")["accuracy"].astype(float)
    common = method.index.intersection(base.index)
    delta = (method.loc[common] - base.loc[common]).astype(float)

    accept = None
    if "stack_multi_accept" in df_method.columns:
        accept = df_method.set_index("subject").loc[common]["stack_multi_accept"].astype(int).to_numpy()

    x = np.arange(len(common))
    colors = "#4C72B0" if accept is None else np.where(accept > 0, "#DD8452", "#9A9A9A")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.bar(x, delta.to_numpy(), color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in common])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Δacc vs EA")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(
    df_method: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out: Path,
    title: str,
    xlabel: str,
) -> None:
    x = df_method[x_col].astype(float).to_numpy()
    y = df_method[y_col].astype(float).to_numpy()
    subj = df_method["subject"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.4)
    ax.scatter(x, y, s=40, alpha=0.85)
    for xi, yi, si in zip(x, y, subj):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(float(xi), float(yi), str(int(si)), fontsize=8, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("True Δacc vs EA")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_family(
    df_method: pd.DataFrame,
    *,
    out: Path,
    title: str,
) -> None:
    # Use a stable vertical ordering for families.
    fam_order = ["ea", "rpa", "tsa", "chan", "fbcsp", "other"]
    fam_y = {f: i for i, f in enumerate(fam_order)}

    subjects = df_method["subject"].astype(int).to_numpy()
    fam_pre = df_method.get("stack_multi_pre_family", pd.Series(["ea"] * len(df_method))).astype(str).to_numpy()
    fam_final = df_method.get("stack_multi_family", pd.Series(["ea"] * len(df_method))).astype(str).to_numpy()
    accept = df_method.get("stack_multi_accept", pd.Series([0] * len(df_method))).astype(int).to_numpy()
    fbcsp_blocked = df_method.get("stack_multi_fbcsp_blocked", pd.Series([0] * len(df_method))).astype(int).to_numpy()
    tsa_blocked = df_method.get("stack_multi_tsa_blocked", pd.Series([0] * len(df_method))).astype(int).to_numpy()

    def _y(f: str) -> float:
        f = str(f)
        if f in fam_y:
            return float(fam_y[f])
        return float(fam_y["other"])

    x = np.arange(len(subjects))
    y_pre = np.array([_y(f) for f in fam_pre], dtype=float)
    y_fin = np.array([_y(f) for f in fam_final], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, y_pre, marker="x", s=70, color="#4C72B0", label="pre (before gates)")
    colors_fin = np.where(accept > 0, "#DD8452", "#9A9A9A")
    ax.scatter(x, y_fin, marker="o", s=60, color=colors_fin, label="final (selected)")

    # Mark explicitly blocked high-risk families.
    for i in range(len(subjects)):
        if int(fbcsp_blocked[i]) > 0:
            ax.scatter(x[i], fam_y["fbcsp"], marker="X", s=90, color="#C44E52", label="_fbcsp_blocked")
        if int(tsa_blocked[i]) > 0:
            ax.scatter(x[i], fam_y["tsa"], marker="X", s=90, color="#8172B3", label="_tsa_blocked")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in subjects])
    ax.set_yticks([fam_y[f] for f in fam_order])
    ax.set_yticklabels(fam_order)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Family")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    # De-duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen or l.startswith("_"):
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot EA-STACK-MULTI-SAFE per-subject diagnostics from *_results.txt.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--base-method", type=str, default="ea-csp-lda")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_txts = sorted(run_dir.glob("*_results.txt"))
    if not results_txts:
        raise RuntimeError(f"No *_results.txt found in {run_dir}")
    results_txt = results_txts[0]

    df_base = _extract_method_table(results_txt, method=str(args.base_method))
    df_method = _extract_method_table(results_txt, method=str(args.method))

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    _plot_delta(
        df_method,
        df_base,
        out=out_dir / f"{args.prefix}_delta.png",
        title=f"{args.prefix}: per-subject Δacc vs EA",
    )

    if "stack_multi_ridge_pred_improve" in df_method.columns and "stack_multi_improve" in df_method.columns:
        _plot_scatter(
            df_method,
            x_col="stack_multi_ridge_pred_improve",
            y_col="stack_multi_improve",
            out=out_dir / f"{args.prefix}_ridge_vs_true.png",
            title=f"{args.prefix}: ridge predicted vs true Δacc",
            xlabel="Ridge predicted improvement",
        )

    if "stack_multi_guard_pos" in df_method.columns and "stack_multi_improve" in df_method.columns:
        _plot_scatter(
            df_method,
            x_col="stack_multi_guard_pos",
            y_col="stack_multi_improve",
            out=out_dir / f"{args.prefix}_guard_vs_true.png",
            title=f"{args.prefix}: guard p(pos) vs true Δacc",
            xlabel="Guard p(pos)",
        )

    if "stack_multi_family" in df_method.columns:
        _plot_family(
            df_method,
            out=out_dir / f"{args.prefix}_family.png",
            title=f"{args.prefix}: pre vs final family (blocked arms marked)",
        )


if __name__ == "__main__":
    main()
