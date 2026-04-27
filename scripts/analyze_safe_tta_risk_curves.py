from __future__ import annotations

"""
Analyze offline SAFE-TTA selection results and plot risk/utility curves.

Usage example (PhysioNetMI top-m sweep):
  python3 scripts/analyze_safe_tta_risk_curves.py \\
    --root outputs/20260129/4class/review_physio_s1-109_alpha0.20_topm_sweep \\
    --out-dir docs/experiments/figures/20260129_physionetmi_crc_cp_alpha0.20_topm_sweep_riskcurves_v1 \\
    --title "PhysioNetMI alpha=0.20 delta=0.05 topm sweep"
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def _parse_eps_list(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",")]
    out: list[float] = []
    for p in parts:
        if not p:
            continue
        out.append(float(p))
    if not out:
        raise RuntimeError("--eps parsed to empty list")
    return out


def _col(df: pd.DataFrame, name: str) -> str | None:
    want = str(name).lower()
    for c in df.columns:
        if str(c).lower() == want:
            return str(c)
    return None


def _require_col(df: pd.DataFrame, name: str, *, context: str) -> str:
    c = _col(df, name)
    if c is None:
        raise RuntimeError(f"{context}: missing required column {name!r}. Available: {list(df.columns)}")
    return c


def _find_newest_per_subject_csv(variant_dir: Path) -> Path:
    candidates = sorted(variant_dir.rglob("*per_subject_selection.csv"))
    if not candidates:
        raise RuntimeError(f"No '*per_subject_selection.csv' found under: {variant_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_newest_method_comparison_csv(variant_dir: Path) -> Path | None:
    candidates = sorted(variant_dir.rglob("*_method_comparison.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _eps_suffix(eps: float) -> str:
    return f"{float(eps):g}"


def _cp_two_sided_ci(*, k: int, n: int, delta: float) -> tuple[float, float]:
    """Two-sided Clopper–Pearson CI at level 1-delta for a binomial proportion."""
    k = int(k)
    n = int(n)
    delta = float(delta)
    if n <= 0:
        return float("nan"), float("nan")
    if not (0.0 < delta < 1.0):
        raise ValueError("--delta must be in (0,1).")
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}.")

    # Edge cases: Beta quantiles with 0 shape parameters are undefined.
    if k == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(delta / 2.0, k, n - k + 1))
    if k == n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1.0 - delta / 2.0, k + 1, n - k))
    return float(lo), float(hi)


def _compute_variant_metrics(
    *,
    variant: str,
    csv_path: Path,
    method_comparison_path: Path | None,
    delta: float,
    eps_list: list[float],
) -> dict[str, float | str]:
    df = pd.read_csv(csv_path)
    context = str(csv_path)

    _require_col(df, "subject", context=context)
    acc_anchor_col = _require_col(df, "acc_anchor", context=context)
    acc_selected_col = _require_col(df, "acc_selected", context=context)

    accept_col = _col(df, "accept")
    if accept_col is None:
        accept_col = _col(df, "accepted")
    if accept_col is None:
        raise RuntimeError(f"{context}: missing acceptance column 'accept' or 'accepted'")

    selected_by_topm_col = _col(df, "selected_by_topm")

    n_subjects = int(len(df))
    if n_subjects <= 0:
        raise RuntimeError(f"{context}: empty CSV")

    acc_anchor = pd.to_numeric(df[acc_anchor_col], errors="coerce").to_numpy(dtype=float)
    acc_selected = pd.to_numeric(df[acc_selected_col], errors="coerce").to_numpy(dtype=float)

    accepted_raw = pd.to_numeric(df[accept_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    accepted = (accepted_raw > 0.5).astype(int)

    final_delta = np.where(accepted == 1, acc_selected - acc_anchor, 0.0).astype(float)
    if not np.isfinite(final_delta).all():
        bad = int(np.sum(~np.isfinite(final_delta)))
        raise RuntimeError(
            f"{context}: computed final_delta has {bad} non-finite entries (check acc/accept columns)"
        )

    # Optional post-review deployed delta (for review-proxy experiments).
    delta_final_col = _col(df, "delta_final_vs_anchor")
    acc_final_col = _col(df, "acc_final")
    if delta_final_col is not None:
        deployed_delta = pd.to_numeric(df[delta_final_col], errors="coerce").to_numpy(dtype=float)
    elif acc_final_col is not None:
        acc_final = pd.to_numeric(df[acc_final_col], errors="coerce").to_numpy(dtype=float)
        deployed_delta = acc_final - acc_anchor
    else:
        deployed_delta = final_delta.copy()
    if not np.isfinite(deployed_delta).all():
        bad = int(np.sum(~np.isfinite(deployed_delta)))
        raise RuntimeError(f"{context}: deployed_delta has {bad} non-finite entries.")

    accepted_n = int(accepted.sum())
    accept_rate = float(accepted_n / n_subjects)

    selected_n: float
    selection_coverage: float
    if selected_by_topm_col is None:
        selected_n = float("nan")
        selection_coverage = float("nan")
    else:
        selected_by_topm = pd.to_numeric(df[selected_by_topm_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        selected_n = float(selected_by_topm.sum())
        selection_coverage = float(selected_n / n_subjects)

    mean_delta_all = float(final_delta.mean())

    if accepted_n > 0:
        delta_acc = final_delta[accepted == 1]
        mean_delta_accepted = float(delta_acc.mean())
        q20_delta_accepted = float(np.quantile(delta_acc, 0.20))
        worst_delta_accepted = float(delta_acc.min())
    else:
        mean_delta_accepted = float("nan")
        q20_delta_accepted = float("nan")
        worst_delta_accepted = float("nan")

    row: dict[str, float | str] = {
        "variant": str(variant),
        "per_subject_csv": str(csv_path),
        "n_subjects": float(n_subjects),
        "selected_n": float(selected_n),
        "selection_coverage": float(selection_coverage),
        "accepted_n": float(accepted_n),
        "accept_rate": float(accept_rate),
        "mean_delta_all": float(mean_delta_all),
        "mean_delta_accepted": float(mean_delta_accepted),
        "q20_delta_accepted": float(q20_delta_accepted),
        "worst_delta_accepted": float(worst_delta_accepted),
        "mean_delta_all_after_review": float(np.mean(deployed_delta)),
        "residual_harm_rate_after_review": float(np.mean(deployed_delta < 0.0)),
    }

    if method_comparison_path is not None:
        try:
            comp = pd.read_csv(method_comparison_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read method_comparison.csv at {method_comparison_path}: {e}") from e
        if not comp.empty:
            r0 = comp.iloc[0]
            for c in [
                "chosen_review_budget_frac",
                "chosen_review_budget_n",
                "review_infeasible",
                "review_budget_mode",
                "cond_risk_target",
            ]:
                if c not in comp.columns:
                    continue
                if c in {"review_budget_mode", "cond_risk_target"}:
                    row[c] = str(r0.get(c, ""))
                else:
                    row[c] = float(pd.to_numeric(r0.get(c, np.nan), errors="coerce"))

    review_flag_col = _col(df, "review_flag")
    if review_flag_col is not None:
        review_flag = pd.to_numeric(df[review_flag_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reviewed_mask = review_flag > 0.5
        row["review_rate"] = float(np.mean(reviewed_mask))
        if int(np.sum(reviewed_mask)) > 0:
            gain = deployed_delta[reviewed_mask] - final_delta[reviewed_mask]
            row["review_yield"] = float(np.mean(gain))
            row["reviewed_n"] = float(np.sum(reviewed_mask))
        else:
            row["review_yield"] = 0.0
            row["reviewed_n"] = 0.0
    else:
        row["review_rate"] = 0.0
        row["review_yield"] = 0.0
        row["reviewed_n"] = 0.0

    risk_mode_col = _col(df, "risk_mode")
    if risk_mode_col is not None:
        vals = df[risk_mode_col].astype(str).dropna().unique().tolist()
        if vals:
            row["risk_mode"] = str(vals[0])

    for eps in eps_list:
        k = int(np.sum(final_delta < -float(eps)))
        rate = float(k / n_subjects)
        if k == n_subjects:
            ucb = 1.0
        else:
            ucb = float(beta.ppf(1.0 - float(delta), k + 1, n_subjects - k))

        suf = _eps_suffix(eps)
        row[f"neg_event_count_eps{suf}"] = float(k)
        row[f"neg_event_rate_eps{suf}"] = float(rate)
        row[f"clopper_pearson_ucb_eps{suf}"] = float(ucb)

        # Conditional risk among accepted subjects.
        if accepted_n > 0:
            cond_rate = float(k / accepted_n)
            if k >= accepted_n:
                cond_ucb = 1.0
            else:
                cond_ucb = float(beta.ppf(1.0 - float(delta), k + 1, accepted_n - k))
            ci_lo, ci_hi = _cp_two_sided_ci(k=k, n=accepted_n, delta=float(delta))
        else:
            cond_rate = float("nan")
            cond_ucb = float("nan")
            ci_lo, ci_hi = float("nan"), float("nan")

        row[f"cond_neg_event_rate_eps{suf}"] = float(cond_rate)
        row[f"cond_clopper_pearson_ucb_eps{suf}"] = float(cond_ucb)
        row[f"cond_clopper_pearson_ci_low_eps{suf}"] = float(ci_lo)
        row[f"cond_clopper_pearson_ci_high_eps{suf}"] = float(ci_hi)

        # Conditional risk after review-proxy among accepted subjects (same accepted set).
        if accepted_n > 0:
            k_post = int(np.sum((deployed_delta < -float(eps)) & (accepted == 1)))
            cond_rate_post = float(k_post / accepted_n)
            if k_post >= accepted_n:
                cond_ucb_post = 1.0
            else:
                cond_ucb_post = float(beta.ppf(1.0 - float(delta), k_post + 1, accepted_n - k_post))
            ci_lo_post, ci_hi_post = _cp_two_sided_ci(k=k_post, n=accepted_n, delta=float(delta))
        else:
            k_post = 0
            cond_rate_post = float("nan")
            cond_ucb_post = float("nan")
            ci_lo_post, ci_hi_post = float("nan"), float("nan")

        row[f"cond_neg_event_count_after_review_eps{suf}"] = float(k_post) if accepted_n > 0 else float("nan")
        row[f"cond_neg_event_rate_after_review_eps{suf}"] = float(cond_rate_post)
        row[f"cond_clopper_pearson_ucb_after_review_eps{suf}"] = float(cond_ucb_post)
        row[f"cond_clopper_pearson_ci_low_after_review_eps{suf}"] = float(ci_lo_post)
        row[f"cond_clopper_pearson_ci_high_after_review_eps{suf}"] = float(ci_hi_post)

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze offline SAFE-TTA selection sweeps and plot risk/utility curves.")
    ap.add_argument("--root", type=Path, required=True, help="Directory containing variant subdirs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--delta", type=float, default=0.05, help="Confidence level is 1-delta for CP upper bound.")
    ap.add_argument("--eps", type=str, default="0,0.005,0.01", help="Comma-separated eps thresholds for risk events.")
    ap.add_argument(
        "--x-axis",
        type=str,
        default="accept_rate",
        choices=["accept_rate", "selection_coverage"],
        help="X axis for curves.",
    )
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise RuntimeError(f"--root does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"--root is not a directory: {root}")

    eps_list = _parse_eps_list(args.eps)

    variant_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not variant_dirs:
        raise RuntimeError(f"No subdirectories found under --root: {root}")

    rows: list[dict[str, float | str]] = []
    for vdir in variant_dirs:
        csv_path = _find_newest_per_subject_csv(vdir)
        comp_path = _find_newest_method_comparison_csv(vdir)
        rows.append(
            _compute_variant_metrics(
                variant=vdir.name,
                csv_path=csv_path,
                method_comparison_path=comp_path,
                delta=float(args.delta),
                eps_list=eps_list,
            )
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(args.x_axis, na_position="last").reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Plots.
    dfp = summary.copy()
    x = pd.to_numeric(dfp[args.x_axis], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(x)
    dfp = dfp.loc[finite_mask].reset_index(drop=True)
    x = pd.to_numeric(dfp[args.x_axis], errors="coerce").to_numpy(dtype=float)

    x_label = args.x_axis
    title_base = str(args.title).strip() if args.title else None

    # (a) Risk curve.
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, eps in enumerate(eps_list):
        suf = _eps_suffix(eps)
        y = pd.to_numeric(dfp[f"neg_event_rate_eps{suf}"], errors="coerce").to_numpy(dtype=float)
        u = pd.to_numeric(dfp[f"clopper_pearson_ucb_eps{suf}"], errors="coerce").to_numpy(dtype=float)
        color = colors[i % len(colors)] if colors else None
        ax.plot(x, y, marker="o", color=color, label=f"risk (eps={eps:g})")
        ax.plot(x, u, linestyle="--", color=color, label=f"ucb (eps={eps:g})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("neg_event_rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title_base} | risk curve" if title_base else "risk curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    risk_png = out_dir / "risk_curve.png"
    fig.savefig(risk_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) Utility curve.
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    y_all = pd.to_numeric(dfp["mean_delta_all"], errors="coerce").to_numpy(dtype=float)
    y_acc = pd.to_numeric(dfp["mean_delta_accepted"], errors="coerce").to_numpy(dtype=float)
    y_q20 = pd.to_numeric(dfp["q20_delta_accepted"], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, y_all, marker="o", color="#4C72B0", label="mean_delta_all")
    ax.plot(x, y_acc, marker="o", color="#DD8452", label="mean_delta_accepted")
    ax.plot(x, y_q20, marker="o", color="#55A868", label="q20_delta_accepted")
    ax.set_xlabel(x_label)
    ax.set_ylabel("delta (selected - anchor)")
    ax.set_title(f"{title_base} | utility curve" if title_base else "utility curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    util_png = out_dir / "utility_curve.png"
    fig.savefig(util_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Variants: {len(variant_dirs)} under {root}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {risk_png}")
    print(f"Wrote: {util_png}")


if __name__ == "__main__":
    main()
