from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 file match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore

        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        # Fallback: kappa = (p_o - p_e) / (1 - p_e)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
        if labels.size == 0:
            return float("nan")
        label_to_idx = {lab: i for i, lab in enumerate(labels.tolist())}
        conf = np.zeros((labels.size, labels.size), dtype=np.float64)
        for yt, yp in zip(y_true, y_pred, strict=False):
            conf[label_to_idx[yt], label_to_idx[yp]] += 1.0
        n = float(conf.sum())
        if n <= 0.0:
            return float("nan")
        p_o = float(np.trace(conf) / n)
        p_true = conf.sum(axis=1) / n
        p_pred = conf.sum(axis=0) / n
        p_e = float((p_true * p_pred).sum())
        denom = 1.0 - p_e
        if denom <= 0.0:
            return float("nan")
        return float((p_o - p_e) / denom)


def _wilcoxon(delta: np.ndarray, *, alternative: str) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    delta = delta[delta != 0.0]
    if delta.size < 1:
        return float("nan")
    try:
        from scipy.stats import wilcoxon  # type: ignore

        _stat, p = wilcoxon(delta, alternative=alternative)
        return float(p)
    except Exception:
        return float("nan")


def _binom_pmf(k: int, n: int, p: float) -> float:
    from math import comb

    return float(comb(n, k) * (p**k) * ((1 - p) ** (n - k)))


def _sign_test(delta: np.ndarray, *, alternative: str) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    delta = delta[delta != 0.0]
    if delta.size < 1:
        return float("nan")
    n_pos = int((delta > 0.0).sum())
    n = int(delta.size)
    try:
        from scipy.stats import binomtest  # type: ignore

        return float(binomtest(n_pos, n, 0.5, alternative=alternative).pvalue)
    except Exception:
        probs = np.array([_binom_pmf(i, n, 0.5) for i in range(n + 1)], dtype=float)
        if alternative == "greater":
            return float(probs[n_pos:].sum())
        if alternative == "less":
            return float(probs[: n_pos + 1].sum())
        if alternative == "two-sided":
            p_obs = probs[n_pos]
            return float(probs[probs <= p_obs + 1e-15].sum())
        return float("nan")


def _holm(pvals: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full_like(p, fill_value=float("nan"), dtype=float)
    mask = np.isfinite(p)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = int(idx.size)
    prev = 0.0
    for j, i in enumerate(order):
        adj = float(min(1.0, (m - j) * float(p[i])))
        adj = float(max(prev, adj))
        out[i] = adj
        prev = adj
    return out.tolist()


def _format_pm_std(mean: float, std: float, *, digits: int) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def _bootstrap_ci_mean(x: np.ndarray, *, iters: int = 20000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 1:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(int(seed))
    n = int(x.size)
    means = np.empty(int(iters), dtype=float)
    for i in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


@dataclass(frozen=True)
class MethodSummary:
    method: str
    n_subjects: int
    mean_accuracy: float
    std_accuracy: float
    worst_accuracy: float
    mean_kappa: float
    std_kappa: float
    worst_kappa: float
    mean_delta_vs_baseline: float
    delta_ci95_low: float
    delta_ci95_high: float
    neg_transfer_rate_vs_baseline: float
    n_pos: int
    n_neg: int
    n_zero: int
    p_wilcoxon_two_sided: float
    p_wilcoxon_greater: float
    p_sign_greater: float
    accept_rate: float


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    pred_path = _find_single(run_dir, "*_predictions_all_methods.csv")
    df = pd.read_csv(pred_path)
    required = {"method", "subject", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {pred_path}: {sorted(missing)}")
    return df


def _load_accept_rate(run_dir: Path) -> dict[str, float]:
    try:
        path = _find_single(run_dir, "*_method_comparison.csv")
    except Exception:
        return {}
    df = pd.read_csv(path)
    if "method" not in df.columns or "accept_rate" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        m = str(row["method"])
        v = float(row["accept_rate"]) if np.isfinite(row["accept_rate"]) else float("nan")
        out[m] = v
    return out


def compute_table(
    *,
    run_dir: Path,
    baseline_method: str,
    include_methods: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_predictions(run_dir)
    accept_rate = _load_accept_rate(run_dir)

    df["correct"] = (df["y_true"].astype(str) == df["y_pred"].astype(str)).astype(float)
    acc = df.groupby(["method", "subject"], sort=True)["correct"].mean().unstack("subject")

    kappas: dict[tuple[str, int], float] = {}
    for (m, s), g in df.groupby(["method", "subject"], sort=True):
        y_t = g["y_true"].astype(str).to_numpy()
        y_p = g["y_pred"].astype(str).to_numpy()
        kappas[(str(m), int(s))] = _cohen_kappa(y_t, y_p)
    kap = pd.Series(kappas).unstack(1).sort_index(axis=0).sort_index(axis=1)

    if include_methods is not None:
        keep = [m for m in include_methods if m in acc.index]
        acc = acc.loc[keep]
        kap = kap.loc[keep]

    if baseline_method not in acc.index:
        raise RuntimeError(f"baseline_method={baseline_method} not found in methods: {list(acc.index)}")
    base_acc = acc.loc[baseline_method]

    subjects = [int(s) for s in acc.columns.tolist()]
    per_subject = pd.DataFrame(index=subjects)
    for m in acc.index:
        per_subject[f"acc_{m}"] = acc.loc[m].to_numpy()
        per_subject[f"kappa_{m}"] = kap.loc[m].to_numpy()
        per_subject[f"delta_acc_vs_{baseline_method}_{m}"] = (acc.loc[m] - base_acc).to_numpy()

    rows: list[MethodSummary] = []
    for m in acc.index:
        a = acc.loc[m].to_numpy(dtype=float)
        k = kap.loc[m].to_numpy(dtype=float)
        delta = (acc.loc[m] - base_acc).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci_mean(delta)
        n_pos = int((delta > 0.0).sum())
        n_neg = int((delta < 0.0).sum())
        n_zero = int((delta == 0.0).sum())
        rows.append(
            MethodSummary(
                method=str(m),
                n_subjects=int(np.isfinite(a).sum()),
                mean_accuracy=float(np.nanmean(a)),
                std_accuracy=float(np.nanstd(a, ddof=1)),
                worst_accuracy=float(np.nanmin(a)),
                mean_kappa=float(np.nanmean(k)),
                std_kappa=float(np.nanstd(k, ddof=1)),
                worst_kappa=float(np.nanmin(k)),
                mean_delta_vs_baseline=float(np.nanmean(delta)),
                delta_ci95_low=float(ci_low),
                delta_ci95_high=float(ci_high),
                neg_transfer_rate_vs_baseline=float(n_neg / len(delta)) if len(delta) else float("nan"),
                n_pos=n_pos,
                n_neg=n_neg,
                n_zero=n_zero,
                p_wilcoxon_two_sided=_wilcoxon(delta, alternative="two-sided") if m != baseline_method else float("nan"),
                p_wilcoxon_greater=_wilcoxon(delta, alternative="greater") if m != baseline_method else float("nan"),
                p_sign_greater=_sign_test(delta, alternative="greater") if m != baseline_method else float("nan"),
                accept_rate=float(accept_rate.get(str(m), float("nan"))),
            )
        )

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values("mean_accuracy", ascending=False)
    out["p_wilcoxon_two_sided_holm"] = _holm(out["p_wilcoxon_two_sided"].to_list())
    out["p_sign_greater_holm"] = _holm(out["p_sign_greater"].to_list())
    return out, per_subject


def _to_markdown_table(df: pd.DataFrame, *, baseline_method: str, digits: int) -> str:
    view = df.copy()
    view.insert(
        1,
        "acc(mean±std)",
        [_format_pm_std(float(r["mean_accuracy"]), float(r["std_accuracy"]), digits=digits) for _, r in df.iterrows()],
    )
    view = view[
        [
            "method",
            "acc(mean±std)",
            "worst_accuracy",
            "mean_delta_vs_baseline",
            "neg_transfer_rate_vs_baseline",
            "p_wilcoxon_two_sided",
            "p_wilcoxon_two_sided_holm",
            "accept_rate",
        ]
    ].rename(
        columns={
            "mean_delta_vs_baseline": f"meanΔacc_vs_{baseline_method}",
            "neg_transfer_rate_vs_baseline": f"neg_transfer_vs_{baseline_method}",
            "p_wilcoxon_two_sided": "p_wilcoxon_2s",
            "p_wilcoxon_two_sided_holm": "p_wilcoxon_2s_holm",
        }
    )

    # Numeric formatting (keep method + acc(mean±std) as-is).
    for c in view.columns:
        if c in {"method", "acc(mean±std)"}:
            continue
        view[c] = view[c].apply(lambda x: f"{float(x):.{digits}f}" if np.isfinite(x) else "nan")
    headers = [str(c) for c in view.columns.tolist()]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join([str(x) for x in row]) + " |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Make main table + paired stats from *_predictions_all_methods.csv.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--baseline-method", type=str, default="ea-csp-lda")
    p.add_argument("--include-methods", type=str, default="")
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--out-per-subject-csv", type=str, default="")
    p.add_argument("--out-md", type=str, default="")
    p.add_argument("--digits", type=int, default=4)
    args = p.parse_args()

    include = [m.strip() for m in str(args.include_methods).split(",") if m.strip()] if args.include_methods else None
    table, per_subject = compute_table(
        run_dir=Path(args.run_dir), baseline_method=str(args.baseline_method), include_methods=include
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    if args.out_per_subject_csv:
        out_ps = Path(args.out_per_subject_csv)
        out_ps.parent.mkdir(parents=True, exist_ok=True)
        per_subject.to_csv(out_ps, index=True)

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(
            _to_markdown_table(table, baseline_method=str(args.baseline_method), digits=int(args.digits)) + "\n",
            encoding="utf-8",
        )

    best = table.iloc[0]
    print(f"Loaded run: {args.run_dir}")
    print(f"Baseline: {args.baseline_method}")
    print(
        "Best by mean_accuracy: "
        f"{best['method']} mean={best['mean_accuracy']:.4f} worst={best['worst_accuracy']:.4f} "
        f"meanΔ={best['mean_delta_vs_baseline']:.4f} neg={best['neg_transfer_rate_vs_baseline']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
