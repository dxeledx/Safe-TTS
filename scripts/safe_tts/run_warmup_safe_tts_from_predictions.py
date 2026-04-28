#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "csp_lda").is_dir()), None)
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csp_lda.certificate import train_evidential_selector


DEFAULT_CANDIDATES = "ALL"
DEFAULT_WARMUP_GRID = "8,16,24,32"


@dataclass(frozen=True)
class Sample:
    subject: int
    method: str
    x: np.ndarray
    suffix_gain: float
    e2e_gain: float
    acc_anchor_full: float
    acc_anchor_suffix: float
    acc_candidate_suffix: float
    acc_oracle_e2e: float
    n_total: int
    n_prefix: int
    n_suffix: int


@dataclass(frozen=True)
class ScoredCandidate:
    sample: Sample
    risk: float
    utility: float
    uncertainty: float
    p_harm: float
    p_neutral: float
    p_benefit: float


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in _parse_csv_list(raw)]


def _parse_eval_subjects(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip() or str(raw).strip().upper() == "ALL":
        return None
    out: list[int] = []
    for chunk in _parse_csv_list(str(raw)):
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def _safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return p / np.sum(p, axis=1, keepdims=True)


def _entropy_rows(p: np.ndarray) -> np.ndarray:
    p = _safe_probs(p)
    return -np.sum(p * np.log(p), axis=1)


def _entropy_vec(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    p = p / float(np.sum(p))
    return float(-np.sum(p * np.log(p)))


def _js_rows(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    p0 = _safe_probs(p0)
    p1 = _safe_probs(p1)
    mid = 0.5 * (p0 + p1)
    kl0 = np.sum(p0 * (np.log(p0) - np.log(mid)), axis=1)
    kl1 = np.sum(p1 * (np.log(p1) - np.log(mid)), axis=1)
    return 0.5 * (kl0 + kl1)


def _balanced_decision_reliability(p: np.ndarray) -> float:
    p = _safe_probs(p)
    if p.shape[0] <= 0:
        return 0.0
    c = int(p.shape[1])
    log_c = max(math.log(float(c)), 1e-12)
    certainty = 1.0 - float(np.mean(_entropy_rows(p))) / log_c
    p_bar = np.mean(p, axis=0)
    balance = _entropy_vec(p_bar) / log_c
    return float(np.clip(certainty * balance, 0.0, 1.0))


def _high_conflict_mask(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    tau: float,
) -> np.ndarray:
    p0 = _safe_probs(p0)
    p1 = _safe_probs(p1)
    y0 = np.asarray(y0, dtype=object).reshape(-1)
    y1 = np.asarray(y1, dtype=object).reshape(-1)
    return (y0 != y1) & (np.max(p0, axis=1) >= float(tau)) & (np.max(p1, axis=1) >= float(tau))


def _relative_js_risk(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    conflict_tau: float,
    conflict_lambda: float,
) -> float:
    p0 = _safe_probs(p0)
    p1 = _safe_probs(p1)
    if p1.shape[0] <= 0:
        return 0.0
    js = _js_rows(p0, p1) / math.log(2.0)
    hc = _high_conflict_mask(p0=p0, p1=p1, y0=y0, y1=y1, tau=float(conflict_tau)).astype(np.float64)
    return float(np.mean(js * (1.0 + float(conflict_lambda) * hc)))


def _chunk_bounds(n_rows: int, n_chunks: int) -> list[tuple[int, int]]:
    n_rows = int(n_rows)
    if n_rows <= 0:
        return []
    n_chunks = min(max(1, int(n_chunks)), n_rows)
    edges = np.linspace(0, n_rows, num=n_chunks + 1, dtype=int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if int(b) > int(a)]


def _ridge_koopman(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    # Rows are observations, columns are the low-dimensional block state.
    x_t = np.asarray(x, dtype=np.float64).T
    y_t = np.asarray(y, dtype=np.float64).T
    dim = int(x_t.shape[0])
    return y_t @ x_t.T @ np.linalg.pinv(x_t @ x_t.T + float(lam) * np.eye(dim))


def _spectral_radius(k: np.ndarray) -> float:
    try:
        eig = np.linalg.eigvals(np.asarray(k, dtype=np.float64))
        return float(np.max(np.abs(eig))) if eig.size else 0.0
    except Exception:
        return 0.0


def _koopman_temporal_instability(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    n_chunks: int,
    ridge_lambda: float,
    spectral_gamma: float,
    conflict_tau: float,
    conflict_lambda: float,
) -> float:
    p0 = _safe_probs(p0)
    p1 = _safe_probs(p1)
    y0 = np.asarray(y0, dtype=object).reshape(-1)
    y1 = np.asarray(y1, dtype=object).reshape(-1)
    states: list[np.ndarray] = []
    for a, b in _chunk_bounds(int(p1.shape[0]), int(n_chunks)):
        p0_b = p0[a:b]
        p1_b = p1[a:b]
        y0_b = y0[a:b]
        y1_b = y1[a:b]
        a_b = _balanced_decision_reliability(p1_b)
        r_b = float(np.mean(_js_rows(p0_b, p1_b) / math.log(2.0))) if p1_b.shape[0] else 0.0
        c_b = float(
            np.mean(_high_conflict_mask(p0=p0_b, p1=p1_b, y0=y0_b, y1=y1_b, tau=float(conflict_tau)).astype(np.float64))
        )
        # Keep conflict intensity in R through the global feature; C is explicit in the Koopman state.
        _ = conflict_lambda
        states.append(np.asarray([a_b, r_b, c_b], dtype=np.float64))
    if len(states) < 2:
        return 0.0
    z = np.vstack(states)
    x = z[:-1]
    y = z[1:]
    k = _ridge_koopman(x, y, lam=float(ridge_lambda))
    pred = (k @ x.T).T
    resid = float(np.sum((y - pred) ** 2))
    denom = float(np.sum(y**2) + 1e-12)
    rho = _spectral_radius(k)
    penalty = float(spectral_gamma) * max(0.0, rho - 1.0) ** 2
    return float(resid / denom + penalty)


def _core_features(
    *,
    df_anchor: pd.DataFrame,
    df_candidate: pd.DataFrame,
    proba_cols: list[str],
    n_chunks: int,
    ridge_lambda: float,
    spectral_gamma: float,
    conflict_tau: float,
    conflict_lambda: float,
) -> np.ndarray:
    p0 = df_anchor[proba_cols].to_numpy(np.float64)
    p1 = df_candidate[proba_cols].to_numpy(np.float64)
    y0 = df_anchor["y_pred"].to_numpy(object)
    y1 = df_candidate["y_pred"].to_numpy(object)
    absolute = _balanced_decision_reliability(p1)
    relative = _relative_js_risk(
        p0=p0,
        p1=p1,
        y0=y0,
        y1=y1,
        conflict_tau=float(conflict_tau),
        conflict_lambda=float(conflict_lambda),
    )
    temporal = _koopman_temporal_instability(
        p0=p0,
        p1=p1,
        y0=y0,
        y1=y1,
        n_chunks=int(n_chunks),
        ridge_lambda=float(ridge_lambda),
        spectral_gamma=float(spectral_gamma),
        conflict_tau=float(conflict_tau),
        conflict_lambda=float(conflict_lambda),
    )
    return np.asarray([absolute, relative, temporal], dtype=np.float64)


def _state_from_gain(gain: float, delta: float) -> int:
    if float(gain) < -float(delta):
        return 0
    if float(gain) > float(delta):
        return 2
    return 1


def _cp_upper(k: int, n: int, confidence: float) -> float:
    k = int(k)
    n = int(n)
    if n <= 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(beta.ppf(float(confidence), k + 1, n - k))


def _split_folds(subjects: list[int], n_splits: int, seed: int) -> list[list[int]]:
    subjects = [int(s) for s in subjects]
    rng = np.random.RandomState(int(seed))
    shuffled = np.asarray(subjects, dtype=int)
    rng.shuffle(shuffled)
    n_splits = min(max(2, int(n_splits)), int(len(shuffled)))
    return [arr.astype(int).tolist() for arr in np.array_split(shuffled, n_splits) if arr.size > 0]


def _dev_cal_split(subjects: list[int], calib_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    subjects = [int(s) for s in subjects]
    rng = np.random.RandomState(int(seed))
    shuffled = np.asarray(subjects, dtype=int)
    rng.shuffle(shuffled)
    n_cal = max(1, int(round(float(calib_fraction) * len(shuffled))))
    n_cal = min(n_cal, max(1, len(shuffled) - 1))
    cal = sorted(int(x) for x in shuffled[:n_cal])
    dev = sorted(int(x) for x in shuffled[n_cal:])
    return dev, cal


def _train_selector(
    samples_by_subject: dict[int, list[Sample]],
    train_subjects: Iterable[int],
    *,
    outcome_delta: float,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_rank: float,
    lambda_kl: float,
    rank_margin: float,
    rho: float,
    eta: float,
    seed: int,
    progress_label: str | None,
):
    rows: list[np.ndarray] = []
    states: list[int] = []
    gains: list[float] = []
    groups: list[int] = []
    for s in train_subjects:
        for sample in samples_by_subject[int(s)]:
            rows.append(np.asarray(sample.x, dtype=np.float64).reshape(-1))
            states.append(_state_from_gain(sample.suffix_gain, delta=float(outcome_delta)))
            gains.append(float(sample.suffix_gain))
            groups.append(int(s))
    if not rows:
        raise RuntimeError("No warm-up training rows.")
    x = np.vstack(rows).astype(np.float64, copy=False)
    y_state = np.asarray(states, dtype=int)
    y_gain = np.asarray(gains, dtype=np.float64)
    group_ids = np.asarray(groups, dtype=int)
    return train_evidential_selector(
        x,
        y_state,
        y_gain,
        group_ids,
        feature_names=("absolute_core", "relative_core", "koopman_temporal"),
        view_slices={
            "absolute_core": (0, 1),
            "relative_core": (1, 2),
            "koopman_temporal": (2, 3),
        },
        hidden_dim=int(hidden_dim),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        lambda_rank=float(lambda_rank),
        lambda_kl=float(lambda_kl),
        pair_margin=float(rank_margin),
        rho=float(rho),
        eta=float(eta),
        seed=int(seed),
        progress_label=progress_label,
        progress_every=0,
    )


def _score_samples(selector, samples: list[Sample]) -> list[ScoredCandidate]:
    if not samples:
        return []
    x = np.vstack([np.asarray(s.x, dtype=np.float64).reshape(-1) for s in samples])
    stats = selector.predict_stats(x)
    probs = np.asarray(stats["probs"], dtype=np.float64)
    risk = np.asarray(stats["risk"], dtype=np.float64).reshape(-1)
    utility = np.asarray(stats["utility"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(stats["uncertainty"], dtype=np.float64).reshape(-1)
    out: list[ScoredCandidate] = []
    for i, sample in enumerate(samples):
        out.append(
            ScoredCandidate(
                sample=sample,
                risk=float(risk[i]),
                utility=float(utility[i]),
                uncertainty=float(uncertainty[i]),
                p_harm=float(probs[i, 0]),
                p_neutral=float(probs[i, 1]),
                p_benefit=float(probs[i, 2]),
            )
        )
    return out


def _select_candidate(
    scored: list[ScoredCandidate],
    *,
    risk_threshold: float,
    utility_threshold: float,
) -> ScoredCandidate | None:
    feasible = [
        c
        for c in scored
        if float(c.risk) <= float(risk_threshold) and float(c.utility) >= float(utility_threshold)
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda c: (float(c.utility), -float(c.risk), float(c.sample.e2e_gain)))


def _evaluate_subject_actions(
    scored_by_subject: dict[int, list[ScoredCandidate]],
    subjects: Iterable[int],
    *,
    risk_threshold: float,
    utility_threshold: float,
    neg_eps: float,
) -> dict[str, float]:
    gains: list[float] = []
    suffix_gains: list[float] = []
    accepted = 0
    harm = 0
    for s in subjects:
        chosen = _select_candidate(
            scored_by_subject[int(s)],
            risk_threshold=float(risk_threshold),
            utility_threshold=float(utility_threshold),
        )
        if chosen is None:
            gains.append(0.0)
            suffix_gains.append(0.0)
            continue
        accepted += 1
        gains.append(float(chosen.sample.e2e_gain))
        suffix_gains.append(float(chosen.sample.suffix_gain))
        if float(chosen.sample.suffix_gain) < -float(neg_eps):
            harm += 1
    n = len(list(subjects)) if not isinstance(subjects, list) else len(subjects)
    return {
        "n_subjects": float(n),
        "accepted": float(accepted),
        "harm": float(harm),
        "mean_e2e_gain": float(np.mean(gains)) if gains else 0.0,
        "mean_suffix_gain": float(np.mean(suffix_gains)) if suffix_gains else 0.0,
        "accept_rate": float(accepted / max(1, n)),
        "cond_harm_rate": float(harm / accepted) if accepted > 0 else 0.0,
    }


def _threshold_candidates(
    scored_by_subject: dict[int, list[ScoredCandidate]],
    *,
    min_utility_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    risks = np.asarray([c.risk for rows in scored_by_subject.values() for c in rows], dtype=np.float64)
    utils = np.asarray([c.utility for rows in scored_by_subject.values() for c in rows], dtype=np.float64)
    if risks.size == 0 or utils.size == 0:
        return np.asarray([0.0]), np.asarray([math.inf])
    r_grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 21), np.quantile(risks, np.linspace(0.0, 1.0, 21))]))
    q_grid = np.unique(
        np.concatenate(
            [
                [float(min_utility_threshold)],
                np.linspace(float(min_utility_threshold), 1.0, 21),
                np.quantile(utils, np.linspace(0.0, 1.0, 21)),
            ]
        )
    )
    q_grid = q_grid[q_grid >= float(min_utility_threshold)]
    return np.asarray(r_grid, dtype=np.float64), np.asarray(q_grid, dtype=np.float64)


def _choose_and_verify_thresholds(
    *,
    oof_scored: dict[int, list[ScoredCandidate]],
    dev_subjects: list[int],
    cal_subjects: list[int],
    risk_alpha: float,
    cp_delta: float,
    neg_eps: float,
    min_accept_rate: float,
    min_utility_threshold: float,
) -> dict[str, float | bool | str]:
    r_grid, q_grid = _threshold_candidates(
        {s: oof_scored[int(s)] for s in dev_subjects},
        min_utility_threshold=float(min_utility_threshold),
    )
    best: dict[str, float] | None = None
    for r_thr in r_grid:
        for q_thr in q_grid:
            m = _evaluate_subject_actions(
                oof_scored,
                dev_subjects,
                risk_threshold=float(r_thr),
                utility_threshold=float(q_thr),
                neg_eps=float(neg_eps),
            )
            if float(m["accept_rate"]) < float(min_accept_rate):
                continue
            obj = float(m["mean_e2e_gain"])
            if best is None or obj > float(best["dev_mean_e2e_gain"]):
                best = {
                    "risk_threshold": float(r_thr),
                    "utility_threshold": float(q_thr),
                    "dev_mean_e2e_gain": float(m["mean_e2e_gain"]),
                    "dev_mean_suffix_gain": float(m["mean_suffix_gain"]),
                    "dev_accept_rate": float(m["accept_rate"]),
                    "dev_cond_harm_rate": float(m["cond_harm_rate"]),
                }
    if best is None:
        return {"verified": False, "fallback_stage": "no_dev_threshold"}

    cal = _evaluate_subject_actions(
        oof_scored,
        cal_subjects,
        risk_threshold=float(best["risk_threshold"]),
        utility_threshold=float(best["utility_threshold"]),
        neg_eps=float(neg_eps),
    )
    n_acc = int(cal["accepted"])
    n_harm = int(cal["harm"])
    ucb = _cp_upper(n_harm, n_acc, confidence=1.0 - float(cp_delta))
    verified = bool(ucb <= float(risk_alpha))
    return {
        **best,
        "verified": verified,
        "fallback_stage": "verified" if verified else "calibration_failed",
        "cal_accept": float(n_acc),
        "cal_harm": float(n_harm),
        "cal_cond_harm_rate": float(cal["cond_harm_rate"]),
        "cal_harm_ucb": float(ucb),
        "cal_mean_e2e_gain": float(cal["mean_e2e_gain"]),
        "cal_mean_suffix_gain": float(cal["mean_suffix_gain"]),
    }


def _load_predictions(path: Path, anchor_method: str, candidate_methods: str) -> tuple[pd.DataFrame, list[str], list[str], list[int]]:
    df = pd.read_csv(path)
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in predictions CSV: {missing}")
    proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
    if not proba_cols:
        raise RuntimeError("No proba_* columns found.")
    methods = sorted(str(m) for m in df["method"].unique().tolist())
    if str(anchor_method) not in methods:
        raise RuntimeError(f"Anchor method {anchor_method!r} not found. Available: {methods}")
    if str(candidate_methods).strip().upper() == "ALL":
        cands = [m for m in methods if m != str(anchor_method)]
    else:
        cands = _parse_csv_list(candidate_methods)
    missing_cands = [m for m in cands if m not in methods]
    if missing_cands:
        raise RuntimeError(f"Candidate methods not found: {missing_cands}. Available: {methods}")
    subjects = sorted(int(s) for s in df["subject"].unique().tolist())
    return df, proba_cols, cands, subjects


def _build_subject_maps(df: pd.DataFrame, methods: list[str]) -> dict[str, dict[int, pd.DataFrame]]:
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for m in methods:
        dm = df[df["method"].astype(str) == str(m)].copy()
        by_s: dict[int, pd.DataFrame] = {}
        for s, g in dm.groupby("subject", sort=True):
            by_s[int(s)] = g.sort_values("trial").reset_index(drop=True)
        out[str(m)] = by_s
    return out


def _validate_alignment(by_method: dict[str, dict[int, pd.DataFrame]], methods: list[str], subjects: list[int]) -> None:
    ref = by_method[methods[0]]
    for s in subjects:
        base = ref[int(s)][["subject", "trial", "y_true"]].reset_index(drop=True)
        for m in methods[1:]:
            cur = by_method[str(m)][int(s)][["subject", "trial", "y_true"]].reset_index(drop=True)
            if not base.equals(cur):
                raise RuntimeError(f"Prediction alignment mismatch for subject={s}, method={m}")


def _build_samples_for_w(
    *,
    by_method: dict[str, dict[int, pd.DataFrame]],
    anchor_method: str,
    candidate_methods: list[str],
    subjects: list[int],
    proba_cols: list[str],
    warmup_trials: int,
    n_chunks: int,
    ridge_lambda: float,
    spectral_gamma: float,
    conflict_tau: float,
    conflict_lambda: float,
    min_suffix_trials: int,
) -> dict[int, list[Sample]]:
    samples: dict[int, list[Sample]] = {}
    for s in subjects:
        df_a = by_method[str(anchor_method)][int(s)]
        n_total = int(df_a.shape[0])
        w = int(warmup_trials)
        if n_total <= w + int(min_suffix_trials):
            continue
        y_true = df_a["y_true"].to_numpy(object)
        pred_a = df_a["y_pred"].to_numpy(object)
        anchor_correct = pred_a == y_true
        acc_anchor_full = float(np.mean(anchor_correct))
        acc_anchor_suffix = float(np.mean(anchor_correct[w:]))
        n_suffix = int(n_total - w)
        prefix_correct = int(np.sum(anchor_correct[:w]))
        subject_samples: list[Sample] = []
        candidate_e2e_accs: list[float] = []
        raw: list[tuple[str, np.ndarray, float, float]] = []
        for m in candidate_methods:
            df_c = by_method[str(m)][int(s)]
            pred_c = df_c["y_pred"].to_numpy(object)
            cand_correct = pred_c == y_true
            acc_c_suffix = float(np.mean(cand_correct[w:]))
            suffix_gain = float(acc_c_suffix - acc_anchor_suffix)
            final_acc = float((prefix_correct + int(np.sum(cand_correct[w:]))) / float(n_total))
            e2e_gain = float(final_acc - acc_anchor_full)
            x = _core_features(
                df_anchor=df_a.iloc[:w].reset_index(drop=True),
                df_candidate=df_c.iloc[:w].reset_index(drop=True),
                proba_cols=proba_cols,
                n_chunks=int(n_chunks),
                ridge_lambda=float(ridge_lambda),
                spectral_gamma=float(spectral_gamma),
                conflict_tau=float(conflict_tau),
                conflict_lambda=float(conflict_lambda),
            )
            raw.append((str(m), x, suffix_gain, e2e_gain))
            candidate_e2e_accs.append(final_acc)
        oracle_e2e = float(max([acc_anchor_full] + candidate_e2e_accs))
        for m, x, suffix_gain, e2e_gain in raw:
            df_c = by_method[str(m)][int(s)]
            acc_c_suffix = float(np.mean(df_c["y_pred"].to_numpy(object)[w:] == y_true[w:]))
            subject_samples.append(
                Sample(
                    subject=int(s),
                    method=str(m),
                    x=np.asarray(x, dtype=np.float64),
                    suffix_gain=float(suffix_gain),
                    e2e_gain=float(e2e_gain),
                    acc_anchor_full=float(acc_anchor_full),
                    acc_anchor_suffix=float(acc_anchor_suffix),
                    acc_candidate_suffix=float(acc_c_suffix),
                    acc_oracle_e2e=float(oracle_e2e),
                    n_total=int(n_total),
                    n_prefix=int(w),
                    n_suffix=int(n_suffix),
                )
            )
        samples[int(s)] = subject_samples
    return samples


def _online_fallback_accuracy(
    *,
    selector,
    by_method: dict[str, dict[int, pd.DataFrame]],
    anchor_method: str,
    chosen: ScoredCandidate,
    proba_cols: list[str],
    warmup_trials: int,
    window_trials: int,
    fallback_risk_threshold: float,
    fallback_koopman_threshold: float,
    fallback_patience: int,
    n_chunks: int,
    ridge_lambda: float,
    spectral_gamma: float,
    conflict_tau: float,
    conflict_lambda: float,
) -> dict[str, float | int]:
    s = int(chosen.sample.subject)
    method = str(chosen.sample.method)
    df_a = by_method[str(anchor_method)][s].sort_values("trial").reset_index(drop=True)
    df_c = by_method[method][s].sort_values("trial").reset_index(drop=True)
    y_true = df_a["y_true"].to_numpy(object)
    pred_a = df_a["y_pred"].to_numpy(object)
    pred_c = df_c["y_pred"].to_numpy(object)
    w = int(warmup_trials)
    final_pred = pred_a.copy()
    active = True
    bad_count = 0
    fallback_at = -1
    fallback_risk = float("nan")
    fallback_koop = float("nan")
    for t in range(w, int(len(y_true))):
        if active:
            final_pred[t] = pred_c[t]
        else:
            final_pred[t] = pred_a[t]
        if not active:
            continue
        obs_start = max(w, t - int(window_trials) + 1)
        obs_stop = t + 1
        if obs_stop - obs_start < max(2, min(int(window_trials), int(warmup_trials)) // 2):
            continue
        x_win = _core_features(
            df_anchor=df_a.iloc[obs_start:obs_stop].reset_index(drop=True),
            df_candidate=df_c.iloc[obs_start:obs_stop].reset_index(drop=True),
            proba_cols=proba_cols,
            n_chunks=int(n_chunks),
            ridge_lambda=float(ridge_lambda),
            spectral_gamma=float(spectral_gamma),
            conflict_tau=float(conflict_tau),
            conflict_lambda=float(conflict_lambda),
        ).reshape(1, -1)
        stats = selector.predict_stats(x_win)
        risk = float(np.asarray(stats["risk"]).reshape(-1)[0])
        koop = float(x_win.reshape(-1)[2])
        bad = (risk > float(fallback_risk_threshold)) or (koop > float(fallback_koopman_threshold))
        bad_count = bad_count + 1 if bad else 0
        if bad_count >= int(fallback_patience):
            active = False
            fallback_at = int(t + 1)
            fallback_risk = float(risk)
            fallback_koop = float(koop)
    acc_final = float(np.mean(final_pred == y_true))
    suffix_acc = float(np.mean(final_pred[w:] == y_true[w:]))
    anchor_full = float(np.mean(pred_a == y_true))
    anchor_suffix = float(np.mean(pred_a[w:] == y_true[w:]))
    return {
        "fallback_at_trial": int(fallback_at),
        "fallback_risk": float(fallback_risk),
        "fallback_koopman": float(fallback_koop),
        "acc_final_with_fallback": float(acc_final),
        "suffix_acc_with_fallback": float(suffix_acc),
        "e2e_gain_with_fallback": float(acc_final - anchor_full),
        "suffix_gain_with_fallback": float(suffix_acc - anchor_suffix),
        "used_candidate_suffix_trials": int(np.sum(final_pred[w:] == pred_c[w:])),
    }


def run_one_w(args: argparse.Namespace, *, w: int) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    df, proba_cols, cand_methods, subjects_all = _load_predictions(
        Path(args.preds),
        anchor_method=str(args.anchor_method),
        candidate_methods=str(args.candidate_methods),
    )
    eval_subjects = _parse_eval_subjects(args.eval_subjects)
    subjects = subjects_all if eval_subjects is None else [s for s in subjects_all if int(s) in set(eval_subjects)]
    methods = [str(args.anchor_method)] + cand_methods
    by_method = _build_subject_maps(df, methods=methods)
    _validate_alignment(by_method, methods=methods, subjects=subjects)
    samples_by_subject = _build_samples_for_w(
        by_method=by_method,
        anchor_method=str(args.anchor_method),
        candidate_methods=cand_methods,
        subjects=subjects,
        proba_cols=proba_cols,
        warmup_trials=int(w),
        n_chunks=int(args.koopman_chunks),
        ridge_lambda=float(args.koopman_ridge),
        spectral_gamma=float(args.koopman_gamma),
        conflict_tau=float(args.conflict_tau),
        conflict_lambda=float(args.conflict_lambda),
        min_suffix_trials=int(args.min_suffix_trials),
    )
    subjects = sorted(samples_by_subject)
    if len(subjects) < 4:
        raise RuntimeError(f"Need at least 4 valid subjects after warm-up filtering, got {len(subjects)}")

    rows: list[dict[str, object]] = []
    for t_idx, test_subject in enumerate(subjects):
        fit_subjects = [s for s in subjects if int(s) != int(test_subject)]
        folds = _split_folds(fit_subjects, n_splits=int(args.n_splits), seed=int(args.seed) + 17 * int(test_subject))
        oof: dict[int, list[ScoredCandidate]] = {}
        for fold_idx, heldout in enumerate(folds):
            heldout_set = set(int(s) for s in heldout)
            train_subjects = [s for s in fit_subjects if int(s) not in heldout_set]
            selector_fold = _train_selector(
                samples_by_subject,
                train_subjects,
                outcome_delta=float(args.outcome_delta),
                hidden_dim=int(args.selector_hidden_dim),
                epochs=int(args.selector_epochs),
                lr=float(args.selector_lr),
                weight_decay=float(args.selector_weight_decay),
                lambda_rank=float(args.selector_lambda_rank),
                lambda_kl=float(args.selector_lambda_kl),
                rank_margin=float(args.selector_rank_margin),
                rho=float(args.selector_rho),
                eta=float(args.selector_eta),
                seed=int(args.seed) + 1000 * int(test_subject) + int(fold_idx),
                progress_label=None,
            )
            for s in heldout:
                oof[int(s)] = _score_samples(selector_fold, samples_by_subject[int(s)])

        dev_subjects, cal_subjects = _dev_cal_split(
            fit_subjects,
            calib_fraction=float(args.calib_fraction),
            seed=int(args.seed) + 31 * int(test_subject),
        )
        thr = _choose_and_verify_thresholds(
            oof_scored=oof,
            dev_subjects=dev_subjects,
            cal_subjects=cal_subjects,
            risk_alpha=float(args.risk_alpha),
            cp_delta=float(args.cp_delta),
            neg_eps=float(args.neg_transfer_eps),
            min_accept_rate=float(args.min_accept_rate),
            min_utility_threshold=float(args.min_utility_threshold),
        )
        selector_final = _train_selector(
            samples_by_subject,
            fit_subjects,
            outcome_delta=float(args.outcome_delta),
            hidden_dim=int(args.selector_hidden_dim),
            epochs=int(args.selector_epochs),
            lr=float(args.selector_lr),
            weight_decay=float(args.selector_weight_decay),
            lambda_rank=float(args.selector_lambda_rank),
            lambda_kl=float(args.selector_lambda_kl),
            rank_margin=float(args.selector_rank_margin),
            rho=float(args.selector_rho),
            eta=float(args.selector_eta),
            seed=int(args.seed) + 100000 + int(test_subject),
            progress_label=None,
        )
        scored_test = _score_samples(selector_final, samples_by_subject[int(test_subject)])
        if bool(thr.get("verified", False)):
            chosen = _select_candidate(
                scored_test,
                risk_threshold=float(thr["risk_threshold"]),
                utility_threshold=float(thr["utility_threshold"]),
            )
        else:
            chosen = None

        anchor_sample = samples_by_subject[int(test_subject)][0]
        selected_method = str(args.anchor_method) if chosen is None else str(chosen.sample.method)
        accept = int(chosen is not None)
        if chosen is None:
            acc_final = float(anchor_sample.acc_anchor_full)
            suffix_gain = 0.0
            e2e_gain = 0.0
            suffix_acc = float(anchor_sample.acc_anchor_suffix)
            acc_final_no_fallback = float(acc_final)
            suffix_gain_no_fallback = float(suffix_gain)
            e2e_gain_no_fallback = float(e2e_gain)
            risk = float("nan")
            utility = float("nan")
            uncertainty = float("nan")
            p_harm = p_neutral = p_benefit = float("nan")
            selected_x = np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
            fallback_info = {
                "fallback_at_trial": -1,
                "fallback_risk": float("nan"),
                "fallback_koopman": float("nan"),
                "acc_final_with_fallback": acc_final,
                "suffix_acc_with_fallback": suffix_acc,
                "e2e_gain_with_fallback": e2e_gain,
                "suffix_gain_with_fallback": suffix_gain,
                "used_candidate_suffix_trials": 0,
            }
        else:
            acc_final = float(chosen.sample.acc_anchor_full + chosen.sample.e2e_gain)
            suffix_gain = float(chosen.sample.suffix_gain)
            e2e_gain = float(chosen.sample.e2e_gain)
            suffix_acc = float(chosen.sample.acc_candidate_suffix)
            acc_final_no_fallback = float(acc_final)
            suffix_gain_no_fallback = float(suffix_gain)
            e2e_gain_no_fallback = float(e2e_gain)
            risk = float(chosen.risk)
            utility = float(chosen.utility)
            uncertainty = float(chosen.uncertainty)
            p_harm = float(chosen.p_harm)
            p_neutral = float(chosen.p_neutral)
            p_benefit = float(chosen.p_benefit)
            selected_x = np.asarray(chosen.sample.x, dtype=np.float64).reshape(-1)
            if bool(args.online_fallback):
                fb_thr = float(args.fallback_risk_threshold)
                if not np.isfinite(fb_thr):
                    fb_thr = float(thr.get("risk_threshold", 1.0))
                fallback_info = _online_fallback_accuracy(
                    selector=selector_final,
                    by_method=by_method,
                    anchor_method=str(args.anchor_method),
                    chosen=chosen,
                    proba_cols=proba_cols,
                    warmup_trials=int(w),
                    window_trials=int(args.fallback_window_trials or w),
                    fallback_risk_threshold=float(fb_thr),
                    fallback_koopman_threshold=float(args.fallback_koopman_threshold),
                    fallback_patience=int(args.fallback_patience),
                    n_chunks=int(args.koopman_chunks),
                    ridge_lambda=float(args.koopman_ridge),
                    spectral_gamma=float(args.koopman_gamma),
                    conflict_tau=float(args.conflict_tau),
                    conflict_lambda=float(args.conflict_lambda),
                )
                acc_final = float(fallback_info["acc_final_with_fallback"])
                suffix_acc = float(fallback_info["suffix_acc_with_fallback"])
                e2e_gain = float(fallback_info["e2e_gain_with_fallback"])
                suffix_gain = float(fallback_info["suffix_gain_with_fallback"])
            else:
                fallback_info = {
                    "fallback_at_trial": -1,
                    "fallback_risk": float("nan"),
                    "fallback_koopman": float("nan"),
                    "acc_final_with_fallback": acc_final,
                    "suffix_acc_with_fallback": suffix_acc,
                    "e2e_gain_with_fallback": e2e_gain,
                    "suffix_gain_with_fallback": suffix_gain,
                    "used_candidate_suffix_trials": int(chosen.sample.n_suffix),
                }

        best_pre = max(scored_test, key=lambda c: (float(c.utility), -float(c.risk)))
        oracle_e2e = float(anchor_sample.acc_oracle_e2e)
        rows.append(
            {
                "warmup_trials": int(w),
                "subject": int(test_subject),
                "selected_method": selected_method,
                "accept": int(accept),
                "acc_anchor_full": float(anchor_sample.acc_anchor_full),
                "acc_anchor_suffix": float(anchor_sample.acc_anchor_suffix),
                "acc_final": float(acc_final),
                "acc_final_no_fallback": float(acc_final_no_fallback),
                "suffix_acc_selected": float(suffix_acc),
                "e2e_gain": float(e2e_gain),
                "e2e_gain_no_fallback": float(e2e_gain_no_fallback),
                "suffix_gain": float(suffix_gain),
                "suffix_gain_no_fallback": float(suffix_gain_no_fallback),
                "neg_transfer_suffix": int(float(suffix_gain) < -float(args.neg_transfer_eps) and accept),
                "neg_transfer_e2e": int(float(e2e_gain) < -float(args.neg_transfer_eps) and accept),
                "oracle_e2e_acc": float(oracle_e2e),
                "oracle_gap_e2e": float(oracle_e2e - acc_final),
                "selector_risk": risk,
                "selector_utility": utility,
                "selector_uncertainty": uncertainty,
                "p_harm": p_harm,
                "p_neutral": p_neutral,
                "p_benefit": p_benefit,
                "selected_absolute_core": float(selected_x[0]),
                "selected_relative_core": float(selected_x[1]),
                "selected_koopman_temporal": float(selected_x[2]),
                "pre_best_method": str(best_pre.sample.method),
                "pre_best_risk": float(best_pre.risk),
                "pre_best_utility": float(best_pre.utility),
                "pre_best_absolute_core": float(best_pre.sample.x[0]),
                "pre_best_relative_core": float(best_pre.sample.x[1]),
                "pre_best_koopman_temporal": float(best_pre.sample.x[2]),
                "threshold_verified": bool(thr.get("verified", False)),
                "fallback_stage": str(thr.get("fallback_stage", "")),
                "risk_threshold": float(thr.get("risk_threshold", float("nan"))),
                "utility_threshold": float(thr.get("utility_threshold", float("nan"))),
                "dev_mean_e2e_gain": float(thr.get("dev_mean_e2e_gain", float("nan"))),
                "dev_accept_rate": float(thr.get("dev_accept_rate", float("nan"))),
                "cal_accept": float(thr.get("cal_accept", float("nan"))),
                "cal_harm": float(thr.get("cal_harm", float("nan"))),
                "cal_harm_ucb": float(thr.get("cal_harm_ucb", float("nan"))),
                **fallback_info,
            }
        )
        if int(args.progress_every) > 0 and (t_idx + 1) % int(args.progress_every) == 0:
            print(f"[warmup W={w}] {t_idx + 1}/{len(subjects)} subjects", flush=True)

    per_subject = pd.DataFrame(rows).sort_values(["warmup_trials", "subject"]).reset_index(drop=True)
    summary = {
        "warmup_trials": int(w),
        "method": str(args.method_name),
        "anchor_method": str(args.anchor_method),
        "n_subjects": int(per_subject.shape[0]),
        "mean_accuracy": float(per_subject["acc_final"].mean()),
        "anchor_mean_accuracy": float(per_subject["acc_anchor_full"].mean()),
        "mean_e2e_gain": float(per_subject["e2e_gain"].mean()),
        "mean_suffix_gain": float(per_subject["suffix_gain"].mean()),
        "worst_accuracy": float(per_subject["acc_final"].min()),
        "worst_subject": int(per_subject.loc[per_subject["acc_final"].idxmin(), "subject"]),
        "accept_rate": float(per_subject["accept"].mean()),
        "neg_transfer_suffix_rate": float(per_subject["neg_transfer_suffix"].mean()),
        "neg_transfer_e2e_rate": float(per_subject["neg_transfer_e2e"].mean()),
        "oracle_e2e_mean_accuracy": float(per_subject["oracle_e2e_acc"].mean()),
        "oracle_gap_e2e_mean": float(per_subject["oracle_gap_e2e"].mean()),
        "threshold_verified_rate": float(per_subject["threshold_verified"].mean()),
        "online_fallback": int(bool(args.online_fallback)),
        "fallback_rate_among_accepted": float(
            (per_subject.loc[per_subject["accept"] == 1, "fallback_at_trial"] >= 0).mean()
        )
        if int(per_subject["accept"].sum()) > 0
        else 0.0,
    }
    return per_subject, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warm-up Safe-TTS selector from merged predictions_all_methods.csv."
    )
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--anchor-method", type=str, default="eegnet_noea")
    p.add_argument("--candidate-methods", type=str, default=DEFAULT_CANDIDATES)
    p.add_argument("--warmup-trials", type=str, default=DEFAULT_WARMUP_GRID)
    p.add_argument("--method-name", type=str, default="safe-tts-warmup-koopman")
    p.add_argument("--date-prefix", type=str, default=datetime.now().strftime("%Y%m%d"))
    p.add_argument("--eval-subjects", type=str, default="ALL")

    p.add_argument("--risk-alpha", type=float, default=0.40)
    p.add_argument("--cp-delta", type=float, default=0.05)
    p.add_argument("--calib-fraction", type=float, default=0.25)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-accept-rate", type=float, default=0.0)
    p.add_argument("--min-utility-threshold", type=float, default=0.0)
    p.add_argument("--neg-transfer-eps", type=float, default=0.0)
    p.add_argument("--outcome-delta", type=float, default=0.02)

    p.add_argument("--conflict-tau", type=float, default=0.80)
    p.add_argument("--conflict-lambda", type=float, default=1.0)
    p.add_argument("--koopman-chunks", type=int, default=4)
    p.add_argument("--koopman-ridge", type=float, default=1e-3)
    p.add_argument("--koopman-gamma", type=float, default=0.25)
    p.add_argument("--min-suffix-trials", type=int, default=8)

    p.add_argument("--selector-hidden-dim", type=int, default=16)
    p.add_argument("--selector-epochs", type=int, default=80)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-weight-decay", type=float, default=1e-4)
    p.add_argument("--selector-lambda-rank", type=float, default=0.5)
    p.add_argument("--selector-lambda-kl", type=float, default=1e-3)
    p.add_argument("--selector-rank-margin", type=float, default=0.005)
    p.add_argument("--selector-rho", type=float, default=0.20)
    p.add_argument("--selector-eta", type=float, default=0.05)

    p.add_argument("--online-fallback", action="store_true")
    p.add_argument("--fallback-window-trials", type=int, default=0, help="0 means use W.")
    p.add_argument("--fallback-risk-threshold", type=float, default=float("nan"), help="NaN means reuse calibrated risk threshold.")
    p.add_argument("--fallback-koopman-threshold", type=float, default=float("inf"))
    p.add_argument("--fallback-patience", type=int, default=2)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--no-diagnostics", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for w in _parse_int_list(args.warmup_trials):
        per_subject, summary = run_one_w(args, w=int(w))
        all_rows.append(per_subject)
        summary_rows.append(summary)
        if not bool(args.no_diagnostics):
            per_subject.to_csv(out_dir / f"{args.date_prefix}_warmup_W{int(w)}_per_subject.csv", index=False)
    per_all = pd.concat(all_rows, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("warmup_trials").reset_index(drop=True)
    per_all.to_csv(out_dir / f"{args.date_prefix}_warmup_per_subject_selection.csv", index=False)
    summary_df.to_csv(out_dir / f"{args.date_prefix}_warmup_method_comparison.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"[done] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
