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

SCRIPTS_ROOT = REPO_ROOT / "scripts" if REPO_ROOT is not None else None
if SCRIPTS_ROOT is not None and str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from offline_safe_tta_multi_select_crc_from_predictions import (  # noqa: E402
    _infer_family as _legacy_infer_family,
    _parse_selector_views as _legacy_parse_selector_views,
    _selector_feature_bundle as _legacy_selector_feature_bundle,
)


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


BASE_FEATURE_NAMES = ("absolute_core", "relative_core", "koopman_temporal")
_COMPACT_VIEW_ALIASES = {
    "absolute": "absolute_core",
    "absolute_core": "absolute_core",
    "relative": "relative_core",
    "relative_core": "relative_core",
    "koopman": "koopman_temporal",
    "temporal": "koopman_temporal",
    "koopman_temporal": "koopman_temporal",
}


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_compact_selector_views(raw: str) -> list[str]:
    if not str(raw).strip():
        raise ValueError("--compact-selector-views cannot be empty")
    out: list[str] = []
    for token in _parse_csv_list(raw):
        key = str(token).strip().lower()
        if key not in _COMPACT_VIEW_ALIASES:
            allowed = ", ".join(sorted(_COMPACT_VIEW_ALIASES))
            raise ValueError(f"Unknown compact selector view {token!r}. Allowed: {allowed}")
        value = _COMPACT_VIEW_ALIASES[key]
        if value not in out:
            out.append(value)
    return out


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


def _effective_koopman_chunks(
    *,
    n_rows: int,
    max_chunks: int,
    dynamic: bool,
    disable_below: int,
) -> int:
    n_rows = int(n_rows)
    if int(disable_below) > 0 and n_rows < int(disable_below):
        return 0
    if bool(dynamic):
        return int(min(max(1, int(max_chunks)), max(2, n_rows // 8)))
    return int(max(1, int(max_chunks)))


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
    mu = np.mean(z, axis=0, keepdims=True)
    sigma = np.std(z, axis=0, keepdims=True)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    z = (z - mu) / sigma
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
    if int(n_chunks) < 2:
        temporal = 0.0
    else:
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


def _select_compact_features(
    x: np.ndarray,
    selected_views: list[str],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[int, int]]]:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(BASE_FEATURE_NAMES):
        raise RuntimeError(f"Expected {len(BASE_FEATURE_NAMES)} compact features, got {values.shape[0]}")
    index = {name: i for i, name in enumerate(BASE_FEATURE_NAMES)}
    cols: list[int] = []
    names: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for name in selected_views:
        if name not in index:
            raise RuntimeError(f"Unknown compact feature name {name!r}")
        slices[name] = (len(cols), len(cols) + 1)
        cols.append(int(index[name]))
        names.append(str(name))
    if not cols:
        raise RuntimeError("At least one compact selector view is required.")
    return values[np.asarray(cols, dtype=int)], tuple(names), slices


def _method_onehot(method: str, candidate_methods: list[str]) -> np.ndarray:
    z = np.zeros((len(candidate_methods),), dtype=np.float64)
    if str(method) in candidate_methods:
        z[int(candidate_methods.index(str(method)))] = 1.0
    return z


def _augment_feature_vector(
    x: np.ndarray,
    *,
    method: str,
    candidate_methods: list[str],
    method_prior_mode: str,
) -> np.ndarray:
    base = np.asarray(x, dtype=np.float64).reshape(-1)
    mode = str(method_prior_mode).strip().lower()
    if mode == "none":
        return base
    if mode == "onehot":
        return np.concatenate([base, _method_onehot(method, candidate_methods)], axis=0)
    raise ValueError(f"Unsupported method_prior_mode={method_prior_mode!r}")


def _feature_metadata(
    *,
    base_feature_names: tuple[str, ...],
    base_view_slices: dict[str, tuple[int, int]],
    candidate_methods: list[str],
    method_prior_mode: str,
) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    names: list[str] = list(base_feature_names)
    view_slices: dict[str, tuple[int, int]] = dict(base_view_slices)
    if str(method_prior_mode).strip().lower() == "onehot":
        start = len(names)
        names.extend([f"method={m}" for m in candidate_methods])
        view_slices["strategy_prior"] = (start, start + len(candidate_methods))
    return tuple(names), view_slices


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
    base_feature_names: tuple[str, ...],
    base_view_slices: dict[str, tuple[int, int]],
    candidate_methods: list[str],
    method_prior_mode: str,
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
            rows.append(
                _augment_feature_vector(
                    sample.x,
                    method=str(sample.method),
                    candidate_methods=candidate_methods,
                    method_prior_mode=str(method_prior_mode),
                )
            )
            states.append(_state_from_gain(sample.suffix_gain, delta=float(outcome_delta)))
            gains.append(float(sample.suffix_gain))
            groups.append(int(s))
    if not rows:
        raise RuntimeError("No warm-up training rows.")
    x = np.vstack(rows).astype(np.float64, copy=False)
    y_state = np.asarray(states, dtype=int)
    y_gain = np.asarray(gains, dtype=np.float64)
    group_ids = np.asarray(groups, dtype=int)
    feature_names, view_slices = _feature_metadata(
        base_feature_names=base_feature_names,
        base_view_slices=base_view_slices,
        candidate_methods=candidate_methods,
        method_prior_mode=str(method_prior_mode),
    )
    return train_evidential_selector(
        x,
        y_state,
        y_gain,
        group_ids,
        feature_names=feature_names,
        view_slices=view_slices,
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


def _score_samples(
    selector,
    samples: list[Sample],
    *,
    candidate_methods: list[str],
    method_prior_mode: str,
) -> list[ScoredCandidate]:
    if not samples:
        return []
    x = np.vstack(
        [
            _augment_feature_vector(
                s.x,
                method=str(s.method),
                candidate_methods=candidate_methods,
                method_prior_mode=str(method_prior_mode),
            )
            for s in samples
        ]
    )
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
    risk_only_selection: bool,
    selection_policy: str,
    default_candidate_method: str,
    switch_utility_margin: float,
) -> ScoredCandidate | None:
    def is_feasible(c: ScoredCandidate) -> bool:
        utility_ok = bool(risk_only_selection) or float(c.utility) >= float(utility_threshold)
        return float(c.risk) <= float(risk_threshold) and utility_ok

    feasible = [
        c
        for c in scored
        if is_feasible(c)
    ]
    if not feasible:
        return None

    policy = str(selection_policy).strip().lower()
    if policy == "free":
        return max(feasible, key=lambda c: (float(c.utility), -float(c.risk), float(c.sample.e2e_gain)))

    if policy != "default_veto_switch":
        raise ValueError(f"Unsupported selection_policy={selection_policy!r}")

    default_method = str(default_candidate_method).strip()
    default = next((c for c in scored if str(c.sample.method) == default_method), None)
    if default is None:
        return max(feasible, key=lambda c: (float(c.utility), -float(c.risk), float(c.sample.e2e_gain)))

    margin = float(switch_utility_margin)
    switchers = [
        c
        for c in feasible
        if str(c.sample.method) != default_method and float(c.utility) >= float(default.utility) + margin
    ]
    if is_feasible(default):
        if switchers:
            return max(switchers, key=lambda c: (float(c.utility), -float(c.risk), float(c.sample.e2e_gain)))
        return default
    if not switchers:
        return None
    return max(switchers, key=lambda c: (float(c.utility), -float(c.risk), float(c.sample.e2e_gain)))


def _evaluate_subject_actions(
    scored_by_subject: dict[int, list[ScoredCandidate]],
    subjects: Iterable[int],
    *,
    risk_threshold: float,
    utility_threshold: float,
    neg_eps: float,
    risk_only_selection: bool,
    selection_policy: str,
    default_candidate_method: str,
    switch_utility_margin: float,
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
            risk_only_selection=bool(risk_only_selection),
            selection_policy=str(selection_policy),
            default_candidate_method=str(default_candidate_method),
            switch_utility_margin=float(switch_utility_margin),
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
        "tail_e2e_gain": _tail_mean(gains, frac=0.10),
        "tail_suffix_gain": _tail_mean(suffix_gains, frac=0.10),
        "accept_rate": float(accepted / max(1, n)),
        "cond_harm_rate": float(harm / accepted) if accepted > 0 else 0.0,
    }


def _tail_mean(values: Iterable[float], frac: float = 0.10) -> float:
    arr = np.sort(np.asarray(list(values), dtype=np.float64).reshape(-1))
    if arr.size == 0:
        return 0.0
    k = max(1, int(math.ceil(float(frac) * int(arr.size))))
    return float(np.mean(arr[:k]))


def _threshold_candidates(
    scored_by_subject: dict[int, list[ScoredCandidate]],
    *,
    min_utility_threshold: float,
    risk_only_selection: bool,
) -> tuple[np.ndarray, np.ndarray]:
    risks = np.asarray([c.risk for rows in scored_by_subject.values() for c in rows], dtype=np.float64)
    utils = np.asarray([c.utility for rows in scored_by_subject.values() for c in rows], dtype=np.float64)
    if risks.size == 0 or utils.size == 0:
        return np.asarray([0.0]), np.asarray([math.inf])
    r_grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 21), np.quantile(risks, np.linspace(0.0, 1.0, 21))]))
    if bool(risk_only_selection):
        return np.asarray(r_grid, dtype=np.float64), np.asarray([float(min_utility_threshold)], dtype=np.float64)
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
    risk_only_selection: bool,
    selection_policy: str,
    default_candidate_method: str,
    switch_utility_margin: float,
    dev_objective: str,
    lambda_tail: float,
    lambda_accept: float,
    tail_frac: float,
) -> dict[str, float | bool | str]:
    r_grid, q_grid = _threshold_candidates(
        {s: oof_scored[int(s)] for s in dev_subjects},
        min_utility_threshold=float(min_utility_threshold),
        risk_only_selection=bool(risk_only_selection),
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
                risk_only_selection=bool(risk_only_selection),
                selection_policy=str(selection_policy),
                default_candidate_method=str(default_candidate_method),
                switch_utility_margin=float(switch_utility_margin),
            )
            if float(m["accept_rate"]) < float(min_accept_rate):
                continue
            if str(dev_objective).strip().lower() == "tail_accept":
                obj = (
                    float(m["mean_e2e_gain"])
                    + float(lambda_tail) * _tail_mean(
                        [
                            float(
                                _select_candidate(
                                    oof_scored[int(s)],
                                    risk_threshold=float(r_thr),
                                    utility_threshold=float(q_thr),
                                    risk_only_selection=bool(risk_only_selection),
                                    selection_policy=str(selection_policy),
                                    default_candidate_method=str(default_candidate_method),
                                    switch_utility_margin=float(switch_utility_margin),
                                ).sample.e2e_gain
                            )
                            if _select_candidate(
                                oof_scored[int(s)],
                                risk_threshold=float(r_thr),
                                utility_threshold=float(q_thr),
                                risk_only_selection=bool(risk_only_selection),
                                selection_policy=str(selection_policy),
                                default_candidate_method=str(default_candidate_method),
                                switch_utility_margin=float(switch_utility_margin),
                            )
                            is not None
                            else 0.0
                            for s in dev_subjects
                        ],
                        frac=float(tail_frac),
                    )
                    + float(lambda_accept) * float(m["accept_rate"])
                )
            else:
                obj = float(m["mean_e2e_gain"])
            if best is None or obj > float(best["dev_objective_value"]):
                best = {
                    "risk_threshold": float(r_thr),
                    "utility_threshold": float(q_thr),
                    "dev_objective_value": float(obj),
                    "dev_mean_e2e_gain": float(m["mean_e2e_gain"]),
                    "dev_mean_suffix_gain": float(m["mean_suffix_gain"]),
                    "dev_tail_e2e_gain": float(_tail_mean(
                        [
                            float(
                                _select_candidate(
                                    oof_scored[int(s)],
                                    risk_threshold=float(r_thr),
                                    utility_threshold=float(q_thr),
                                    risk_only_selection=bool(risk_only_selection),
                                    selection_policy=str(selection_policy),
                                    default_candidate_method=str(default_candidate_method),
                                    switch_utility_margin=float(switch_utility_margin),
                                ).sample.e2e_gain
                            )
                            if _select_candidate(
                                oof_scored[int(s)],
                                risk_threshold=float(r_thr),
                                utility_threshold=float(q_thr),
                                risk_only_selection=bool(risk_only_selection),
                                selection_policy=str(selection_policy),
                                default_candidate_method=str(default_candidate_method),
                                switch_utility_margin=float(switch_utility_margin),
                            )
                            is not None
                            else 0.0
                            for s in dev_subjects
                        ],
                        frac=float(tail_frac),
                    )),
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
        risk_only_selection=bool(risk_only_selection),
        selection_policy=str(selection_policy),
        default_candidate_method=str(default_candidate_method),
        switch_utility_margin=float(switch_utility_margin),
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
    warmup_feature_set: str,
    compact_selector_views: list[str],
    legacy_selector_views: list[str],
    legacy_feature_mode: str,
    legacy_dynamic_chunks: int,
    legacy_stochastic_bootstrap_rounds: int,
    legacy_stochastic_bootstrap_seed: int,
    n_chunks: int,
    dynamic_koopman_chunks: bool,
    disable_koopman_below_w: int,
    ridge_lambda: float,
    spectral_gamma: float,
    conflict_tau: float,
    conflict_lambda: float,
    min_suffix_trials: int,
) -> tuple[dict[int, list[Sample]], tuple[str, ...], dict[str, tuple[int, int]]]:
    samples: dict[int, list[Sample]] = {}
    effective_chunks = _effective_koopman_chunks(
        n_rows=int(warmup_trials),
        max_chunks=int(n_chunks),
        dynamic=bool(dynamic_koopman_chunks),
        disable_below=int(disable_koopman_below_w),
    )
    feature_names: tuple[str, ...] | None = None
    view_slices: dict[str, tuple[int, int]] | None = None
    feature_set = str(warmup_feature_set).strip().lower()
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
            df_a_prefix = df_a.iloc[:w].reset_index(drop=True)
            df_c_prefix = df_c.iloc[:w].reset_index(drop=True)
            if feature_set == "compact":
                x_all = _core_features(
                    df_anchor=df_a_prefix,
                    df_candidate=df_c_prefix,
                    proba_cols=proba_cols,
                    n_chunks=int(effective_chunks),
                    ridge_lambda=float(ridge_lambda),
                    spectral_gamma=float(spectral_gamma),
                    conflict_tau=float(conflict_tau),
                    conflict_lambda=float(conflict_lambda),
                )
                x, cur_names, cur_slices = _select_compact_features(
                    x_all,
                    selected_views=list(compact_selector_views),
                )
            elif feature_set == "legacy":
                x, cur_names, cur_slices = _legacy_selector_feature_bundle(
                    p_id=df_a_prefix[proba_cols].to_numpy(np.float64),
                    p_c=df_c_prefix[proba_cols].to_numpy(np.float64),
                    y_pred_id=df_a_prefix["y_pred"].to_numpy(object),
                    y_pred_c=df_c_prefix["y_pred"].to_numpy(object),
                    anchor_family=_legacy_infer_family(str(anchor_method)),
                    cand_family=_legacy_infer_family(str(m)),
                    n_classes=len(proba_cols),
                    feature_mode=str(legacy_feature_mode),
                    selector_views=list(legacy_selector_views),
                    dynamic_chunks=int(legacy_dynamic_chunks),
                    stochastic_bootstrap_rounds=int(legacy_stochastic_bootstrap_rounds),
                    stochastic_bootstrap_seed=int(legacy_stochastic_bootstrap_seed) + 1009 * int(s),
                )
            else:
                raise ValueError(f"Unsupported warmup_feature_set={warmup_feature_set!r}")
            if feature_names is None:
                feature_names = tuple(cur_names)
                view_slices = dict(cur_slices)
            elif tuple(cur_names) != feature_names or dict(cur_slices) != dict(view_slices or {}):
                raise RuntimeError("Warm-up feature metadata changed across samples.")
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
    if feature_names is None or view_slices is None:
        raise RuntimeError("No warm-up features were built.")
    return samples, feature_names, view_slices


def _online_fallback_accuracy(
    *,
    selector,
    by_method: dict[str, dict[int, pd.DataFrame]],
    anchor_method: str,
    chosen: ScoredCandidate,
    proba_cols: list[str],
    candidate_methods: list[str],
    method_prior_mode: str,
    warmup_feature_set: str,
    compact_selector_views: list[str],
    legacy_selector_views: list[str],
    legacy_feature_mode: str,
    legacy_dynamic_chunks: int,
    legacy_stochastic_bootstrap_rounds: int,
    legacy_stochastic_bootstrap_seed: int,
    warmup_trials: int,
    window_trials: int,
    fallback_risk_threshold: float,
    fallback_koopman_threshold: float,
    fallback_patience: int,
    n_chunks: int,
    dynamic_koopman_chunks: bool,
    disable_koopman_below_w: int,
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
        effective_chunks = _effective_koopman_chunks(
            n_rows=int(obs_stop - obs_start),
            max_chunks=int(n_chunks),
            dynamic=bool(dynamic_koopman_chunks),
            disable_below=int(disable_koopman_below_w),
        )
        df_a_win = df_a.iloc[obs_start:obs_stop].reset_index(drop=True)
        df_c_win = df_c.iloc[obs_start:obs_stop].reset_index(drop=True)
        if str(warmup_feature_set).strip().lower() == "legacy":
            x_core, _names, _slices = _legacy_selector_feature_bundle(
                p_id=df_a_win[proba_cols].to_numpy(np.float64),
                p_c=df_c_win[proba_cols].to_numpy(np.float64),
                y_pred_id=df_a_win["y_pred"].to_numpy(object),
                y_pred_c=df_c_win["y_pred"].to_numpy(object),
                anchor_family=_legacy_infer_family(str(anchor_method)),
                cand_family=_legacy_infer_family(str(method)),
                n_classes=len(proba_cols),
                feature_mode=str(legacy_feature_mode),
                selector_views=list(legacy_selector_views),
                dynamic_chunks=int(legacy_dynamic_chunks),
                stochastic_bootstrap_rounds=int(legacy_stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(legacy_stochastic_bootstrap_seed) + 1009 * int(s),
            )
            koop = float("nan")
        else:
            x_all = _core_features(
                df_anchor=df_a_win,
                df_candidate=df_c_win,
                proba_cols=proba_cols,
                n_chunks=int(effective_chunks),
                ridge_lambda=float(ridge_lambda),
                spectral_gamma=float(spectral_gamma),
                conflict_tau=float(conflict_tau),
                conflict_lambda=float(conflict_lambda),
            )
            x_core, x_names, _x_slices = _select_compact_features(
                x_all,
                selected_views=list(compact_selector_views),
            )
            if "koopman_temporal" in x_names:
                koop = float(x_core.reshape(-1)[int(x_names.index("koopman_temporal"))])
            else:
                koop = float("nan")
        x_win = _augment_feature_vector(
            x_core,
            method=method,
            candidate_methods=candidate_methods,
            method_prior_mode=str(method_prior_mode),
        ).reshape(1, -1)
        stats = selector.predict_stats(x_win)
        risk = float(np.asarray(stats["risk"]).reshape(-1)[0])
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
    if str(args.selection_policy).strip().lower() == "default_veto_switch":
        default_method = str(args.default_candidate_method).strip() or (cand_methods[0] if cand_methods else "")
        if default_method not in cand_methods:
            raise RuntimeError(f"default_candidate_method={default_method!r} is not in candidate_methods={cand_methods}")
    else:
        default_method = str(args.default_candidate_method).strip() or (cand_methods[0] if cand_methods else "")
    methods = [str(args.anchor_method)] + cand_methods
    by_method = _build_subject_maps(df, methods=methods)
    _validate_alignment(by_method, methods=methods, subjects=subjects)
    samples_by_subject, base_feature_names, base_view_slices = _build_samples_for_w(
        by_method=by_method,
        anchor_method=str(args.anchor_method),
        candidate_methods=cand_methods,
        subjects=subjects,
        proba_cols=proba_cols,
        warmup_trials=int(w),
        warmup_feature_set=str(args.warmup_feature_set),
        compact_selector_views=_parse_compact_selector_views(str(args.compact_selector_views)),
        legacy_selector_views=_legacy_parse_selector_views(str(args.legacy_selector_views)),
        legacy_feature_mode=str(args.legacy_feature_mode),
        legacy_dynamic_chunks=int(args.legacy_dynamic_chunks),
        legacy_stochastic_bootstrap_rounds=int(args.legacy_stochastic_bootstrap_rounds),
        legacy_stochastic_bootstrap_seed=int(args.legacy_stochastic_bootstrap_seed),
        n_chunks=int(args.koopman_chunks),
        dynamic_koopman_chunks=bool(args.dynamic_koopman_chunks),
        disable_koopman_below_w=int(args.disable_koopman_below_w),
        ridge_lambda=float(args.koopman_ridge),
        spectral_gamma=float(args.koopman_gamma),
        conflict_tau=float(args.conflict_tau),
        conflict_lambda=float(args.conflict_lambda),
        min_suffix_trials=int(args.min_suffix_trials),
    )
    subjects = sorted(samples_by_subject)
    if len(subjects) < 4:
        raise RuntimeError(f"Need at least 4 valid subjects after warm-up filtering, got {len(subjects)}")
    base_feature_index = {str(name): int(i) for i, name in enumerate(base_feature_names)}

    def _feature_value(vec: np.ndarray, name: str) -> float:
        idx = base_feature_index.get(str(name))
        if idx is None:
            return float("nan")
        values = np.asarray(vec, dtype=np.float64).reshape(-1)
        if idx >= int(values.shape[0]):
            return float("nan")
        return float(values[idx])

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
                base_feature_names=base_feature_names,
                base_view_slices=base_view_slices,
                candidate_methods=cand_methods,
                method_prior_mode=str(args.method_prior_mode),
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
                oof[int(s)] = _score_samples(
                    selector_fold,
                    samples_by_subject[int(s)],
                    candidate_methods=cand_methods,
                    method_prior_mode=str(args.method_prior_mode),
                )

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
            risk_only_selection=bool(args.risk_only_selection),
            selection_policy=str(args.selection_policy),
            default_candidate_method=str(default_method),
            switch_utility_margin=float(args.switch_utility_margin),
            dev_objective=str(args.dev_objective),
            lambda_tail=float(args.lambda_tail),
            lambda_accept=float(args.lambda_accept),
            tail_frac=float(args.tail_frac),
        )
        selector_final = _train_selector(
            samples_by_subject,
            fit_subjects,
            base_feature_names=base_feature_names,
            base_view_slices=base_view_slices,
            candidate_methods=cand_methods,
            method_prior_mode=str(args.method_prior_mode),
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
        scored_test = _score_samples(
            selector_final,
            samples_by_subject[int(test_subject)],
            candidate_methods=cand_methods,
            method_prior_mode=str(args.method_prior_mode),
        )
        if bool(thr.get("verified", False)):
            chosen = _select_candidate(
                scored_test,
                risk_threshold=float(thr["risk_threshold"]),
                utility_threshold=float(thr["utility_threshold"]),
                risk_only_selection=bool(args.risk_only_selection),
                selection_policy=str(args.selection_policy),
                default_candidate_method=str(default_method),
                switch_utility_margin=float(args.switch_utility_margin),
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
            selected_x = np.full((len(base_feature_names),), float("nan"), dtype=np.float64)
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
                    candidate_methods=cand_methods,
                    method_prior_mode=str(args.method_prior_mode),
                    warmup_feature_set=str(args.warmup_feature_set),
                    compact_selector_views=_parse_compact_selector_views(str(args.compact_selector_views)),
                    legacy_selector_views=_legacy_parse_selector_views(str(args.legacy_selector_views)),
                    legacy_feature_mode=str(args.legacy_feature_mode),
                    legacy_dynamic_chunks=int(args.legacy_dynamic_chunks),
                    legacy_stochastic_bootstrap_rounds=int(args.legacy_stochastic_bootstrap_rounds),
                    legacy_stochastic_bootstrap_seed=int(args.legacy_stochastic_bootstrap_seed),
                    warmup_trials=int(w),
                    window_trials=int(args.fallback_window_trials or w),
                    fallback_risk_threshold=float(fb_thr),
                    fallback_koopman_threshold=float(args.fallback_koopman_threshold),
                    fallback_patience=int(args.fallback_patience),
                    n_chunks=int(args.koopman_chunks),
                    dynamic_koopman_chunks=bool(args.dynamic_koopman_chunks),
                    disable_koopman_below_w=int(args.disable_koopman_below_w),
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
                "selected_absolute_core": _feature_value(selected_x, "absolute_core"),
                "selected_relative_core": _feature_value(selected_x, "relative_core"),
                "selected_koopman_temporal": _feature_value(selected_x, "koopman_temporal"),
                "pre_best_method": str(best_pre.sample.method),
                "pre_best_risk": float(best_pre.risk),
                "pre_best_utility": float(best_pre.utility),
                "pre_best_absolute_core": _feature_value(best_pre.sample.x, "absolute_core"),
                "pre_best_relative_core": _feature_value(best_pre.sample.x, "relative_core"),
                "pre_best_koopman_temporal": _feature_value(best_pre.sample.x, "koopman_temporal"),
                "threshold_verified": bool(thr.get("verified", False)),
                "fallback_stage": str(thr.get("fallback_stage", "")),
                "risk_threshold": float(thr.get("risk_threshold", float("nan"))),
                "utility_threshold": float(thr.get("utility_threshold", float("nan"))),
                "dev_mean_e2e_gain": float(thr.get("dev_mean_e2e_gain", float("nan"))),
                "dev_accept_rate": float(thr.get("dev_accept_rate", float("nan"))),
                "cal_accept": float(thr.get("cal_accept", float("nan"))),
                "cal_harm": float(thr.get("cal_harm", float("nan"))),
                "cal_harm_ucb": float(thr.get("cal_harm_ucb", float("nan"))),
                "dev_objective_value": float(thr.get("dev_objective_value", float("nan"))),
                "dev_tail_e2e_gain": float(thr.get("dev_tail_e2e_gain", float("nan"))),
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
        "method_prior_mode": str(args.method_prior_mode),
        "warmup_feature_set": str(args.warmup_feature_set),
        "compact_selector_views": str(args.compact_selector_views),
        "legacy_selector_views": str(args.legacy_selector_views),
        "selection_policy": str(args.selection_policy),
        "default_candidate_method": str(default_method),
        "risk_only_selection": int(bool(args.risk_only_selection)),
        "dev_objective": str(args.dev_objective),
        "candidate_methods": ",".join(cand_methods),
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
    p.add_argument("--warmup-feature-set", type=str, choices=("compact", "legacy"), default="compact")
    p.add_argument(
        "--compact-selector-views",
        type=str,
        default="absolute,relative,koopman",
        help="Comma-separated compact views. Choices/aliases: absolute,relative,koopman.",
    )
    p.add_argument("--legacy-selector-views", type=str, default="stats,decision,relative,dynamic")
    p.add_argument("--legacy-feature-mode", type=str, choices=("delta", "anchor_delta"), default="delta")
    p.add_argument("--legacy-dynamic-chunks", type=int, default=4)
    p.add_argument("--legacy-stochastic-bootstrap-rounds", type=int, default=16)
    p.add_argument("--legacy-stochastic-bootstrap-seed", type=int, default=0)
    p.add_argument("--method-prior-mode", type=str, choices=("none", "onehot"), default="none")
    p.add_argument("--selection-policy", type=str, choices=("free", "default_veto_switch"), default="free")
    p.add_argument("--default-candidate-method", type=str, default="")
    p.add_argument("--switch-utility-margin", type=float, default=0.0)
    p.add_argument("--risk-only-selection", action="store_true")

    p.add_argument("--risk-alpha", type=float, default=0.40)
    p.add_argument("--cp-delta", type=float, default=0.05)
    p.add_argument("--calib-fraction", type=float, default=0.25)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-accept-rate", type=float, default=0.0)
    p.add_argument("--min-utility-threshold", type=float, default=0.0)
    p.add_argument("--neg-transfer-eps", type=float, default=0.0)
    p.add_argument("--outcome-delta", type=float, default=0.02)
    p.add_argument("--dev-objective", type=str, choices=("mean", "tail_accept"), default="mean")
    p.add_argument("--lambda-tail", type=float, default=0.0)
    p.add_argument("--lambda-accept", type=float, default=0.0)
    p.add_argument("--tail-frac", type=float, default=0.10)

    p.add_argument("--conflict-tau", type=float, default=0.80)
    p.add_argument("--conflict-lambda", type=float, default=1.0)
    p.add_argument("--koopman-chunks", type=int, default=4)
    p.add_argument("--dynamic-koopman-chunks", action="store_true")
    p.add_argument("--disable-koopman-below-w", type=int, default=0)
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
