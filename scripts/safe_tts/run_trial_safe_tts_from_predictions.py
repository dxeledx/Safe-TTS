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
COMPACT_FEATURE_NAMES = ("absolute_core", "relative_core", "koopman_temporal")
TRIAL_FEATURE_GROUPS = {
    "absolute_view": (
        "absolute_core",
        "candidate_confidence",
        "candidate_margin",
    ),
    "relative_view": (
        "relative_core",
        "js_drift",
        "confidence_delta",
        "entropy_delta",
        "prediction_disagree",
        "high_conflict",
    ),
    "temporal_view": (
        "candidate_confidence_step",
        "candidate_prob_step",
        "candidate_flip",
        "relative_js_step",
        "koopman_temporal",
    ),
}
FEATURE_ALIASES = {
}
FEATURE_PRESETS = {
    "compact": COMPACT_FEATURE_NAMES,
    "rich": tuple(name for names in TRIAL_FEATURE_GROUPS.values() for name in names),
    "core": COMPACT_FEATURE_NAMES,
}


@dataclass(frozen=True)
class TrialSample:
    subject: int
    trial: int
    method: str
    x: np.ndarray
    gain: float
    y_true: object
    anchor_pred: object
    candidate_pred: object
    anchor_correct: int
    candidate_correct: int


@dataclass(frozen=True)
class ScoredTrial:
    sample: TrialSample
    risk: float
    utility: float
    uncertainty: float
    p_harm: float
    p_neutral: float
    p_benefit: float


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _canonical_feature_name(raw: str) -> str:
    name = str(raw).strip()
    return FEATURE_ALIASES.get(name, name)


def _parse_trial_feature_names(raw: str, preset: str) -> tuple[str, ...]:
    preset_key = str(preset).strip().lower()
    if preset_key not in FEATURE_PRESETS:
        raise ValueError(f"Unknown trial feature preset {preset!r}. Available: {sorted(FEATURE_PRESETS)}")
    if not str(raw).strip() or str(raw).strip().upper() == "PRESET":
        names = tuple(_canonical_feature_name(x) for x in FEATURE_PRESETS[preset_key])
    else:
        names = tuple(_canonical_feature_name(x) for x in _parse_csv_list(raw))
    allowed = {name for group in TRIAL_FEATURE_GROUPS.values() for name in group}
    missing = [name for name in names if name not in allowed]
    if missing:
        raise ValueError(f"Unknown trial feature names: {missing}. Allowed: {sorted(allowed)}")
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        raise ValueError("At least one trial feature is required.")
    grouped = [
        name
        for group_names in TRIAL_FEATURE_GROUPS.values()
        for name in group_names
        if name in deduped
    ]
    return tuple(grouped)


def _view_slices_for_features(feature_names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for view_name, group_names in TRIAL_FEATURE_GROUPS.items():
        n_view = sum(1 for name in feature_names if name in set(group_names))
        if n_view > 0:
            slices[str(view_name)] = (int(offset), int(offset + n_view))
            offset += int(n_view)
    if offset != len(feature_names):
        raise RuntimeError("Feature/view slice construction failed.")
    return slices


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
    return p / np.sum(p, axis=-1, keepdims=True)


def _entropy_vec(p: np.ndarray) -> float:
    p = _safe_probs(np.asarray(p, dtype=np.float64).reshape(1, -1))[0]
    return float(-np.sum(p * np.log(p)))


def _js_vec(p0: np.ndarray, p1: np.ndarray) -> float:
    p0 = _safe_probs(np.asarray(p0, dtype=np.float64).reshape(1, -1))[0]
    p1 = _safe_probs(np.asarray(p1, dtype=np.float64).reshape(1, -1))[0]
    mid = 0.5 * (p0 + p1)
    kl0 = float(np.sum(p0 * (np.log(p0) - np.log(mid))))
    kl1 = float(np.sum(p1 * (np.log(p1) - np.log(mid))))
    return 0.5 * (kl0 + kl1)


def _top2_margin(p: np.ndarray) -> float:
    p = _safe_probs(np.asarray(p, dtype=np.float64).reshape(1, -1))[0]
    if p.size <= 1:
        return 0.0
    top2 = np.sort(p)[-2:]
    return float(top2[-1] - top2[-2])


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
    slices = dict(base_view_slices)
    if str(method_prior_mode).strip().lower() == "onehot":
        start = len(names)
        names.extend([f"method={m}" for m in candidate_methods])
        slices["strategy_prior"] = (start, start + len(candidate_methods))
    return tuple(names), slices


def _trial_feature_values(
    *,
    p_anchor: np.ndarray,
    p_candidate: np.ndarray,
    pred_anchor: object,
    pred_candidate: object,
    prev_trace: dict[str, object] | None,
    conflict_tau: float,
    conflict_lambda: float,
) -> tuple[dict[str, float], dict[str, object]]:
    p_anchor = _safe_probs(np.asarray(p_anchor, dtype=np.float64).reshape(1, -1))[0]
    p_candidate = _safe_probs(np.asarray(p_candidate, dtype=np.float64).reshape(1, -1))[0]
    c = int(p_candidate.shape[0])
    log_c = max(math.log(float(c)), 1e-12)
    candidate_entropy = _entropy_vec(p_candidate) / log_c
    anchor_entropy = _entropy_vec(p_anchor) / log_c
    candidate_confidence = float(np.max(p_candidate))
    anchor_confidence = float(np.max(p_anchor))
    candidate_certainty = 1.0 - float(candidate_entropy)
    js_drift = _js_vec(p_anchor, p_candidate) / math.log(2.0)
    high_conflict = float(
        pred_anchor != pred_candidate
        and float(np.max(p_anchor)) >= float(conflict_tau)
        and float(np.max(p_candidate)) >= float(conflict_tau)
    )
    relative_core = float(js_drift) * (1.0 + float(conflict_lambda) * high_conflict)
    state = np.asarray([candidate_certainty, relative_core, high_conflict], dtype=np.float64)
    if prev_trace is None:
        confidence_step = 0.0
        prob_step = 0.0
        candidate_flip = 0.0
        js_step = 0.0
        temporal = 0.0
    else:
        prev_prob = np.asarray(prev_trace.get("p_candidate", p_candidate), dtype=np.float64).reshape(-1)
        prev_state = np.asarray(prev_trace.get("state", state), dtype=np.float64).reshape(-1)
        prev_conf = float(prev_trace.get("candidate_confidence", candidate_confidence))
        prev_js = float(prev_trace.get("js_drift", js_drift))
        prev_pred = prev_trace.get("pred_candidate", pred_candidate)
        confidence_step = abs(float(candidate_confidence) - prev_conf)
        prob_step = float(np.linalg.norm(p_candidate - prev_prob) / math.sqrt(float(c)))
        candidate_flip = float(pred_candidate != prev_pred)
        js_step = abs(float(js_drift) - prev_js)
        temporal = float(np.linalg.norm(state - prev_state) / math.sqrt(3.0))
    values = {
        "absolute_core": float(candidate_certainty),
        "candidate_confidence": float(candidate_confidence),
        "candidate_margin": _top2_margin(p_candidate),
        "relative_core": float(relative_core),
        "js_drift": float(js_drift),
        "confidence_delta": float(candidate_confidence - anchor_confidence),
        "entropy_delta": float(candidate_entropy - anchor_entropy),
        "prediction_disagree": float(pred_anchor != pred_candidate),
        "high_conflict": float(high_conflict),
        "candidate_confidence_step": float(confidence_step),
        "candidate_prob_step": float(prob_step),
        "candidate_flip": float(candidate_flip),
        "relative_js_step": float(js_step),
        "koopman_temporal": float(temporal),
    }
    trace = {
        "p_candidate": p_candidate,
        "pred_candidate": pred_candidate,
        "candidate_confidence": float(candidate_confidence),
        "js_drift": float(js_drift),
        "state": state,
    }
    return values, trace


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


def _load_predictions(path: Path, anchor_method: str, candidate_methods: str) -> tuple[pd.DataFrame, list[str], list[str], list[int]]:
    df = pd.read_csv(path)
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in predictions CSV: {missing}")
    proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
    if not proba_cols:
        raise RuntimeError("No proba_* columns found.")
    df = df.copy()
    df["method"] = df["method"].astype(str)
    df["subject"] = df["subject"].astype(int)
    df["trial"] = df["trial"].astype(int)
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
    subjects = sorted(int(s) for s in df.loc[df["method"] == str(anchor_method), "subject"].unique().tolist())
    return df, proba_cols, cands, subjects


def _build_subject_maps(
    df: pd.DataFrame,
    *,
    methods: list[str],
) -> dict[str, dict[int, pd.DataFrame]]:
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for method in methods:
        method_df = df[df["method"] == str(method)].copy()
        out[str(method)] = {
            int(s): g.sort_values("trial").reset_index(drop=True)
            for s, g in method_df.groupby("subject", sort=True)
        }
    return out


def _validate_alignment(
    by_method: dict[str, dict[int, pd.DataFrame]],
    *,
    methods: list[str],
    subjects: list[int],
) -> None:
    anchor = str(methods[0])
    for s in subjects:
        if int(s) not in by_method[anchor]:
            raise RuntimeError(f"Missing anchor subject {s}")
        ref = by_method[anchor][int(s)][["trial", "y_true"]].reset_index(drop=True)
        for m in methods[1:]:
            if int(s) not in by_method[str(m)]:
                raise RuntimeError(f"Missing method={m} subject={s}")
            cur = by_method[str(m)][int(s)][["trial", "y_true"]].reset_index(drop=True)
            if ref.shape[0] != cur.shape[0] or not ref.equals(cur):
                raise RuntimeError(f"Trial/y_true mismatch for method={m} subject={s}")


def _build_trial_samples(
    *,
    by_method: dict[str, dict[int, pd.DataFrame]],
    anchor_method: str,
    candidate_methods: list[str],
    subjects: list[int],
    proba_cols: list[str],
    feature_names: tuple[str, ...],
    conflict_tau: float,
    conflict_lambda: float,
) -> dict[int, list[TrialSample]]:
    samples: dict[int, list[TrialSample]] = {}
    for s in subjects:
        df_a = by_method[str(anchor_method)][int(s)]
        p_anchor_all = df_a[proba_cols].to_numpy(np.float64)
        pred_anchor_all = df_a["y_pred"].to_numpy(object)
        y_true_all = df_a["y_true"].to_numpy(object)
        subject_samples: list[TrialSample] = []
        for m in candidate_methods:
            df_c = by_method[str(m)][int(s)]
            p_candidate_all = df_c[proba_cols].to_numpy(np.float64)
            pred_candidate_all = df_c["y_pred"].to_numpy(object)
            prev_trace: dict[str, object] | None = None
            for idx in range(int(df_a.shape[0])):
                values, prev_trace = _trial_feature_values(
                    p_anchor=p_anchor_all[idx],
                    p_candidate=p_candidate_all[idx],
                    pred_anchor=pred_anchor_all[idx],
                    pred_candidate=pred_candidate_all[idx],
                    prev_trace=prev_trace,
                    conflict_tau=float(conflict_tau),
                    conflict_lambda=float(conflict_lambda),
                )
                x = np.asarray([float(values[name]) for name in feature_names], dtype=np.float64)
                anchor_correct = int(pred_anchor_all[idx] == y_true_all[idx])
                candidate_correct = int(pred_candidate_all[idx] == y_true_all[idx])
                subject_samples.append(
                    TrialSample(
                        subject=int(s),
                        trial=int(df_a.loc[idx, "trial"]),
                        method=str(m),
                        x=x,
                        gain=float(candidate_correct - anchor_correct),
                        y_true=y_true_all[idx],
                        anchor_pred=pred_anchor_all[idx],
                        candidate_pred=pred_candidate_all[idx],
                        anchor_correct=int(anchor_correct),
                        candidate_correct=int(candidate_correct),
                    )
                )
        samples[int(s)] = subject_samples
    return samples


def _split_subjects(subjects: list[int], fractions: tuple[float, float, float], seed: int) -> tuple[list[int], list[int], list[int]]:
    subjects = [int(s) for s in subjects]
    rng = np.random.RandomState(int(seed))
    arr = np.asarray(subjects, dtype=int)
    rng.shuffle(arr)
    n = int(arr.size)
    n_fit = max(1, int(round(float(fractions[0]) * n)))
    n_dev = max(1, int(round(float(fractions[1]) * n)))
    if n_fit + n_dev >= n:
        n_fit = max(1, n - 2)
        n_dev = 1
    fit = sorted(int(x) for x in arr[:n_fit])
    dev = sorted(int(x) for x in arr[n_fit : n_fit + n_dev])
    cal = sorted(int(x) for x in arr[n_fit + n_dev :])
    if not cal:
        cal = dev[-1:]
        dev = dev[:-1] or fit[-1:]
    return fit, dev, cal


def _make_subject_folds(subjects: list[int], n_folds: int, seed: int) -> list[list[int]]:
    subjects = [int(s) for s in subjects]
    if int(n_folds) <= 1:
        return [[int(s)] for s in subjects]
    rng = np.random.RandomState(int(seed))
    arr = np.asarray(subjects, dtype=int)
    rng.shuffle(arr)
    folds = [sorted(int(x) for x in fold.tolist()) for fold in np.array_split(arr, min(int(n_folds), len(subjects)))]
    return [fold for fold in folds if fold]


def _train_selector(
    samples_by_subject: dict[int, list[TrialSample]],
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
            states.append(_state_from_gain(sample.gain, delta=float(outcome_delta)))
            gains.append(float(sample.gain))
            # Ranking compares candidate methods for the same source trial.
            groups.append(int(sample.subject) * 100000 + int(sample.trial))
    if not rows:
        raise RuntimeError("No trial-level training rows.")
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
        progress_label=None,
        progress_every=0,
    )


def _score_subject(
    selector,
    samples: list[TrialSample],
    *,
    candidate_methods: list[str],
    method_prior_mode: str,
) -> dict[int, list[ScoredTrial]]:
    if not samples:
        return {}
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
    out: dict[int, list[ScoredTrial]] = {}
    for i, sample in enumerate(samples):
        out.setdefault(int(sample.trial), []).append(
            ScoredTrial(
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


def _select_scored(
    scored: list[ScoredTrial],
    *,
    risk_threshold: float,
    utility_threshold: float,
    risk_only_selection: bool,
    selection_policy: str,
    default_candidate_method: str,
    switch_utility_margin: float,
) -> ScoredTrial | None:
    def is_feasible(c: ScoredTrial) -> bool:
        utility_ok = bool(risk_only_selection) or float(c.utility) >= float(utility_threshold)
        return float(c.risk) <= float(risk_threshold) and utility_ok

    feasible = [c for c in scored if is_feasible(c)]
    if not feasible:
        return None
    policy = str(selection_policy).strip().lower()
    if policy == "free":
        return max(
            feasible,
            key=lambda c: (float(c.utility), -float(c.risk), -float(c.uncertainty), str(c.sample.method)),
        )
    if policy != "default_veto_switch":
        raise ValueError(f"Unsupported selection_policy={selection_policy!r}")
    default_method = str(default_candidate_method).strip()
    default = next((c for c in scored if str(c.sample.method) == default_method), None)
    if default is None:
        return max(
            feasible,
            key=lambda c: (float(c.utility), -float(c.risk), -float(c.uncertainty), str(c.sample.method)),
        )
    margin = float(switch_utility_margin)
    switchers = [
        c
        for c in feasible
        if str(c.sample.method) != default_method and float(c.utility) >= float(default.utility) + margin
    ]
    if is_feasible(default):
        if switchers:
            return max(
                switchers,
                key=lambda c: (float(c.utility), -float(c.risk), -float(c.uncertainty), str(c.sample.method)),
            )
        return default
    if not switchers:
        return None
    return max(
        switchers,
        key=lambda c: (float(c.utility), -float(c.risk), -float(c.uncertainty), str(c.sample.method)),
    )


def _threshold_grids(scored_by_subject: dict[int, dict[int, list[ScoredTrial]]], *, min_utility: float, risk_only: bool) -> tuple[np.ndarray, np.ndarray]:
    rows = [c for subj in scored_by_subject.values() for trials in subj.values() for c in trials]
    if not rows:
        return np.asarray([0.0]), np.asarray([math.inf])
    risks = np.asarray([c.risk for c in rows], dtype=np.float64)
    utils = np.asarray([c.utility for c in rows], dtype=np.float64)
    r_grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 21), np.quantile(risks, np.linspace(0.0, 1.0, 21))]))
    if bool(risk_only):
        return np.asarray(r_grid, dtype=np.float64), np.asarray([float(min_utility)], dtype=np.float64)
    q_grid = np.unique(
        np.concatenate(
            [
                [float(min_utility)],
                np.linspace(float(min_utility), 1.0, 21),
                np.quantile(utils, np.linspace(0.0, 1.0, 21)),
            ]
        )
    )
    q_grid = q_grid[q_grid >= float(min_utility)]
    return np.asarray(r_grid, dtype=np.float64), np.asarray(q_grid, dtype=np.float64)


def _evaluate_scored(
    scored_by_subject: dict[int, dict[int, list[ScoredTrial]]],
    subjects: Iterable[int],
    *,
    risk_threshold: float,
    utility_threshold: float,
    risk_only_selection: bool,
    selection_policy: str,
    default_candidate_method: str,
    switch_utility_margin: float,
) -> dict[str, float]:
    gains: list[float] = []
    correct: list[int] = []
    selected = 0
    harm = 0
    for s in subjects:
        for trial, scored in scored_by_subject.get(int(s), {}).items():
            if not scored:
                continue
            chosen = _select_scored(
                scored,
                risk_threshold=float(risk_threshold),
                utility_threshold=float(utility_threshold),
                risk_only_selection=bool(risk_only_selection),
                selection_policy=str(selection_policy),
                default_candidate_method=str(default_candidate_method),
                switch_utility_margin=float(switch_utility_margin),
            )
            anchor_correct = int(scored[0].sample.anchor_correct)
            if chosen is None:
                gain = 0.0
                final_correct = anchor_correct
            else:
                selected += 1
                gain = float(chosen.sample.gain)
                final_correct = int(chosen.sample.candidate_correct)
                if gain < 0.0:
                    harm += 1
            gains.append(float(gain))
            correct.append(int(final_correct))
    n = int(len(gains))
    return {
        "n_trials": float(n),
        "selected": float(selected),
        "harm": float(harm),
        "mean_gain": float(np.mean(gains)) if gains else 0.0,
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "accept_rate": float(selected / max(1, n)),
        "harm_all_rate": float(harm / max(1, n)),
        "harm_selected_rate": float(harm / selected) if selected > 0 else 0.0,
    }


def _choose_thresholds(
    *,
    scored_by_subject: dict[int, dict[int, list[ScoredTrial]]],
    dev_subjects: list[int],
    cal_subjects: list[int],
    risk_alpha: float,
    cp_delta: float,
    min_accept_rate: float,
    min_utility_threshold: float,
    risk_only_selection: bool,
    selection_policy: str,
    default_candidate_method: str,
    switch_utility_margin: float,
    lambda_accept: float,
    lambda_harm: float,
) -> dict[str, float | bool | str]:
    r_grid, q_grid = _threshold_grids(
        {s: scored_by_subject[int(s)] for s in dev_subjects},
        min_utility=float(min_utility_threshold),
        risk_only=bool(risk_only_selection),
    )
    best: dict[str, float] | None = None
    for r_thr in r_grid:
        for q_thr in q_grid:
            m = _evaluate_scored(
                scored_by_subject,
                dev_subjects,
                risk_threshold=float(r_thr),
                utility_threshold=float(q_thr),
                risk_only_selection=bool(risk_only_selection),
                selection_policy=str(selection_policy),
                default_candidate_method=str(default_candidate_method),
                switch_utility_margin=float(switch_utility_margin),
            )
            if float(m["accept_rate"]) < float(min_accept_rate):
                continue
            obj = float(m["mean_gain"]) + float(lambda_accept) * float(m["accept_rate"]) - float(lambda_harm) * float(m["harm_all_rate"])
            if best is None or obj > float(best["dev_objective"]):
                best = {
                    "risk_threshold": float(r_thr),
                    "utility_threshold": float(q_thr),
                    "dev_objective": float(obj),
                    "dev_mean_gain": float(m["mean_gain"]),
                    "dev_accuracy": float(m["accuracy"]),
                    "dev_accept_rate": float(m["accept_rate"]),
                    "dev_harm_all_rate": float(m["harm_all_rate"]),
                    "dev_harm_selected_rate": float(m["harm_selected_rate"]),
                }
    if best is None:
        return {"verified": False, "fallback_stage": "no_dev_threshold"}
    cal = _evaluate_scored(
        scored_by_subject,
        cal_subjects,
        risk_threshold=float(best["risk_threshold"]),
        utility_threshold=float(best["utility_threshold"]),
        risk_only_selection=bool(risk_only_selection),
        selection_policy=str(selection_policy),
        default_candidate_method=str(default_candidate_method),
        switch_utility_margin=float(switch_utility_margin),
    )
    n_acc = int(cal["selected"])
    n_harm = int(cal["harm"])
    ucb = _cp_upper(n_harm, n_acc, confidence=1.0 - float(cp_delta))
    verified = bool(ucb <= float(risk_alpha))
    return {
        **best,
        "verified": verified,
        "fallback_stage": "verified" if verified else "calibration_failed",
        "cal_accept": float(n_acc),
        "cal_harm": float(n_harm),
        "cal_harm_ucb": float(ucb),
        "cal_mean_gain": float(cal["mean_gain"]),
        "cal_accuracy": float(cal["accuracy"]),
        "cal_accept_rate": float(cal["accept_rate"]),
        "cal_harm_all_rate": float(cal["harm_all_rate"]),
        "cal_harm_selected_rate": float(cal["harm_selected_rate"]),
    }


def _oracle_trial_correct(scored: list[ScoredTrial]) -> int:
    if not scored:
        return 0
    if int(scored[0].sample.anchor_correct):
        return 1
    return int(any(int(c.sample.candidate_correct) for c in scored))


def _run_for_subject_fold(
    *,
    test_subjects: list[int],
    subjects: list[int],
    samples_by_subject: dict[int, list[TrialSample]],
    base_feature_names: tuple[str, ...],
    base_view_slices: dict[str, tuple[int, int]],
    candidate_methods: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_set = {int(s) for s in test_subjects}
    train_subjects = [s for s in subjects if int(s) not in test_set]
    fit_subjects, dev_subjects, cal_subjects = _split_subjects(
        train_subjects,
        fractions=(float(args.fit_fraction), float(args.dev_fraction), float(args.cal_fraction)),
        seed=int(args.seed) + 17 * int(sum(test_set)) + 997 * int(len(test_set)),
    )
    selector = _train_selector(
        samples_by_subject,
        fit_subjects,
        base_feature_names=base_feature_names,
        base_view_slices=base_view_slices,
        candidate_methods=candidate_methods,
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
        seed=int(args.seed) + 100000 + int(sum(test_set)) + 997 * int(len(test_set)),
    )
    scored_train = {
        int(s): _score_subject(
            selector,
            samples_by_subject[int(s)],
            candidate_methods=candidate_methods,
            method_prior_mode=str(args.method_prior_mode),
        )
        for s in dev_subjects + cal_subjects
    }
    thr = _choose_thresholds(
        scored_by_subject=scored_train,
        dev_subjects=dev_subjects,
        cal_subjects=cal_subjects,
        risk_alpha=float(args.risk_alpha),
        cp_delta=float(args.cp_delta),
        min_accept_rate=float(args.min_accept_rate),
        min_utility_threshold=float(args.min_utility_threshold),
        risk_only_selection=bool(args.risk_only_selection),
        selection_policy=str(args.selection_policy),
        default_candidate_method=str(args.default_candidate_method),
        switch_utility_margin=float(args.switch_utility_margin),
        lambda_accept=float(args.lambda_accept),
        lambda_harm=float(args.lambda_harm),
    )
    rows: list[dict[str, object]] = []
    subject_summaries: list[dict[str, object]] = []
    for test_subject in test_subjects:
        scored_test = _score_subject(
            selector,
            samples_by_subject[int(test_subject)],
            candidate_methods=candidate_methods,
            method_prior_mode=str(args.method_prior_mode),
        )
        anchor_corrects: list[int] = []
        final_corrects: list[int] = []
        oracle_corrects: list[int] = []
        selected_count = 0
        harm_count = 0
        for trial in sorted(scored_test):
            scored = scored_test[int(trial)]
            chosen = None
            if bool(thr.get("verified", False)):
                chosen = _select_scored(
                    scored,
                    risk_threshold=float(thr["risk_threshold"]),
                    utility_threshold=float(thr["utility_threshold"]),
                    risk_only_selection=bool(args.risk_only_selection),
                    selection_policy=str(args.selection_policy),
                    default_candidate_method=str(args.default_candidate_method),
                    switch_utility_margin=float(args.switch_utility_margin),
                )
            best_pre = max(scored, key=lambda c: (float(c.utility), -float(c.risk)))
            anchor = scored[0].sample
            if chosen is None:
                selected_method = str(args.anchor_method)
                final_pred = anchor.anchor_pred
                final_correct = int(anchor.anchor_correct)
                gain = 0.0
                risk = utility = uncertainty = float("nan")
                selected_x = np.full((3,), float("nan"), dtype=np.float64)
            else:
                selected_method = str(chosen.sample.method)
                final_pred = chosen.sample.candidate_pred
                final_correct = int(chosen.sample.candidate_correct)
                gain = float(chosen.sample.gain)
                selected_count += 1
                if gain < 0.0:
                    harm_count += 1
                risk = float(chosen.risk)
                utility = float(chosen.utility)
                uncertainty = float(chosen.uncertainty)
                selected_x = np.asarray(chosen.sample.x, dtype=np.float64).reshape(-1)
            oracle_correct = _oracle_trial_correct(scored)
            anchor_corrects.append(int(anchor.anchor_correct))
            final_corrects.append(int(final_correct))
            oracle_corrects.append(int(oracle_correct))
            row = {
                "subject": int(test_subject),
                "trial": int(trial),
                "y_true": anchor.y_true,
                "anchor_pred": anchor.anchor_pred,
                "selected_method": selected_method,
                "final_pred": final_pred,
                "anchor_correct": int(anchor.anchor_correct),
                "final_correct": int(final_correct),
                "oracle_correct": int(oracle_correct),
                "gain": float(gain),
                "accept": int(chosen is not None),
                "harm": int(float(gain) < 0.0 and chosen is not None),
                "selector_risk": risk,
                "selector_utility": utility,
                "selector_uncertainty": uncertainty,
                "pre_best_method": str(best_pre.sample.method),
                "pre_best_risk": float(best_pre.risk),
                "pre_best_utility": float(best_pre.utility),
                "threshold_verified": bool(thr.get("verified", False)),
                "fallback_stage": str(thr.get("fallback_stage", "")),
                "risk_threshold": float(thr.get("risk_threshold", float("nan"))),
                "utility_threshold": float(thr.get("utility_threshold", float("nan"))),
                "fit_n_subjects": int(len(fit_subjects)),
                "dev_n_subjects": int(len(dev_subjects)),
                "cal_n_subjects": int(len(cal_subjects)),
            }
            for j, feat_name in enumerate(base_feature_names):
                selected_value = float(selected_x[j]) if j < selected_x.size and np.isfinite(selected_x[j]) else float("nan")
                row[f"selected_{feat_name}"] = selected_value
                row[f"pre_best_{feat_name}"] = float(best_pre.sample.x[j])
            rows.append(row)
        subject_summaries.append(
            {
                "subject": int(test_subject),
                "n_trials": int(len(final_corrects)),
                "anchor_acc": float(np.mean(anchor_corrects)) if anchor_corrects else 0.0,
                "final_acc": float(np.mean(final_corrects)) if final_corrects else 0.0,
                "gain": float(np.mean(final_corrects) - np.mean(anchor_corrects)) if final_corrects else 0.0,
                "oracle_acc": float(np.mean(oracle_corrects)) if oracle_corrects else 0.0,
                "accept_rate": float(selected_count / max(1, len(final_corrects))),
                "harm_all_rate": float(harm_count / max(1, len(final_corrects))),
                "harm_selected_rate": float(harm_count / max(1, selected_count)) if selected_count > 0 else 0.0,
                "threshold_verified": bool(thr.get("verified", False)),
                "fallback_stage": str(thr.get("fallback_stage", "")),
                "risk_threshold": float(thr.get("risk_threshold", float("nan"))),
                "utility_threshold": float(thr.get("utility_threshold", float("nan"))),
                "cal_harm_ucb": float(thr.get("cal_harm_ucb", float("nan"))),
                "cal_accept_rate": float(thr.get("cal_accept_rate", float("nan"))),
            }
        )
    return rows, subject_summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trial-level Safe-TTS selector from merged predictions_all_methods.csv.")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--anchor-method", type=str, default="eegnet_noea")
    p.add_argument("--candidate-methods", type=str, default=DEFAULT_CANDIDATES)
    p.add_argument("--method-name", type=str, default="trial_safe_tts")
    p.add_argument("--date-prefix", type=str, default=datetime.now().strftime("%Y%m%d"))
    p.add_argument("--eval-subjects", type=str, default="ALL")
    p.add_argument("--trial-feature-preset", type=str, choices=tuple(sorted(FEATURE_PRESETS)), default="core")
    p.add_argument(
        "--trial-feature-names",
        type=str,
        default="PRESET",
        help="Comma-separated feature names. PRESET uses --trial-feature-preset.",
    )
    p.add_argument("--method-prior-mode", type=str, choices=("none", "onehot"), default="none")
    p.add_argument("--selection-policy", type=str, choices=("free", "default_veto_switch"), default="free")
    p.add_argument("--default-candidate-method", type=str, default="")
    p.add_argument("--switch-utility-margin", type=float, default=0.0)
    p.add_argument("--risk-only-selection", action="store_true")
    p.add_argument(
        "--subject-folds",
        type=int,
        default=1,
        help="1 keeps strict LOSO. Values >1 use subject-level K-fold cross-fitting for faster diagnostics.",
    )

    p.add_argument("--fit-fraction", type=float, default=0.60)
    p.add_argument("--dev-fraction", type=float, default=0.20)
    p.add_argument("--cal-fraction", type=float, default=0.20)
    p.add_argument("--risk-alpha", type=float, default=0.40)
    p.add_argument("--cp-delta", type=float, default=0.05)
    p.add_argument("--min-accept-rate", type=float, default=0.0)
    p.add_argument("--min-utility-threshold", type=float, default=0.0)
    p.add_argument("--outcome-delta", type=float, default=0.02)
    p.add_argument("--lambda-accept", type=float, default=0.0)
    p.add_argument("--lambda-harm", type=float, default=0.0)

    p.add_argument("--conflict-tau", type=float, default=0.80)
    p.add_argument("--conflict-lambda", type=float, default=1.0)

    p.add_argument("--selector-hidden-dim", type=int, default=16)
    p.add_argument("--selector-epochs", type=int, default=10)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-weight-decay", type=float, default=1e-4)
    p.add_argument("--selector-lambda-rank", type=float, default=0.5)
    p.add_argument("--selector-lambda-kl", type=float, default=1e-3)
    p.add_argument("--selector-rank-margin", type=float, default=0.005)
    p.add_argument("--selector-rho", type=float, default=0.20)
    p.add_argument("--selector-eta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, proba_cols, candidate_methods, subjects = _load_predictions(
        args.preds,
        anchor_method=str(args.anchor_method),
        candidate_methods=str(args.candidate_methods),
    )
    eval_subjects = _parse_eval_subjects(args.eval_subjects)
    if eval_subjects is not None:
        wanted = set(int(s) for s in eval_subjects)
        subjects = [s for s in subjects if int(s) in wanted]
    if str(args.selection_policy) == "default_veto_switch":
        if not str(args.default_candidate_method).strip():
            raise RuntimeError("--default-candidate-method is required for default_veto_switch.")
        if str(args.default_candidate_method).strip() not in candidate_methods:
            raise RuntimeError("--default-candidate-method must be one of candidate methods.")

    base_feature_names = _parse_trial_feature_names(
        str(args.trial_feature_names),
        preset=str(args.trial_feature_preset),
    )
    base_view_slices = _view_slices_for_features(base_feature_names)
    methods = [str(args.anchor_method)] + candidate_methods
    by_method = _build_subject_maps(df, methods=methods)
    _validate_alignment(by_method, methods=methods, subjects=subjects)
    samples_by_subject = _build_trial_samples(
        by_method=by_method,
        anchor_method=str(args.anchor_method),
        candidate_methods=candidate_methods,
        subjects=subjects,
        proba_cols=proba_cols,
        feature_names=base_feature_names,
        conflict_tau=float(args.conflict_tau),
        conflict_lambda=float(args.conflict_lambda),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = _make_subject_folds(subjects, n_folds=int(args.subject_folds), seed=int(args.seed) + 2026)
    all_rows: list[dict[str, object]] = []
    subject_summaries: list[dict[str, object]] = []
    done_subjects = 0
    for i, fold_subjects in enumerate(folds, start=1):
        rows, summaries = _run_for_subject_fold(
            test_subjects=[int(s) for s in fold_subjects],
            subjects=subjects,
            samples_by_subject=samples_by_subject,
            base_feature_names=base_feature_names,
            base_view_slices=base_view_slices,
            candidate_methods=candidate_methods,
            args=args,
        )
        all_rows.extend(rows)
        subject_summaries.extend(summaries)
        done_subjects += int(len(fold_subjects))
        if int(args.progress_every) > 0 and (
            i % int(args.progress_every) == 0 or i == len(folds) or done_subjects >= len(subjects)
        ):
            print(f"[trial-safe] fold {i}/{len(folds)}; subjects {done_subjects}/{len(subjects)}", flush=True)

    per_trial = pd.DataFrame(all_rows).sort_values(["subject", "trial"]).reset_index(drop=True)
    per_subject = pd.DataFrame(subject_summaries).sort_values("subject").reset_index(drop=True)
    summary = {
        "method": str(args.method_name),
        "anchor_method": str(args.anchor_method),
        "candidate_methods": ",".join(candidate_methods),
        "n_subjects": int(per_subject.shape[0]),
        "n_trials": int(per_trial.shape[0]),
        "mean_accuracy": float(per_subject["final_acc"].mean()),
        "anchor_mean_accuracy": float(per_subject["anchor_acc"].mean()),
        "mean_gain": float(per_subject["gain"].mean()),
        "worst_accuracy": float(per_subject["final_acc"].min()),
        "worst_subject": int(per_subject.loc[per_subject["final_acc"].idxmin(), "subject"]),
        "accept_rate": float(per_trial["accept"].mean()),
        "harm_all_rate": float(per_trial["harm"].mean()),
        "harm_selected_rate": float(per_trial.loc[per_trial["accept"] == 1, "harm"].mean())
        if int(per_trial["accept"].sum()) > 0
        else 0.0,
        "subject_neg_transfer_rate": float((per_subject["gain"] < 0.0).mean()),
        "oracle_mean_accuracy": float(per_subject["oracle_acc"].mean()),
        "oracle_gap_mean": float((per_subject["oracle_acc"] - per_subject["final_acc"]).mean()),
        "threshold_verified_rate": float(per_subject["threshold_verified"].mean()),
        "method_prior_mode": str(args.method_prior_mode),
        "selection_policy": str(args.selection_policy),
        "default_candidate_method": str(args.default_candidate_method),
        "risk_only_selection": int(bool(args.risk_only_selection)),
        "risk_alpha": float(args.risk_alpha),
        "selector_epochs": int(args.selector_epochs),
        "subject_folds": int(args.subject_folds),
        "trial_feature_preset": str(args.trial_feature_preset),
        "trial_feature_names": ",".join(base_feature_names),
        "trial_view_slices": ";".join(f"{k}:{v[0]}-{v[1]}" for k, v in base_view_slices.items()),
    }
    summary_df = pd.DataFrame([summary])
    prefix = str(args.date_prefix)
    per_trial.to_csv(out_dir / f"{prefix}_trial_per_trial_selection.csv", index=False)
    per_subject.to_csv(out_dir / f"{prefix}_trial_per_subject_summary.csv", index=False)
    summary_df.to_csv(out_dir / f"{prefix}_trial_method_comparison.csv", index=False)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
