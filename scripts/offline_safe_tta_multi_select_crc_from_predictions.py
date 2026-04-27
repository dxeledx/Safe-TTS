#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csp_lda.certificate import (
    EvidentialSelector,
    candidate_features_from_record,
    train_evidential_selector,
    train_hgb_certificate,
    train_hgb_guard,
    train_logistic_guard,
    train_ridge_certificate,
)


_LEGACY_MIN_PRED_GRID = "0,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.01,0.02"
_EVIDENTIAL_MIN_PRED_GRID = "-0.30,-0.20,-0.15,-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.15,0.20,0.30"


def _parse_csv_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_float_list(raw: str) -> list[float]:
    raw = str(raw).strip()
    if not raw:
        return []
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_method_family_map(raw: str) -> dict[str, str]:
    raw = str(raw).strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --candidate-family-map entry: {part!r} (expected method=family)")
        m, fam = part.split("=", 1)
        m = m.strip()
        fam = fam.strip()
        if not m or not fam:
            raise ValueError(f"Invalid --candidate-family-map entry: {part!r} (empty method/family)")
        out[m] = fam
    return out


def _parse_selector_views(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return ["stats"]
    allowed = {"stats", "decision", "relative", "dynamic", "stochastic"}
    views: list[str] = []
    for part in raw.split(","):
        name = str(part).strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(f"Invalid selector view {name!r}; expected one of {sorted(allowed)}.")
        if name not in views:
            views.append(name)
    if not views:
        raise ValueError("At least one selector view is required.")
    return views


def _infer_family(method: str) -> str:
    m = str(method).strip().lower()
    if m in {"ea", "ea-csp-lda"} or m.startswith("ea-"):
        return "ea"
    if "lea" in m or m.startswith("rpa") or "logea" in m:
        return "rpa"
    if "tsa" in m or "rot" in m:
        return "tsa"
    if "fbcsp" in m:
        return "fbcsp"
    if "mdm" in m:
        return "mdm"
    if "ts-" in m or "tangent" in m:
        return "ts_svc"
    return "other"


def _row_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def _row_margin(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] <= 0:
        return np.asarray([], dtype=np.float64)
    if p.shape[1] < 2:
        return np.max(p, axis=1)
    ps = np.sort(p, axis=1)
    return ps[:, -1] - ps[:, -2]


def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p0 = np.clip(p0, 1e-12, 1.0)
    p1 = np.clip(p1, 1e-12, 1.0)
    p0 = p0 / np.sum(p0, axis=1, keepdims=True)
    p1 = p1 / np.sum(p1, axis=1, keepdims=True)
    return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)


def _js_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p0 = np.clip(p0, 1e-12, 1.0)
    p1 = np.clip(p1, 1e-12, 1.0)
    p0 = p0 / np.sum(p0, axis=1, keepdims=True)
    p1 = p1 / np.sum(p1, axis=1, keepdims=True)
    m = 0.5 * (p0 + p1)
    return 0.5 * _drift_vec(p0, m) + 0.5 * _drift_vec(p1, m)


def _safe_quantile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    return float(np.quantile(arr, float(q)))


def _record_from_proba(
    *,
    p_id: np.ndarray,
    p_c: np.ndarray,
    y_pred_id: np.ndarray,
    y_pred_c: np.ndarray,
    cand_family: str,
    kind: str,
    drift_delta: float = 0.0,
) -> dict:
    p_id = np.asarray(p_id, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    p_id_clip = np.clip(p_id, 1e-12, 1.0)
    p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
    p_bar = p_bar / float(np.sum(p_bar))
    ent = _row_entropy(p_c)
    ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))
    mean_conf = float(np.mean(np.max(np.clip(p_c, 1e-12, 1.0), axis=1)))
    conf_id = np.max(p_id_clip, axis=1)
    conf_c = np.max(np.clip(p_c, 1e-12, 1.0), axis=1)

    d = _drift_vec(p_id, p_c)
    y_pred_id = np.asarray(y_pred_id, dtype=object).reshape(-1)
    y_pred_c = np.asarray(y_pred_c, dtype=object).reshape(-1)
    pred_disagree = float(np.mean(y_pred_id != y_pred_c)) if y_pred_id.shape == y_pred_c.shape else 0.0
    disagree_mask = (y_pred_id != y_pred_c) if y_pred_id.shape == y_pred_c.shape else np.zeros_like(conf_id, dtype=bool)
    q80 = float(np.quantile(conf_id, 0.80))
    q20 = float(np.quantile(conf_id, 0.20))
    override_high = float(np.mean(disagree_mask & (conf_id >= q80)))
    override_low = float(np.mean(disagree_mask & (conf_id <= q20)))
    if bool(np.any(disagree_mask)):
        mean_conf_id_dis = float(np.mean(conf_id[disagree_mask]))
        mean_conf_c_dis = float(np.mean(conf_c[disagree_mask]))
        mean_delta_conf_dis = float(np.mean((conf_c - conf_id)[disagree_mask]))
    else:
        mean_conf_id_dis = 0.0
        mean_conf_c_dis = 0.0
        mean_delta_conf_dis = 0.0

    rec = {
        "kind": str(kind),
        "cand_family": str(cand_family).strip().lower(),
        "cand_rank": 0.0,
        "cand_lambda": 0.0,
        "objective_base": float(np.mean(ent)),
        "pen_marginal": 0.0,
        "mean_entropy": float(np.mean(ent)),
        "entropy_bar": float(ent_bar),
        "mean_confidence": float(mean_conf),
        "pred_disagree": float(pred_disagree),
        "override_high_conf_id": float(override_high),
        "override_low_conf_id": float(override_low),
        "conf_id_on_disagree": float(mean_conf_id_dis),
        "conf_c_on_disagree": float(mean_conf_c_dis),
        "delta_conf_on_disagree": float(mean_delta_conf_dis),
        "drift_best": float(np.mean(d)),
        "drift_best_std": float(np.std(d)),
        "drift_best_q90": float(np.quantile(d, 0.90)),
        "drift_best_q95": float(np.quantile(d, 0.95)),
        "drift_best_max": float(np.max(d)),
        "drift_best_tail_frac": float(np.mean(d > float(drift_delta))) if float(drift_delta) > 0.0 else 0.0,
        "p_bar_full": p_bar.astype(np.float64),
        "q_bar": np.zeros_like(p_bar, dtype=np.float64),
    }
    rec["objective"] = float(rec["objective_base"])
    rec["score"] = float(rec["objective_base"])
    return rec


def _features_from_anchor_and_candidate(
    *,
    x_anchor: np.ndarray,
    x_candidate: np.ndarray,
    names: tuple[str, ...],
    feature_mode: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build features for cert/guard from anchor/candidate label-free vectors.

    feature_mode:
    - delta: use (x_candidate - x_anchor)
    - anchor_delta: concatenate [x_anchor, (x_candidate - x_anchor)]
    """

    x_anchor = np.asarray(x_anchor, dtype=np.float64).reshape(-1)
    x_candidate = np.asarray(x_candidate, dtype=np.float64).reshape(-1)
    if x_anchor.shape != x_candidate.shape:
        raise ValueError("Anchor/candidate feature shape mismatch.")
    if len(names) != int(x_anchor.shape[0]):
        raise ValueError("Feature name length mismatch.")

    mode = str(feature_mode).strip().lower()
    x_delta = x_candidate - x_anchor
    if mode == "anchor_delta":
        feats = np.concatenate([x_anchor, x_delta], axis=0)
        out_names = tuple([f"anchor_{n}" for n in names] + [f"delta_{n}" for n in names])
        return np.asarray(feats, dtype=np.float64), out_names

    out_names = tuple([f"delta_{n}" for n in names])
    return np.asarray(x_delta, dtype=np.float64), out_names


def _decision_view_features(
    *,
    p_c: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    p_c = np.asarray(p_c, dtype=np.float64)
    conf = np.max(p_c, axis=1)
    ent = _row_entropy(p_c)
    margin = _row_margin(p_c)
    y_pred_idx = np.asarray(np.argmax(p_c, axis=1), dtype=int).reshape(-1)

    feats: list[float] = []
    names: list[str] = []
    for prefix, arr in (
        ("conf", conf),
        ("entropy", ent),
        ("margin", margin),
    ):
        feats.extend(
            [
                float(np.mean(arr)),
                float(np.std(arr)),
                _safe_quantile(arr, 0.10),
                _safe_quantile(arr, 0.90),
            ]
        )
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_q10", f"{prefix}_q90"])

    for k in range(int(n_classes)):
        pk = p_c[:, k]
        feats.extend([float(np.mean(pk)), float(np.std(pk))])
        names.extend([f"prob_mean_{k}", f"prob_std_{k}"])
        feats.append(float(np.mean(y_pred_idx == int(k))))
        names.append(f"pred_freq_{k}")
    return np.asarray(feats, dtype=np.float64), tuple(names)


def _relative_view_features(
    *,
    p_id: np.ndarray,
    p_c: np.ndarray,
    y_pred_id: np.ndarray,
    y_pred_c: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    p_id = np.asarray(p_id, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    y_pred_id = np.asarray(y_pred_id, dtype=object).reshape(-1)
    y_pred_c = np.asarray(y_pred_c, dtype=object).reshape(-1)

    delta_prob = p_c - p_id
    conf_id = np.max(p_id, axis=1)
    conf_c = np.max(p_c, axis=1)
    ent_id = _row_entropy(p_id)
    ent_c = _row_entropy(p_c)
    js = _js_vec(p_id, p_c)
    kl_id_c = _drift_vec(p_id, p_c)
    kl_c_id = _drift_vec(p_c, p_id)
    flip = (y_pred_id != y_pred_c).astype(np.float64)
    conflict = ((y_pred_id != y_pred_c) & (conf_id >= 0.8) & (conf_c >= 0.8)).astype(np.float64)
    delta_conf = conf_c - conf_id
    delta_ent = ent_c - ent_id

    feats: list[float] = []
    names: list[str] = []
    for k in range(int(n_classes)):
        dk = delta_prob[:, k]
        feats.extend([float(np.mean(dk)), float(np.std(dk)), float(np.mean(np.abs(dk)))])
        names.extend([f"delta_prob_mean_{k}", f"delta_prob_std_{k}", f"delta_prob_abs_mean_{k}"])

    for prefix, arr in (
        ("js", js),
        ("kl_id_c", kl_id_c),
        ("kl_c_id", kl_c_id),
        ("delta_conf", delta_conf),
        ("delta_entropy", delta_ent),
    ):
        feats.extend(
            [
                float(np.mean(arr)),
                float(np.std(arr)),
                _safe_quantile(arr, 0.90),
            ]
        )
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_q90"])

    feats.extend([float(np.mean(flip)), float(np.mean(conflict))])
    names.extend(["flip_rate", "high_conflict_rate"])
    return np.asarray(feats, dtype=np.float64), tuple(names)


def _split_chunk_bounds(n_rows: int, n_chunks: int) -> list[tuple[int, int]]:
    n_rows = int(n_rows)
    n_chunks = max(1, int(n_chunks))
    if n_rows <= 0:
        return []
    n_chunks = min(n_chunks, n_rows)
    bounds = np.linspace(0, n_rows, num=n_chunks + 1, dtype=int)
    out: list[tuple[int, int]] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if int(stop) > int(start):
            out.append((int(start), int(stop)))
    return out


def _sequence_summary_features(values: list[float], prefix: str) -> tuple[list[float], list[str]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        arr = np.zeros((1,), dtype=np.float64)
    if arr.size == 1:
        slope = 0.0
        delta_last = 0.0
        max_step = 0.0
    else:
        x = np.arange(arr.size, dtype=np.float64)
        slope = float(np.polyfit(x, arr, deg=1)[0])
        delta_last = float(arr[-1] - arr[0])
        max_step = float(np.max(np.abs(np.diff(arr))))
    feats = [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(arr[-1]),
        float(delta_last),
        float(slope),
        float(max_step),
    ]
    names = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_last",
        f"{prefix}_delta_last",
        f"{prefix}_slope",
        f"{prefix}_max_step",
    ]
    return feats, names


def _chunk_class_entropy(prob: np.ndarray, n_classes: int) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[0] <= 0:
        return 0.0
    pred_idx = np.asarray(np.argmax(prob, axis=1), dtype=int).reshape(-1)
    freq = np.bincount(pred_idx, minlength=int(n_classes)).astype(np.float64)
    freq /= max(1.0, float(np.sum(freq)))
    freq = np.clip(freq, 1e-12, 1.0)
    return float(-np.sum(freq * np.log(freq)))


def _dynamic_view_features(
    *,
    p_id: np.ndarray,
    p_c: np.ndarray,
    y_pred_id: np.ndarray,
    y_pred_c: np.ndarray,
    n_classes: int,
    n_chunks: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    p_id = np.asarray(p_id, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    y_pred_id = np.asarray(y_pred_id, dtype=object).reshape(-1)
    y_pred_c = np.asarray(y_pred_c, dtype=object).reshape(-1)
    bounds = _split_chunk_bounds(int(p_c.shape[0]), int(n_chunks))

    ent_seq: list[float] = []
    margin_seq: list[float] = []
    js_seq: list[float] = []
    flip_seq: list[float] = []
    conflict_seq: list[float] = []
    class_ent_seq: list[float] = []
    conf_seq: list[float] = []
    for start, stop in bounds:
        p_id_b = p_id[start:stop]
        p_c_b = p_c[start:stop]
        y_id_b = y_pred_id[start:stop]
        y_c_b = y_pred_c[start:stop]
        ent_seq.append(float(np.mean(_row_entropy(p_c_b))))
        margin_seq.append(float(np.mean(_row_margin(p_c_b))))
        js_seq.append(float(np.mean(_js_vec(p_id_b, p_c_b))))
        conf_c_b = np.max(p_c_b, axis=1)
        conf_id_b = np.max(p_id_b, axis=1)
        flip_mask = (y_id_b != y_c_b)
        conflict_mask = flip_mask & (conf_id_b >= 0.8) & (conf_c_b >= 0.8)
        flip_seq.append(float(np.mean(flip_mask.astype(np.float64))))
        conflict_seq.append(float(np.mean(conflict_mask.astype(np.float64))))
        class_ent_seq.append(float(_chunk_class_entropy(p_c_b, n_classes=int(n_classes))))
        conf_seq.append(float(np.mean(conf_c_b)))

    feats: list[float] = []
    names: list[str] = []
    for prefix, seq in (
        ("dyn_entropy", ent_seq),
        ("dyn_margin", margin_seq),
        ("dyn_js", js_seq),
        ("dyn_flip", flip_seq),
        ("dyn_conflict", conflict_seq),
        ("dyn_class_entropy", class_ent_seq),
        ("dyn_confidence", conf_seq),
    ):
        seq_feats, seq_names = _sequence_summary_features(seq, prefix)
        feats.extend(seq_feats)
        names.extend(seq_names)
    feats.append(float(len(bounds)))
    names.append("dyn_num_chunks")
    return np.asarray(feats, dtype=np.float64), tuple(names)


def _stochastic_view_features(
    *,
    p_id: np.ndarray,
    p_c: np.ndarray,
    y_pred_id: np.ndarray,
    y_pred_c: np.ndarray,
    n_classes: int,
    bootstrap_rounds: int,
    bootstrap_seed: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    p_id = np.asarray(p_id, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    y_pred_id = np.asarray(y_pred_id, dtype=object).reshape(-1)
    y_pred_c = np.asarray(y_pred_c, dtype=object).reshape(-1)
    n_trials = int(p_c.shape[0])
    rounds = max(2, int(bootstrap_rounds))
    rng = np.random.RandomState(int(bootstrap_seed))

    pbar_samples: list[np.ndarray] = []
    conf_means: list[float] = []
    ent_means: list[float] = []
    margin_means: list[float] = []
    js_means: list[float] = []
    flip_rates: list[float] = []
    top_classes: list[int] = []
    for _ in range(rounds):
        if n_trials <= 0:
            idx = np.zeros((0,), dtype=int)
        else:
            idx = rng.randint(0, n_trials, size=n_trials)
        p_c_b = p_c[idx] if idx.size > 0 else p_c
        p_id_b = p_id[idx] if idx.size > 0 else p_id
        y_id_b = y_pred_id[idx] if idx.size > 0 else y_pred_id
        y_c_b = y_pred_c[idx] if idx.size > 0 else y_pred_c
        pbar = np.mean(p_c_b, axis=0) if p_c_b.shape[0] > 0 else np.zeros((int(n_classes),), dtype=np.float64)
        pbar_samples.append(np.asarray(pbar, dtype=np.float64).reshape(-1))
        conf_means.append(float(np.mean(np.max(p_c_b, axis=1))) if p_c_b.shape[0] > 0 else 0.0)
        ent_means.append(float(np.mean(_row_entropy(p_c_b))) if p_c_b.shape[0] > 0 else 0.0)
        margin_means.append(float(np.mean(_row_margin(p_c_b))) if p_c_b.shape[0] > 0 else 0.0)
        js_means.append(float(np.mean(_js_vec(p_id_b, p_c_b))) if p_c_b.shape[0] > 0 else 0.0)
        flip_rates.append(float(np.mean((y_id_b != y_c_b).astype(np.float64))) if y_c_b.shape[0] > 0 else 0.0)
        top_classes.append(int(np.argmax(pbar)) if pbar.size > 0 else 0)

    pbar_arr = np.stack(pbar_samples, axis=0) if pbar_samples else np.zeros((1, int(n_classes)), dtype=np.float64)
    pbar_std = np.std(pbar_arr, axis=0)
    top_classes_arr = np.asarray(top_classes, dtype=int)
    mode_freq = np.bincount(top_classes_arr, minlength=int(n_classes)).astype(np.float64)
    top_agree = float(np.max(mode_freq) / max(1.0, float(np.sum(mode_freq))))

    feats: list[float] = [
        float(np.mean(pbar_std)),
        float(np.max(pbar_std)),
        float(np.std(conf_means)),
        float(np.std(ent_means)),
        float(np.std(margin_means)),
        float(np.std(js_means)),
        float(np.std(flip_rates)),
        float(1.0 - top_agree),
    ]
    names: list[str] = [
        "stoch_pbar_std_mean",
        "stoch_pbar_std_max",
        "stoch_conf_mean_std",
        "stoch_entropy_mean_std",
        "stoch_margin_mean_std",
        "stoch_js_mean_std",
        "stoch_flip_rate_std",
        "stoch_topclass_disagree",
    ]
    for k in range(int(n_classes)):
        feats.append(float(pbar_std[k]))
        names.append(f"stoch_pbar_std_{k}")
    feats.append(float(rounds))
    names.append("stoch_bootstrap_rounds")
    return np.asarray(feats, dtype=np.float64), tuple(names)


def _selector_feature_bundle(
    *,
    p_id: np.ndarray,
    p_c: np.ndarray,
    y_pred_id: np.ndarray,
    y_pred_c: np.ndarray,
    anchor_family: str,
    cand_family: str,
    n_classes: int,
    feature_mode: str,
    selector_views: list[str],
    dynamic_chunks: int,
    stochastic_bootstrap_rounds: int,
    stochastic_bootstrap_seed: int,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[int, int]]]:
    rec_id = _record_from_proba(
        p_id=p_id,
        p_c=p_id,
        y_pred_id=y_pred_id,
        y_pred_c=y_pred_id,
        cand_family=str(anchor_family),
        kind="identity",
    )
    rec_c = _record_from_proba(
        p_id=p_id,
        p_c=p_c,
        y_pred_id=y_pred_id,
        y_pred_c=y_pred_c,
        cand_family=str(cand_family),
        kind="candidate",
    )
    x0, names0 = candidate_features_from_record(rec_id, n_classes=n_classes, include_pbar=True)
    x, names = candidate_features_from_record(rec_c, n_classes=n_classes, include_pbar=True)
    if names != names0:
        raise RuntimeError("Feature name mismatch between anchor and candidate features.")
    stats_feats, stats_names = _features_from_anchor_and_candidate(
        x_anchor=np.asarray(x0, dtype=np.float64).reshape(-1),
        x_candidate=np.asarray(x, dtype=np.float64).reshape(-1),
        names=tuple(names0),
        feature_mode=str(feature_mode),
    )
    dec_feats, dec_names = _decision_view_features(
        p_c=p_c,
        n_classes=int(n_classes),
    )
    rel_feats, rel_names = _relative_view_features(
        p_id=p_id,
        p_c=p_c,
        y_pred_id=y_pred_id,
        y_pred_c=y_pred_c,
        n_classes=int(n_classes),
    )
    dyn_feats, dyn_names = _dynamic_view_features(
        p_id=p_id,
        p_c=p_c,
        y_pred_id=y_pred_id,
        y_pred_c=y_pred_c,
        n_classes=int(n_classes),
        n_chunks=int(dynamic_chunks),
    )
    stoch_feats, stoch_names = _stochastic_view_features(
        p_id=p_id,
        p_c=p_c,
        y_pred_id=y_pred_id,
        y_pred_c=y_pred_c,
        n_classes=int(n_classes),
        bootstrap_rounds=int(stochastic_bootstrap_rounds),
        bootstrap_seed=int(stochastic_bootstrap_seed),
    )

    bundles = {
        "stats": (np.asarray(stats_feats, dtype=np.float64), tuple(stats_names)),
        "decision": (np.asarray(dec_feats, dtype=np.float64), tuple(dec_names)),
        "relative": (np.asarray(rel_feats, dtype=np.float64), tuple(rel_names)),
        "dynamic": (np.asarray(dyn_feats, dtype=np.float64), tuple(dyn_names)),
        "stochastic": (np.asarray(stoch_feats, dtype=np.float64), tuple(stoch_names)),
    }
    feats_parts: list[np.ndarray] = []
    names_parts: list[str] = []
    view_slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for view in selector_views:
        arr, names_view = bundles[str(view)]
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        feats_parts.append(arr)
        names_parts.extend([f"{view}_{name}" for name in names_view])
        view_slices[str(view)] = (int(offset), int(offset + arr.shape[0]))
        offset += int(arr.shape[0])
    feats = np.concatenate(feats_parts, axis=0) if feats_parts else np.zeros((0,), dtype=np.float64)
    return np.asarray(feats, dtype=np.float64), tuple(names_parts), view_slices


class _ConstantGuard:
    def __init__(self, *, p_pos: float, feature_names: tuple[str, ...]) -> None:
        self._p_pos = float(p_pos)
        self.feature_names = tuple(feature_names)

    def predict_pos_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        n = int(features.shape[0]) if features.ndim == 2 else 1
        return np.full((n,), fill_value=self._p_pos, dtype=np.float64)


def _resolve_min_pred_grid(raw: str, *, selector_model: str) -> list[float]:
    raw = str(raw).strip()
    if str(selector_model).strip().lower() == "evidential" and raw == _LEGACY_MIN_PRED_GRID:
        raw = _EVIDENTIAL_MIN_PRED_GRID
    return sorted({float(x) for x in _parse_float_list(raw)})


def _parse_eval_subjects(raw: str) -> list[int]:
    text = str(raw).strip()
    if not text:
        return []
    out: list[int] = []
    for chunk in text.split(","):
        token = str(chunk).strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid subject range {token!r}: end < start.")
            out.extend(range(start, end + 1))
        else:
            out.append(int(token))
    return sorted({int(x) for x in out})


def _outcome_state_from_improve(*, improve: float, outcome_delta: float) -> int:
    margin = max(0.0, float(outcome_delta))
    if float(improve) <= -margin:
        return 0
    if float(improve) >= margin:
        return 2
    return 1


def _selector_outputs(
    *,
    cert: object,
    guard: object,
    feats: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    pred_improve = float(cert.predict_accuracy(feats)[0])
    p_pos = float(guard.predict_pos_proba(feats)[0])
    extras: dict[str, float] = {}
    if isinstance(cert, EvidentialSelector):
        stats = cert.predict_stats(feats)
        probs = np.asarray(stats["probs"], dtype=np.float64)
        beliefs = np.asarray(stats["beliefs"], dtype=np.float64)
        uncertainty = np.asarray(stats["uncertainty"], dtype=np.float64).reshape(-1)
        risk = np.asarray(stats["risk"], dtype=np.float64).reshape(-1)
        utility = np.asarray(stats["utility"], dtype=np.float64).reshape(-1)
        non_harm = np.asarray(stats["non_harm"], dtype=np.float64).reshape(-1)
        pred_improve = float(utility[0])
        p_pos = float(non_harm[0])
        extras = {
            "selector_p_harm": float(probs[0, 0]),
            "selector_p_neutral": float(probs[0, 1]),
            "selector_p_help": float(probs[0, 2]),
            "selector_b_harm": float(beliefs[0, 0]),
            "selector_b_neutral": float(beliefs[0, 1]),
            "selector_b_help": float(beliefs[0, 2]),
            "selector_uncertainty": float(uncertainty[0]),
            "selector_risk": float(risk[0]),
            "selector_utility": float(utility[0]),
        }
        for key, value in stats.items():
            if str(key).startswith("reliability_"):
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                extras[f"selector_{key}"] = float(arr[0]) if arr.size > 0 else float("nan")
    return float(pred_improve), float(p_pos), extras


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Offline SAFE-TTA multi-armed selection with CRC-style risk calibration.\n\n"
            "For each test subject t, split the remaining subjects into fit/calib. Train a ridge certificate + "
            "logistic guard on the fit set, then calibrate min_pred_improve by sweeping a grid on the calib set "
            "to satisfy an upper-confidence bound on neg_transfer_rate <= risk_alpha (Clopper–Pearson). "
            "Apply the calibrated threshold to t.\n"
        )
    )
    p.add_argument("--preds", type=Path, required=True, help="Merged predictions_all_methods.csv (multiple methods).")
    p.add_argument("--anchor-method", type=str, default="ea-csp-lda", help="Anchor method (default: ea-csp-lda).")
    p.add_argument(
        "--candidate-methods",
        type=str,
        default="ALL",
        help="Comma-separated candidate methods, or ALL to use all methods except anchor.",
    )
    p.add_argument(
        "--candidate-family-map",
        type=str,
        default="",
        help="Optional mapping 'method=family,method=family'. Families are used for feature one-hots.",
    )
    p.add_argument("--guard-threshold", type=float, default=0.5)
    p.add_argument("--anchor-guard-delta", type=float, default=0.05)
    p.add_argument(
        "--policy",
        type=str,
        default="full",
        choices=["full", "pred_only", "guard_only"],
        help=(
            "Selection policy preset. "
            "full: guard + certificate thresholding (default). "
            "pred_only: disable guard, threshold only on certificate. "
            "guard_only: threshold only on guard_p_pos."
        ),
    )
    p.add_argument(
        "--threshold-score",
        type=str,
        default="pred_improve",
        choices=["pred_improve", "guard_p_pos", "pred_improve_x_guard"],
        help="Score used for thresholding during calibration and deployment (default: pred_improve).",
    )
    p.add_argument(
        "--candidate-choice-score",
        type=str,
        default="pred_improve",
        choices=["pred_improve", "guard_p_pos", "pred_improve_x_guard"],
        help="Score used to pick the best candidate among feasible candidates (default: pred_improve).",
    )
    p.add_argument(
        "--feature-mode",
        type=str,
        default="delta",
        choices=["delta", "anchor_delta"],
        help="Cert/guard feature mode: delta (default) or anchor_delta (anchor features + delta).",
    )
    p.add_argument(
        "--disable-guard",
        action="store_true",
        help="Disable the guard (treat all candidates as guard_ok). Useful for pred-only baselines.",
    )
    p.add_argument(
        "--guard-target",
        type=str,
        default="improve",
        choices=["improve", "harm"],
        help=(
            "Guard training target. "
            "improve trains P(Δacc>0) (default). "
            "harm trains P(Δacc<0) but logs/stores guard_p_pos as P(non-harm)=1-P(harm). "
            "In harm mode, guard_ok uses an absolute threshold guard_p_pos>=guard_threshold "
            "(anchor_guard_delta is ignored for guard_ok)."
        ),
    )
    p.add_argument(
        "--guard-gray-margin",
        type=float,
        default=0.02,
        help=(
            "Symmetric gray-zone margin for guard labels. "
            "When >0, rows with |Δacc| <= margin are excluded from guard training. "
            "Default: 0.02 (paper-aligned 2pp safety interval)."
        ),
    )
    p.add_argument(
        "--selector-model",
        type=str,
        default="legacy",
        choices=["legacy", "evidential"],
        help=(
            "Meta-selector family. legacy keeps ridge certificate + logistic guard. "
            "evidential uses a 3-state evidential selector that derives both risk and utility "
            "from a shared deployment-outcome posterior."
        ),
    )
    p.add_argument(
        "--selector-hidden-dim",
        type=int,
        default=64,
        help="Hidden width for the evidential selector MLP (default: 64).",
    )
    p.add_argument(
        "--selector-epochs",
        type=int,
        default=400,
        help="Training epochs for the evidential selector (default: 400).",
    )
    p.add_argument(
        "--selector-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the evidential selector (default: 1e-3).",
    )
    p.add_argument(
        "--selector-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the evidential selector (default: 1e-4).",
    )
    p.add_argument(
        "--selector-lambda-rank",
        type=float,
        default=0.5,
        help="Ranking-loss weight for the evidential selector (default: 0.5).",
    )
    p.add_argument(
        "--selector-lambda-kl",
        type=float,
        default=1e-3,
        help="KL regularization weight for the evidential selector (default: 1e-3).",
    )
    p.add_argument(
        "--selector-outcome-delta",
        type=float,
        default=0.005,
        help="Dead-zone half-width for harmful/neutral/helpful labels in the evidential selector (default: 0.005).",
    )
    p.add_argument(
        "--selector-rank-margin",
        type=float,
        default=0.005,
        help="Pairwise ranking margin on true Δacc for the evidential selector (default: 0.005).",
    )
    p.add_argument(
        "--selector-rho",
        type=float,
        default=0.05,
        help="Uncertainty penalty used in risk score r=b_H+rho*u (default: 0.05).",
    )
    p.add_argument(
        "--selector-eta",
        type=float,
        default=0.05,
        help="Uncertainty penalty used in utility score q=(b_B-b_H)-eta*u (default: 0.05).",
    )
    p.add_argument(
        "--log-stage-breakdown",
        action="store_true",
        help="Log SCRC-style stage breakdown columns in per_subject_selection.csv.",
    )
    p.add_argument(
        "--select-topm",
        type=int,
        default=0,
        help="Select up to top-m subjects by selection score before thresholding (0 disables).",
    )
    p.add_argument(
        "--select-fraction",
        type=float,
        default=0.0,
        help="Select top ceil(fraction * n_total) subjects by selection score before thresholding (0 disables).",
    )
    p.add_argument(
        "--select-score",
        type=str,
        default="guard_p_pos",
        choices=["guard_p_pos", "pred_improve"],
        help="Subject-level selection score computed from guard_ok candidates (default: guard_p_pos).",
    )
    p.add_argument(
        "--select-score-scope",
        type=str,
        default="guard_ok",
        choices=["guard_ok", "all"],
        help=(
            "Scope for computing subject-level selection score. "
            "guard_ok uses only candidates with guard_ok==True (default). "
            "all uses all candidates (ignores guard_ok for scoring) but still logs has_guard_ok."
        ),
    )
    p.add_argument("--risk-alpha", type=float, default=0.0, help="Target neg-transfer risk budget α on calibration set.")
    p.add_argument(
        "--risk-mode",
        type=str,
        default="marginal",
        choices=["marginal", "dual"],
        help="Calibration mode: marginal-only (default) or dual constraints (marginal + conditional + min-accept).",
    )
    p.add_argument(
        "--risk-beta-cond",
        type=float,
        default=None,
        help=(
            "Conditional risk budget β among accepted subjects (required when --risk-mode=dual). "
            "In dual-risk mode, this constrains the one-sided Clopper–Pearson UCB of Pr(neg | acc)."
        ),
    )
    p.add_argument(
        "--min-accept-rate",
        type=float,
        default=0.0,
        help="Minimum acceptance-rate constraint ρ_min used in calibration (default: 0.0).",
    )
    p.add_argument(
        "--calib-objective",
        type=str,
        default="mean_delta_all",
        choices=["mean_delta_all", "mean_acc"],
        help="Objective to maximize among feasible thresholds (default: mean_delta_all).",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Confidence δ for the (one-sided) Clopper–Pearson UCB on neg-transfer rate (default: 0.05).",
    )
    p.add_argument(
        "--neg-transfer-eps",
        type=float,
        default=0.0,
        help="Define negative transfer as Δ < -eps (default: 0.0). Useful to ignore tiny accuracy drops.",
    )
    p.add_argument(
        "--calib-fraction",
        type=float,
        default=0.25,
        help="Fraction of (non-test) subjects used for calibration when --n-splits=1.",
    )
    p.add_argument("--calib-seed", type=int, default=0, help="Seed for fit/calib split / cross-fitting folds.")
    p.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help=(
            "Cross-fitting folds. 1 uses a single fit/calib split. "
            "K>1 uses K-fold cross-fitting to pool calibration subjects (larger effective n_calib)."
        ),
    )
    p.add_argument(
        "--calibration-protocol",
        type=str,
        default="legacy_pooled",
        choices=["legacy_pooled", "paper_oof_dev_cal"],
        help=(
            "Threshold calibration protocol. "
            "legacy_pooled reproduces the previous pooled fit/cal or cross-fit calibration. "
            "paper_oof_dev_cal first forms OOF subject scores via cross-fitting, then splits "
            "pseudo-target subjects into D_dev/D_cal, selects the threshold on D_dev, and verifies it on D_cal."
        ),
    )
    p.add_argument(
        "--min-pred-grid",
        type=str,
        default=_LEGACY_MIN_PRED_GRID,
        help=(
            "Comma-separated threshold grid swept during calibration. "
            "Interpreted in the units of --threshold-score (pred_improve or guard_p_pos)."
        ),
    )
    p.add_argument(
        "--cert-model",
        type=str,
        default="ridge",
        choices=["ridge", "hgb"],
        help="Certificate model family: ridge (default) or tree-based HGB.",
    )
    p.add_argument(
        "--guard-model",
        type=str,
        default="logreg",
        choices=["logreg", "hgb"],
        help="Guard model family: logistic regression (default) or tree-based HGB.",
    )
    p.add_argument(
        "--hgb-max-iter",
        type=int,
        default=200,
        help="Max boosting iterations for HGB certificate/guard when enabled.",
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--guard-c", type=float, default=1.0)
    p.add_argument(
        "--selector-views",
        type=str,
        default="stats,decision,relative",
        help="Comma-separated evidential selector views. Choices: stats,decision,relative,dynamic,stochastic.",
    )
    p.add_argument(
        "--dynamic-chunks",
        type=int,
        default=4,
        help="Number of temporal chunks used by the dynamic view (default: 4).",
    )
    p.add_argument(
        "--stochastic-bootstrap-rounds",
        type=int,
        default=16,
        help="Bootstrap rounds used by the stochastic stability view (default: 16).",
    )
    p.add_argument(
        "--stochastic-bootstrap-seed",
        type=int,
        default=0,
        help="Base seed for stochastic bootstrap view construction (default: 0).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method-name", type=str, default="safe-tta-offline-multi-crc")
    p.add_argument("--date-prefix", type=str, default="", help="Output file prefix (default: YYYYMMDD).")
    p.add_argument(
        "--eval-subjects",
        type=str,
        default="",
        help=(
            "Optional subset of test subjects to evaluate, e.g. '1-27' or '1-10,21,35-40'. "
            "Training/calibration still uses the full remaining subject pool."
        ),
    )
    p.add_argument("--no-diagnostics", action="store_true", help="Do not write per-subject candidates.csv.")
    p.add_argument(
        "--dump-calib-grid",
        action="store_true",
        help=(
            "Dump the calibration grid (per test subject, per threshold) to <out_dir>/<date>_calib_grid.csv. "
            "Useful for diagnosing why dual-risk constraints are (in)feasible."
        ),
    )
    p.add_argument(
        "--enable-review-proxy",
        action="store_true",
        help="Enable offline human-review proxy on abstained subjects.",
    )
    p.add_argument(
        "--review-budget-frac",
        type=float,
        default=0.0,
        help=(
            "Review budget fraction q in [0,1] (used when --review-budget-mode=fixed). "
            "Reviewed subjects are selected according to --review-scope."
        ),
    )
    p.add_argument(
        "--review-budget-mode",
        type=str,
        default="fixed",
        choices=["fixed", "auto_min_beta"],
        help=(
            "Review budget selection mode. "
            "fixed uses --review-budget-frac (default). "
            "auto_min_beta selects the minimal q (from --review-budget-grid) such that "
            "the (target) conditional-risk UCB is <= beta."
        ),
    )
    p.add_argument(
        "--review-budget-grid",
        type=str,
        default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40",
        help=(
            "Comma-separated candidate q values in [0,1] for --review-budget-mode=auto_min_beta "
            "(default: 0..0.40 in steps of 0.05)."
        ),
    )
    p.add_argument(
        "--cond-risk-target",
        type=str,
        default="pre",
        choices=["pre", "post"],
        help=(
            "Conditional-risk target used by dual-risk + review calibration. "
            "pre constrains Pr(neg|acc) on pre-review decisions. "
            "post constrains the residual Pr(neg|acc) after review-proxy (default: pre)."
        ),
    )
    p.add_argument(
        "--review-scope",
        type=str,
        default="abstain",
        choices=["abstain", "accepted", "all"],
        help=(
            "Which subjects are eligible for review-proxy ranking (default: abstain). "
            "abstain: review abstained subjects (current behavior). "
            "accepted: review accepted subjects (useful to reduce conditional harm). "
            "all: review from all subjects."
        ),
    )
    p.add_argument(
        "--review-score",
        type=str,
        default="opportunity",
        choices=["opportunity", "uncertainty", "harm"],
        help=(
            "Ranking score for selecting subjects for review. "
            "opportunity ranks by predicted headroom; uncertainty ranks by 1-guard_p_pos; "
            "harm ranks by 1-guard_p_pos_selected (estimated P(harm) when --guard-target=harm)."
        ),
    )
    p.add_argument(
        "--review-cost",
        type=float,
        default=0.0,
        help="Linear review cost coefficient c_review for net utility computation.",
    )
    return p.parse_args()


def _hoeffding_ucb(*, p_hat: float, n: int, delta: float) -> float:
    p_hat = float(p_hat)
    n = int(n)
    delta = float(delta)
    if not np.isfinite(p_hat):
        return float("nan")
    if n <= 0:
        return float("nan")
    if not (0.0 < delta < 1.0):
        raise ValueError("--delta must be in (0,1).")
    rad = float(np.sqrt(np.log(1.0 / float(delta)) / (2.0 * float(n))))
    return float(min(1.0, p_hat + rad))


def _binom_cdf(*, k: int, n: int, p: float) -> float:
    """Binomial CDF: P[X <= k] for X~Bin(n,p)."""
    k = int(k)
    n = int(n)
    p = float(p)
    if n < 0:
        raise ValueError("n must be >= 0")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0

    q = 1.0 - p
    # For better numerical stability, sum the shorter side.
    if k < (n // 2):
        prob = float(q**n)
        cdf = prob
        ratio = float(p / q)
        for i in range(0, k):
            prob *= float(n - i) / float(i + 1) * ratio
            cdf += prob
        return float(min(1.0, max(0.0, cdf)))

    # Complement: 1 - P[X >= k+1].
    prob = float(p**n)  # P[X=n]
    tail = prob
    ratio = float(q / p)
    # Accumulate P[X=i] for i=n-1,...,k+1.
    for i in range(n, k + 1, -1):
        prob *= float(i) / float(n - i + 1) * ratio  # P[X=i-1]
        tail += prob
    cdf = 1.0 - tail
    return float(min(1.0, max(0.0, cdf)))


def _clopper_pearson_ucb(*, k: int, n: int, delta: float) -> float:
    """One-sided Clopper–Pearson upper confidence bound for a binomial proportion.

    Returns p_ucb such that P[X <= k | p_ucb] = delta for X ~ Bin(n, p_ucb).
    Equivalently, p_ucb = Beta^{-1}(1-delta; k+1, n-k).
    """
    k = int(k)
    n = int(n)
    delta = float(delta)
    if n <= 0:
        return float("nan")
    if not (0.0 < delta < 1.0):
        raise ValueError("--delta must be in (0,1).")
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}.")
    if k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - delta ** (1.0 / float(n)))

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = _binom_cdf(k=k, n=n, p=mid)
        if cdf > delta:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-12:
            break
    return float(hi)


def _clopper_pearson_ci(*, k: int, n: int, delta: float) -> tuple[float, float]:
    """Two-sided Clopper–Pearson confidence interval at level 1-delta."""
    k = int(k)
    n = int(n)
    delta = float(delta)
    if n <= 0:
        return float("nan"), float("nan")
    if not (0.0 < delta < 1.0):
        raise ValueError("--delta must be in (0,1).")
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}.")

    if k == 0:
        lo = 0.0
    else:
        # Lower bound can be computed from the one-sided upper bound on failure rate q=1-p.
        # failures = n-k, so p_low = 1 - UCB_q(delta/2).
        lo = float(1.0 - _clopper_pearson_ucb(k=n - k, n=n, delta=delta / 2.0))

    if k == n:
        hi = 1.0
    else:
        hi = float(_clopper_pearson_ucb(k=k, n=n, delta=delta / 2.0))
    return float(lo), float(hi)


def _precompute_subject_candidates(
    *,
    subjects_eval: list[int],
    by_subject_anchor: dict[int, pd.DataFrame],
    by_subject_cands: dict[str, dict[int, pd.DataFrame]],
    proba_cols: list[str],
    n_classes: int,
    cand_methods: list[str],
    cand_families: dict[str, str],
    anchor_family: str,
    cert,
    guard,
    p_pos_anchor: float,
    guard_threshold: float,
    anchor_guard_delta: float,
    disable_guard: bool,
    guard_target: str,
    feature_mode: str,
    selector_model: str,
    selector_views: list[str],
    dynamic_chunks: int,
    stochastic_bootstrap_rounds: int,
    stochastic_bootstrap_seed: int,
) -> list[dict[str, object]]:
    out = []
    for s in subjects_eval:
        df_id = by_subject_anchor[int(s)]
        y_true = df_id["y_true"].to_numpy(object)
        y_pred_id = df_id["y_pred"].to_numpy(object)
        p_id = df_id[proba_cols].to_numpy(np.float64)
        acc_id = float(np.mean(y_pred_id == y_true))

        rec_id = _record_from_proba(
            p_id=p_id,
            p_c=p_id,
            y_pred_id=y_pred_id,
            y_pred_c=y_pred_id,
            cand_family=str(anchor_family).strip().lower(),
            kind="identity",
        )
        cands = []
        for m in cand_methods:
            df_c = by_subject_cands[m][int(s)]
            y_pred_c = df_c["y_pred"].to_numpy(object)
            p_c = df_c[proba_cols].to_numpy(np.float64)
            acc_c = float(np.mean(y_pred_c == y_true))

            rec_c = _record_from_proba(
                p_id=p_id,
                p_c=p_c,
                y_pred_id=y_pred_id,
                y_pred_c=y_pred_c,
                cand_family=str(cand_families[m]),
                kind="candidate",
            )
            if str(selector_model).strip().lower() == "evidential":
                feats, _feat_names, _view_slices = _selector_feature_bundle(
                    p_id=p_id,
                    p_c=p_c,
                    y_pred_id=y_pred_id,
                    y_pred_c=y_pred_c,
                    anchor_family=str(anchor_family).strip().lower(),
                    cand_family=str(cand_families[m]),
                    n_classes=int(n_classes),
                    feature_mode=str(feature_mode),
                    selector_views=selector_views,
                    dynamic_chunks=int(dynamic_chunks),
                    stochastic_bootstrap_rounds=int(stochastic_bootstrap_rounds),
                    stochastic_bootstrap_seed=int(stochastic_bootstrap_seed) + 1009 * int(s),
                )
            else:
                x0, names0 = candidate_features_from_record(rec_id, n_classes=n_classes, include_pbar=True)
                x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
                x, names = candidate_features_from_record(rec_c, n_classes=n_classes, include_pbar=True)
                if names != names0:
                    raise RuntimeError("Feature name mismatch between anchor and candidate features.")
                x = np.asarray(x, dtype=np.float64).reshape(-1)
                feats, _feat_names = _features_from_anchor_and_candidate(
                    x_anchor=x0,
                    x_candidate=x,
                    names=tuple(names0),
                    feature_mode=str(feature_mode),
                )
            feats = np.asarray(feats, dtype=np.float64).reshape(1, -1)

            pred_improve, p_pos, selector_extra = _selector_outputs(
                cert=cert,
                guard=guard,
                feats=feats,
            )
            if str(selector_model).strip().lower() == "evidential":
                guard_ok = p_pos >= float(guard_threshold)
            elif str(guard_target).strip().lower() == "harm":
                p_pos = 1.0 - p_pos  # P(non-harm)
                guard_ok = p_pos >= float(guard_threshold)
            else:
                guard_ok = (p_pos >= float(guard_threshold)) and (p_pos >= float(p_pos_anchor) + float(anchor_guard_delta))
            if bool(disable_guard):
                guard_ok = True
            cands.append(
                {
                    "method": str(m),
                    "pred_improve": float(pred_improve),
                    "guard_p_pos": float(p_pos),
                    "guard_ok": bool(guard_ok),
                    "acc": float(acc_c),
                    **selector_extra,
                }
            )

        out.append({"subject": int(s), "acc_anchor": float(acc_id), "cands": cands})
    return out


def _subject_has_guard_ok_and_score(
    row: dict[str, object],
    *,
    select_score: str,
    score_scope: str,
) -> tuple[bool, float]:
    cands = row.get("cands", [])
    if str(score_scope) not in {"guard_ok", "all"}:
        raise ValueError(f"Invalid score_scope={score_scope!r} (expected 'guard_ok' or 'all').")

    has_guard_ok = False
    vals: list[float] = []
    for c in cands:
        c = dict(c)
        guard_ok = bool(c.get("guard_ok", False))
        if guard_ok:
            has_guard_ok = True
        if str(score_scope) == "guard_ok" and not guard_ok:
            continue
        v = float(c.get(str(select_score), float("nan")))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return bool(has_guard_ok), float("-inf")
    return bool(has_guard_ok), float(max(vals))


def _select_subject_ids(
    subjects: list[int],
    scores: list[float],
    *,
    select_topm: int,
    select_fraction: float,
) -> set[int]:
    n_total = int(len(subjects))
    if int(select_topm) <= 0 and float(select_fraction) <= 0.0:
        return set(int(s) for s in subjects)

    finite: list[tuple[int, float]] = []
    for s, sc in zip(subjects, scores):
        sc = float(sc)
        if np.isfinite(sc):
            finite.append((int(s), sc))

    if not finite or n_total <= 0:
        return set()

    finite.sort(key=lambda t: (-float(t[1]), int(t[0])))
    if int(select_topm) > 0:
        k = min(int(select_topm), int(len(finite)))
    else:
        k_target = int(np.ceil(float(select_fraction) * float(n_total)))
        k = min(int(max(0, k_target)), int(len(finite)))
    if k <= 0:
        return set()
    return set(s for s, _ in finite[:k])


def _eval_precomputed(
    precomputed: list[dict[str, object]],
    *,
    min_pred_improve: float,
    neg_transfer_eps: float = 0.0,
    select_topm: int = 0,
    select_fraction: float = 0.0,
    select_score: str = "guard_p_pos",
    select_score_scope: str = "guard_ok",
    threshold_score: str = "pred_improve",
    candidate_choice_score: str = "pred_improve",
    require_guard_ok: bool = True,
) -> dict[str, float]:
    subjects = [int(r.get("subject", -1)) for r in precomputed]
    has_guard_ok = []
    scores = []
    for row in precomputed:
        # When guard is disabled, treat score_scope as "all" to avoid dropping all candidates.
        eff_scope = str(select_score_scope)
        if not bool(require_guard_ok) and eff_scope == "guard_ok":
            eff_scope = "all"

        hg, sc = _subject_has_guard_ok_and_score(
            row,
            select_score=str(select_score),
            score_scope=str(eff_scope),
        )
        # For stage breakdown: when guard is disabled, the "has_guard_ok" gate is always satisfied
        # as long as there exists at least one candidate.
        if not bool(require_guard_ok):
            has_guard_ok.append(True)
        else:
            has_guard_ok.append(bool(hg))
        scores.append(float(sc))

    selected_ids = _select_subject_ids(
        subjects,
        scores,
        select_topm=int(select_topm),
        select_fraction=float(select_fraction),
    )

    acc_sel = []
    delta = []
    accepted = []
    selected_flags = []
    for row in precomputed:
        subj = int(row.get("subject", -1))
        selected = int(subj in selected_ids)
        selected_flags.append(int(selected))
        acc_id = float(row["acc_anchor"])
        if not bool(selected):
            acc_sel.append(float(acc_id))
            delta.append(0.0)
            accepted.append(0)
            continue

        best = None
        best_score = float("-inf")
        for c in row["cands"]:
            c = dict(c)
            if bool(require_guard_ok) and not bool(c.get("guard_ok", False)):
                continue
            if str(threshold_score) == "pred_improve_x_guard":
                thr_val = float(c.get("pred_improve", float("-inf"))) * float(c.get("guard_p_pos", float("nan")))
            else:
                thr_val = float(c.get(str(threshold_score), float("-inf")))
            if not np.isfinite(thr_val) or thr_val < float(min_pred_improve):
                continue
            if str(candidate_choice_score) == "pred_improve_x_guard":
                sc_val = float(c.get("pred_improve", float("-inf"))) * float(c.get("guard_p_pos", float("nan")))
            else:
                sc_val = float(c.get(str(candidate_choice_score), float("-inf")))
            if not np.isfinite(sc_val):
                continue
            if sc_val > best_score:
                best_score = sc_val
                best = c
        if best is None:
            acc_s = float(acc_id)
            accepted.append(0)
        else:
            acc_s = float(best["acc"])
            accepted.append(1)
        acc_sel.append(float(acc_s))
        delta.append(float(acc_s - acc_id))

    arr_acc = np.asarray(acc_sel, dtype=float)
    arr_delta = np.asarray(delta, dtype=float)
    arr_sel = np.asarray(selected_flags, dtype=int)
    n_eval = int(arr_delta.size)
    n_selected = int(np.sum(arr_sel)) if arr_sel.size else 0
    sel_mask = arr_sel.astype(bool)

    eps = float(neg_transfer_eps)
    n_neg = int(np.sum(arr_delta < -eps)) if n_eval > 0 else 0
    n_accept = int(np.sum(np.asarray(accepted, dtype=int))) if accepted else 0

    n_neg_sel = int(np.sum(arr_delta[sel_mask] < -eps)) if n_selected > 0 else 0
    n_accept_sel = int(np.sum(np.asarray(accepted, dtype=int)[sel_mask])) if n_selected > 0 else 0

    mean_delta_all = float(np.mean(arr_delta)) if arr_delta.size else float("nan")
    if n_accept > 0:
        mean_delta_accepted = float(np.mean(arr_delta[np.asarray(accepted, dtype=int) == 1]))
    else:
        mean_delta_accepted = float("nan")

    return {
        # Back-compat keys (total, i.e., denominator=n_subjects_total).
        "n_subjects": float(n_eval),
        "n_neg_transfer": float(n_neg),
        "n_accept": float(n_accept),
        "mean_acc": float(np.mean(arr_acc)) if arr_acc.size else float("nan"),
        "worst_acc": float(np.min(arr_acc)) if arr_acc.size else float("nan"),
        "neg_transfer_rate": float(n_neg / n_eval) if n_eval > 0 else float("nan"),
        "accept_rate": float(n_accept / n_eval) if n_eval > 0 else float("nan"),
        # SCRC-style selection metrics.
        "n_subjects_total": float(n_eval),
        "n_selected": float(n_selected),
        "coverage": float(n_selected / n_eval) if n_eval > 0 else float("nan"),
        "n_neg_transfer_total": float(n_neg),
        "n_accept_total": float(n_accept),
        "n_neg_transfer_selected": float(n_neg_sel),
        "n_accept_selected": float(n_accept_sel),
        "neg_transfer_rate_total": float(n_neg / n_eval) if n_eval > 0 else float("nan"),
        "accept_rate_total": float(n_accept / n_eval) if n_eval > 0 else float("nan"),
        "neg_transfer_rate_selected": float(n_neg_sel / n_selected) if n_selected > 0 else float("nan"),
        "accept_rate_selected": float(n_accept_sel / n_selected) if n_selected > 0 else float("nan"),
        "cond_neg_transfer_rate": float(n_neg_sel / n_accept_sel) if n_accept_sel > 0 else float("nan"),
        "mean_delta_all": float(mean_delta_all),
        "mean_delta_accepted": float(mean_delta_accepted),
    }


def _calib_row_from_eval(*, thr: float, m: dict[str, float], delta: float) -> dict[str, float]:
    n_total = int(m.get("n_subjects_total", m.get("n_subjects", 0)))
    k_marg = int(m.get("n_neg_transfer_total", m.get("n_neg_transfer", 0)))
    n_selected = int(m.get("n_selected", n_total))
    n_accept = int(m.get("n_accept_total", m.get("n_accept", 0)))
    n_cond = int(m.get("n_accept_selected", 0))
    k_cond = int(m.get("n_neg_transfer_selected", 0))

    marg_rate = float(k_marg / n_total) if n_total > 0 else float("nan")
    marg_ucb = _clopper_pearson_ucb(k=k_marg, n=n_total, delta=float(delta)) if n_total > 0 else float("nan")

    if n_cond > 0:
        cond_rate = float(k_cond / n_cond)
        cond_ucb = _clopper_pearson_ucb(k=k_cond, n=n_cond, delta=float(delta))
        cond_ci_low, cond_ci_high = _clopper_pearson_ci(k=k_cond, n=n_cond, delta=float(delta))
    else:
        cond_rate = float("nan")
        cond_ucb = float("nan")
        cond_ci_low, cond_ci_high = float("nan"), float("nan")

    return {
        "min_pred": float(thr),
        "mean_acc": float(m.get("mean_acc", float("nan"))),
        "worst_acc": float(m.get("worst_acc", float("nan"))),
        "mean_delta_all": float(m.get("mean_delta_all", float("nan"))),
        "mean_delta_accepted": float(m.get("mean_delta_accepted", float("nan"))),
        "marg_rate": float(marg_rate),
        "marg_ucb": float(marg_ucb),
        "cond_rate": float(cond_rate),
        "cond_ucb": float(cond_ucb),
        "cond_ci_low": float(cond_ci_low),
        "cond_ci_high": float(cond_ci_high),
        "accept": float(m.get("accept_rate_total", m.get("accept_rate", float("nan")))),
        "coverage": float(m.get("coverage", float("nan"))),
        "accept_selected": float(m.get("accept_rate_selected", float("nan"))),
        "negT_total": float(m.get("neg_transfer_rate_total", m.get("neg_transfer_rate", float("nan")))),
        "negT_selected": float(m.get("neg_transfer_rate_selected", float("nan"))),
        "n_subjects_total": float(n_total),
        "n_selected": float(n_selected),
        "n_accept_total": float(n_accept),
        "n_accept_selected": float(n_cond),
        "k_neg_marg": float(k_marg),
        "k_neg_cond": float(k_cond),
    }


def _constraint_flags(
    *,
    row: dict[str, float],
    risk_alpha: float,
    risk_mode: str,
    risk_beta_cond: float | None,
    min_accept_rate: float,
) -> dict[str, bool]:
    feasible_marg = bool(np.isfinite(row.get("marg_ucb", float("nan")))) and (
        float(row.get("marg_ucb", float("nan"))) <= float(risk_alpha) + 1e-12
    )
    if str(risk_mode) != "dual":
        return {
            "feasible_marg": bool(feasible_marg),
            "feasible_cond": False,
            "feasible_accept": False,
            "feasible_all": bool(feasible_marg),
        }

    beta = float(risk_beta_cond) if risk_beta_cond is not None else 1.0
    feasible_cond = bool(np.isfinite(row.get("cond_ucb", float("nan")))) and (
        float(row.get("cond_ucb", float("nan"))) <= beta + 1e-12
    )
    feasible_accept = bool(np.isfinite(row.get("accept", float("nan")))) and (
        float(row.get("accept", float("nan"))) >= float(min_accept_rate) - 1e-12
    )
    return {
        "feasible_marg": bool(feasible_marg),
        "feasible_cond": bool(feasible_cond),
        "feasible_accept": bool(feasible_accept),
        "feasible_all": bool(feasible_marg and feasible_cond and feasible_accept),
    }


def _select_best_row_by_objective(*, rows_df: pd.DataFrame, calib_objective: str) -> dict[str, float]:
    obj_col = "mean_delta_all" if str(calib_objective) == "mean_delta_all" else "mean_acc"
    df = rows_df.copy()
    if obj_col not in df.columns:
        obj_col = "mean_acc"
    df = df.sort_values([obj_col, "accept", "min_pred"], ascending=[False, False, True]).reset_index(drop=True)
    return df.iloc[0].to_dict()


def _subset_precomputed_by_subjects(
    precomputed: list[dict[str, object]],
    *,
    subjects: list[int],
) -> list[dict[str, object]]:
    subjects_set = {int(s) for s in subjects}
    return [row for row in precomputed if int(row.get("subject", -1)) in subjects_set]


def _choose_threshold_by_constraints(
    *,
    rows_df: pd.DataFrame,
    fallback_row: dict[str, float],
    risk_alpha: float,
    risk_mode: str,
    risk_beta_cond: float | None,
    min_accept_rate: float,
    calib_objective: str,
) -> tuple[float, dict[str, float]]:
    df = rows_df.sort_values("min_pred").reset_index(drop=True)

    feasible_marg = df[np.isfinite(df["marg_ucb"]) & (df["marg_ucb"] <= float(risk_alpha) + 1e-12)].copy()
    feasible = feasible_marg.copy()
    fallback_stage = "marginal_only"

    if str(risk_mode) == "dual":
        beta = float(risk_beta_cond) if risk_beta_cond is not None else 1.0
        feasible_cond = feasible_marg[
            np.isfinite(feasible_marg["cond_ucb"]) & (feasible_marg["cond_ucb"] <= beta + 1e-12)
        ].copy()
        if float(min_accept_rate) > 0.0:
            feasible_cond_acc = feasible_cond[
                np.isfinite(feasible_cond["accept"]) & (feasible_cond["accept"] >= float(min_accept_rate) - 1e-12)
            ].copy()
            feasible_marg_acc = feasible_marg[
                np.isfinite(feasible_marg["accept"]) & (feasible_marg["accept"] >= float(min_accept_rate) - 1e-12)
            ].copy()
        else:
            feasible_cond_acc = feasible_cond
            feasible_marg_acc = feasible_marg

        if not feasible_cond_acc.empty:
            feasible = feasible_cond_acc
            fallback_stage = "dual+accept" if float(min_accept_rate) > 0.0 else "dual"
        elif not feasible_cond.empty:
            feasible = feasible_cond
            fallback_stage = "fallback_drop_accept"
        elif not feasible_marg_acc.empty:
            feasible = feasible_marg_acc
            fallback_stage = "fallback_drop_cond"
        else:
            feasible = feasible_marg
            fallback_stage = "fallback_marginal_only"

    if feasible.empty:
        if feasible_marg.empty:
            best_row = dict(fallback_row)
            best_row["feasible_empty"] = 1.0
            best_row["fallback_stage"] = "fallback_all_abstain"
        else:
            feasible = feasible_marg.copy()
            fallback_stage = "fallback_marginal_only"
            obj_col = "mean_delta_all" if str(calib_objective) == "mean_delta_all" else "mean_acc"
            if obj_col not in feasible.columns:
                obj_col = "mean_acc"
            feasible = feasible.sort_values([obj_col, "accept", "min_pred"], ascending=[False, False, True])
            best_row = feasible.iloc[0].to_dict()
            best_row["feasible_empty"] = 0.0
            best_row["fallback_stage"] = fallback_stage
    else:
        obj_col = "mean_delta_all" if str(calib_objective) == "mean_delta_all" else "mean_acc"
        if obj_col not in feasible.columns:
            obj_col = "mean_acc"
        feasible = feasible.sort_values([obj_col, "accept", "min_pred"], ascending=[False, False, True])
        best_row = feasible.iloc[0].to_dict()
        best_row["feasible_empty"] = 0.0
        best_row["fallback_stage"] = fallback_stage

    thr = float(best_row.get("min_pred", float("inf")))
    n_calib = int(best_row.get("n_subjects_total", 0))
    n_sel = int(best_row.get("n_selected", 0))
    stats = {
        "calib_mean_acc": float(best_row.get("mean_acc", float("nan"))),
        "calib_worst_acc": float(best_row.get("worst_acc", float("nan"))),
        "calib_mean_delta_all": float(best_row.get("mean_delta_all", float("nan"))),
        "calib_mean_delta_accepted": float(best_row.get("mean_delta_accepted", float("nan"))),
        "calib_emp_negT": float(best_row.get("marg_rate", float("nan"))),
        "calib_ucb_negT": float(best_row.get("marg_ucb", float("nan"))),
        "calib_cond_rate": float(best_row.get("cond_rate", float("nan"))),
        "calib_cond_ucb": float(best_row.get("cond_ucb", float("nan"))),
        "calib_cond_ci_low": float(best_row.get("cond_ci_low", float("nan"))),
        "calib_cond_ci_high": float(best_row.get("cond_ci_high", float("nan"))),
        "calib_accept": float(best_row.get("accept", float("nan"))),
        "calib_coverage": float(best_row.get("coverage", float("nan"))),
        "calib_accept_selected": float(best_row.get("accept_selected", float("nan"))),
        "calib_negT_total": float(best_row.get("negT_total", float("nan"))),
        "calib_negT_selected": float(best_row.get("negT_selected", float("nan"))),
        "calib_n_subjects": int(n_calib),
        "calib_n_subjects_total": int(n_calib),
        "calib_n_selected": int(n_sel),
        "calib_risk_mode": 1.0 if str(risk_mode) == "dual" else 0.0,
        "calib_risk_beta_cond": float(risk_beta_cond) if risk_beta_cond is not None else float("nan"),
        "calib_min_accept_rate": float(min_accept_rate),
        "feasible_empty": float(best_row.get("feasible_empty", 0.0)),
        "fallback_stage": str(best_row.get("fallback_stage", "marginal_only")),
        "calib_threshold_objective": str(calib_objective),
    }
    return thr, stats


def _guard_label_from_improve(
    *,
    improve: float,
    guard_target: str,
    neg_transfer_eps: float,
    guard_gray_margin: float,
) -> int | None:
    improve = float(improve)
    mode = str(guard_target).strip().lower()
    margin = max(0.0, float(guard_gray_margin))
    eps = max(0.0, float(neg_transfer_eps))

    if margin > 0.0:
        if mode == "harm":
            if improve < -margin:
                return 1
            if improve > margin:
                return 0
            return None
        if improve > margin:
            return 1
        if improve < -margin:
            return 0
        return None

    if mode == "harm":
        return 1 if improve < -eps else 0
    return 1 if improve > 0.0 else 0


def _train_cert_guard(
    *,
    fit_subjects: list[int],
    by_subject_anchor: dict[int, pd.DataFrame],
    by_subject_cands: dict[str, dict[int, pd.DataFrame]],
    proba_cols: list[str],
    n_classes: int,
    cand_methods: list[str],
    cand_families: dict[str, str],
    anchor_family: str,
    ridge_alpha: float,
    guard_c: float,
    guard_target: str,
    cert_model: str,
    guard_model: str,
    hgb_max_iter: int,
    model_seed: int,
    feature_mode: str,
    selector_views: list[str],
    dynamic_chunks: int,
    stochastic_bootstrap_rounds: int,
    stochastic_bootstrap_seed: int,
    neg_transfer_eps: float,
    guard_gray_margin: float,
    selector_model: str,
    selector_hidden_dim: int,
    selector_epochs: int,
    selector_lr: float,
    selector_weight_decay: float,
    selector_lambda_rank: float,
    selector_lambda_kl: float,
    selector_outcome_delta: float,
    selector_rank_margin: float,
    selector_rho: float,
    selector_eta: float,
    progress_label: str | None = None,
) -> tuple[object, object, float, tuple[str, ...]]:
    # Build training set: one row per (fit subject, candidate method).
    selector_mode = str(selector_model).strip().lower()
    X_rows: list[np.ndarray] = []
    y_improve: list[float] = []
    X_guard_rows: list[np.ndarray] = []
    y_guard: list[int] = []
    y_state: list[int] = []
    group_ids: list[int] = []
    X_anchor_rows: list[np.ndarray] = []
    feat_names: tuple[str, ...] | None = None
    selector_view_slices: dict[str, tuple[int, int]] | None = None

    for s in fit_subjects:
        df_s_id = by_subject_anchor[int(s)]
        y_true_s = df_s_id["y_true"].to_numpy(object)
        p_id = df_s_id[proba_cols].to_numpy(np.float64)
        y_pred_id = df_s_id["y_pred"].to_numpy(object)
        acc_id = float(np.mean(y_pred_id == y_true_s))

        rec_id = _record_from_proba(
            p_id=p_id,
            p_c=p_id,
            y_pred_id=y_pred_id,
            y_pred_c=y_pred_id,
            cand_family=str(anchor_family).strip().lower(),
            kind="identity",
        )
        x0, names0 = candidate_features_from_record(rec_id, n_classes=n_classes, include_pbar=True)
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if selector_mode == "evidential":
            x_anchor_feat, names_feat, view_slices = _selector_feature_bundle(
                p_id=p_id,
                p_c=p_id,
                y_pred_id=y_pred_id,
                y_pred_c=y_pred_id,
                anchor_family=str(anchor_family).strip().lower(),
                cand_family=str(anchor_family).strip().lower(),
                n_classes=int(n_classes),
                feature_mode=str(feature_mode),
                selector_views=selector_views,
                dynamic_chunks=int(dynamic_chunks),
                stochastic_bootstrap_rounds=int(stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(stochastic_bootstrap_seed) + 1009 * int(s),
            )
            selector_view_slices = dict(view_slices)
        else:
            x_anchor_feat, _anchor_names = _features_from_anchor_and_candidate(
                x_anchor=x0,
                x_candidate=x0,
                names=tuple(names0),
                feature_mode=str(feature_mode),
            )
            names_feat = tuple([f"anchor_{n}" for n in range(int(np.asarray(x_anchor_feat).reshape(-1).shape[0]))])
        X_anchor_rows.append(np.asarray(x_anchor_feat, dtype=np.float64).reshape(-1))

        for m in cand_methods:
            df_s_c = by_subject_cands[m][int(s)]
            y_pred_c = df_s_c["y_pred"].to_numpy(object)
            p_c = df_s_c[proba_cols].to_numpy(np.float64)
            acc_c = float(np.mean(y_pred_c == y_true_s))
            improve = float(acc_c - acc_id)

            rec_c = _record_from_proba(
                p_id=p_id,
                p_c=p_c,
                y_pred_id=y_pred_id,
                y_pred_c=y_pred_c,
                cand_family=str(cand_families[m]),
                kind="candidate",
            )
            x, names = candidate_features_from_record(rec_c, n_classes=n_classes, include_pbar=True)
            if names != names0:
                raise RuntimeError("Feature name mismatch between anchor and candidate features.")
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if selector_mode == "evidential":
                x_feat, names_feat, view_slices = _selector_feature_bundle(
                    p_id=p_id,
                    p_c=p_c,
                    y_pred_id=y_pred_id,
                    y_pred_c=y_pred_c,
                    anchor_family=str(anchor_family).strip().lower(),
                    cand_family=str(cand_families[m]),
                    n_classes=int(n_classes),
                    feature_mode=str(feature_mode),
                    selector_views=selector_views,
                    dynamic_chunks=int(dynamic_chunks),
                    stochastic_bootstrap_rounds=int(stochastic_bootstrap_rounds),
                    stochastic_bootstrap_seed=int(stochastic_bootstrap_seed) + 1009 * int(s),
                )
                selector_view_slices = dict(view_slices)
            else:
                x_feat, names_feat = _features_from_anchor_and_candidate(
                    x_anchor=x0,
                    x_candidate=x,
                    names=tuple(names0),
                    feature_mode=str(feature_mode),
                )
            if feat_names is None:
                feat_names = tuple(names_feat)

            X_rows.append(np.asarray(x_feat, dtype=np.float64).reshape(-1))
            y_improve.append(improve)
            y_state.append(
                _outcome_state_from_improve(
                    improve=float(improve),
                    outcome_delta=float(selector_outcome_delta),
                )
            )
            group_ids.append(int(s))
            guard_label = _guard_label_from_improve(
                improve=float(improve),
                guard_target=str(guard_target),
                neg_transfer_eps=float(neg_transfer_eps),
                guard_gray_margin=float(guard_gray_margin),
            )
            if guard_label is not None:
                X_guard_rows.append(np.asarray(x_feat, dtype=np.float64).reshape(-1))
                y_guard.append(int(guard_label))

    if feat_names is None or not X_rows:
        raise RuntimeError("No training rows.")
    X_tr = np.vstack(X_rows).astype(np.float64, copy=False)
    y_tr = np.asarray(y_improve, dtype=np.float64)
    y_state_tr = np.asarray(y_state, dtype=int)
    group_ids_tr = np.asarray(group_ids, dtype=int)
    X_guard = np.vstack(X_guard_rows).astype(np.float64, copy=False) if X_guard_rows else np.zeros((0, X_tr.shape[1]), dtype=np.float64)
    yb_tr = np.asarray(y_guard, dtype=int)

    cert_model = str(cert_model).strip().lower()
    if progress_label:
        print(
            f"[train] {progress_label} fit_subjects={len(fit_subjects)} rows={int(X_tr.shape[0])} guard_rows={int(X_guard.shape[0])} selector={selector_mode}",
            flush=True,
        )
    if selector_mode == "evidential":
        selector = train_evidential_selector(
            X_tr,
            y_state_tr,
            y_tr,
            group_ids_tr,
            feature_names=feat_names,
            view_slices=selector_view_slices,
            hidden_dim=int(selector_hidden_dim),
            lr=float(selector_lr),
            weight_decay=float(selector_weight_decay),
            epochs=int(selector_epochs),
            lambda_rank=float(selector_lambda_rank),
            lambda_kl=float(selector_lambda_kl),
            pair_margin=float(selector_rank_margin),
            rho=float(selector_rho),
            eta=float(selector_eta),
            seed=int(model_seed),
            progress_label=progress_label,
            progress_every=max(1, int(selector_epochs) // 4),
        )
        cert = selector
        guard = selector
    else:
        if cert_model == "hgb":
            cert = train_hgb_certificate(
                X_tr,
                y_tr,
                feature_names=feat_names,
                max_iter=int(hgb_max_iter),
                random_state=int(model_seed),
            )
        else:
            cert = train_ridge_certificate(X_tr, y_tr, feature_names=feat_names, alpha=float(ridge_alpha))
        try:
            guard_model = str(guard_model).strip().lower()
            if X_guard.shape[0] <= 0 or yb_tr.size <= 0:
                raise RuntimeError("No non-gray rows available for guard training.")
            if guard_model == "hgb":
                guard = train_hgb_guard(
                    X_guard,
                    yb_tr,
                    feature_names=feat_names,
                    max_iter=int(hgb_max_iter),
                    random_state=int(model_seed),
                )
            else:
                guard = train_logistic_guard(X_guard, yb_tr, feature_names=feat_names, c=float(guard_c))
        except Exception:
            guard = _ConstantGuard(p_pos=float(np.mean(yb_tr)) if yb_tr.size else 0.5, feature_names=feat_names)

    if X_anchor_rows:
        X_anchor = np.vstack(X_anchor_rows).astype(np.float64, copy=False)
        p_guard_anchor = float(np.nanmedian(guard.predict_pos_proba(X_anchor)))
    else:
        p_guard_anchor = float(guard.predict_pos_proba(np.zeros((1, X_tr.shape[1]), dtype=np.float64))[0])
    if selector_mode == "evidential":
        p_pos_anchor = float(p_guard_anchor)
    elif str(guard_target).strip().lower() == "harm":
        p_pos_anchor = 1.0 - p_guard_anchor
    else:
        p_pos_anchor = p_guard_anchor
    return cert, guard, p_pos_anchor, feat_names


def _calibrate_min_pred(
    *,
    test_subject: int,
    calib_subjects: list[int],
    by_subject_anchor: dict[int, pd.DataFrame],
    by_subject_cands: dict[str, dict[int, pd.DataFrame]],
    proba_cols: list[str],
    n_classes: int,
    cand_methods: list[str],
    cand_families: dict[str, str],
    anchor_family: str,
    cert: object,
    guard: object,
    p_pos_anchor: float,
    guard_threshold: float,
    anchor_guard_delta: float,
    min_pred_grid: list[float],
    risk_alpha: float,
    delta: float,
    neg_transfer_eps: float,
    select_topm: int,
    select_fraction: float,
    select_score: str,
    select_score_scope: str,
    threshold_score: str,
    candidate_choice_score: str,
    require_guard_ok: bool,
    disable_guard: bool,
    guard_target: str,
    feature_mode: str,
    selector_model: str,
    selector_views: list[str],
    dynamic_chunks: int,
    stochastic_bootstrap_rounds: int,
    stochastic_bootstrap_seed: int,
    risk_mode: str,
    risk_beta_cond: float | None,
    min_accept_rate: float,
    calib_objective: str,
    calib_grid_dump_rows: list[dict[str, float]] | None = None,
) -> tuple[float, dict[str, object]]:
    n_calib = int(len(calib_subjects))
    precomputed = _precompute_subject_candidates(
        subjects_eval=calib_subjects,
        by_subject_anchor=by_subject_anchor,
        by_subject_cands=by_subject_cands,
        proba_cols=proba_cols,
        n_classes=n_classes,
        cand_methods=cand_methods,
        cand_families=cand_families,
        anchor_family=str(anchor_family),
        cert=cert,
        guard=guard,
        p_pos_anchor=p_pos_anchor,
        guard_threshold=float(guard_threshold),
        anchor_guard_delta=float(anchor_guard_delta),
        disable_guard=bool(disable_guard),
        guard_target=str(guard_target),
        feature_mode=str(feature_mode),
        selector_model=str(selector_model),
        selector_views=selector_views,
        dynamic_chunks=int(dynamic_chunks),
        stochastic_bootstrap_rounds=int(stochastic_bootstrap_rounds),
        stochastic_bootstrap_seed=int(stochastic_bootstrap_seed),
    )
    rows: list[dict[str, float]] = []
    for thr in min_pred_grid:
        m = _eval_precomputed(
            precomputed,
            min_pred_improve=float(thr),
            neg_transfer_eps=float(neg_transfer_eps),
            select_topm=int(select_topm),
            select_fraction=float(select_fraction),
            select_score=str(select_score),
            select_score_scope=str(select_score_scope),
            threshold_score=str(threshold_score),
            candidate_choice_score=str(candidate_choice_score),
            require_guard_ok=bool(require_guard_ok),
        )
        rows.append(_calib_row_from_eval(thr=float(thr), m=m, delta=float(delta)))

    m0 = _eval_precomputed(
        precomputed,
        min_pred_improve=float("inf"),
        neg_transfer_eps=float(neg_transfer_eps),
        select_topm=int(select_topm),
        select_fraction=float(select_fraction),
        select_score=str(select_score),
        select_score_scope=str(select_score_scope),
        threshold_score=str(threshold_score),
        candidate_choice_score=str(candidate_choice_score),
        require_guard_ok=bool(require_guard_ok),
    )
    fallback_row = _calib_row_from_eval(thr=float("inf"), m=m0, delta=float(delta))
    df = pd.DataFrame(rows).sort_values("min_pred")
    if calib_grid_dump_rows is not None:
        dump = df.copy()
        dump["is_fallback"] = 0
        dump["test_subject"] = int(test_subject)
        dump["risk_alpha"] = float(risk_alpha)
        dump["risk_beta_cond"] = float(risk_beta_cond) if risk_beta_cond is not None else float("nan")
        dump["min_accept_rate"] = float(min_accept_rate)
        dump["risk_mode"] = str(risk_mode)
        dump["feasible_marg"] = np.isfinite(dump["marg_ucb"]) & (dump["marg_ucb"] <= float(risk_alpha) + 1e-12)
        if str(risk_mode) == "dual":
            beta = float(risk_beta_cond) if risk_beta_cond is not None else 1.0
            dump["feasible_cond"] = dump["feasible_marg"] & np.isfinite(dump["cond_ucb"]) & (dump["cond_ucb"] <= beta + 1e-12)
            dump["feasible_dual"] = dump["feasible_cond"] & np.isfinite(dump["accept"]) & (
                dump["accept"] >= float(min_accept_rate) - 1e-12
            )
        else:
            dump["feasible_cond"] = False
            dump["feasible_dual"] = False

        dump_records = dump.to_dict(orient="records")
        fb = dict(fallback_row)
        fb["is_fallback"] = 1
        fb["test_subject"] = int(test_subject)
        fb["risk_alpha"] = float(risk_alpha)
        fb["risk_beta_cond"] = float(risk_beta_cond) if risk_beta_cond is not None else float("nan")
        fb["min_accept_rate"] = float(min_accept_rate)
        fb["risk_mode"] = str(risk_mode)
        fb["feasible_marg"] = False
        fb["feasible_cond"] = False
        fb["feasible_dual"] = False
        dump_records.append(fb)
        calib_grid_dump_rows.extend(dump_records)

    return _choose_threshold_by_constraints(
        rows_df=df,
        fallback_row=fallback_row,
        risk_alpha=float(risk_alpha),
        risk_mode=str(risk_mode),
        risk_beta_cond=(None if risk_beta_cond is None else float(risk_beta_cond)),
        min_accept_rate=float(min_accept_rate),
        calib_objective=str(calib_objective),
    )


def _calibrate_min_pred_crossfit(
    *,
    test_subject: int,
    folds: list[list[int]],
    fold_models: list[dict[str, object]],
    by_subject_anchor: dict[int, pd.DataFrame],
    by_subject_cands: dict[str, dict[int, pd.DataFrame]],
    proba_cols: list[str],
    n_classes: int,
    cand_methods: list[str],
    cand_families: dict[str, str],
    anchor_family: str,
    min_pred_grid: list[float],
    risk_alpha: float,
    delta: float,
    neg_transfer_eps: float,
    guard_threshold: float,
    anchor_guard_delta: float,
    select_topm: int,
    select_fraction: float,
    select_score: str,
    select_score_scope: str,
    threshold_score: str,
    candidate_choice_score: str,
    require_guard_ok: bool,
    disable_guard: bool,
    guard_target: str,
    feature_mode: str,
    selector_model: str,
    selector_views: list[str],
    dynamic_chunks: int,
    stochastic_bootstrap_rounds: int,
    stochastic_bootstrap_seed: int,
    risk_mode: str,
    risk_beta_cond: float | None,
    min_accept_rate: float,
    calib_objective: str,
    calib_grid_dump_rows: list[dict[str, float]] | None = None,
) -> tuple[float, dict[str, object]]:
    rows: list[dict[str, float]] = []
    precomputed_parts = []
    all_subjects = []
    for fold_idx, fold_subjs in enumerate(folds):
        all_subjects.extend([int(s) for s in fold_subjs])
        cert = fold_models[fold_idx]["cert"]
        guard = fold_models[fold_idx]["guard"]
        p_pos_anchor = float(fold_models[fold_idx]["p_pos_anchor"])
        precomputed_parts.append(
            _precompute_subject_candidates(
                subjects_eval=[int(s) for s in fold_subjs],
                by_subject_anchor=by_subject_anchor,
                by_subject_cands=by_subject_cands,
                proba_cols=proba_cols,
                n_classes=n_classes,
                cand_methods=cand_methods,
                cand_families=cand_families,
                anchor_family=str(anchor_family),
                cert=cert,
                guard=guard,
                p_pos_anchor=p_pos_anchor,
                guard_threshold=float(guard_threshold),
                anchor_guard_delta=float(anchor_guard_delta),
                disable_guard=bool(disable_guard),
                guard_target=str(guard_target),
                feature_mode=str(feature_mode),
                selector_model=str(selector_model),
                selector_views=selector_views,
                dynamic_chunks=int(dynamic_chunks),
                stochastic_bootstrap_rounds=int(stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(stochastic_bootstrap_seed),
            )
        )

    n_total = int(len(all_subjects))
    if n_total <= 0:
        raise RuntimeError("Empty crossfit calibration set.")

    precomputed = []
    for part in precomputed_parts:
        precomputed.extend(list(part))

    for thr in min_pred_grid:
        m = _eval_precomputed(
            precomputed,
            min_pred_improve=float(thr),
            neg_transfer_eps=float(neg_transfer_eps),
            select_topm=int(select_topm),
            select_fraction=float(select_fraction),
            select_score=str(select_score),
            select_score_scope=str(select_score_scope),
            threshold_score=str(threshold_score),
            candidate_choice_score=str(candidate_choice_score),
            require_guard_ok=bool(require_guard_ok),
        )
        rows.append(_calib_row_from_eval(thr=float(thr), m=m, delta=float(delta)))

    m0 = _eval_precomputed(
        precomputed,
        min_pred_improve=float("inf"),
        neg_transfer_eps=float(neg_transfer_eps),
        select_topm=int(select_topm),
        select_fraction=float(select_fraction),
        select_score=str(select_score),
        select_score_scope=str(select_score_scope),
        threshold_score=str(threshold_score),
        candidate_choice_score=str(candidate_choice_score),
        require_guard_ok=bool(require_guard_ok),
    )
    fallback_row = _calib_row_from_eval(thr=float("inf"), m=m0, delta=float(delta))
    df = pd.DataFrame(rows).sort_values("min_pred")
    if calib_grid_dump_rows is not None:
        dump = df.copy()
        dump["is_fallback"] = 0
        dump["test_subject"] = int(test_subject)
        dump["risk_alpha"] = float(risk_alpha)
        dump["risk_beta_cond"] = float(risk_beta_cond) if risk_beta_cond is not None else float("nan")
        dump["min_accept_rate"] = float(min_accept_rate)
        dump["risk_mode"] = str(risk_mode)
        dump["feasible_marg"] = np.isfinite(dump["marg_ucb"]) & (dump["marg_ucb"] <= float(risk_alpha) + 1e-12)
        if str(risk_mode) == "dual":
            beta = float(risk_beta_cond) if risk_beta_cond is not None else 1.0
            dump["feasible_cond"] = dump["feasible_marg"] & np.isfinite(dump["cond_ucb"]) & (dump["cond_ucb"] <= beta + 1e-12)
            dump["feasible_dual"] = dump["feasible_cond"] & np.isfinite(dump["accept"]) & (
                dump["accept"] >= float(min_accept_rate) - 1e-12
            )
        else:
            dump["feasible_cond"] = False
            dump["feasible_dual"] = False

        dump_records = dump.to_dict(orient="records")
        fb = dict(fallback_row)
        fb["is_fallback"] = 1
        fb["test_subject"] = int(test_subject)
        fb["risk_alpha"] = float(risk_alpha)
        fb["risk_beta_cond"] = float(risk_beta_cond) if risk_beta_cond is not None else float("nan")
        fb["min_accept_rate"] = float(min_accept_rate)
        fb["risk_mode"] = str(risk_mode)
        fb["feasible_marg"] = False
        fb["feasible_cond"] = False
        fb["feasible_dual"] = False
        dump_records.append(fb)
        calib_grid_dump_rows.extend(dump_records)

    return _choose_threshold_by_constraints(
        rows_df=df,
        fallback_row=fallback_row,
        risk_alpha=float(risk_alpha),
        risk_mode=str(risk_mode),
        risk_beta_cond=(None if risk_beta_cond is None else float(risk_beta_cond)),
        min_accept_rate=float(min_accept_rate),
        calib_objective=str(calib_objective),
    )


def _calibrate_min_pred_paper_dev_cal(
    *,
    test_subject: int,
    precomputed_oof: list[dict[str, object]],
    dev_subjects: list[int],
    cal_subjects: list[int],
    min_pred_grid: list[float],
    risk_alpha: float,
    delta: float,
    neg_transfer_eps: float,
    select_topm: int,
    select_fraction: float,
    select_score: str,
    select_score_scope: str,
    threshold_score: str,
    candidate_choice_score: str,
    require_guard_ok: bool,
    risk_mode: str,
    risk_beta_cond: float | None,
    min_accept_rate: float,
    calib_objective: str,
    calib_grid_dump_rows: list[dict[str, float]] | None = None,
) -> tuple[float, dict[str, object]]:
    precomputed_dev = _subset_precomputed_by_subjects(precomputed_oof, subjects=dev_subjects)
    precomputed_cal = _subset_precomputed_by_subjects(precomputed_oof, subjects=cal_subjects)
    if not precomputed_dev:
        raise RuntimeError("Empty D_dev in paper_oof_dev_cal protocol.")
    if not precomputed_cal:
        raise RuntimeError("Empty D_cal in paper_oof_dev_cal protocol.")

    dev_rows: list[dict[str, float]] = []
    for thr in min_pred_grid:
        m_dev = _eval_precomputed(
            precomputed_dev,
            min_pred_improve=float(thr),
            neg_transfer_eps=float(neg_transfer_eps),
            select_topm=int(select_topm),
            select_fraction=float(select_fraction),
            select_score=str(select_score),
            select_score_scope=str(select_score_scope),
            threshold_score=str(threshold_score),
            candidate_choice_score=str(candidate_choice_score),
            require_guard_ok=bool(require_guard_ok),
        )
        dev_rows.append(_calib_row_from_eval(thr=float(thr), m=m_dev, delta=float(delta)))

    m_dev_fb = _eval_precomputed(
        precomputed_dev,
        min_pred_improve=float("inf"),
        neg_transfer_eps=float(neg_transfer_eps),
        select_topm=int(select_topm),
        select_fraction=float(select_fraction),
        select_score=str(select_score),
        select_score_scope=str(select_score_scope),
        threshold_score=str(threshold_score),
        candidate_choice_score=str(candidate_choice_score),
        require_guard_ok=bool(require_guard_ok),
    )
    dev_fallback_row = _calib_row_from_eval(thr=float("inf"), m=m_dev_fb, delta=float(delta))
    dev_df = pd.DataFrame(dev_rows + [dev_fallback_row]).sort_values("min_pred").reset_index(drop=True)
    best_dev_row = _select_best_row_by_objective(rows_df=dev_df, calib_objective=str(calib_objective))
    chosen_thr = float(best_dev_row.get("min_pred", float("inf")))

    if np.isfinite(chosen_thr):
        m_cal = _eval_precomputed(
            precomputed_cal,
            min_pred_improve=float(chosen_thr),
            neg_transfer_eps=float(neg_transfer_eps),
            select_topm=int(select_topm),
            select_fraction=float(select_fraction),
            select_score=str(select_score),
            select_score_scope=str(select_score_scope),
            threshold_score=str(threshold_score),
            candidate_choice_score=str(candidate_choice_score),
            require_guard_ok=bool(require_guard_ok),
        )
        cal_row = _calib_row_from_eval(thr=float(chosen_thr), m=m_cal, delta=float(delta))
    else:
        cal_row = None

    m_cal_fb = _eval_precomputed(
        precomputed_cal,
        min_pred_improve=float("inf"),
        neg_transfer_eps=float(neg_transfer_eps),
        select_topm=int(select_topm),
        select_fraction=float(select_fraction),
        select_score=str(select_score),
        select_score_scope=str(select_score_scope),
        threshold_score=str(threshold_score),
        candidate_choice_score=str(candidate_choice_score),
        require_guard_ok=bool(require_guard_ok),
    )
    cal_fallback_row = _calib_row_from_eval(thr=float("inf"), m=m_cal_fb, delta=float(delta))

    verified = False
    if cal_row is not None:
        flags = _constraint_flags(
            row=cal_row,
            risk_alpha=float(risk_alpha),
            risk_mode=str(risk_mode),
            risk_beta_cond=(None if risk_beta_cond is None else float(risk_beta_cond)),
            min_accept_rate=float(min_accept_rate),
        )
        verified = bool(flags["feasible_all"])
    else:
        flags = {
            "feasible_marg": False,
            "feasible_cond": False,
            "feasible_accept": False,
            "feasible_all": False,
        }

    best_row = dict(cal_row) if (cal_row is not None and verified) else dict(cal_fallback_row)
    best_row["feasible_empty"] = 0.0
    best_row["fallback_stage"] = "paper_dev_cal_verified" if verified else "paper_dev_cal_fallback"

    if calib_grid_dump_rows is not None:
        dev_dump = dev_df.copy()
        dev_dump["grid_stage"] = "dev"
        dev_dump["test_subject"] = int(test_subject)
        dev_dump["risk_alpha"] = float(risk_alpha)
        dev_dump["risk_beta_cond"] = float(risk_beta_cond) if risk_beta_cond is not None else float("nan")
        dev_dump["min_accept_rate"] = float(min_accept_rate)
        dev_dump["risk_mode"] = str(risk_mode)
        calib_grid_dump_rows.extend(dev_dump.to_dict(orient="records"))

        verify_row = dict(cal_row) if cal_row is not None else dict(cal_fallback_row)
        verify_row["grid_stage"] = "cal_verify"
        verify_row["test_subject"] = int(test_subject)
        verify_row["risk_alpha"] = float(risk_alpha)
        verify_row["risk_beta_cond"] = float(risk_beta_cond) if risk_beta_cond is not None else float("nan")
        verify_row["min_accept_rate"] = float(min_accept_rate)
        verify_row["risk_mode"] = str(risk_mode)
        verify_row.update(flags)
        verify_row["verified"] = int(bool(verified))
        calib_grid_dump_rows.append(verify_row)

    n_calib = int(best_row.get("n_subjects_total", 0))
    n_sel = int(best_row.get("n_selected", 0))
    stats = {
        "calib_mean_acc": float(best_row.get("mean_acc", float("nan"))),
        "calib_worst_acc": float(best_row.get("worst_acc", float("nan"))),
        "calib_mean_delta_all": float(best_row.get("mean_delta_all", float("nan"))),
        "calib_mean_delta_accepted": float(best_row.get("mean_delta_accepted", float("nan"))),
        "calib_emp_negT": float(best_row.get("marg_rate", float("nan"))),
        "calib_ucb_negT": float(best_row.get("marg_ucb", float("nan"))),
        "calib_cond_rate": float(best_row.get("cond_rate", float("nan"))),
        "calib_cond_ucb": float(best_row.get("cond_ucb", float("nan"))),
        "calib_cond_ci_low": float(best_row.get("cond_ci_low", float("nan"))),
        "calib_cond_ci_high": float(best_row.get("cond_ci_high", float("nan"))),
        "calib_accept": float(best_row.get("accept", float("nan"))),
        "calib_coverage": float(best_row.get("coverage", float("nan"))),
        "calib_accept_selected": float(best_row.get("accept_selected", float("nan"))),
        "calib_negT_total": float(best_row.get("negT_total", float("nan"))),
        "calib_negT_selected": float(best_row.get("negT_selected", float("nan"))),
        "calib_n_subjects": int(n_calib),
        "calib_n_subjects_total": int(n_calib),
        "calib_n_selected": int(n_sel),
        "calib_risk_mode": 1.0 if str(risk_mode) == "dual" else 0.0,
        "calib_risk_beta_cond": float(risk_beta_cond) if risk_beta_cond is not None else float("nan"),
        "calib_min_accept_rate": float(min_accept_rate),
        "feasible_empty": 0.0,
        "fallback_stage": str(best_row.get("fallback_stage", "paper_dev_cal_fallback")),
        "calib_threshold_objective": str(calib_objective),
        "dev_subjects": int(len(dev_subjects)),
        "cal_subjects": int(len(cal_subjects)),
        "dev_chosen_min_pred": float(chosen_thr),
        "dev_mean_acc": float(best_dev_row.get("mean_acc", float("nan"))),
        "dev_mean_delta_all": float(best_dev_row.get("mean_delta_all", float("nan"))),
        "dev_accept": float(best_dev_row.get("accept", float("nan"))),
        "cal_verification_failed": 0.0 if verified else 1.0,
    }
    return float(best_row.get("min_pred", float("inf"))), stats


def _eval_subjects_for_threshold(
    *,
    subjects_eval: list[int],
    by_subject_anchor: dict[int, pd.DataFrame],
    by_subject_cands: dict[str, dict[int, pd.DataFrame]],
    proba_cols: list[str],
    n_classes: int,
    cand_methods: list[str],
    cand_families: dict[str, str],
    anchor_family: str,
    cert,
    guard,
    p_pos_anchor: float,
    guard_threshold: float,
    anchor_guard_delta: float,
    min_pred_improve: float,
    neg_transfer_eps: float = 0.0,
    select_topm: int = 0,
    select_fraction: float = 0.0,
    select_score: str = "guard_p_pos",
    select_score_scope: str = "guard_ok",
    threshold_score: str = "pred_improve",
    candidate_choice_score: str = "pred_improve",
    require_guard_ok: bool = True,
    disable_guard: bool = False,
    guard_target: str = "improve",
    feature_mode: str = "delta",
    selector_model: str = "legacy",
    selector_views: list[str] | None = None,
    dynamic_chunks: int = 4,
    stochastic_bootstrap_rounds: int = 16,
    stochastic_bootstrap_seed: int = 0,
) -> dict[str, float]:
    if selector_views is None:
        selector_views = ["stats"]
    precomputed = _precompute_subject_candidates(
        subjects_eval=subjects_eval,
        by_subject_anchor=by_subject_anchor,
        by_subject_cands=by_subject_cands,
        proba_cols=proba_cols,
        n_classes=n_classes,
        cand_methods=cand_methods,
        cand_families=cand_families,
        anchor_family=str(anchor_family),
        cert=cert,
        guard=guard,
        p_pos_anchor=p_pos_anchor,
        guard_threshold=float(guard_threshold),
        anchor_guard_delta=float(anchor_guard_delta),
        disable_guard=bool(disable_guard),
        guard_target=str(guard_target),
        feature_mode=str(feature_mode),
        selector_model=str(selector_model),
        selector_views=selector_views,
        dynamic_chunks=int(dynamic_chunks),
        stochastic_bootstrap_rounds=int(stochastic_bootstrap_rounds),
        stochastic_bootstrap_seed=int(stochastic_bootstrap_seed),
    )
    return _eval_precomputed(
        precomputed,
        min_pred_improve=float(min_pred_improve),
        neg_transfer_eps=float(neg_transfer_eps),
        select_topm=int(select_topm),
        select_fraction=float(select_fraction),
        select_score=str(select_score),
        select_score_scope=str(select_score_scope),
        threshold_score=str(threshold_score),
        candidate_choice_score=str(candidate_choice_score),
        require_guard_ok=bool(require_guard_ok),
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = str(args.date_prefix).strip() or datetime.now().strftime("%Y%m%d")

    if not (0.0 <= float(args.risk_alpha) <= 1.0):
        raise ValueError("--risk-alpha must be in [0,1].")
    if str(args.risk_mode).strip() == "dual" and args.risk_beta_cond is None:
        raise ValueError("--risk-beta-cond is required when --risk-mode=dual.")
    if args.risk_beta_cond is not None and not (0.0 <= float(args.risk_beta_cond) <= 1.0):
        raise ValueError("--risk-beta-cond must be in [0,1].")
    if not (0.0 <= float(args.min_accept_rate) <= 1.0):
        raise ValueError("--min-accept-rate must be in [0,1].")
    if not (0.0 < float(args.delta) < 1.0):
        raise ValueError("--delta must be in (0,1).")
    if float(args.neg_transfer_eps) < 0.0:
        raise ValueError("--neg-transfer-eps must be >= 0.")
    if float(args.guard_gray_margin) < 0.0:
        raise ValueError("--guard-gray-margin must be >= 0.")
    if not (0.0 < float(args.calib_fraction) < 1.0):
        raise ValueError("--calib-fraction must be in (0,1).")
    if int(args.n_splits) < 1:
        raise ValueError("--n-splits must be >= 1.")
    if str(args.calibration_protocol).strip() == "paper_oof_dev_cal" and int(args.n_splits) < 2:
        raise ValueError("--calibration-protocol=paper_oof_dev_cal requires --n-splits >= 2.")
    if not (0.0 <= float(args.review_budget_frac) <= 1.0):
        raise ValueError("--review-budget-frac must be in [0,1].")
    if float(args.review_cost) < 0.0:
        raise ValueError("--review-cost must be >= 0.")
    if int(args.selector_hidden_dim) < 1:
        raise ValueError("--selector-hidden-dim must be >= 1.")
    if int(args.selector_epochs) < 1:
        raise ValueError("--selector-epochs must be >= 1.")
    if float(args.selector_lr) <= 0.0:
        raise ValueError("--selector-lr must be > 0.")
    if float(args.selector_weight_decay) < 0.0:
        raise ValueError("--selector-weight-decay must be >= 0.")
    if float(args.selector_outcome_delta) < 0.0:
        raise ValueError("--selector-outcome-delta must be >= 0.")
    if float(args.selector_rank_margin) < 0.0:
        raise ValueError("--selector-rank-margin must be >= 0.")
    if int(args.dynamic_chunks) < 1:
        raise ValueError("--dynamic-chunks must be >= 1.")
    if int(args.stochastic_bootstrap_rounds) < 2:
        raise ValueError("--stochastic-bootstrap-rounds must be >= 2.")

    review_budget_mode = str(args.review_budget_mode).strip().lower()
    cond_risk_target = str(args.cond_risk_target).strip().lower()
    review_budget_grid = sorted({float(x) for x in _parse_float_list(str(args.review_budget_grid))})
    if not review_budget_grid:
        raise ValueError("Empty --review-budget-grid.")
    if any((q < 0.0) or (q > 1.0) for q in review_budget_grid):
        raise ValueError("--review-budget-grid entries must be in [0,1].")
    if review_budget_mode == "auto_min_beta":
        if not bool(args.enable_review_proxy):
            raise ValueError("--review-budget-mode=auto_min_beta requires --enable-review-proxy.")
        if str(args.review_scope).strip().lower() != "accepted":
            raise ValueError("--review-budget-mode=auto_min_beta currently supports only --review-scope=accepted.")
        if str(args.risk_mode).strip().lower() != "dual":
            raise ValueError("--review-budget-mode=auto_min_beta requires --risk-mode=dual (needs beta).")
        if args.risk_beta_cond is None:
            raise ValueError("--review-budget-mode=auto_min_beta requires --risk-beta-cond (needs beta).")

    policy = str(args.policy).strip().lower()
    disable_guard = bool(args.disable_guard) or policy == "pred_only"
    guard_target = str(args.guard_target).strip().lower()
    feature_mode = str(args.feature_mode).strip().lower()
    selector_views = _parse_selector_views(str(args.selector_views))
    calibration_protocol = str(args.calibration_protocol).strip().lower()
    neg_transfer_eps = float(args.neg_transfer_eps)
    threshold_score = str(args.threshold_score).strip()
    candidate_choice_score = str(args.candidate_choice_score).strip()
    if policy == "guard_only":
        threshold_score = "guard_p_pos"
        candidate_choice_score = "guard_p_pos"
    # When calibrating directly on guard_p_pos ("guard_only"), the guard_ok boolean becomes redundant.
    # We treat all candidates as eligible and let the calibrated threshold control the acceptance rate.
    require_guard_ok = (not bool(disable_guard)) and policy != "guard_only"

    min_pred_grid = _resolve_min_pred_grid(
        str(args.min_pred_grid),
        selector_model=str(args.selector_model),
    )
    if not min_pred_grid:
        raise ValueError("Empty --min-pred-grid.")

    df = pd.read_csv(Path(args.preds))
    required = {"method", "subject", "trial", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"Preds CSV missing columns {sorted(required - set(df.columns))}.")

    proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
    if not proba_cols:
        raise ValueError("Preds CSV has no proba_* columns.")
    n_classes = int(len(proba_cols))
    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")

    anchor_method = str(args.anchor_method).strip()
    all_methods = sorted({str(m) for m in df["method"].unique().tolist()})
    if anchor_method not in all_methods:
        raise ValueError(f"Anchor method {anchor_method!r} not found in preds. Methods: {all_methods}")

    cand_methods_raw = str(args.candidate_methods).strip()
    if cand_methods_raw.upper() == "ALL":
        cand_methods = [m for m in all_methods if m != anchor_method]
    else:
        cand_methods = [m for m in _parse_csv_list(cand_methods_raw) if m != anchor_method]
    if not cand_methods:
        raise ValueError("No candidate methods specified (or only anchor given).")
    missing = [m for m in cand_methods if m not in all_methods]
    if missing:
        raise ValueError(f"Candidate methods missing in preds: {missing}. Available: {all_methods}")

    fam_map = _parse_method_family_map(args.candidate_family_map)
    anchor_family = str(fam_map.get(anchor_method, _infer_family(anchor_method))).strip().lower()
    cand_families: dict[str, str] = {m: str(fam_map.get(m, _infer_family(m))).strip().lower() for m in cand_methods}

    # Slice per method; require full alignment (same subject/trial/y_true keys).
    df_anchor = df[df["method"].astype(str) == anchor_method].copy()
    if df_anchor.empty:
        raise RuntimeError("Anchor dataframe is empty after filtering.")
    df_anchor = df_anchor.sort_values(["subject", "trial"]).reset_index(drop=True)
    subjects_anchor = sorted({int(s) for s in df_anchor["subject"].unique().tolist()})
    key_anchor = df_anchor[["subject", "trial", "y_true"]].copy()

    df_cands: dict[str, pd.DataFrame] = {}
    common_subjects = set(subjects_anchor)
    for m in cand_methods:
        dm = df[df["method"].astype(str) == str(m)].copy()
        if dm.empty:
            raise RuntimeError(f"Candidate dataframe is empty for method={m}")
        dm = dm.sort_values(["subject", "trial"]).reset_index(drop=True)
        if not dm[["subject", "trial", "y_true"]].equals(key_anchor):
            raise RuntimeError(
                f"Key mismatch between anchor and candidate method={m}. "
                "Ensure the runs were merged from the same trials/protocol."
            )
        df_cands[m] = dm
        common_subjects &= set(int(s) for s in dm["subject"].unique().tolist())

    subjects = sorted(common_subjects)
    all_subjects = list(subjects)
    eval_subjects_raw = _parse_eval_subjects(str(args.eval_subjects))
    if eval_subjects_raw:
        eval_subjects_set = set(int(s) for s in eval_subjects_raw)
        missing_eval = sorted(eval_subjects_set - set(all_subjects))
        if missing_eval:
            raise ValueError(
                f"--eval-subjects contains subjects absent from preds/common pool: {missing_eval}"
            )
        subjects = [int(s) for s in subjects if int(s) in eval_subjects_set]
    if len(all_subjects) < 4:
        raise RuntimeError(f"Need at least 4 common subjects for fit/calib split, got {len(all_subjects)}: {all_subjects}")
    if not subjects:
        raise RuntimeError("No eval subjects remain after applying --eval-subjects.")

    by_subject_anchor: dict[int, pd.DataFrame] = {int(s): g.copy() for s, g in df_anchor.groupby("subject", sort=True)}
    by_subject_cands: dict[str, dict[int, pd.DataFrame]] = {
        m: {int(s): g.copy() for s, g in dm.groupby("subject", sort=True)} for m, dm in df_cands.items()
    }

    diagnostics_root = out_dir / "diagnostics" / str(args.method_name)
    if not bool(args.no_diagnostics):
        diagnostics_root.mkdir(parents=True, exist_ok=True)

    per_subject_records: list[dict[str, object]] = []
    calib_grid_dump_rows: list[dict[str, float]] | None = [] if bool(args.dump_calib_grid) else None
    calib_coverage_vals: list[float] = []
    calib_accept_selected_vals: list[float] = []
    calib_negT_selected_vals: list[float] = []
    calib_n_selected_vals: list[float] = []

    total_subjects = int(len(subjects))
    run_t0 = time.time()
    print(
        f"[run] start method={args.method_name} subjects={total_subjects} selector={args.selector_model} protocol={calibration_protocol}",
        flush=True,
    )

    for subject_idx, t in enumerate(subjects, start=1):
        subject_t0 = time.time()
        print(f"[subject] start test_subject={int(t)} ({subject_idx}/{total_subjects})", flush=True)
        train_subjects = [s for s in all_subjects if s != int(t)]
        if len(train_subjects) < 3:
            raise RuntimeError("Need at least 4 total subjects for fit/calib split.")

        rng = np.random.default_rng(int(args.calib_seed))
        perm = [int(x) for x in rng.permutation(np.asarray(train_subjects, dtype=int)).tolist()]

        n_splits = int(args.n_splits)
        if n_splits > int(len(perm)):
            raise RuntimeError(f"--n-splits={n_splits} is too large for {len(perm)} train subjects (test subject {t}).")

        if calibration_protocol == "paper_oof_dev_cal":
            folds = [xs.tolist() for xs in np.array_split(np.asarray(perm, dtype=int), n_splits) if xs.size > 0]
            precomputed_oof: list[dict[str, object]] = []
            for fold_idx, fold_subjs in enumerate(folds):
                fold_set = set(int(x) for x in fold_subjs)
                fit_subjects = [int(s) for s in train_subjects if int(s) not in fold_set]
                if len(fold_subjs) < 1 or len(fit_subjects) < 2:
                    raise RuntimeError(
                        f"Invalid crossfit split for test subject {t}: fit={len(fit_subjects)} calib={len(fold_subjs)}."
                    )
                print(
                    f"[subject] test_subject={int(t)} crossfit_fold={fold_idx + 1}/{len(folds)} fit={len(fit_subjects)} eval={len(fold_subjs)}",
                    flush=True,
                )
                cert_k, guard_k, p_pos_anchor_k, _feat_names = _train_cert_guard(
                    fit_subjects=fit_subjects,
                    by_subject_anchor=by_subject_anchor,
                    by_subject_cands=by_subject_cands,
                    proba_cols=proba_cols,
                    n_classes=n_classes,
                    cand_methods=cand_methods,
                    cand_families=cand_families,
                    anchor_family=str(anchor_family),
                    ridge_alpha=float(args.ridge_alpha),
                    guard_c=float(args.guard_c),
                    guard_target=str(guard_target),
                    cert_model=str(args.cert_model),
                    guard_model=str(args.guard_model),
                    hgb_max_iter=int(args.hgb_max_iter),
                    model_seed=int(args.calib_seed) + 9973 * int(fold_idx),
                    feature_mode=str(feature_mode),
                    selector_views=selector_views,
                    dynamic_chunks=int(args.dynamic_chunks),
                    stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                    stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                    neg_transfer_eps=float(args.neg_transfer_eps),
                    guard_gray_margin=float(args.guard_gray_margin),
                    selector_model=str(args.selector_model),
                    selector_hidden_dim=int(args.selector_hidden_dim),
                    selector_epochs=int(args.selector_epochs),
                    selector_lr=float(args.selector_lr),
                    selector_weight_decay=float(args.selector_weight_decay),
                    selector_lambda_rank=float(args.selector_lambda_rank),
                    selector_lambda_kl=float(args.selector_lambda_kl),
                    selector_outcome_delta=float(args.selector_outcome_delta),
                    selector_rank_margin=float(args.selector_rank_margin),
                    selector_rho=float(args.selector_rho),
                    selector_eta=float(args.selector_eta),
                    progress_label=f"test_subject={int(t)} fold={fold_idx + 1}/{len(folds)} oof_train",
                )
                precomputed_oof.extend(
                    _precompute_subject_candidates(
                        subjects_eval=[int(s) for s in fold_subjs],
                        by_subject_anchor=by_subject_anchor,
                        by_subject_cands=by_subject_cands,
                        proba_cols=proba_cols,
                        n_classes=n_classes,
                        cand_methods=cand_methods,
                        cand_families=cand_families,
                        anchor_family=str(anchor_family),
                        cert=cert_k,
                        guard=guard_k,
                        p_pos_anchor=float(p_pos_anchor_k),
                        guard_threshold=float(args.guard_threshold),
                        anchor_guard_delta=float(args.anchor_guard_delta),
                        disable_guard=bool(disable_guard),
                        guard_target=str(guard_target),
                        feature_mode=str(feature_mode),
                        selector_model=str(args.selector_model),
                        selector_views=selector_views,
                        dynamic_chunks=int(args.dynamic_chunks),
                        stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                        stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                    )
                )

            calib_n = int(round(float(args.calib_fraction) * float(len(train_subjects))))
            calib_n = max(1, min(calib_n, int(len(train_subjects)) - 1))
            chosen_calib_subjects = [int(x) for x in perm[:calib_n]]
            chosen_fit_subjects = [int(x) for x in perm[calib_n:]]
            if len(chosen_calib_subjects) < 1 or len(chosen_fit_subjects) < 1:
                raise RuntimeError("Invalid D_dev/D_cal split; adjust --calib-fraction.")

            chosen_min_pred, stats = _calibrate_min_pred_paper_dev_cal(
                test_subject=int(t),
                precomputed_oof=precomputed_oof,
                dev_subjects=chosen_fit_subjects,
                cal_subjects=chosen_calib_subjects,
                min_pred_grid=min_pred_grid,
                risk_alpha=float(args.risk_alpha),
                delta=float(args.delta),
                neg_transfer_eps=float(args.neg_transfer_eps),
                select_topm=int(args.select_topm),
                select_fraction=float(args.select_fraction),
                select_score=str(args.select_score),
                select_score_scope=str(args.select_score_scope),
                threshold_score=str(threshold_score),
                candidate_choice_score=str(candidate_choice_score),
                require_guard_ok=bool(require_guard_ok),
                risk_mode=str(args.risk_mode),
                risk_beta_cond=(None if args.risk_beta_cond is None else float(args.risk_beta_cond)),
                min_accept_rate=float(args.min_accept_rate),
                calib_objective=str(args.calib_objective),
                calib_grid_dump_rows=calib_grid_dump_rows,
            )
            cert, guard, p_pos_anchor, _feat_names = _train_cert_guard(
                fit_subjects=train_subjects,
                by_subject_anchor=by_subject_anchor,
                by_subject_cands=by_subject_cands,
                proba_cols=proba_cols,
                n_classes=n_classes,
                cand_methods=cand_methods,
                cand_families=cand_families,
                anchor_family=str(anchor_family),
                ridge_alpha=float(args.ridge_alpha),
                guard_c=float(args.guard_c),
                guard_target=str(guard_target),
                cert_model=str(args.cert_model),
                guard_model=str(args.guard_model),
                hgb_max_iter=int(args.hgb_max_iter),
                model_seed=int(args.calib_seed),
                feature_mode=str(feature_mode),
                selector_views=selector_views,
                dynamic_chunks=int(args.dynamic_chunks),
                stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                neg_transfer_eps=float(args.neg_transfer_eps),
                guard_gray_margin=float(args.guard_gray_margin),
                selector_model=str(args.selector_model),
                selector_hidden_dim=int(args.selector_hidden_dim),
                selector_epochs=int(args.selector_epochs),
                selector_lr=float(args.selector_lr),
                selector_weight_decay=float(args.selector_weight_decay),
                selector_lambda_rank=float(args.selector_lambda_rank),
                selector_lambda_kl=float(args.selector_lambda_kl),
                selector_outcome_delta=float(args.selector_outcome_delta),
                selector_rank_margin=float(args.selector_rank_margin),
                selector_rho=float(args.selector_rho),
                selector_eta=float(args.selector_eta),
                progress_label=f"test_subject={int(t)} final_train_all",
            )
            chosen_split_idx = -2
            calib_mean_acc = float(stats.get("calib_mean_acc", float("nan")))
            calib_worst_acc = float(stats.get("calib_worst_acc", float("nan")))
            calib_emp_negT = float(stats.get("calib_emp_negT", float("nan")))
            calib_ucb_negT = float(stats.get("calib_ucb_negT", float("nan")))
            calib_accept = float(stats.get("calib_accept", float("nan")))
            feasible_empty = int(float(stats.get("feasible_empty", 0.0)) > 0.0)
        elif n_splits == 1:
            calib_n = int(round(float(args.calib_fraction) * float(len(train_subjects))))
            calib_n = max(1, min(calib_n, int(len(train_subjects)) - 2))
            chosen_calib_subjects = [int(x) for x in perm[:calib_n]]
            chosen_fit_subjects = [int(x) for x in perm[calib_n:]]
            if len(chosen_calib_subjects) < 1 or len(chosen_fit_subjects) < 2:
                raise RuntimeError("Invalid fit/calib split; adjust --calib-fraction.")

            cert, guard, p_pos_anchor, _feat_names = _train_cert_guard(
                fit_subjects=chosen_fit_subjects,
                by_subject_anchor=by_subject_anchor,
                by_subject_cands=by_subject_cands,
                proba_cols=proba_cols,
                n_classes=n_classes,
                cand_methods=cand_methods,
                cand_families=cand_families,
                anchor_family=str(anchor_family),
                ridge_alpha=float(args.ridge_alpha),
                guard_c=float(args.guard_c),
                guard_target=str(guard_target),
                cert_model=str(args.cert_model),
                guard_model=str(args.guard_model),
                hgb_max_iter=int(args.hgb_max_iter),
                model_seed=int(args.calib_seed),
                feature_mode=str(feature_mode),
                selector_views=selector_views,
                dynamic_chunks=int(args.dynamic_chunks),
                stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                neg_transfer_eps=float(args.neg_transfer_eps),
                guard_gray_margin=float(args.guard_gray_margin),
                selector_model=str(args.selector_model),
                selector_hidden_dim=int(args.selector_hidden_dim),
                selector_epochs=int(args.selector_epochs),
                selector_lr=float(args.selector_lr),
                selector_weight_decay=float(args.selector_weight_decay),
                selector_lambda_rank=float(args.selector_lambda_rank),
                selector_lambda_kl=float(args.selector_lambda_kl),
                selector_outcome_delta=float(args.selector_outcome_delta),
                selector_rank_margin=float(args.selector_rank_margin),
                selector_rho=float(args.selector_rho),
                selector_eta=float(args.selector_eta),
                progress_label=f"test_subject={int(t)} single_split_train",
            )
            chosen_min_pred, stats = _calibrate_min_pred(
                test_subject=int(t),
                calib_subjects=chosen_calib_subjects,
                by_subject_anchor=by_subject_anchor,
                by_subject_cands=by_subject_cands,
                proba_cols=proba_cols,
                n_classes=n_classes,
                cand_methods=cand_methods,
                cand_families=cand_families,
                anchor_family=str(anchor_family),
                cert=cert,
                guard=guard,
                p_pos_anchor=float(p_pos_anchor),
                guard_threshold=float(args.guard_threshold),
                anchor_guard_delta=float(args.anchor_guard_delta),
                min_pred_grid=min_pred_grid,
                risk_alpha=float(args.risk_alpha),
                delta=float(args.delta),
                neg_transfer_eps=float(args.neg_transfer_eps),
                select_topm=int(args.select_topm),
                select_fraction=float(args.select_fraction),
                select_score=str(args.select_score),
                select_score_scope=str(args.select_score_scope),
                threshold_score=str(threshold_score),
                candidate_choice_score=str(candidate_choice_score),
                require_guard_ok=bool(require_guard_ok),
                disable_guard=bool(disable_guard),
                guard_target=str(guard_target),
                feature_mode=str(feature_mode),
                selector_model=str(args.selector_model),
                selector_views=selector_views,
                dynamic_chunks=int(args.dynamic_chunks),
                stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                risk_mode=str(args.risk_mode),
                risk_beta_cond=(None if args.risk_beta_cond is None else float(args.risk_beta_cond)),
                min_accept_rate=float(args.min_accept_rate),
                calib_objective=str(args.calib_objective),
                calib_grid_dump_rows=calib_grid_dump_rows,
            )
            chosen_split_idx = 0
            calib_mean_acc = float(stats.get("calib_mean_acc", float("nan")))
            calib_worst_acc = float(stats.get("calib_worst_acc", float("nan")))
            calib_emp_negT = float(stats.get("calib_emp_negT", float("nan")))
            calib_ucb_negT = float(stats.get("calib_ucb_negT", float("nan")))
            calib_accept = float(stats.get("calib_accept", float("nan")))
            feasible_empty = int(float(stats.get("feasible_empty", 0.0)) > 0.0)
        else:
            folds = [xs.tolist() for xs in np.array_split(np.asarray(perm, dtype=int), n_splits) if xs.size > 0]
            fold_models: list[dict[str, object]] = []
            for fold_idx, fold_subjs in enumerate(folds):
                fold_set = set(int(x) for x in fold_subjs)
                fit_subjects = [int(s) for s in train_subjects if int(s) not in fold_set]
                if len(fold_subjs) < 1 or len(fit_subjects) < 2:
                    raise RuntimeError(
                        f"Invalid crossfit split for test subject {t}: fit={len(fit_subjects)} calib={len(fold_subjs)}."
                    )
                cert_k, guard_k, p_pos_anchor_k, _feat_names = _train_cert_guard(
                    fit_subjects=fit_subjects,
                    by_subject_anchor=by_subject_anchor,
                    by_subject_cands=by_subject_cands,
                    proba_cols=proba_cols,
                    n_classes=n_classes,
                    cand_methods=cand_methods,
                    cand_families=cand_families,
                    anchor_family=str(anchor_family),
                    ridge_alpha=float(args.ridge_alpha),
                    guard_c=float(args.guard_c),
                    guard_target=str(guard_target),
                    cert_model=str(args.cert_model),
                    guard_model=str(args.guard_model),
                    hgb_max_iter=int(args.hgb_max_iter),
                    model_seed=int(args.calib_seed) + 9973 * int(fold_idx),
                    feature_mode=str(feature_mode),
                    selector_views=selector_views,
                    dynamic_chunks=int(args.dynamic_chunks),
                    stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                    stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                    neg_transfer_eps=float(args.neg_transfer_eps),
                    guard_gray_margin=float(args.guard_gray_margin),
                    selector_model=str(args.selector_model),
                    selector_hidden_dim=int(args.selector_hidden_dim),
                    selector_epochs=int(args.selector_epochs),
                    selector_lr=float(args.selector_lr),
                    selector_weight_decay=float(args.selector_weight_decay),
                    selector_lambda_rank=float(args.selector_lambda_rank),
                    selector_lambda_kl=float(args.selector_lambda_kl),
                    selector_outcome_delta=float(args.selector_outcome_delta),
                    selector_rank_margin=float(args.selector_rank_margin),
                    selector_rho=float(args.selector_rho),
                    selector_eta=float(args.selector_eta),
                    progress_label=f"test_subject={int(t)} fold={fold_idx + 1}/{len(folds)} crossfit_train",
                )
                fold_models.append({"cert": cert_k, "guard": guard_k, "p_pos_anchor": float(p_pos_anchor_k)})

            chosen_min_pred, stats = _calibrate_min_pred_crossfit(
                test_subject=int(t),
                folds=folds,
                fold_models=fold_models,
                by_subject_anchor=by_subject_anchor,
                by_subject_cands=by_subject_cands,
                proba_cols=proba_cols,
                n_classes=n_classes,
                cand_methods=cand_methods,
                cand_families=cand_families,
                anchor_family=str(anchor_family),
                min_pred_grid=min_pred_grid,
                risk_alpha=float(args.risk_alpha),
                delta=float(args.delta),
                neg_transfer_eps=float(args.neg_transfer_eps),
                guard_threshold=float(args.guard_threshold),
                anchor_guard_delta=float(args.anchor_guard_delta),
                select_topm=int(args.select_topm),
                select_fraction=float(args.select_fraction),
                select_score=str(args.select_score),
                select_score_scope=str(args.select_score_scope),
                threshold_score=str(threshold_score),
                candidate_choice_score=str(candidate_choice_score),
                require_guard_ok=bool(require_guard_ok),
                disable_guard=bool(disable_guard),
                guard_target=str(guard_target),
                feature_mode=str(feature_mode),
                selector_model=str(args.selector_model),
                selector_views=selector_views,
                dynamic_chunks=int(args.dynamic_chunks),
                stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                risk_mode=str(args.risk_mode),
                risk_beta_cond=(None if args.risk_beta_cond is None else float(args.risk_beta_cond)),
                min_accept_rate=float(args.min_accept_rate),
                calib_objective=str(args.calib_objective),
                calib_grid_dump_rows=calib_grid_dump_rows,
            )
            # Train final cert/guard on all available (non-test) subjects.
            cert, guard, p_pos_anchor, _feat_names = _train_cert_guard(
                fit_subjects=train_subjects,
                by_subject_anchor=by_subject_anchor,
                by_subject_cands=by_subject_cands,
                proba_cols=proba_cols,
                n_classes=n_classes,
                cand_methods=cand_methods,
                cand_families=cand_families,
                anchor_family=str(anchor_family),
                ridge_alpha=float(args.ridge_alpha),
                guard_c=float(args.guard_c),
                guard_target=str(guard_target),
                cert_model=str(args.cert_model),
                guard_model=str(args.guard_model),
                hgb_max_iter=int(args.hgb_max_iter),
                model_seed=int(args.calib_seed),
                feature_mode=str(feature_mode),
                selector_views=selector_views,
                dynamic_chunks=int(args.dynamic_chunks),
                stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed),
                neg_transfer_eps=float(args.neg_transfer_eps),
                guard_gray_margin=float(args.guard_gray_margin),
                selector_model=str(args.selector_model),
                selector_hidden_dim=int(args.selector_hidden_dim),
                selector_epochs=int(args.selector_epochs),
                selector_lr=float(args.selector_lr),
                selector_weight_decay=float(args.selector_weight_decay),
                selector_lambda_rank=float(args.selector_lambda_rank),
                selector_lambda_kl=float(args.selector_lambda_kl),
                selector_outcome_delta=float(args.selector_outcome_delta),
                selector_rank_margin=float(args.selector_rank_margin),
                selector_rho=float(args.selector_rho),
                selector_eta=float(args.selector_eta),
                progress_label=f"test_subject={int(t)} final_train_all",
            )
            chosen_fit_subjects = list(train_subjects)
            chosen_calib_subjects = list(train_subjects)
            chosen_split_idx = -1
            calib_mean_acc = float(stats.get("calib_mean_acc", float("nan")))
            calib_worst_acc = float(stats.get("calib_worst_acc", float("nan")))
            calib_emp_negT = float(stats.get("calib_emp_negT", float("nan")))
            calib_ucb_negT = float(stats.get("calib_ucb_negT", float("nan")))
            calib_accept = float(stats.get("calib_accept", float("nan")))
            feasible_empty = int(float(stats.get("feasible_empty", 0.0)) > 0.0)

        calib_coverage_vals.append(float(stats.get("calib_coverage", float("nan"))))
        calib_accept_selected_vals.append(float(stats.get("calib_accept_selected", float("nan"))))
        calib_negT_selected_vals.append(float(stats.get("calib_negT_selected", float("nan"))))
        calib_n_selected_vals.append(float(stats.get("calib_n_selected", float("nan"))))

        # Evaluate on test subject t using the calibrated threshold.
        df_t_id = by_subject_anchor[int(t)]
        y_true_t = df_t_id["y_true"].to_numpy(object)
        y_pred_id_t = df_t_id["y_pred"].to_numpy(object)
        p_id_t = df_t_id[proba_cols].to_numpy(np.float64)
        acc_id_t = float(np.mean(y_pred_id_t == y_true_t))

        rec_id_t = _record_from_proba(
            p_id=p_id_t,
            p_c=p_id_t,
            y_pred_id=y_pred_id_t,
            y_pred_c=y_pred_id_t,
            cand_family=str(anchor_family).strip().lower(),
            kind="identity",
        )

        cand_rows = []
        cand_rows.append(
            {
                **{
                    "idx": 0,
                    "is_selected": 0,
                    "kind": "identity",
                    "cand_family": str(anchor_family).strip().lower(),
                    "cand_key": str(anchor_method),
                    "ridge_pred_improve": 0.0,
                    "guard_p_pos": float(p_pos_anchor),
                    "accept": 0,
                    "accuracy": float(acc_id_t),
                    "selector_p_harm": 0.0,
                    "selector_p_neutral": 1.0,
                    "selector_p_help": 0.0,
                    "selector_b_harm": 0.0,
                    "selector_b_neutral": 0.0,
                    "selector_b_help": 0.0,
                    "selector_uncertainty": 1.0,
                    "selector_risk": float(max(0.0, 1.0 - float(p_pos_anchor))),
                    "selector_utility": 0.0,
                },
                **{k: v for k, v in rec_id_t.items() if k not in {"kind", "cand_family"}},
            }
        )

        cand_eval = []
        for j, m in enumerate(cand_methods, start=1):
            df_t_c = by_subject_cands[m][int(t)]
            y_pred_c_t = df_t_c["y_pred"].to_numpy(object)
            p_c_t = df_t_c[proba_cols].to_numpy(np.float64)
            acc_c_t = float(np.mean(y_pred_c_t == y_true_t))

            rec_c_t = _record_from_proba(
                p_id=p_id_t,
                p_c=p_c_t,
                y_pred_id=y_pred_id_t,
                y_pred_c=y_pred_c_t,
                cand_family=str(cand_families[m]),
                kind="candidate",
            )
            if str(args.selector_model).strip().lower() == "evidential":
                feats_t, _feat_names, _view_slices = _selector_feature_bundle(
                    p_id=p_id_t,
                    p_c=p_c_t,
                    y_pred_id=y_pred_id_t,
                    y_pred_c=y_pred_c_t,
                    anchor_family=str(anchor_family).strip().lower(),
                    cand_family=str(cand_families[m]),
                    n_classes=int(n_classes),
                    feature_mode=str(feature_mode),
                    selector_views=selector_views,
                    dynamic_chunks=int(args.dynamic_chunks),
                    stochastic_bootstrap_rounds=int(args.stochastic_bootstrap_rounds),
                    stochastic_bootstrap_seed=int(args.stochastic_bootstrap_seed) + 1009 * int(t),
                )
            else:
                x0_t, names0_t = candidate_features_from_record(rec_id_t, n_classes=n_classes, include_pbar=True)
                x0_t = np.asarray(x0_t, dtype=np.float64).reshape(-1)
                x_t, names_t = candidate_features_from_record(rec_c_t, n_classes=n_classes, include_pbar=True)
                if names_t != names0_t:
                    raise RuntimeError("Anchor/candidate feature name mismatch on test subject.")
                x_t = np.asarray(x_t, dtype=np.float64).reshape(-1)
                feats_t, _feat_names = _features_from_anchor_and_candidate(
                    x_anchor=x0_t,
                    x_candidate=x_t,
                    names=tuple(names0_t),
                    feature_mode=str(feature_mode),
                )
            feats_t = np.asarray(feats_t, dtype=np.float64).reshape(1, -1)

            pred_improve, p_pos, selector_extra = _selector_outputs(
                cert=cert,
                guard=guard,
                feats=feats_t,
            )
            if str(args.selector_model).strip().lower() == "evidential":
                guard_ok = p_pos >= float(args.guard_threshold)
            elif str(guard_target).strip().lower() == "harm":
                p_pos = 1.0 - p_pos  # P(non-harm)
                guard_ok = p_pos >= float(args.guard_threshold)
            else:
                guard_ok = (p_pos >= float(args.guard_threshold)) and (
                    p_pos >= float(p_pos_anchor) + float(args.anchor_guard_delta)
                )
            if bool(disable_guard):
                guard_ok = True
            thr_score = str(threshold_score)
            if thr_score == "pred_improve_x_guard":
                thr_val = float(pred_improve) * float(p_pos)
            else:
                thr_val = float(pred_improve) if thr_score == "pred_improve" else float(p_pos)
            accept = (bool(guard_ok) if bool(require_guard_ok) else True) and (thr_val >= float(chosen_min_pred))

            cand_rows.append(
                {
                    **{
                        "idx": int(j),
                        "is_selected": 0,
                        "kind": "candidate",
                        "cand_family": str(cand_families[m]),
                        "cand_key": str(m),
                        "ridge_pred_improve": float(pred_improve),
                        "guard_p_pos": float(p_pos),
                        "accept": int(bool(accept)),
                        "accuracy": float(acc_c_t),
                        **selector_extra,
                    },
                    **{k: v for k, v in rec_c_t.items() if k not in {"kind", "cand_family"}},
                }
            )
            cand_eval.append(
                {
                    "method": str(m),
                    "family": str(cand_families[m]),
                    "pred_improve": float(pred_improve),
                    "guard_p_pos": float(p_pos),
                    "guard_ok": bool(guard_ok),
                    "accept": bool(accept),
                    "acc": float(acc_c_t),
                    **selector_extra,
                }
            )

        best_pre = max(cand_eval, key=lambda r: float(r["pred_improve"])) if cand_eval else None
        accepted = [r for r in cand_eval if bool(r["accept"])]
        best_acc = max([acc_id_t] + [float(r["acc"]) for r in cand_eval]) if cand_eval else acc_id_t
        oracle_acc = float(best_acc)
        max_pred_improve = float(max([float(r["pred_improve"]) for r in cand_eval], default=0.0))
        max_guard_p = float(max([float(r["guard_p_pos"]) for r in cand_eval], default=float(rec_id_t.get("mean_confidence", 0.0))))

        has_guard_ok, sel_score = _subject_has_guard_ok_and_score(
            {"cands": cand_eval},
            select_score=str(args.select_score),
            score_scope=str(args.select_score_scope),
        )
        choice_score = str(candidate_choice_score)
        if choice_score == "pred_improve_x_guard":
            best_accept = (
                max(accepted, key=lambda r: float(r.get("pred_improve", float("-inf"))) * float(r.get("guard_p_pos", 0.0)))
                if accepted
                else None
            )
        else:
            best_accept = max(accepted, key=lambda r: float(r.get(choice_score, float("-inf")))) if accepted else None
        per_subject_records.append(
            {
                "subject": int(t),
                "p_pos_anchor": float(p_pos_anchor),
                "acc_anchor": float(acc_id_t),
                "oracle_acc": float(oracle_acc),
                "headroom": float(oracle_acc - acc_id_t),
                "pre_best": dict(best_pre) if best_pre is not None else None,
                "cand_rows": cand_rows,
                "review_opportunity_score": float(max_pred_improve * max_guard_p),
                "review_uncertainty_score": float(1.0 - max_guard_p),
                "has_guard_ok": bool(has_guard_ok),
                "has_pred_ok": bool(best_accept is not None),
                "sel_score": float(sel_score),
                "best_accept": dict(best_accept) if best_accept is not None else None,
                "calib_min_pred": float(chosen_min_pred),
                "calib_mean_acc": float(calib_mean_acc),
                "calib_worst_acc": float(calib_worst_acc),
                "calib_emp_negT": float(calib_emp_negT),
                "calib_ucb_negT": float(calib_ucb_negT),
                "calib_mean_delta_all": float(stats.get("calib_mean_delta_all", float("nan"))),
                "calib_mean_delta_accepted": float(stats.get("calib_mean_delta_accepted", float("nan"))),
                "calib_cond_rate": float(stats.get("calib_cond_rate", float("nan"))),
                "calib_cond_ucb": float(stats.get("calib_cond_ucb", float("nan"))),
                "calib_cond_ci_low": float(stats.get("calib_cond_ci_low", float("nan"))),
                "calib_cond_ci_high": float(stats.get("calib_cond_ci_high", float("nan"))),
                "calib_accept": float(calib_accept),
                "calib_risk_mode": str(args.risk_mode),
                "calib_risk_beta_cond": (
                    float(args.risk_beta_cond) if args.risk_beta_cond is not None else float("nan")
                ),
                "calib_min_accept_rate": float(args.min_accept_rate),
                "calib_threshold_objective": str(args.calib_objective),
                "calib_fallback_stage": str(stats.get("fallback_stage", "")),
                "dev_chosen_min_pred": float(stats.get("dev_chosen_min_pred", float("nan"))),
                "dev_mean_acc": float(stats.get("dev_mean_acc", float("nan"))),
                "dev_mean_delta_all": float(stats.get("dev_mean_delta_all", float("nan"))),
                "dev_accept": float(stats.get("dev_accept", float("nan"))),
                "dev_n_subjects": int(stats.get("dev_subjects", 0)),
                "cal_verification_failed": int(float(stats.get("cal_verification_failed", 0.0)) > 0.0),
                "feasible_empty": int(feasible_empty),
                "n_splits": int(args.n_splits),
                "chosen_split_idx": int(chosen_split_idx),
                "calib_n_subjects": int(len(chosen_calib_subjects)),
                "fit_n_subjects": int(len(chosen_fit_subjects)),
            }
        )

        print(
            f"[subject] done test_subject={int(t)} chosen_min_pred={float(chosen_min_pred):.4f} anchor_acc={float(acc_id_t):.4f} elapsed={time.time() - subject_t0:.1f}s",
            flush=True,
        )

    # Global test-time selection across ALL eval subjects (coverage knob).
    subjects_all = [int(r["subject"]) for r in per_subject_records]
    scores_all = [float(r.get("sel_score", float("-inf"))) for r in per_subject_records]
    selected_ids = _select_subject_ids(
        subjects_all,
        scores_all,
        select_topm=int(args.select_topm),
        select_fraction=float(args.select_fraction),
    )

    rows_all: list[pd.DataFrame] = []
    per_subject_summary_rows: list[dict] = []
    eval_n_subjects_total = int(len(subjects_all))
    eval_n_selected = 0
    eval_n_accept_selected = 0
    eval_n_neg_transfer_selected = 0

    for rec in per_subject_records:
        t = int(rec["subject"])
        selected = bool(t in selected_ids)
        sel_score = float(rec.get("sel_score", float("-inf")))
        sel_score_finite = bool(np.isfinite(sel_score))
        selected_by_topm = bool(selected)
        has_guard_ok = bool(rec.get("has_guard_ok", False))
        has_pred_ok = bool(rec.get("has_pred_ok", False))

        selected_method = str(anchor_method)
        selected_family = str(anchor_family)
        selected_pred_improve = 0.0
        selected_p_pos = float(rec["p_pos_anchor"])
        selected_selector_risk = float(max(0.0, 1.0 - float(rec["p_pos_anchor"])))
        selected_selector_utility = 0.0
        selected_selector_uncertainty = 1.0
        accept_any = False

        best_accept = rec.get("best_accept", None)
        if bool(selected) and best_accept is not None:
            best = dict(best_accept)
            selected_method = str(best["method"])
            selected_family = str(best["family"])
            selected_pred_improve = float(best["pred_improve"])
            selected_p_pos = float(best["guard_p_pos"])
            selected_selector_risk = float(best.get("selector_risk", max(0.0, 1.0 - float(selected_p_pos))))
            selected_selector_utility = float(best.get("selector_utility", selected_pred_improve))
            selected_selector_uncertainty = float(best.get("selector_uncertainty", float("nan")))
            accept_any = True

        # For non-selected subjects: force abstain (anchor, accept=0, delta=0).
        if not bool(selected):
            selected_method = str(anchor_method)
            selected_family = str(anchor_family)
            selected_pred_improve = 0.0
            selected_p_pos = float(rec["p_pos_anchor"])
            selected_selector_risk = float(max(0.0, 1.0 - float(rec["p_pos_anchor"])))
            selected_selector_utility = 0.0
            selected_selector_uncertainty = 1.0
            accept_any = False

        cand_rows = rec.get("cand_rows", [])
        for r in cand_rows:
            if r.get("kind", "") == "identity" and str(selected_method) == str(anchor_method):
                r["is_selected"] = 1
            elif r.get("kind", "") == "candidate" and str(r.get("cand_key", "")) == str(selected_method):
                r["is_selected"] = 1
            else:
                r["is_selected"] = 0

        if not bool(args.no_diagnostics):
            subj_dir = diagnostics_root / f"subject_{int(t):02d}"
            subj_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cand_rows).to_csv(subj_dir / "candidates.csv", index=False)

        df_t_id = by_subject_anchor[int(t)]
        y_true_t = df_t_id["y_true"].to_numpy(object)
        if selected_method == anchor_method:
            df_sel = df_t_id.copy()
        else:
            df_sel = by_subject_cands[selected_method][int(t)].copy()

        df_out = pd.DataFrame(
            {
                "method": str(args.method_name),
                "subject": df_sel["subject"].astype(int),
                "trial": df_sel["trial"].astype(int),
                "y_true": df_sel["y_true"].astype(str),
                "y_pred": df_sel["y_pred"].astype(str),
            }
        )
        for c in proba_cols:
            df_out[c] = df_sel[c].astype(float)
        rows_all.append(df_out)

        acc_sel_t = float(np.mean(df_sel["y_pred"].to_numpy(object) == y_true_t))
        acc_id_t = float(rec["acc_anchor"])
        delta_sel = float(acc_sel_t - acc_id_t)

        if bool(selected):
            eval_n_selected += 1
            if bool(accept_any):
                eval_n_accept_selected += 1
            if float(delta_sel) < -float(neg_transfer_eps):
                eval_n_neg_transfer_selected += 1

        oracle_acc = float(rec["oracle_acc"])
        oracle_gap = float(oracle_acc - acc_sel_t)

        if bool(args.log_stage_breakdown):
            # Stage attribution order:
            # (1) top-m selection gate (if enabled)
            # (2) existence of any guard-qualified candidate
            # (3) existence of any candidate meeting min_pred threshold
            # (4) accepted
            #
            # Some subjects can have no finite selection score when score_scope='guard_ok' and has_guard_ok=False.
            # For stage breakdown, we still want those to show up as 'no_guard_ok' (rather than an opaque 'no_score').
            if not bool(selected_by_topm):
                reason = "not_in_topm" if bool(sel_score_finite) else "no_guard_ok"
            elif not bool(has_guard_ok):
                reason = "no_guard_ok"
            elif not bool(has_pred_ok):
                reason = "no_pred_ok"
            elif bool(accept_any):
                reason = "accepted"
            else:
                # Should not happen (accept_any implies selected_by_topm & has_pred_ok), but keep a safe fallback.
                reason = "no_pred_ok"
        else:
            reason = "accepted"
            if not bool(selected):
                reason = "not_selected"
            elif not bool(has_guard_ok):
                reason = "no_guard_ok"
            elif not bool(has_pred_ok):
                reason = "no_pred_ok"

        pre_best = rec.get("pre_best", None)
        per_row = {
            "subject": int(t),
            "selected_method": str(selected_method),
            "selected_family": str(selected_family),
            "accept": int(bool(accept_any)),
            "guard_p_pos_selected": float(selected_p_pos),
            "guard_p_pos_anchor": float(rec["p_pos_anchor"]),
            "ridge_pred_improve_selected": float(selected_pred_improve),
            "selector_risk_selected": float(selected_selector_risk),
            "selector_utility_selected": float(selected_selector_utility),
            "selector_uncertainty_selected": float(selected_selector_uncertainty),
            "pre_best_method": str(pre_best.get("method", "")) if isinstance(pre_best, dict) else "",
            "pre_best_pred_improve": (
                float(pre_best.get("pred_improve", float("nan"))) if isinstance(pre_best, dict) else float("nan")
            ),
            "pre_best_selector_risk": (
                float(pre_best.get("selector_risk", float("nan"))) if isinstance(pre_best, dict) else float("nan")
            ),
            "pre_best_selector_utility": (
                float(pre_best.get("selector_utility", float("nan"))) if isinstance(pre_best, dict) else float("nan")
            ),
            "acc_anchor": float(acc_id_t),
            "acc_selected": float(acc_sel_t),
            "acc_final": float(acc_sel_t),
            "oracle_acc": float(oracle_acc),
            "headroom": float(rec["headroom"]),
            "oracle_gap": float(oracle_gap),
            "delta_selected_vs_anchor": float(delta_sel),
            "delta_final_vs_anchor": float(delta_sel),
            "calib_min_pred": float(rec["calib_min_pred"]),
            "calib_mean_acc": float(rec["calib_mean_acc"]),
            "calib_worst_acc": float(rec["calib_worst_acc"]),
            "calib_mean_delta_all": float(rec.get("calib_mean_delta_all", float("nan"))),
            "calib_mean_delta_accepted": float(rec.get("calib_mean_delta_accepted", float("nan"))),
            "calib_negT": float(rec["calib_emp_negT"]),
            "calib_cond_rate": float(rec.get("calib_cond_rate", float("nan"))),
            "calib_cond_ucb": float(rec.get("calib_cond_ucb", float("nan"))),
            "calib_cond_ci_low": float(rec.get("calib_cond_ci_low", float("nan"))),
            "calib_cond_ci_high": float(rec.get("calib_cond_ci_high", float("nan"))),
            "calib_accept": float(rec.get("calib_accept", float("nan"))),
            "chosen_min_pred": float(rec["calib_min_pred"]),
            "calib_emp_negT": float(rec["calib_emp_negT"]),
            "calib_ucb_negT": float(rec["calib_ucb_negT"]),
            "calib_delta": float(args.delta),
            "calib_risk_alpha": float(args.risk_alpha),
            "calib_risk_mode": str(args.risk_mode),
            "calib_risk_beta_cond": float(args.risk_beta_cond) if args.risk_beta_cond is not None else float("nan"),
            "calib_min_accept_rate": float(args.min_accept_rate),
            "calib_threshold_objective": str(args.calib_objective),
            "calib_fallback_stage": str(rec.get("calib_fallback_stage", "")),
            "feasible_empty": int(rec["feasible_empty"]),
            "n_splits": int(rec["n_splits"]),
            "chosen_split_idx": int(rec["chosen_split_idx"]),
            "calib_n_subjects": int(rec["calib_n_subjects"]),
            "fit_n_subjects": int(rec["fit_n_subjects"]),
            "risk_mode": str(args.risk_mode),
            "review_flag": 0,
            "review_score": float("nan"),
            "review_rank": float("nan"),
            "review_reason": "not_reviewed",
            "is_oracle_corrected": 0,
            "review_opportunity_score": float(rec.get("review_opportunity_score", float("nan"))),
            "review_uncertainty_score": float(rec.get("review_uncertainty_score", float("nan"))),
            "review_harm_score": (
                float(1.0 - float(selected_p_pos))
                if bool(accept_any) and np.isfinite(float(selected_p_pos))
                else float("nan")
            ),
        }
        if bool(args.log_stage_breakdown):
            per_row.update(
                {
                    "selected": int(bool(selected)),
                    "selected_by_topm": int(bool(selected_by_topm)),
                    "has_guard_ok": int(bool(has_guard_ok)),
                    "has_pred_ok": int(bool(has_pred_ok)),
                    "sel_score": float(sel_score),
                    "sel_score_finite": int(bool(sel_score_finite)),
                    "accepted": int(bool(accept_any)),
                    "reason": str(reason),
                }
            )
        per_subject_summary_rows.append(per_row)

    pred_all = pd.concat(rows_all, axis=0, ignore_index=True).sort_values(["subject", "trial"])
    pred_all.to_csv(out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False)

    per_subj = pd.DataFrame(per_subject_summary_rows).sort_values("subject").reset_index(drop=True)

    # Optional human-review proxy (offline oracle agent on abstained subjects).
    chosen_review_budget_frac = 0.0
    chosen_review_budget_n = 0
    review_infeasible = 0

    review_budget_frac = float(args.review_budget_frac) if bool(args.enable_review_proxy) else 0.0
    review_scope = str(args.review_scope).strip().lower()
    review_score_name = str(args.review_score).strip().lower()
    if review_score_name == "opportunity":
        score_col = "review_opportunity_score"
    elif review_score_name == "harm":
        score_col = "review_harm_score"
    else:
        score_col = "review_uncertainty_score"
    if bool(args.enable_review_proxy):
        accept_arr = pd.to_numeric(per_subj["accept"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        abstain_mask = accept_arr <= 0.5
        accepted_mask_arr = accept_arr > 0.5
        if review_scope == "accepted":
            cand_mask = accepted_mask_arr
            chosen_reason = "accept_high_uncertainty"
            exhausted_reason = "accept_budget_exhausted"
            zero_reason = "accept_budget_zero"
        elif review_scope == "all":
            cand_mask = np.ones_like(accept_arr, dtype=bool)
            chosen_reason = "all_scope"
            exhausted_reason = "all_budget_exhausted"
            zero_reason = "all_budget_zero"
        else:
            cand_mask = abstain_mask
            chosen_reason = "abstain_low_evidence"
            exhausted_reason = "abstain_budget_exhausted"
            zero_reason = "abstain_budget_zero"

        n_total_subj = int(per_subj.shape[0])
        beta = float(args.risk_beta_cond) if args.risk_beta_cond is not None else 1.0

        if review_budget_mode == "auto_min_beta":
            score_vals = pd.to_numeric(per_subj[score_col], errors="coerce").to_numpy(dtype=float)
            score_vals = np.where(np.isfinite(score_vals), score_vals, -np.inf)
            acc_anchor_arr = per_subj["acc_anchor"].to_numpy(dtype=float)
            acc_selected_arr = per_subj["acc_selected"].to_numpy(dtype=float)
            oracle_acc_arr = per_subj["oracle_acc"].to_numpy(dtype=float)
            delta_pre = per_subj["delta_selected_vs_anchor"].to_numpy(dtype=float)

            n_acc = int(np.sum(accepted_mask_arr))
            k_pre = int(np.sum((delta_pre < -float(neg_transfer_eps)) & accepted_mask_arr))
            ucb_pre = _clopper_pearson_ucb(k=k_pre, n=n_acc, delta=float(args.delta)) if n_acc > 0 else float("nan")

            best_q: float | None = None
            best_budget_n: int | None = None

            if cond_risk_target == "pre":
                if np.isfinite(ucb_pre) and ucb_pre <= beta + 1e-12:
                    best_q = 0.0
                    best_budget_n = 0
                else:
                    review_infeasible = 1
                    best_q = float(max(review_budget_grid))
                    best_budget_n = int(np.ceil(best_q * float(n_total_subj)))
            else:
                accepted_idx = np.where(accepted_mask_arr)[0]
                order = accepted_idx[np.argsort(-score_vals[accepted_idx], kind="mergesort")] if accepted_idx.size else np.array([], dtype=int)

                for q in review_budget_grid:
                    budget_n = int(np.ceil(float(q) * float(n_total_subj)))
                    chosen = order[: min(int(budget_n), int(order.size))]
                    acc_final = np.asarray(acc_selected_arr, dtype=float).copy()
                    if chosen.size > 0:
                        acc_final[chosen] = oracle_acc_arr[chosen]
                    delta_final = acc_final - acc_anchor_arr
                    k_post = int(np.sum((delta_final < -float(neg_transfer_eps)) & accepted_mask_arr))
                    ucb_post = (
                        _clopper_pearson_ucb(k=k_post, n=n_acc, delta=float(args.delta)) if n_acc > 0 else float("nan")
                    )
                    if np.isfinite(ucb_post) and ucb_post <= beta + 1e-12:
                        best_q = float(q)
                        best_budget_n = int(budget_n)
                        break

                if best_q is None or best_budget_n is None:
                    review_infeasible = 1
                    best_q = float(max(review_budget_grid))
                    best_budget_n = int(np.ceil(best_q * float(n_total_subj)))

            chosen_review_budget_frac = float(best_q)
            chosen_review_budget_n = int(best_budget_n)
            review_budget_frac = float(best_q)
        else:
            chosen_review_budget_frac = float(review_budget_frac)
            chosen_review_budget_n = int(np.ceil(float(review_budget_frac) * float(n_total_subj)))

        review_budget_n = max(0, int(chosen_review_budget_n))

        if review_budget_n > 0 and int(np.sum(cand_mask)) > 0:
            score_vals = pd.to_numeric(per_subj[score_col], errors="coerce").to_numpy(dtype=float)
            score_vals = np.where(np.isfinite(score_vals), score_vals, -np.inf)

            cand_idx = np.where(cand_mask)[0]
            order = cand_idx[np.argsort(-score_vals[cand_idx], kind="mergesort")]
            chosen = order[: min(review_budget_n, int(order.size))]

            if chosen.size > 0:
                per_subj.loc[chosen, "review_flag"] = 1
                per_subj.loc[chosen, "review_score"] = score_vals[chosen]
                per_subj.loc[chosen, "review_rank"] = np.arange(1, int(chosen.size) + 1, dtype=float)
                per_subj.loc[chosen, "review_reason"] = str(chosen_reason)
                per_subj.loc[chosen, "acc_final"] = per_subj.loc[chosen, "oracle_acc"].astype(float)
                per_subj.loc[chosen, "delta_final_vs_anchor"] = (
                    per_subj.loc[chosen, "acc_final"].astype(float) - per_subj.loc[chosen, "acc_anchor"].astype(float)
                )
                per_subj.loc[chosen, "is_oracle_corrected"] = (
                    per_subj.loc[chosen, "acc_final"].astype(float) > per_subj.loc[chosen, "acc_selected"].astype(float) + 1e-12
                ).astype(int)

            not_reviewed = np.where(
                cand_mask
                & (pd.to_numeric(per_subj["review_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 0.5)
            )[0]
            if not_reviewed.size > 0:
                per_subj.loc[not_reviewed, "review_reason"] = str(exhausted_reason)
        else:
            not_reviewed = np.where(cand_mask)[0]
            if not_reviewed.size > 0:
                per_subj.loc[not_reviewed, "review_reason"] = str(zero_reason)

    # Finalize review annotations for non-reviewed subjects.
    if per_subj.shape[0] > 0:
        unreviewed_mask = pd.to_numeric(per_subj["review_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 0.5
        if review_scope == "abstain":
            # Preserve old behavior: accepted subjects are not eligible for review.
            accept_arr = pd.to_numeric(per_subj["accept"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            accepted_mask_arr = accept_arr > 0.5
            idx = np.where(unreviewed_mask & accepted_mask_arr)[0]
            if idx.size > 0:
                per_subj.loc[idx, "review_reason"] = "not_abstain"
        else:
            idx = np.where(unreviewed_mask & (per_subj["review_reason"].astype(str) == "not_reviewed"))[0]
            if idx.size > 0:
                per_subj.loc[idx, "review_reason"] = "not_in_scope"

    per_subj["neg_transfer_eps"] = float(neg_transfer_eps)
    per_subj.to_csv(out_dir / f"{date_prefix}_per_subject_selection.csv", index=False)

    if calib_grid_dump_rows is not None:
        grid_df = pd.DataFrame(calib_grid_dump_rows)
        grid_df.to_csv(out_dir / f"{date_prefix}_calib_grid.csv", index=False)

    mean_acc = float(per_subj["acc_selected"].mean())
    worst_acc = float(per_subj["acc_selected"].min())
    mean_delta = float(per_subj["delta_selected_vs_anchor"].mean())
    neg_transfer = float((per_subj["delta_selected_vs_anchor"] < -float(neg_transfer_eps)).mean())
    accept_rate = float(per_subj["accept"].mean())
    mean_oracle = float(per_subj["oracle_acc"].mean())
    mean_headroom = float(per_subj["headroom"].mean())
    mean_oracle_gap = float(per_subj["oracle_gap"].mean())

    # Metrics after review-proxy.
    mean_acc_after_review = float(per_subj["acc_final"].mean())
    mean_delta_after_review = float(per_subj["delta_final_vs_anchor"].mean())
    residual_harm_rate_after_review = float((per_subj["delta_final_vs_anchor"] < -float(neg_transfer_eps)).mean())
    residual_harm_count_after_review = int(
        np.sum(per_subj["delta_final_vs_anchor"].to_numpy(dtype=float) < -float(neg_transfer_eps))
    )
    review_rate = float(pd.to_numeric(per_subj["review_flag"], errors="coerce").fillna(0.0).mean())
    reviewed_mask = pd.to_numeric(per_subj["review_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
    if int(np.sum(reviewed_mask)) > 0:
        review_gain = per_subj.loc[reviewed_mask, "acc_final"].astype(float) - per_subj.loc[reviewed_mask, "acc_selected"].astype(float)
        review_yield = float(review_gain.mean())
    else:
        review_yield = 0.0
    net_utility = float(mean_delta_after_review - float(args.review_cost) * review_rate)

    # Dual-risk summary on deployed (pre-review) decisions.
    n_subj = int(per_subj.shape[0])
    k_neg_marg = int(np.sum(per_subj["delta_selected_vs_anchor"].to_numpy(dtype=float) < -float(neg_transfer_eps)))
    marg_rate = float(k_neg_marg / n_subj) if n_subj > 0 else float("nan")
    marg_ucb = _clopper_pearson_ucb(k=k_neg_marg, n=n_subj, delta=float(args.delta)) if n_subj > 0 else float("nan")
    accepted_mask = pd.to_numeric(per_subj["accept"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
    n_accept_total = int(np.sum(accepted_mask))
    k_neg_cond = int(
        np.sum((per_subj["delta_selected_vs_anchor"].to_numpy(dtype=float) < -float(neg_transfer_eps)) & accepted_mask)
    )
    if n_accept_total > 0:
        cond_rate = float(k_neg_cond / n_accept_total)
        cond_ci_low, cond_ci_high = _clopper_pearson_ci(k=k_neg_cond, n=n_accept_total, delta=float(args.delta))
    else:
        cond_rate = float("nan")
        cond_ci_low, cond_ci_high = float("nan"), float("nan")

    # Dual-risk summary after review-proxy (final decisions).
    k_neg_marg_post = int(np.sum(per_subj["delta_final_vs_anchor"].to_numpy(dtype=float) < -float(neg_transfer_eps)))
    marg_rate_after_review = float(k_neg_marg_post / n_subj) if n_subj > 0 else float("nan")
    marg_ucb_after_review = (
        _clopper_pearson_ucb(k=k_neg_marg_post, n=n_subj, delta=float(args.delta)) if n_subj > 0 else float("nan")
    )
    k_neg_cond_post = int(
        np.sum((per_subj["delta_final_vs_anchor"].to_numpy(dtype=float) < -float(neg_transfer_eps)) & accepted_mask)
    )
    if n_accept_total > 0:
        cond_rate_after_review = float(k_neg_cond_post / n_accept_total)
        cond_ucb_after_review = _clopper_pearson_ucb(k=k_neg_cond_post, n=n_accept_total, delta=float(args.delta))
        cond_ci_low_after_review, cond_ci_high_after_review = _clopper_pearson_ci(
            k=k_neg_cond_post, n=n_accept_total, delta=float(args.delta)
        )
    else:
        cond_rate_after_review = float("nan")
        cond_ucb_after_review = float("nan")
        cond_ci_low_after_review, cond_ci_high_after_review = float("nan"), float("nan")

    coverage = float(eval_n_selected / eval_n_subjects_total) if eval_n_subjects_total > 0 else float("nan")
    accept_selected = float(eval_n_accept_selected / eval_n_selected) if eval_n_selected > 0 else float("nan")
    negT_selected = float(eval_n_neg_transfer_selected / eval_n_selected) if eval_n_selected > 0 else float("nan")

    calib_coverage = float(np.nanmean(np.asarray(calib_coverage_vals, dtype=float))) if calib_coverage_vals else float("nan")
    calib_accept_selected = (
        float(np.nanmean(np.asarray(calib_accept_selected_vals, dtype=float))) if calib_accept_selected_vals else float("nan")
    )
    calib_negT_selected = float(np.nanmean(np.asarray(calib_negT_selected_vals, dtype=float))) if calib_negT_selected_vals else float("nan")
    calib_n_selected = float(np.nanmean(np.asarray(calib_n_selected_vals, dtype=float))) if calib_n_selected_vals else float("nan")

    comp = pd.DataFrame(
        [
            {
                "method": str(args.method_name),
                "anchor_method": str(anchor_method),
                "anchor_family": str(anchor_family),
                "n_subjects": int(per_subj.shape[0]),
                "mean_accuracy": float(mean_acc),
                "worst_accuracy": float(worst_acc),
                "mean_delta_vs_anchor": float(mean_delta),
                "neg_transfer_vs_anchor": float(neg_transfer),
                "meanΔacc_vs_ea-csp-lda": float(mean_delta),
                "neg_transfer_vs_ea-csp-lda": float(neg_transfer),
                "accept_rate": float(accept_rate),
                "marg_rate": float(marg_rate),
                "marg_ucb": float(marg_ucb),
                "cond_rate": float(cond_rate),
                "cond_ci_low": float(cond_ci_low),
                "cond_ci_high": float(cond_ci_high),
                "neg_transfer_eps": float(neg_transfer_eps),
                "oracle_mean_accuracy": float(mean_oracle),
                "headroom_mean": float(mean_headroom),
                "oracle_gap_mean": float(mean_oracle_gap),
                "n_candidates": int(len(cand_methods) + 1),
                "coverage": float(coverage),
                "accept_selected": float(accept_selected),
                "negT_selected": float(negT_selected),
                "calib_coverage": float(calib_coverage),
                "calib_accept_selected": float(calib_accept_selected),
                "calib_negT_selected": float(calib_negT_selected),
                "calib_n_selected": float(calib_n_selected),
                "risk_mode": str(args.risk_mode),
                "risk_beta_cond": float(args.risk_beta_cond) if args.risk_beta_cond is not None else float("nan"),
                "min_accept_rate": float(args.min_accept_rate),
                "calib_objective": str(args.calib_objective),
                "mean_accuracy_after_review": float(mean_acc_after_review),
                "mean_delta_all_after_review": float(mean_delta_after_review),
                "residual_harm_rate_after_review": float(residual_harm_rate_after_review),
                "residual_harm_count_after_review": int(residual_harm_count_after_review),
                "review_rate": float(review_rate),
                "review_yield": float(review_yield),
                "review_budget_frac": float(review_budget_frac),
                "chosen_review_budget_frac": float(chosen_review_budget_frac),
                "chosen_review_budget_n": int(chosen_review_budget_n),
                "review_budget_mode": str(review_budget_mode),
                "cond_risk_target": str(cond_risk_target),
                "review_infeasible": int(review_infeasible),
                "review_cost": float(args.review_cost),
                "review_score_mode": str(review_score_name),
                "review_scope": str(review_scope),
                "net_utility": float(net_utility),
                "marg_rate_after_review": float(marg_rate_after_review),
                "marg_ucb_after_review": float(marg_ucb_after_review),
                "cond_rate_after_review": float(cond_rate_after_review),
                "cond_ucb_after_review": float(cond_ucb_after_review),
                "cond_ci_low_after_review": float(cond_ci_low_after_review),
                "cond_ci_high_after_review": float(cond_ci_high_after_review),
            }
        ]
    )
    comp_path = out_dir / f"{date_prefix}_method_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"[run] done method={args.method_name} subjects={total_subjects} elapsed={time.time() - run_t0:.1f}s", flush=True)
    print(f"[done] wrote: {out_dir / (date_prefix + '_predictions_all_methods.csv')}", flush=True)
    print(f"[done] wrote: {out_dir / (date_prefix + '_per_subject_selection.csv')}", flush=True)
    print(f"[done] wrote: {comp_path}", flush=True)


if __name__ == "__main__":
    main()
