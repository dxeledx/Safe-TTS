from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence, TYPE_CHECKING

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import nn

from .alignment import apply_spatial_transform
from .proba import reorder_proba_columns as _reorder_proba_columns

if TYPE_CHECKING:
    from .model import TrainedModel


@dataclass(frozen=True)
class RidgeCertificate:
    """Simple calibrated certificate model for candidate selection.

    This regresses (unlabeled features) -> (expected improvement over the EA anchor)
    on pseudo-target subjects.
    """

    model: Pipeline
    feature_names: tuple[str, ...]

    def predict_accuracy(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return np.asarray(self.model.predict(features), dtype=np.float64)


@dataclass(frozen=True)
class LogisticGuard:
    """Binary guard model for rejecting likely negative-transfer candidates.

    Models: P(improvement_over_identity > margin | unlabeled_features).
    """

    model: Pipeline
    feature_names: tuple[str, ...]

    def predict_pos_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = np.asarray(self.model.predict_proba(features), dtype=np.float64)
        if proba.ndim != 2:
            raise ValueError("Unexpected proba shape.")
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]


@dataclass(frozen=True)
class SoftmaxBanditPolicy:
    """Linear softmax contextual bandit policy for candidate selection.

    Given a candidate feature vector x, assigns a score s(x)=θᵀ·z where z is the standardized feature.
    Training uses pseudo-target subjects with full-information rewards (Δacc per candidate) and maximizes
    the expected reward under the softmax distribution over candidates in each pseudo-target set.
    """

    scaler: StandardScaler
    theta: np.ndarray  # (d,)
    feature_names: tuple[str, ...]

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        z = self.scaler.transform(features)
        return np.asarray(z @ self.theta.reshape(-1, 1), dtype=np.float64).reshape(-1)

    def action_probs(self, features: np.ndarray) -> np.ndarray:
        s = self.score(features).reshape(-1)
        if s.size == 0:
            return np.asarray([], dtype=np.float64)
        s = s - float(np.max(s))
        p = np.exp(s)
        p = p / float(np.sum(p))
        return np.asarray(p, dtype=np.float64)


def _safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def _safe_float_or(x, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def candidate_features_from_record(
    rec: dict,
    *,
    n_classes: int,
    include_pbar: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a feature vector from a candidate record (label-free)."""

    p_bar = np.asarray(rec.get("p_bar_full", np.zeros(n_classes)), dtype=np.float64).reshape(-1)
    if p_bar.shape[0] != n_classes:
        p_bar = np.zeros(n_classes, dtype=np.float64)
    p_bar = np.clip(p_bar, 1e-12, 1.0)
    p_bar = p_bar / float(np.sum(p_bar))
    p_bar_ent = float(-np.sum(p_bar * np.log(p_bar)))

    n_keep = int(rec.get("n_keep", -1))
    n_best_total = int(rec.get("n_best_total", -1))
    keep_ratio = 0.0
    if n_keep >= 0 and n_best_total > 0:
        keep_ratio = float(n_keep) / float(n_best_total)

    # Optional bilevel stats (may be missing on older runs).
    coverage = _safe_float(rec.get("coverage", 0.0))
    eff_n = _safe_float(rec.get("eff_n", 0.0))
    mean_entropy_q = _safe_float(rec.get("mean_entropy_q", 0.0))

    drift_best = _safe_float(rec.get("drift_best", 0.0))
    drift_best_std = _safe_float(rec.get("drift_best_std", 0.0))
    drift_best_q90 = _safe_float(rec.get("drift_best_q90", 0.0))
    drift_best_q95 = _safe_float(rec.get("drift_best_q95", 0.0))
    drift_best_max = _safe_float(rec.get("drift_best_max", 0.0))
    drift_best_tail_frac = _safe_float(rec.get("drift_best_tail_frac", 0.0))

    q_bar = np.asarray(rec.get("q_bar", np.zeros(n_classes)), dtype=np.float64).reshape(-1)
    if q_bar.shape[0] != n_classes:
        q_bar = np.zeros(n_classes, dtype=np.float64)
    q_bar = np.clip(q_bar, 1e-12, 1.0)
    q_bar = q_bar / float(np.sum(q_bar))
    q_bar_ent = float(-np.sum(q_bar * np.log(q_bar)))

    feats: list[float] = [
        _safe_float(rec.get("objective_base", 0.0)),
        _safe_float(rec.get("pen_marginal", 0.0)),
        drift_best,
        drift_best_std,
        drift_best_q90,
        drift_best_q95,
        drift_best_max,
        drift_best_tail_frac,
        _safe_float(rec.get("mean_entropy", 0.0)),
        mean_entropy_q,
        _safe_float(rec.get("entropy_bar", 0.0)),
        _safe_float(keep_ratio),
        coverage,
        eff_n,
        _safe_float(p_bar_ent),
        _safe_float(q_bar_ent),
    ]
    names: list[str] = [
        "objective_base",
        "pen_marginal",
        "drift_best",
        "drift_best_std",
        "drift_best_q90",
        "drift_best_q95",
        "drift_best_max",
        "drift_best_tail_frac",
        "mean_entropy",
        "mean_entropy_q",
        "entropy_bar",
        "keep_ratio",
        "coverage",
        "eff_n",
        "pbar_entropy",
        "qbar_entropy",
    ]

    # Candidate meta info (optional; useful when mixing candidate families in a single selector).
    cand_family = str(rec.get("cand_family", "")).strip().lower()
    for fam in ("ea", "rpa", "tsa", "chan", "mdm", "fbcsp", "ts_svc", "tsa_ts_svc", "fgmdm"):
        feats.append(1.0 if cand_family == fam else 0.0)
        names.append(f"cand_family_{fam}")
    feats.append(_safe_float(rec.get("cand_rank", 0.0)))
    names.append("cand_rank")
    feats.append(_safe_float(rec.get("cand_lambda", 0.0)))
    names.append("cand_lambda")

    if include_pbar:
        feats.extend([_safe_float(x) for x in p_bar.tolist()])
        names.extend([f"pbar_{k}" for k in range(n_classes)])
        feats.extend([_safe_float(x) for x in q_bar.tolist()])
        names.extend([f"qbar_{k}" for k in range(n_classes)])

    return np.asarray(feats, dtype=np.float64), tuple(names)


def stacked_candidate_features_from_record(
    rec: dict,
    *,
    n_classes: int,
    include_pbar: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Feature vector that also includes multiple certificate signals (evidence/probe/etc).

    Intended for calibrated model selection (certificate stacking).
    """

    base_feats, base_names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=include_pbar)

    # Note: for NLL-like certificates, missing values are set to a large default
    # so that "missing" never looks artificially good.
    LARGE = 1e6
    extras: list[float] = [
        _safe_float(rec.get("objective", 0.0)),
        _safe_float(rec.get("score", 0.0)),
        _safe_float(rec.get("mean_confidence", 0.0)),
        _safe_float_or(rec.get("evidence_nll_best", np.nan), LARGE),
        _safe_float_or(rec.get("evidence_nll_full", np.nan), LARGE),
        _safe_float_or(rec.get("probe_mixup_best", np.nan), LARGE),
        _safe_float_or(rec.get("probe_mixup_full", np.nan), LARGE),
        _safe_float_or(rec.get("probe_mixup_hard_best", np.nan), LARGE),
        _safe_float_or(rec.get("probe_mixup_hard_full", np.nan), LARGE),
        _safe_float(rec.get("probe_mixup_pairs_best", 0.0)),
        _safe_float(rec.get("probe_mixup_pairs_full", 0.0)),
        _safe_float(rec.get("probe_mixup_keep_best", 0.0)),
        _safe_float(rec.get("probe_mixup_keep_full", 0.0)),
        _safe_float(rec.get("probe_mixup_frac_intra_best", 0.0)),
        _safe_float(rec.get("probe_mixup_frac_intra_full", 0.0)),
        _safe_float(rec.get("probe_mixup_hard_pairs_best", 0.0)),
        _safe_float(rec.get("probe_mixup_hard_pairs_full", 0.0)),
        _safe_float(rec.get("probe_mixup_hard_keep_best", 0.0)),
        _safe_float(rec.get("probe_mixup_hard_keep_full", 0.0)),
        _safe_float(rec.get("probe_mixup_hard_frac_intra_best", 0.0)),
        _safe_float(rec.get("probe_mixup_hard_frac_intra_full", 0.0)),
    ]
    extra_names: list[str] = [
        "objective",
        "score",
        "mean_confidence",
        "evidence_nll_best",
        "evidence_nll_full",
        "probe_mixup_best",
        "probe_mixup_full",
        "probe_mixup_hard_best",
        "probe_mixup_hard_full",
        "probe_mixup_pairs_best",
        "probe_mixup_pairs_full",
        "probe_mixup_keep_best",
        "probe_mixup_keep_full",
        "probe_mixup_frac_intra_best",
        "probe_mixup_frac_intra_full",
        "probe_mixup_hard_pairs_best",
        "probe_mixup_hard_pairs_full",
        "probe_mixup_hard_keep_best",
        "probe_mixup_hard_keep_full",
        "probe_mixup_hard_frac_intra_best",
        "probe_mixup_hard_frac_intra_full",
    ]

    feats = np.concatenate([base_feats, np.asarray(extras, dtype=np.float64)], axis=0)
    names = tuple(base_names) + tuple(extra_names)
    return feats, names


def stacked_candidate_features_delta_from_records(
    rec: dict,
    *,
    anchor: dict,
    n_classes: int,
    include_pbar: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Anchor-relative stacked feature vector.

    Computes Δ-features = features(rec) - features(anchor) for all non-meta features.

    Meta features are kept absolute:
    - cand_family_* one-hots
    - cand_rank / cand_lambda

    This is intended for calibrated selection where the target is Δacc over the EA anchor.
    """

    x, names = stacked_candidate_features_from_record(rec, n_classes=n_classes, include_pbar=include_pbar)
    x0, names0 = stacked_candidate_features_from_record(anchor, n_classes=n_classes, include_pbar=include_pbar)
    if names != names0:
        raise ValueError("Anchor/record feature name mismatch.")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    out = x.copy()
    out_names = list(names)
    for i, name in enumerate(names):
        if name.startswith("cand_family_") or name in {"cand_rank", "cand_lambda"}:
            continue
        out[i] = float(x[i]) - float(x0[i])
        out_names[i] = f"delta_{name}"
    return out, tuple(out_names)


def candidate_features_delta_from_records(
    rec: dict,
    *,
    anchor: dict,
    n_classes: int,
    include_pbar: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Anchor-relative *base* feature vector.

    Computes Δ-features = base_features(rec) - base_features(anchor) for all non-meta features.

    Meta features are kept absolute:
    - cand_family_* one-hots
    - cand_rank / cand_lambda

    This is the low-dimensional analogue of `stacked_candidate_features_delta_from_records`.
    """

    x, names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=include_pbar)
    x0, names0 = candidate_features_from_record(anchor, n_classes=n_classes, include_pbar=include_pbar)
    if names != names0:
        raise ValueError("Anchor/record feature name mismatch.")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    out = x.copy()
    out_names = list(names)
    for i, name in enumerate(names):
        if name.startswith("cand_family_") or name in {"cand_rank", "cand_lambda"}:
            continue
        out[i] = float(x[i]) - float(x0[i])
        out_names[i] = f"delta_{name}"
    return out, tuple(out_names)


def _csp_logvar_features(*, model: TrainedModel, X: np.ndarray) -> np.ndarray:
    """Compute CSP log-variance features with the already-fitted CSP filters."""

    X = np.asarray(X, dtype=np.float64)
    csp = model.csp
    F = np.asarray(csp.filters_[: int(csp.n_components)], dtype=np.float64)
    use_log = True if (getattr(csp, "log", None) is None) else bool(getattr(csp, "log"))
    Y = np.einsum("kc,nct->nkt", F, X, optimize=True)
    power = np.mean(Y * Y, axis=2)
    power = np.maximum(power, 1e-20)
    return np.log(power) if use_log else power


def _fit_domain_logreg_ratio(
    *,
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
    c: float = 1.0,
    clip_max: float = 20.0,
) -> np.ndarray:
    """Estimate density ratio w(x)=p_T(x)/p_S(x) via a domain classifier in feature space.

    Uses a balanced training set (subsampling) so that w(x) = p(d=1|x)/p(d=0|x).
    """

    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError("Expected 2D feature arrays for domain ratio.")
    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError("Source/target feature dim mismatch.")
    n_s = int(X_source.shape[0])
    n_t = int(X_target.shape[0])
    if n_s < 2 or n_t < 2:
        return np.ones(n_s, dtype=np.float64)

    rng = np.random.RandomState(int(seed))
    n = int(min(n_s, n_t))
    idx_s = rng.choice(n_s, size=n, replace=False)
    idx_t = rng.choice(n_t, size=n, replace=False)
    X = np.concatenate([X_source[idx_s], X_target[idx_t]], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)], axis=0)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(c),
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(X, y)

    p_t = np.asarray(clf.predict_proba(X_source), dtype=np.float64)[:, 1]
    p_t = np.clip(p_t, 1e-6, 1.0 - 1e-6)
    w = p_t / (1.0 - p_t)
    w = np.clip(w, 0.0, float(clip_max))
    return np.asarray(w, dtype=np.float64)


def select_by_iwcv_nll(
    records: Iterable[dict],
    *,
    model: TrainedModel,
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target: np.ndarray,
    class_order: Sequence[str],
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
    seed: int = 0,
) -> dict:
    """Select candidate by importance-weighted (covariate-shift) NLL on labeled source.

    For each candidate A, we:
    - compute CSP feature distributions on (A·z_source) and (A·z_target),
    - estimate density ratio w(x)=p_T(x)/p_S(x) via a domain classifier in feature space,
    - score the candidate by weighted NLL on source labels under the frozen CSP+LDA model.

    Smaller is better. Falls back to identity if no improvement.
    """

    class_order = [str(c) for c in class_order]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_source = np.asarray(y_source)
    try:
        y_idx = np.fromiter((class_to_idx[str(c)] for c in y_source), dtype=int, count=len(y_source))
    except KeyError as e:
        raise ValueError(f"y_source contains unknown class '{e.args[0]}'.") from e

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec_i, rec in enumerate(records):
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        Q = rec.get("Q", None)
        if Q is None:
            continue
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2:
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        Xs = apply_spatial_transform(Q, z_source)
        Xt = apply_spatial_transform(Q, z_target)
        fs = _csp_logvar_features(model=model, X=Xs)
        ft = _csp_logvar_features(model=model, X=Xt)

        w = _fit_domain_logreg_ratio(X_source=fs, X_target=ft, seed=int(seed) + 9973 * int(rec_i))
        w_sum = float(np.sum(w))
        w_sq_sum = float(np.sum(w * w))
        eff_n = (w_sum * w_sum / w_sq_sum) if w_sq_sum > 1e-12 else 0.0

        proba_s = np.asarray(model.predict_proba(Xs), dtype=np.float64)
        proba_s = _reorder_proba_columns(proba_s, model.classes_, class_order)
        p = np.clip(proba_s, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        nll = -np.log(p[np.arange(p.shape[0]), y_idx])
        score = float(np.sum(w * nll) / max(1e-12, w_sum))

        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        # Store for diagnostics / calibrated models.
        rec["iwcv_nll"] = float(score)
        rec["iwcv_eff_n"] = float(eff_n)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("iwcv_nll", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        # Ensure identity has a score if we computed none for it.
        if best is not None:
            return best
        return identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best


def select_by_iwcv_ucb(
    records: Iterable[dict],
    *,
    model: TrainedModel,
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target: np.ndarray,
    class_order: Sequence[str],
    kappa: float = 1.0,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
    seed: int = 0,
) -> dict:
    """Select candidate by an IWCV-UCB certificate (risk estimate + uncertainty penalty).

    Certificate (smaller is better):
      mean_w[nll] + kappa * sqrt(var_w[nll] / n_eff),
    where n_eff = (sum w)^2 / sum w^2 is the effective sample size of the (clipped) density ratios.

    This is intended to be a more *mathematically grounded* certificate:
    - If weights are unstable (small n_eff), the uncertainty term increases -> safer selection.
    - If the estimated target risk is genuinely low and stable, it can beat the identity anchor.
    """

    if float(kappa) < 0.0:
        raise ValueError("kappa must be >= 0.")

    class_order = [str(c) for c in class_order]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_source = np.asarray(y_source)
    try:
        y_idx = np.fromiter((class_to_idx[str(c)] for c in y_source), dtype=int, count=len(y_source))
    except KeyError as e:
        raise ValueError(f"y_source contains unknown class '{e.args[0]}'.") from e

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec_i, rec in enumerate(records):
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        Q = rec.get("Q", None)
        if Q is None:
            continue
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2:
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        Xs = apply_spatial_transform(Q, z_source)
        Xt = apply_spatial_transform(Q, z_target)
        fs = _csp_logvar_features(model=model, X=Xs)
        ft = _csp_logvar_features(model=model, X=Xt)

        w = _fit_domain_logreg_ratio(X_source=fs, X_target=ft, seed=int(seed) + 9973 * int(rec_i))
        w_sum = float(np.sum(w))
        w_sq_sum = float(np.sum(w * w))
        eff_n = (w_sum * w_sum / w_sq_sum) if w_sq_sum > 1e-12 else 0.0

        proba_s = np.asarray(model.predict_proba(Xs), dtype=np.float64)
        proba_s = _reorder_proba_columns(proba_s, model.classes_, class_order)
        p = np.clip(proba_s, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        nll = -np.log(p[np.arange(p.shape[0]), y_idx])

        if w_sum <= 1e-12:
            score = float("inf")
            mean = float("nan")
            var = float("nan")
            se = float("nan")
        else:
            mean = float(np.sum(w * nll) / max(1e-12, w_sum))
            var = float(np.sum(w * (nll - mean) * (nll - mean)) / max(1e-12, w_sum))
            se = float(np.sqrt(max(0.0, var) / max(1e-12, eff_n)))
            score = float(mean) + float(kappa) * float(se)

        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        # Store for diagnostics / calibrated models.
        rec["iwcv_nll"] = float(mean) if np.isfinite(mean) else float(rec.get("iwcv_nll", float("nan")))
        rec["iwcv_eff_n"] = float(eff_n)
        rec["iwcv_var"] = float(var) if np.isfinite(var) else float("nan")
        rec["iwcv_se"] = float(se) if np.isfinite(se) else float("nan")
        rec["iwcv_ucb"] = float(score)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("iwcv_ucb", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        # Ensure identity has a score if we computed none for it.
        if best is not None:
            return best
        return identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best


def select_by_dev_nll(
    records: Iterable[dict],
    *,
    model: TrainedModel,
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target: np.ndarray,
    class_order: Sequence[str],
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
    seed: int = 0,
) -> dict:
    """Select candidate by a DEV-style control-variate IW certificate.

    Reference: Deep Embedded Validation (DEV), ICML 2019.

    Given density ratios w(x) = p_T(x)/p_S(x) estimated in feature space, DEV forms an
    importance-weighted risk estimate with a control variate:

      score(Q) = mean_s[w * nll] + eta * (mean_s[w] - 1),
      eta = -Cov_s(w*nll, w) / Var_s(w).

    Smaller is better. Falls back to identity if no improvement.
    """

    class_order = [str(c) for c in class_order]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_source = np.asarray(y_source)
    try:
        y_idx = np.fromiter((class_to_idx[str(c)] for c in y_source), dtype=int, count=len(y_source))
    except KeyError as e:
        raise ValueError(f"y_source contains unknown class '{e.args[0]}'.") from e

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec_i, rec in enumerate(records):
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        Q = rec.get("Q", None)
        if Q is None:
            continue
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2:
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        Xs = apply_spatial_transform(Q, z_source)
        Xt = apply_spatial_transform(Q, z_target)
        fs = _csp_logvar_features(model=model, X=Xs)
        ft = _csp_logvar_features(model=model, X=Xt)

        w = _fit_domain_logreg_ratio(X_source=fs, X_target=ft, seed=int(seed) + 9973 * int(rec_i))
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w_sum = float(np.sum(w))
        w_sq_sum = float(np.sum(w * w))
        eff_n = (w_sum * w_sum / w_sq_sum) if w_sq_sum > 1e-12 else 0.0

        proba_s = np.asarray(model.predict_proba(Xs), dtype=np.float64)
        proba_s = _reorder_proba_columns(proba_s, model.classes_, class_order)
        p = np.clip(proba_s, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        nll = -np.log(p[np.arange(p.shape[0]), y_idx])

        # DEV control variate.
        L = w * nll
        mean_L = float(np.mean(L))
        mean_W = float(np.mean(w))
        Wc = w - mean_W
        var_W = float(np.mean(Wc * Wc))
        if var_W > 1e-12:
            cov_LW = float(np.mean((L - mean_L) * Wc))
            eta = -float(cov_LW) / float(var_W)
        else:
            cov_LW = float("nan")
            eta = 0.0
        score = float(mean_L) + float(eta) * float(mean_W - 1.0)

        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        rec["dev_nll"] = float(score)
        rec["dev_eta"] = float(eta)
        rec["dev_mean_w"] = float(mean_W)
        rec["dev_var_w"] = float(var_W) if np.isfinite(var_W) else float("nan")
        rec["dev_cov_LW"] = float(cov_LW) if np.isfinite(cov_LW) else float("nan")
        rec["dev_eff_n"] = float(eff_n)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("dev_nll", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        if best is not None:
            return best
        return identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best


def train_ridge_certificate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    alpha: float = 1.0,
) -> RidgeCertificate:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch.")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be > 0.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(X, y)
    return RidgeCertificate(model=model, feature_names=tuple(feature_names))


def train_logistic_guard(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    c: float = 1.0,
) -> LogisticGuard:
    """Train a negative-transfer guard on pseudo-target subjects."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch.")
    if float(c) <= 0.0:
        raise ValueError("c must be > 0.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(c),
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X, y)
    return LogisticGuard(model=model, feature_names=tuple(feature_names))


def train_hgb_certificate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    max_iter: int = 200,
    random_state: int = 0,
) -> RidgeCertificate:
    """Train a tree-based certificate model (HGB regressor).

    This is a drop-in alternative to the ridge certificate when non-linear effects
    are expected in the unlabeled feature space.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    model.fit(X, y)
    return RidgeCertificate(model=model, feature_names=tuple(feature_names))


def train_hgb_guard(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    max_iter: int = 200,
    random_state: int = 0,
) -> LogisticGuard:
    """Train a tree-based guard model (HGB classifier).

    Models: P(improvement > 0 | unlabeled_features) on pseudo-target subjects.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                ),
            ),
        ]
    )

    n = int(y.shape[0])
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos > 0 and n_neg > 0:
        w_pos = float(n) / (2.0 * float(n_pos))
        w_neg = float(n) / (2.0 * float(n_neg))
        sample_weight = np.where(y == 1, w_pos, w_neg).astype(np.float64)
        model.fit(X, y, hgb__sample_weight=sample_weight)
    else:
        model.fit(X, y)

    return LogisticGuard(model=model, feature_names=tuple(feature_names))


class _EvidentialMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))


class _MultiViewEvidentialHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.evidence = nn.Linear(int(hidden_dim), 3)
        self.reliability = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return F.softplus(self.evidence(h)), torch.sigmoid(self.reliability(h))


class _MultiViewEvidentialMLP(nn.Module):
    def __init__(self, view_dims: dict[str, int], hidden_dim: int) -> None:
        super().__init__()
        self.view_order = tuple(view_dims.keys())
        self.heads = nn.ModuleDict(
            {name: _MultiViewEvidentialHead(int(dim), int(hidden_dim)) for name, dim in view_dims.items()}
        )

    def forward(self, views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused_parts: list[torch.Tensor] = []
        reliabilities: dict[str, torch.Tensor] = {}
        for name in self.view_order:
            evidence, reliability = self.heads[name](views[name])
            fused_parts.append(reliability * evidence)
            reliabilities[name] = reliability.squeeze(1)
        fused_evidence = torch.stack(fused_parts, dim=0).sum(dim=0)
        alpha = fused_evidence + 1.0
        return {"alpha": alpha, "reliability": reliabilities}


@dataclass(frozen=True)
class EvidentialSelector:
    scaler: StandardScaler | None
    model: nn.Module
    feature_names: tuple[str, ...]
    rho: float
    eta: float
    view_slices: tuple[tuple[str, int, int], ...] = ()
    view_stats: tuple[tuple[str, np.ndarray, np.ndarray], ...] = ()

    def _predict_alpha(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        self.model.eval()
        if self.scaler is not None:
            z = self.scaler.transform(features)
            x = torch.as_tensor(z, dtype=torch.float32)
            with torch.no_grad():
                evidence = self.model(x).cpu().numpy().astype(np.float64, copy=False)
            return evidence + 1.0

        if not self.view_slices or not self.view_stats:
            raise RuntimeError("Missing view metadata for multi-view evidential selector.")

        stats_map = {
            str(name): (
                np.asarray(mu, dtype=np.float64).reshape(-1),
                np.asarray(sigma, dtype=np.float64).reshape(-1),
            )
            for name, mu, sigma in self.view_stats
        }
        views: dict[str, torch.Tensor] = {}
        for name, start, stop in self.view_slices:
            mu, sigma = stats_map[str(name)]
            view = features[:, int(start) : int(stop)]
            view = (view - mu.reshape(1, -1)) / sigma.reshape(1, -1)
            views[str(name)] = torch.as_tensor(view.astype(np.float32, copy=False), dtype=torch.float32)
        with torch.no_grad():
            out = self.model(views)
            alpha = out["alpha"].cpu().numpy().astype(np.float64, copy=False)
        return alpha

    def predict_stats(self, features: np.ndarray) -> dict[str, np.ndarray]:
        alpha = self._predict_alpha(features)
        evidence = np.maximum(alpha - 1.0, 0.0)
        s = np.sum(alpha, axis=1, keepdims=True)
        probs = alpha / s
        beliefs = evidence / s
        uncertainty = float(alpha.shape[1]) / np.clip(s[:, 0], 1e-12, None)
        risk = np.clip(beliefs[:, 0] + float(self.rho) * uncertainty, 0.0, 1.0)
        utility = (beliefs[:, 2] - beliefs[:, 0]) - float(self.eta) * uncertainty
        non_harm = np.clip(1.0 - risk, 0.0, 1.0)
        out = {
            "alpha": alpha,
            "probs": probs,
            "beliefs": beliefs,
            "uncertainty": uncertainty,
            "risk": risk,
            "utility": utility,
            "non_harm": non_harm,
        }
        if self.scaler is None and self.view_slices and self.view_stats:
            stats_map = {
                str(name): (
                    np.asarray(mu, dtype=np.float64).reshape(-1),
                    np.asarray(sigma, dtype=np.float64).reshape(-1),
                )
                for name, mu, sigma in self.view_stats
            }
            x = np.asarray(features, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            views: dict[str, torch.Tensor] = {}
            for name, start, stop in self.view_slices:
                mu, sigma = stats_map[str(name)]
                view = x[:, int(start) : int(stop)]
                view = (view - mu.reshape(1, -1)) / sigma.reshape(1, -1)
                views[str(name)] = torch.as_tensor(view.astype(np.float32, copy=False), dtype=torch.float32)
            with torch.no_grad():
                model_out = self.model(views)
            for name, rel in model_out["reliability"].items():
                out[f"reliability_{name}"] = rel.cpu().numpy().astype(np.float64, copy=False)
        return out

    def predict_accuracy(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict_stats(features)["utility"], dtype=np.float64)

    def predict_pos_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict_stats(features)["non_harm"], dtype=np.float64)


def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    beta = torch.ones_like(alpha)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_b_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_b_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(sum_beta)
    digamma_alpha = torch.digamma(alpha)
    digamma_sum = torch.digamma(sum_alpha)
    term = torch.sum((alpha - beta) * (digamma_alpha - digamma_sum), dim=1, keepdim=True)
    return (ln_b_alpha + ln_b_beta + term).reshape(-1)


def _evidential_outcome_loss(
    *,
    alpha: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None,
    lambda_kl: float,
) -> torch.Tensor:
    num_classes = int(alpha.shape[1])
    target_oh = F.one_hot(target, num_classes=num_classes).to(dtype=alpha.dtype)
    s = torch.sum(alpha, dim=1, keepdim=True)
    base = torch.sum(target_oh * (torch.digamma(s) - torch.digamma(alpha)), dim=1)
    if class_weights is not None:
        base = base * class_weights[target]
    evidence = torch.clamp(alpha - 1.0, min=0.0)
    alpha_tilde = evidence * (1.0 - target_oh) + 1.0
    kl = _dirichlet_kl_to_uniform(alpha_tilde)
    return torch.mean(base + float(lambda_kl) * kl)


def _pairwise_rank_loss(
    *,
    alpha: torch.Tensor,
    gains: torch.Tensor,
    group_ids: torch.Tensor,
    eta: float,
    pair_margin: float,
) -> torch.Tensor:
    evidence = torch.clamp(alpha - 1.0, min=0.0)
    s = torch.sum(alpha, dim=1, keepdim=True)
    beliefs = evidence / s
    uncertainty = float(alpha.shape[1]) / torch.clamp(s[:, 0], min=1e-12)
    utility = (beliefs[:, 2] - beliefs[:, 0]) - float(eta) * uncertainty

    losses: list[torch.Tensor] = []
    unique_groups = torch.unique(group_ids)
    for gid in unique_groups:
        idx = torch.where(group_ids == gid)[0]
        if int(idx.numel()) < 2:
            continue
        q_g = utility[idx]
        gain_g = gains[idx]
        delta_g = gain_g[:, None] - gain_g[None, :]
        mask = torch.triu(torch.abs(delta_g) > float(pair_margin), diagonal=1)
        if not bool(torch.any(mask)):
            continue
        q_delta = (q_g[:, None] - q_g[None, :])[mask]
        sign = torch.sign(delta_g[mask])
        losses.append(torch.mean(F.softplus(-sign * q_delta)))

    if not losses:
        return alpha.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def train_evidential_selector(
    X: np.ndarray,
    y_state: np.ndarray,
    gains: np.ndarray,
    group_ids: np.ndarray,
    *,
    feature_names: Sequence[str],
    view_slices: dict[str, tuple[int, int]] | None = None,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 400,
    lambda_rank: float = 0.5,
    lambda_kl: float = 1e-3,
    pair_margin: float = 0.0,
    rho: float = 0.05,
    eta: float = 0.05,
    seed: int = 0,
    progress_label: str | None = None,
    progress_every: int = 0,
) -> EvidentialSelector:
    X = np.asarray(X, dtype=np.float64)
    y_state = np.asarray(y_state, dtype=int).reshape(-1)
    gains = np.asarray(gains, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    n = int(X.shape[0])
    if y_state.shape[0] != n or gains.shape[0] != n or group_ids.shape[0] != n:
        raise ValueError("X/y_state/gains/group_ids length mismatch.")
    if int(hidden_dim) < 1:
        raise ValueError("hidden_dim must be >= 1.")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")

    rng = np.random.RandomState(int(seed))
    torch.manual_seed(int(rng.randint(0, 2**31 - 1)))
    y_t = torch.as_tensor(y_state, dtype=torch.long)
    gain_t = torch.as_tensor(gains, dtype=torch.float32)
    group_t = torch.as_tensor(group_ids, dtype=torch.long)

    counts = np.bincount(y_state, minlength=3).astype(np.float64, copy=False)
    weights = np.ones((3,), dtype=np.float32)
    present = counts > 0
    if bool(np.any(present)):
        weights[present] = float(np.sum(counts[present])) / (float(np.sum(present)) * counts[present])
    class_weights = torch.as_tensor(weights, dtype=torch.float32)

    use_multiview = bool(view_slices)
    scaler: StandardScaler | None = None
    view_stats_out: tuple[tuple[str, np.ndarray, np.ndarray], ...] = ()
    view_slices_out: tuple[tuple[str, int, int], ...] = ()
    if use_multiview:
        view_tensors: dict[str, torch.Tensor] = {}
        view_dims: dict[str, int] = {}
        view_stats_list: list[tuple[str, np.ndarray, np.ndarray]] = []
        slices_sorted = sorted(
            ((str(name), int(bounds[0]), int(bounds[1])) for name, bounds in view_slices.items()),
            key=lambda item: item[1],
        )
        for name, start, stop in slices_sorted:
            view = X[:, int(start) : int(stop)]
            mu = np.mean(view, axis=0, dtype=np.float64)
            sigma = np.std(view, axis=0, dtype=np.float64)
            sigma = np.where(sigma > 1e-8, sigma, 1.0)
            view_scaled = (view - mu.reshape(1, -1)) / sigma.reshape(1, -1)
            view_tensors[str(name)] = torch.as_tensor(view_scaled.astype(np.float32, copy=False), dtype=torch.float32)
            view_dims[str(name)] = int(view_scaled.shape[1])
            view_stats_list.append((str(name), mu.astype(np.float64, copy=False), sigma.astype(np.float64, copy=False)))
        model = _MultiViewEvidentialMLP(view_dims=view_dims, hidden_dim=int(hidden_dim))
        view_stats_out = tuple(view_stats_list)
        view_slices_out = tuple(slices_sorted)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)
        x_t = torch.as_tensor(X_scaled, dtype=torch.float32)
        model = _EvidentialMLP(input_dim=int(X.shape[1]), hidden_dim=int(hidden_dim))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    progress_every = int(progress_every)
    run_label = str(progress_label).strip() if progress_label is not None else ""
    t0 = time.time()
    if run_label:
        print(f"[selector] start {run_label} n={n} dim={int(X.shape[1])} epochs={int(epochs)}", flush=True)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch_idx in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        if use_multiview:
            model_out = model(view_tensors)
            alpha = model_out["alpha"]
        else:
            evidence = model(x_t)
            alpha = evidence + 1.0
        loss_out = _evidential_outcome_loss(
            alpha=alpha,
            target=y_t,
            class_weights=class_weights,
            lambda_kl=float(lambda_kl),
        )
        loss_rank = _pairwise_rank_loss(
            alpha=alpha,
            gains=gain_t,
            group_ids=group_t,
            eta=float(eta),
            pair_margin=float(pair_margin),
        )
        loss = loss_out + float(lambda_rank) * loss_rank
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if run_label and progress_every > 0 and (
            epoch_idx == 0
            or (epoch_idx + 1) % progress_every == 0
            or (epoch_idx + 1) == int(epochs)
        ):
            print(
                f"[selector] {run_label} epoch={epoch_idx + 1}/{int(epochs)} "
                f"loss={loss_value:.6f} best={best_loss:.6f} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if run_label:
        print(f"[selector] done {run_label} best_loss={best_loss:.6f} elapsed={time.time() - t0:.1f}s", flush=True)
    return EvidentialSelector(
        scaler=scaler,
        model=model,
        feature_names=tuple(feature_names),
        rho=float(rho),
        eta=float(eta),
        view_slices=view_slices_out,
        view_stats=view_stats_out,
    )


def train_softmax_bandit_policy(
    X: np.ndarray,
    rewards: np.ndarray,
    group_ids: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2: float = 1.0,
    lr: float = 0.1,
    iters: int = 300,
    seed: int = 0,
) -> SoftmaxBanditPolicy:
    """Train a linear softmax policy to maximize expected reward over candidate sets.

    Parameters
    ----------
    X:
        Candidate features, shape (N, d).
    rewards:
        Reward per candidate (typically Δacc vs identity), shape (N,).
    group_ids:
        Integer group id per row (pseudo-target id), shape (N,).
    l2:
        L2 penalty on θ (>=0).
    lr:
        Learning rate (>0).
    iters:
        Gradient steps (>=1).
    """

    X = np.asarray(X, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != rewards.shape[0] or X.shape[0] != group_ids.shape[0]:
        raise ValueError("X/rewards/group_ids length mismatch.")
    if float(l2) < 0.0:
        raise ValueError("l2 must be >= 0.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if int(iters) < 1:
        raise ValueError("iters must be >= 1.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    uniq = np.unique(group_ids)
    group_indices = [np.where(group_ids == g)[0] for g in uniq]
    if any(idx.size < 1 for idx in group_indices):
        raise ValueError("Each group must have at least 1 candidate.")

    rng = np.random.RandomState(int(seed))
    theta = rng.normal(scale=0.01, size=(Xs.shape[1],)).astype(np.float64)

    n_groups = max(1, len(group_indices))
    for _ in range(int(iters)):
        grad = np.zeros_like(theta)
        for idx in group_indices:
            Xg = Xs[idx]  # (m,d)
            rg = rewards[idx].reshape(-1)  # (m,)
            s = Xg @ theta  # (m,)

            # softmax
            a = s - float(np.max(s))
            Z = float(np.sum(np.exp(a)))
            if not np.isfinite(Z) or Z <= 0.0:
                continue
            p = np.exp(a) / Z  # (m,)

            exp_r = float(np.sum(p * rg))
            centered = (rg - exp_r).reshape(-1, 1)
            grad += np.sum((p.reshape(-1, 1) * centered) * Xg, axis=0)

        if float(l2) > 0.0:
            grad -= float(l2) * theta
        theta = theta + float(lr) * grad / float(n_groups)

    return SoftmaxBanditPolicy(scaler=scaler, theta=theta, feature_names=tuple(feature_names))


def select_by_predicted_improvement(
    records: Iterable[dict],
    *,
    cert: RidgeCertificate,
    n_classes: int,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    feature_set: str = "base",
) -> dict:
    """Select the best candidate record using the calibrated certificate.

    Returns the selected record (dict).
    """

    records = list(records)
    best: dict | None = None
    best_score = -float("inf")
    identity: dict | None = None
    if feature_set in {"stacked_delta", "base_delta"}:
        for rec in records:
            if str(rec.get("kind", "")) == "identity":
                identity = rec
                break
        if identity is None:
            raise ValueError(f"{feature_set} feature_set requires an identity (EA anchor) record.")

    for rec in records:
        if str(rec.get("kind", "")) == "identity" and identity is None:
            identity = rec

        if feature_set == "stacked":
            feats, _names = stacked_candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        elif feature_set == "stacked_delta":
            feats, _names = stacked_candidate_features_delta_from_records(
                rec, anchor=identity, n_classes=n_classes, include_pbar=True
            )
        elif feature_set == "base_delta":
            feats, _names = candidate_features_delta_from_records(
                rec, anchor=identity, n_classes=n_classes, include_pbar=True
            )
        else:
            feats, _names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        pred_improve = float(cert.predict_accuracy(feats)[0])
        # Record for diagnostics / analysis.
        rec["ridge_pred_improve"] = float(pred_improve)
        drift = _safe_float(rec.get("drift_best", 0.0))

        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            pred_improve = float(pred_improve) - float(drift_gamma) * float(drift)

        if pred_improve > best_score:
            best_score = float(pred_improve)
            best = rec

    # Safety: if the best predicted improvement is non-positive, fall back to identity.
    if best is None or best_score <= 0.0:
        if identity is not None:
            return identity
        # If identity missing, fall back to the first record.
        for rec in records:
            return rec
        raise ValueError("No candidates to select from.")

    return best


def select_by_guarded_objective(
    records: Iterable[dict],
    *,
    guard: LogisticGuard,
    n_classes: int,
    threshold: float = 0.5,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
) -> dict:
    """Guarded selection: first reject likely negative-transfer candidates, then pick by objective.

    Selection rule:
    - keep candidates with P(pos_improve) >= threshold (identity is always allowed),
    - apply optional drift hard/penalty,
    - choose the minimum recorded `score` (or `objective` if score missing).
    """

    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0,1].")

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        feats, _names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        p_pos = float(guard.predict_pos_proba(feats)[0])
        # Record for diagnostics / analysis.
        rec["guard_p_pos"] = float(p_pos)
        if p_pos < float(threshold) and str(rec.get("kind", "")) != "identity":
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = _safe_float(rec.get("score", rec.get("objective", 0.0)))
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if best is not None:
        return best
    if identity is not None:
        return identity
    for rec in records:
        return rec
    raise ValueError("No candidates to select from.")


def select_by_guarded_predicted_improvement(
    records: Iterable[dict],
    *,
    cert: RidgeCertificate,
    guard: LogisticGuard,
    cert_by_family: dict[str, RidgeCertificate] | None = None,
    guard_by_family: dict[str, LogisticGuard] | None = None,
    family_key: str = "cand_family",
    family_counts: dict[str, int] | None = None,
    family_blend_mode: str = "hard",
    family_shrinkage: float = 20.0,
    n_classes: int,
    threshold: float = 0.5,
    anchor_guard_delta: float = 0.0,
    anchor_probe_hard_worsen: float = -1.0,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    feature_set: str = "base",
    score_mode: str = "ridge",
) -> dict:
    """Guarded selection: reject likely negative-transfer candidates, then pick by ridge prediction.

    Selection rule:
    - keep candidates with P(pos_improve) >= threshold (identity is always allowed),
    - optionally require P(pos_improve) >= P(pos_improve | identity) + anchor_guard_delta for non-identity,
    - apply optional drift hard/penalty,
    - choose the maximum predicted improvement; if <= 0, fall back to identity.
    """

    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0,1].")
    if float(anchor_guard_delta) < 0.0:
        raise ValueError("anchor_guard_delta must be >= 0.")
    if float(anchor_probe_hard_worsen) < -1.0:
        raise ValueError("anchor_probe_hard_worsen must be -1 (disable) or > -1.")
    if str(family_blend_mode) not in {"hard", "blend"}:
        raise ValueError("family_blend_mode must be one of: 'hard', 'blend'.")
    if float(family_shrinkage) < 0.0:
        raise ValueError("family_shrinkage must be >= 0.")
    if str(score_mode) not in {"ridge", "borda_ridge_probe", "borda_ridge_probe_iwcv"}:
        raise ValueError("score_mode must be one of: 'ridge', 'borda_ridge_probe', 'borda_ridge_probe_iwcv'.")

    records = list(records)
    identity: dict | None = None
    if feature_set in {"stacked_delta", "base_delta"}:
        for rec in records:
            if str(rec.get("kind", "")) == "identity":
                identity = rec
                break
        if identity is None:
            raise ValueError(f"{feature_set} feature_set requires an identity (EA anchor) record.")

    # First pass: compute guard / ridge predictions for all candidates and cache them on records.
    computed: list[dict] = []
    anchor_p_pos: float | None = None
    anchor_probe_hard: float | None = None
    anchor_iwcv_ucb: float | None = None

    for rec in records:
        if str(rec.get("kind", "")) == "identity" and identity is None:
            identity = rec

        if feature_set == "stacked":
            feats, _names = stacked_candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        elif feature_set == "stacked_delta":
            feats, _names = stacked_candidate_features_delta_from_records(
                rec, anchor=identity, n_classes=n_classes, include_pbar=True
            )
        elif feature_set == "base_delta":
            feats, _names = candidate_features_delta_from_records(
                rec, anchor=identity, n_classes=n_classes, include_pbar=True
            )
        else:
            feats, _names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)

        fam = str(rec.get(family_key, "")).strip().lower()
        cert_f = cert_by_family.get(fam) if (cert_by_family is not None) else None
        guard_f = guard_by_family.get(fam) if (guard_by_family is not None) else None

        p_pos_g = float(guard.predict_pos_proba(feats)[0])
        pred_g = float(cert.predict_accuracy(feats)[0])
        p_pos_f = float(guard_f.predict_pos_proba(feats)[0]) if guard_f is not None else float("nan")
        pred_f = float(cert_f.predict_accuracy(feats)[0]) if cert_f is not None else float("nan")

        # Default: use the global model.
        p_pos = float(p_pos_g)
        pred_improve = float(pred_g)
        w = float("nan")

        # Per-family usage (optional).
        if (cert_f is not None) or (guard_f is not None):
            if str(family_blend_mode) == "hard":
                if np.isfinite(p_pos_f):
                    p_pos = float(p_pos_f)
                if np.isfinite(pred_f):
                    pred_improve = float(pred_f)
            else:
                n = int(family_counts.get(fam, 0)) if (family_counts is not None) else 0
                if float(family_shrinkage) <= 0.0:
                    w = 1.0
                else:
                    w = float(n) / float(n + float(family_shrinkage))

                if np.isfinite(p_pos_f):
                    p_pos = float((1.0 - w) * float(p_pos_g) + w * float(p_pos_f))
                if np.isfinite(pred_f):
                    pred_improve = float((1.0 - w) * float(pred_g) + w * float(pred_f))

        # Record for diagnostics / analysis.
        rec["guard_p_pos_global"] = float(p_pos_g)
        rec["ridge_pred_improve_global"] = float(pred_g)
        rec["guard_p_pos_family"] = float(p_pos_f) if np.isfinite(p_pos_f) else float("nan")
        rec["ridge_pred_improve_family"] = float(pred_f) if np.isfinite(pred_f) else float("nan")
        rec["family_blend_w"] = float(w) if np.isfinite(w) else float("nan")
        rec["guard_p_pos"] = float(p_pos)
        rec["ridge_pred_improve"] = float(pred_improve)

        computed.append(rec)
        if str(rec.get("kind", "")) == "identity":
            anchor_p_pos = float(p_pos)
            try:
                anchor_probe_hard = float(rec.get("probe_mixup_hard_best", float("nan")))
            except Exception:
                anchor_probe_hard = float("nan")
            try:
                anchor_iwcv_ucb = float(rec.get("iwcv_ucb", float("nan")))
            except Exception:
                anchor_iwcv_ucb = float("nan")

    best: dict | None = None
    best_score = -float("inf")
    thr_anchor = float("nan")
    if float(anchor_guard_delta) > 0.0 and anchor_p_pos is not None and np.isfinite(anchor_p_pos):
        thr_anchor = float(anchor_p_pos) + float(anchor_guard_delta)
    thr_probe = float("nan")
    if float(anchor_probe_hard_worsen) > -1.0 and anchor_probe_hard is not None and np.isfinite(anchor_probe_hard):
        thr_probe = float(anchor_probe_hard) + float(anchor_probe_hard_worsen)

    def _rank_desc_min(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return np.asarray([], dtype=np.float64)
        v = np.where(np.isfinite(v), v, -np.inf)
        order = np.argsort(-v, kind="mergesort")
        ranks_sorted = (np.arange(v.size, dtype=np.float64) + 1.0).copy()
        # Tie handling: assign min rank for equal values.
        sv = v[order]
        start = 0
        while start < sv.size:
            end = start + 1
            while end < sv.size and sv[end] == sv[start]:
                end += 1
            ranks_sorted[start:end] = ranks_sorted[start]
            start = end
        ranks = np.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted
        return ranks

    eligible_recs: list[dict] = []
    eligible_pred: list[float] = []
    eligible_probe_imp: list[float] = []
    eligible_iwcv_imp: list[float] = []

    for rec in computed:
        p_pos = float(rec.get("guard_p_pos", float("nan")))
        pred_improve = float(rec.get("ridge_pred_improve", float("nan")))
        is_identity = str(rec.get("kind", "")) == "identity"

        # Never rank identity (EA) as an eligible candidate; keep it only as the anchor/fallback.
        if is_identity:
            continue

        if not np.isfinite(p_pos) or float(p_pos) < float(threshold):
            continue
        if np.isfinite(thr_anchor) and float(p_pos) < float(thr_anchor):
            continue
        if np.isfinite(thr_probe):
            probe = float(rec.get("probe_mixup_hard_best", float("nan")))
            if not np.isfinite(probe) or float(probe) > float(thr_probe):
                continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            pred_improve = float(pred_improve) - float(drift_gamma) * float(drift)

        eligible_recs.append(rec)
        eligible_pred.append(float(pred_improve))
        if anchor_probe_hard is not None and np.isfinite(anchor_probe_hard):
            probe_imp = float(anchor_probe_hard) - float(rec.get("probe_mixup_hard_best", float("nan")))
        else:
            probe_imp = float("nan")
        eligible_probe_imp.append(float(probe_imp))
        if anchor_iwcv_ucb is not None and np.isfinite(anchor_iwcv_ucb):
            try:
                cand_iwcv = float(rec.get("iwcv_ucb", float("nan")))
            except Exception:
                cand_iwcv = float("nan")
            iwcv_imp = float(anchor_iwcv_ucb) - float(cand_iwcv)
        else:
            iwcv_imp = float("nan")
        eligible_iwcv_imp.append(float(iwcv_imp))

    # No eligible candidate => safe fallback to identity.
    if not eligible_recs:
        if identity is not None:
            return identity
        for rec in records:
            return rec
        raise ValueError("No candidates to select from.")

    # Default score: ridge predicted improvement (legacy behavior).
    if str(score_mode) == "ridge" or not np.isfinite(_safe_float_or(anchor_probe_hard, float("nan"))):
        best = None
        best_score = -float("inf")
        for rec, pred_improve in zip(eligible_recs, eligible_pred):
            if float(pred_improve) > float(best_score):
                best_score = float(pred_improve)
                best = rec

        # Safety: if the best predicted improvement is non-positive, fall back to identity.
        if best is None or float(best_score) <= 0.0:
            if identity is not None:
                return identity
            for rec in records:
                return rec
            raise ValueError("No candidates to select from.")

        return best

    # Borda aggregation: combine ranks across multiple improvement signals.
    pred_arr = np.asarray(eligible_pred, dtype=np.float64)
    probe_arr = np.asarray(eligible_probe_imp, dtype=np.float64)
    ranks_pred = _rank_desc_min(pred_arr)
    ranks_probe = _rank_desc_min(probe_arr)
    score = ranks_pred + ranks_probe  # smaller is better
    use_iwcv = str(score_mode) == "borda_ridge_probe_iwcv" and np.isfinite(_safe_float_or(anchor_iwcv_ucb, float("nan")))
    iwcv_arr = np.asarray(eligible_iwcv_imp, dtype=np.float64)
    if use_iwcv:
        ranks_iwcv = _rank_desc_min(iwcv_arr)
        score = score + ranks_iwcv

    best_idx = int(np.argmin(score))
    best = eligible_recs[best_idx]
    best_pred = float(pred_arr[best_idx])
    best_probe = float(probe_arr[best_idx])
    best_iwcv = float(iwcv_arr[best_idx]) if use_iwcv else -float("inf")

    # Safety fallback: if both signals indicate non-positive improvement, return identity.
    if float(max(best_pred, best_probe, best_iwcv)) <= 0.0:
        if identity is not None:
            return identity
        for rec in records:
            return rec
        raise ValueError("No candidates to select from.")

    return best


def select_by_guarded_bandit_policy(
    records: Iterable[dict],
    *,
    policy: SoftmaxBanditPolicy,
    guard: LogisticGuard,
    n_classes: int,
    threshold: float = 0.5,
    anchor_guard_delta: float = 0.0,
    anchor_probe_hard_worsen: float = -1.0,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    feature_set: str = "base",
) -> dict:
    """Guarded selection: filter by guard, then pick the highest policy score.

    Notes
    -----
    - Identity is always allowed (safe fallback).
    - Drift guard/penalty can be applied similarly to other selectors.
    """

    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0,1].")
    if float(anchor_guard_delta) < 0.0:
        raise ValueError("anchor_guard_delta must be >= 0.")
    if float(anchor_probe_hard_worsen) < -1.0:
        raise ValueError("anchor_probe_hard_worsen must be -1 (disable) or > -1.")

    records = list(records)
    identity: dict | None = None
    if feature_set in {"stacked_delta", "base_delta"}:
        for rec in records:
            if str(rec.get("kind", "")) == "identity":
                identity = rec
                break
        if identity is None:
            raise ValueError(f"{feature_set} feature_set requires an identity (EA anchor) record.")

    # First pass: compute guard scores for all records so we can apply anchor-relative filtering robustly.
    computed: list[dict] = []
    anchor_p_pos: float | None = None
    anchor_probe_hard: float | None = None

    for rec in records:
        if str(rec.get("kind", "")) == "identity" and identity is None:
            identity = rec

        if feature_set == "stacked":
            feats, _names = stacked_candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        elif feature_set == "stacked_delta":
            feats, _names = stacked_candidate_features_delta_from_records(
                rec, anchor=identity, n_classes=n_classes, include_pbar=True
            )
        elif feature_set == "base_delta":
            feats, _names = candidate_features_delta_from_records(
                rec, anchor=identity, n_classes=n_classes, include_pbar=True
            )
        else:
            feats, _names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)

        p_pos = float(guard.predict_pos_proba(feats)[0])
        score = float(policy.score(feats)[0])

        # Record for diagnostics / analysis.
        rec["guard_p_pos"] = float(p_pos)
        rec["bandit_score"] = float(score)
        computed.append(rec)
        if str(rec.get("kind", "")) == "identity":
            anchor_p_pos = float(p_pos)
            try:
                anchor_probe_hard = float(rec.get("probe_mixup_hard_best", float("nan")))
            except Exception:
                anchor_probe_hard = float("nan")

    thr_anchor = float("nan")
    if float(anchor_guard_delta) > 0.0 and anchor_p_pos is not None and np.isfinite(anchor_p_pos):
        thr_anchor = float(anchor_p_pos) + float(anchor_guard_delta)
    thr_probe = float("nan")
    if float(anchor_probe_hard_worsen) > -1.0 and anchor_probe_hard is not None and np.isfinite(anchor_probe_hard):
        thr_probe = float(anchor_probe_hard) + float(anchor_probe_hard_worsen)

    best: dict | None = None
    best_score = -float("inf")

    for rec in computed:
        p_pos = float(rec.get("guard_p_pos", float("nan")))
        score = float(rec.get("bandit_score", float("nan")))
        is_identity = str(rec.get("kind", "")) == "identity"

        if not is_identity:
            if not np.isfinite(p_pos) or float(p_pos) < float(threshold):
                continue
            if np.isfinite(thr_anchor) and float(p_pos) < float(thr_anchor):
                continue
            if np.isfinite(thr_probe):
                probe = float(rec.get("probe_mixup_hard_best", float("nan")))
                if not np.isfinite(probe) or float(probe) > float(thr_probe):
                    continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) - float(drift_gamma) * float(drift)

        if float(score) > float(best_score):
            best_score = float(score)
            best = rec

    if best is not None:
        return best
    if identity is not None:
        return identity
    for rec in records:
        return rec
    raise ValueError("No candidates to select from.")


def select_by_evidence_nll(
    records: Iterable[dict],
    *,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
) -> dict:
    """Select the best candidate using LDA evidence (-log p(z)).

    Candidates must have `evidence_nll_best` recorded (smaller is better).
    If no candidate improves over the identity anchor, return identity.
    """

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        try:
            ev = float(rec.get("evidence_nll_best", float("nan")))
        except Exception:
            ev = float("nan")
        if not np.isfinite(ev):
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = float(ev)
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        ev_id = float(identity.get("evidence_nll_best", float("nan")))
    except Exception:
        ev_id = float("nan")
    if not np.isfinite(ev_id):
        return best if best is not None else identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(ev_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(ev_id):
        return identity

    return best


def select_by_probe_mixup(
    records: Iterable[dict],
    *,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
) -> dict:
    """Select the best candidate using a MixUp-style probe score.

    Candidates must have `probe_mixup_best` recorded (smaller is better).
    If no candidate improves over the identity anchor, return identity.
    """

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        try:
            s = float(rec.get("probe_mixup_best", float("nan")))
        except Exception:
            s = float("nan")
        if not np.isfinite(s):
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = float(s)
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("probe_mixup_best", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        return best if best is not None else identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best


def select_by_probe_mixup_hard(
    records: Iterable[dict],
    *,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
) -> dict:
    """Select the best candidate using a hard-major MixUp probe score.

    This corresponds to a MixVal-style heuristic: when λ>0.5, assign the (hard)
    pseudo label of the dominant sample (implemented in the probe score).

    Candidates must have `probe_mixup_hard_best` recorded (smaller is better).
    If no candidate improves over the identity anchor, return identity.
    """

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        try:
            s = float(rec.get("probe_mixup_hard_best", float("nan")))
        except Exception:
            s = float("nan")
        if not np.isfinite(s):
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = float(s)
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("probe_mixup_hard_best", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        return best if best is not None else identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best
