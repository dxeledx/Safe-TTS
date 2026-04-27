from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import SubjectData
from .alignment import (
    EuclideanAligner,
    LogEuclideanAligner,
    apply_spatial_transform,
    blend_with_identity,
    class_cov_diff,
    orthogonal_align_tsa_procrustes,
    orthogonal_align_symmetric,
    sorted_eigh,
)
from .model import TrainedModel, fit_csp_lda, fit_fbcsp_lda
from .model import fit_csp_projected_lda
from .subject_invariant import (
    HSICProjectorParams,
    CenteredLinearProjector,
    learn_hsic_subject_invariant_projector,
    ChannelProjectorParams,
    compute_channel_projector_scatter,
    solve_channel_projector_from_scatter,
    learn_subject_invariant_channel_projector,
)
from .certificate import (
    candidate_features_from_record,
    candidate_features_delta_from_records,
    stacked_candidate_features_from_record,
    stacked_candidate_features_delta_from_records,
    select_by_dev_nll,
    select_by_evidence_nll,
    select_by_guarded_bandit_policy,
    select_by_guarded_predicted_improvement,
    select_by_guarded_objective,
    select_by_iwcv_nll,
    select_by_iwcv_ucb,
    select_by_predicted_improvement,
    select_by_probe_mixup,
    select_by_probe_mixup_hard,
    train_logistic_guard,
    train_ridge_certificate,
    train_softmax_bandit_policy,
)
from .metrics import compute_metrics, summarize_results
from .proba import reorder_proba_columns as _reorder_proba_columns
from .probes import evidence_nll as _evidence_nll
from .probes import probe_mixup as _probe_mixup
from .probes import select_keep_indices as _select_keep_indices
from .zo import (
    _optimize_qt_oea_zo,
    _select_pseudo_indices,
    _soft_class_cov_diff,
    _write_zo_diagnostics,
)
from .riemann import covariances_from_epochs


def _maybe_malloc_trim() -> None:
    """Best-effort RSS reduction (glibc): return freed heap pages to the OS.

    This is important for long-running LOSO loops on large datasets where repeated
    allocation/free patterns can otherwise keep RSS high and trigger OOM kills.
    """

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        return


def _compute_tsa_target_rotation(
    *,
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_target: np.ndarray,
    model: TrainedModel,
    class_order: Sequence[str],
    pseudo_mode: str,
    pseudo_iters: int,
    q_blend: float,
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
    eps: float,
    shrinkage: float,
) -> np.ndarray:
    """Compute a TSA-style closed-form target rotation using (pseudo-)class anchors."""

    class_order = [str(c) for c in class_order]
    if pseudo_mode not in {"hard", "soft"}:
        raise ValueError("pseudo_mode must be one of: 'hard', 'soft'")
    if int(pseudo_iters) <= 0:
        return np.eye(int(z_target.shape[1]), dtype=np.float64)

    q_t = np.eye(int(z_target.shape[1]), dtype=np.float64)
    for _ in range(int(pseudo_iters)):
        X_cur = apply_spatial_transform(q_t, z_target)
        proba = model.predict_proba(X_cur)
        proba = _reorder_proba_columns(proba, model.classes_, list(class_order))

        try:
            if pseudo_mode == "soft":
                q_new = orthogonal_align_tsa_procrustes(
                    z_train,
                    y_train,
                    z_target,
                    pseudo_mode="soft",
                    proba_target=proba,
                    y_pseudo_target=None,
                    class_order=class_order,
                    eps=float(eps),
                    shrinkage=float(shrinkage),
                )
            else:
                y_pseudo = np.asarray(model.predict(X_cur))
                keep = _select_pseudo_indices(
                    y_pseudo=y_pseudo,
                    proba=proba,
                    class_order=class_order,
                    confidence=float(pseudo_confidence),
                    topk_per_class=int(pseudo_topk_per_class),
                    balance=bool(pseudo_balance),
                )
                if keep.size == 0:
                    break
                q_new = orthogonal_align_tsa_procrustes(
                    z_train,
                    y_train,
                    z_target[keep],
                    pseudo_mode="hard",
                    proba_target=None,
                    y_pseudo_target=y_pseudo[keep],
                    class_order=class_order,
                    eps=float(eps),
                    shrinkage=float(shrinkage),
                )
        except ValueError:
            break

        q_t = blend_with_identity(q_new, float(q_blend))

    return q_t

def _features_before_lda(*, model: TrainedModel, X: np.ndarray) -> np.ndarray:
    """Compute features fed into the final LDA step (supports CSP/FBCSP/projected pipelines)."""

    X = np.asarray(X, dtype=np.float64)
    pipe = model.pipeline
    if "lda" not in getattr(pipe, "named_steps", {}):
        raise ValueError("Expected model.pipeline to contain a final 'lda' step.")
    feats = pipe[:-1].transform(X)
    return np.asarray(feats, dtype=np.float64)


def _fit_domain_logreg_ratio_feats(
    *,
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
    c: float = 1.0,
    clip_max: float = 20.0,
) -> np.ndarray:
    """Estimate density ratio w(x)=p_T(x)/p_S(x) using a domain classifier in feature space."""

    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError("Expected 2D feature arrays for domain ratio.")
    if int(X_source.shape[1]) != int(X_target.shape[1]):
        raise ValueError("Source/target feature dim mismatch.")

    n_s = int(X_source.shape[0])
    n_t = int(X_target.shape[0])
    if n_s < 2 or n_t < 2:
        return np.ones(n_s, dtype=np.float64)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(int(seed))
    n = int(min(n_s, n_t))
    idx_s = rng.choice(n_s, size=n, replace=False)
    idx_t = rng.choice(n_t, size=n, replace=False)
    X = np.concatenate([X_source[idx_s], X_target[idx_t]], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)], axis=0)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=float(c), max_iter=1000, solver="lbfgs")),
        ]
    )
    clf.fit(X, y)

    p_t = np.asarray(clf.predict_proba(X_source), dtype=np.float64)[:, 1]
    p_t = np.clip(p_t, 1e-6, 1.0 - 1e-6)
    w = p_t / (1.0 - p_t)
    w = np.clip(w, 0.0, float(clip_max))
    return np.asarray(w, dtype=np.float64)


def _iwcv_ucb_stats_for_model(
    *,
    model: TrainedModel,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    class_order: Sequence[str],
    seed: int,
    kappa: float = 1.0,
    domain_c: float = 1.0,
    clip_max: float = 20.0,
) -> dict:
    """Compute IWCV-UCB-style certificate stats in the model's feature space.

    Returns dict fields:
    - iwcv_nll, iwcv_eff_n, iwcv_var, iwcv_se, iwcv_ucb
    """

    class_order = [str(c) for c in class_order]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_source = np.asarray(y_source)
    try:
        y_idx = np.fromiter((class_to_idx[str(c)] for c in y_source), dtype=int, count=len(y_source))
    except KeyError as e:
        raise ValueError(f"y_source contains unknown class '{e.args[0]}'.") from e

    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    if X_source.ndim != 3 or X_target.ndim != 3:
        raise ValueError("Expected epochs with shape (n,C,T).")
    if int(X_source.shape[0]) != int(y_source.shape[0]):
        raise ValueError("X_source/y_source length mismatch.")
    if int(X_source.shape[1]) != int(X_target.shape[1]):
        raise ValueError("Source/target channel mismatch.")

    fs = _features_before_lda(model=model, X=X_source)
    ft = _features_before_lda(model=model, X=X_target)
    w = _fit_domain_logreg_ratio_feats(
        X_source=fs,
        X_target=ft,
        seed=int(seed),
        c=float(domain_c),
        clip_max=float(clip_max),
    )
    w_sum = float(np.sum(w))
    w_sq_sum = float(np.sum(w * w))
    eff_n = (w_sum * w_sum / w_sq_sum) if w_sq_sum > 1e-12 else 0.0

    proba_s = np.asarray(model.predict_proba(X_source), dtype=np.float64)
    proba_s = _reorder_proba_columns(proba_s, model.classes_, class_order)
    p = np.clip(proba_s, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    nll = -np.log(p[np.arange(p.shape[0]), y_idx])

    if w_sum <= 1e-12:
        mean = float("nan")
        var = float("nan")
        se = float("nan")
        ucb = float("inf")
    else:
        mean = float(np.sum(w * nll) / max(1e-12, w_sum))
        var = float(np.sum(w * (nll - mean) * (nll - mean)) / max(1e-12, w_sum))
        se = float(np.sqrt(max(0.0, var) / max(1e-12, eff_n)))
        ucb = float(mean) + float(kappa) * float(se)

    return {
        "iwcv_nll": float(mean) if np.isfinite(mean) else float("nan"),
        "iwcv_eff_n": float(eff_n),
        "iwcv_var": float(var) if np.isfinite(var) else float("nan"),
        "iwcv_se": float(se) if np.isfinite(se) else float("nan"),
        "iwcv_ucb": float(ucb),
    }


def _compute_lda_evidence_params(
    *,
    model: TrainedModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_order: Sequence[str],
    ridge: float = 1e-6,
) -> dict:
    """Build Gaussian-mixture evidence params from *LDA feature space* training data.

    Returns a dict with:
    - mu: (K,d) class means in feature space
    - priors: (K,) class priors
    - cov: (d,d) pooled within-class covariance (ridge-stabilized)
    - cov_inv: (d,d) inverse covariance
    - logdet: log|cov|
    """

    class_order = [str(c) for c in class_order]
    feats = _features_before_lda(model=model, X=X_train)
    y_train = np.asarray(y_train)
    n = int(feats.shape[0])
    if n != int(y_train.shape[0]):
        raise ValueError("X_train/y_train length mismatch for evidence params.")
    k = int(len(class_order))
    if k < 2:
        raise ValueError("Need at least 2 classes for evidence params.")

    mu = np.zeros((k, feats.shape[1]), dtype=np.float64)
    priors = np.zeros(k, dtype=np.float64)
    present = 0
    for i, c in enumerate(class_order):
        mask = y_train == c
        if not np.any(mask):
            continue
        present += 1
        priors[i] = float(np.sum(mask)) / float(n)
        mu[i] = np.mean(feats[mask], axis=0)
    if present < 2:
        raise ValueError("At least two classes must be present to compute evidence params.")

    priors = np.clip(priors, 1e-12, 1.0)
    priors = priors / float(np.sum(priors))

    # Pooled within-class covariance.
    d = int(feats.shape[1])
    scatter = np.zeros((d, d), dtype=np.float64)
    for i, c in enumerate(class_order):
        mask = y_train == c
        if not np.any(mask):
            continue
        fc = feats[mask] - mu[i]
        scatter += fc.T @ fc

    denom = max(1, int(n - present))
    cov = scatter / float(denom)
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov)) / float(d) if float(np.trace(cov)) > 0.0 else 1.0
    cov = cov + float(ridge) * float(scale) * np.eye(d, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0 or not np.isfinite(logdet):
        cov = cov + 1e-3 * np.eye(d, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(cov)
    cov_inv = np.linalg.pinv(cov) if (sign <= 0.0 or not np.isfinite(logdet)) else np.linalg.inv(cov)

    return {
        "mu": mu,
        "priors": priors,
        "cov": cov,
        "cov_inv": cov_inv,
        "logdet": float(logdet) if np.isfinite(logdet) else float("nan"),
    }


def _compute_gaussian_evidence_params_from_feats(
    *,
    feats: np.ndarray,
    y_train: np.ndarray,
    class_order: Sequence[str],
    ridge: float = 1e-6,
) -> dict:
    """Gaussian-mixture evidence params from arbitrary feature space (2D array).

    Returns the same dict format as `_compute_lda_evidence_params`.
    """

    class_order = [str(c) for c in class_order]
    feats = np.asarray(feats, dtype=np.float64)
    y_train = np.asarray(y_train)
    if feats.ndim != 2:
        raise ValueError("feats must be 2D.")
    n = int(feats.shape[0])
    if n != int(y_train.shape[0]):
        raise ValueError("feats/y_train length mismatch for evidence params.")
    k = int(len(class_order))
    if k < 2:
        raise ValueError("Need at least 2 classes for evidence params.")

    mu = np.zeros((k, feats.shape[1]), dtype=np.float64)
    priors = np.zeros(k, dtype=np.float64)
    present = 0
    for i, c in enumerate(class_order):
        mask = y_train == c
        if not np.any(mask):
            continue
        present += 1
        priors[i] = float(np.sum(mask)) / float(n)
        mu[i] = np.mean(feats[mask], axis=0)
    if present < 2:
        raise ValueError("At least two classes must be present to compute evidence params.")

    priors = np.clip(priors, 1e-12, 1.0)
    priors = priors / float(np.sum(priors))

    # Pooled within-class covariance.
    d = int(feats.shape[1])
    scatter = np.zeros((d, d), dtype=np.float64)
    for i, c in enumerate(class_order):
        mask = y_train == c
        if not np.any(mask):
            continue
        fc = feats[mask] - mu[i]
        scatter += fc.T @ fc

    denom = max(1, int(n - present))
    cov = scatter / float(denom)
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov)) / float(d) if float(np.trace(cov)) > 0.0 else 1.0
    cov = cov + float(ridge) * float(scale) * np.eye(d, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0 or not np.isfinite(logdet):
        cov = cov + 1e-3 * np.eye(d, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(cov)
    cov_inv = np.linalg.pinv(cov) if (sign <= 0.0 or not np.isfinite(logdet)) else np.linalg.inv(cov)

    return {
        "mu": mu,
        "priors": priors,
        "cov": cov,
        "cov_inv": cov_inv,
        "logdet": float(logdet) if np.isfinite(logdet) else float("nan"),
    }


@dataclass(frozen=True)
class FoldResult:
    subject: int
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    kappa: float


def loso_cross_subject_evaluation(
    subject_data: Dict[int, SubjectData],
    *,
    class_order: Sequence[str],
    test_subjects: Sequence[int] | None = None,
    channel_names: Sequence[str] | None = None,
    n_components: int = 4,
    average: str = "macro",
    alignment: str = "none",
    inplace_pre_align: bool = False,
    fbcsp_multiclass_strategy: str = "multiclass",
    sfreq: float = 250.0,
    oea_eps: float = 1e-10,
    oea_shrinkage: float = 0.0,
    oea_pseudo_iters: int = 2,
    oea_q_blend: float = 1.0,
    oea_pseudo_mode: str = "hard",
    oea_pseudo_confidence: float = 0.0,
    oea_pseudo_topk_per_class: int = 0,
    oea_pseudo_balance: bool = False,
    oea_zo_objective: str = "entropy",
    oea_zo_transform: str = "orthogonal",
    oea_zo_localmix_neighbors: int = 4,
    oea_zo_localmix_self_bias: float = 3.0,
    oea_zo_infomax_lambda: float = 1.0,
    oea_zo_reliable_metric: str = "none",
    oea_zo_reliable_threshold: float = 0.0,
    oea_zo_reliable_alpha: float = 10.0,
    oea_zo_trust_lambda: float = 0.0,
    oea_zo_trust_q0: str = "identity",
    oea_zo_marginal_mode: str = "none",
    oea_zo_marginal_beta: float = 0.0,
    oea_zo_marginal_tau: float = 0.05,
    oea_zo_marginal_prior: str = "uniform",
    oea_zo_marginal_prior_mix: float = 0.0,
    oea_zo_bilevel_iters: int = 5,
    oea_zo_bilevel_temp: float = 1.0,
    oea_zo_bilevel_step: float = 1.0,
    oea_zo_bilevel_coverage_target: float = 0.5,
    oea_zo_bilevel_coverage_power: float = 1.0,
    oea_zo_drift_mode: str = "none",
    oea_zo_drift_gamma: float = 0.0,
    oea_zo_drift_delta: float = 0.0,
    oea_zo_selector: str = "objective",
    oea_zo_iwcv_kappa: float = 1.0,
    oea_zo_calib_ridge_alpha: float = 1.0,
    oea_zo_calib_max_subjects: int = 0,
    oea_zo_calib_seed: int = 0,
    oea_zo_calib_guard_c: float = 1.0,
    oea_zo_calib_guard_threshold: float = 0.5,
    oea_zo_calib_guard_margin: float = 0.0,
    oea_zo_min_improvement: float = 0.0,
    oea_zo_holdout_fraction: float = 0.0,
    oea_zo_warm_start: str = "none",
    oea_zo_warm_iters: int = 1,
    oea_zo_fallback_min_marginal_entropy: float = 0.0,
    oea_zo_iters: int = 30,
    oea_zo_lr: float = 0.5,
    oea_zo_mu: float = 0.1,
    oea_zo_k: int = 50,
    oea_zo_seed: int = 0,
    oea_zo_l2: float = 0.0,
    mm_safe_mdm_guard_threshold: float = -1.0,
    mm_safe_mdm_min_pred_improve: float = 0.0,
    mm_safe_mdm_drift_delta: float = 0.0,
    stack_safe_fbcsp_guard_threshold: float = -1.0,
    stack_safe_fbcsp_min_pred_improve: float = 0.0,
    stack_safe_fbcsp_drift_delta: float = 0.0,
    stack_safe_fbcsp_max_pred_disagree: float = -1.0,
    stack_safe_tsa_guard_threshold: float = -1.0,
    stack_safe_tsa_min_pred_improve: float = 0.0,
    stack_safe_tsa_drift_delta: float = 0.0,
    stack_safe_anchor_guard_delta: float = 0.0,
    stack_safe_anchor_probe_hard_worsen: float = -1.0,
    stack_safe_min_pred_improve: float = 0.0,
    stack_calib_per_family: bool = False,
    stack_calib_per_family_mode: str = "hard",
    stack_calib_per_family_shrinkage: float = 20.0,
    stack_feature_set: str = "stacked",
    stack_candidate_families: Sequence[str] = ("ea", "fbcsp", "rpa", "tsa", "chan"),
    si_subject_lambda: float = 1.0,
    si_ridge: float = 1e-6,
    si_proj_dim: int = 0,
    si_chan_candidate_ranks: Sequence[int] = (),
    si_chan_candidate_lambdas: Sequence[float] = (),
    diagnostics_dir: Path | None = None,
    diagnostics_subjects: Sequence[int] = (),
    diagnostics_tag: str = "",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[int, TrainedModel],
]:
    """LOSO evaluation: each subject is test once; others are training.

    Returns
    -------
    results_df:
        Per-subject metrics.
    y_true_all, y_pred_all, y_proba_all:
        Aggregated predictions across all folds (aligned to `class_order` columns).
    class_order:
        Class names order used.
    models_by_subject:
        Trained model for each test subject (useful for inspection/plotting).
    """

    subject_data_raw = subject_data
    subject_data_rpa: Dict[int, SubjectData] | None = None

    subjects_all = sorted(subject_data.keys())
    subjects_test = subjects_all
    if test_subjects is not None and len(test_subjects) > 0:
        test_set = {int(s) for s in test_subjects}
        subjects_test = [int(s) for s in subjects_all if int(s) in test_set]
        missing = sorted(test_set - set(subjects_all))
        if missing:
            raise ValueError(f"test_subjects contains unknown subject ids: {missing}")
    fold_rows: List[FoldResult] = []
    models_by_subject: Dict[int, TrainedModel] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    if alignment not in {
        "none",
        "ea",
        "rpa",
        "fbcsp",
        "ea_fbcsp",
        "ea_si",
        "ea_si_chan",
        "ea_si_chan_safe",
        "ea_si_chan_multi_safe",
        "ea_si_chan_spsa_safe",
        "ea_mm_safe",
        "ea_stack_multi_safe",
        "riemann_mdm",
        "rpa_mdm",
        "rpa_rot_mdm",
        "ts_lr",
        "ts_svc",
        "tsa_ts_svc",
        "fgmdm",
        "rpa_ts_lr",
        "ea_ts_lr",
        "ea_si_zo",
        "ea_zo",
        "raw_zo",
        "rpa_zo",
        "tsa",
        "tsa_zo",
        "oea_cov",
        "oea",
        "oea_zo",
    }:
        raise ValueError(
            "alignment must be one of: "
            "'none', 'ea', 'rpa', 'fbcsp', 'ea_fbcsp', 'ea_si', 'ea_si_chan', 'ea_si_chan_safe', 'ea_si_chan_multi_safe', 'ea_si_chan_spsa_safe', 'ea_mm_safe', 'ea_stack_multi_safe', "
            "'riemann_mdm', 'rpa_mdm', 'rpa_rot_mdm', 'ts_lr', 'ts_svc', 'tsa_ts_svc', 'fgmdm', 'rpa_ts_lr', 'ea_ts_lr', "
            "'ea_si_zo', 'ea_zo', 'raw_zo', 'rpa_zo', 'tsa', 'tsa_zo', 'oea_cov', 'oea', 'oea_zo'"
        )

    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")
    if str(fbcsp_multiclass_strategy) not in {"auto", "multiclass", "ovo", "ovr"}:
        raise ValueError("fbcsp_multiclass_strategy must be one of: 'auto', 'multiclass', 'ovo', 'ovr'.")
    if not (0.0 <= float(oea_pseudo_confidence) <= 1.0):
        raise ValueError("oea_pseudo_confidence must be in [0,1].")
    if int(oea_pseudo_topk_per_class) < 0:
        raise ValueError("oea_pseudo_topk_per_class must be >= 0.")

    if oea_zo_objective not in {
        "entropy",
        "pseudo_ce",
        "confidence",
        "infomax",
        "lda_nll",
        "entropy_bilevel",
        "infomax_bilevel",
    }:
        raise ValueError(
            "oea_zo_objective must be one of: "
            "'entropy', 'pseudo_ce', 'confidence', 'infomax', 'lda_nll', 'entropy_bilevel', 'infomax_bilevel'"
        )
    if float(oea_zo_infomax_lambda) <= 0.0:
        raise ValueError("oea_zo_infomax_lambda must be > 0.")
    if oea_zo_reliable_metric not in {"none", "confidence", "entropy"}:
        raise ValueError("oea_zo_reliable_metric must be one of: 'none', 'confidence', 'entropy'")
    if float(oea_zo_reliable_alpha) <= 0.0:
        raise ValueError("oea_zo_reliable_alpha must be > 0.")
    if oea_zo_reliable_metric == "confidence" and not (
        0.0 <= float(oea_zo_reliable_threshold) <= 1.0
    ):
        raise ValueError("oea_zo_reliable_threshold must be in [0,1] when metric='confidence'.")
    if oea_zo_reliable_metric == "entropy" and float(oea_zo_reliable_threshold) < 0.0:
        raise ValueError("oea_zo_reliable_threshold must be >= 0 when metric='entropy'.")
    if float(oea_zo_trust_lambda) < 0.0:
        raise ValueError("oea_zo_trust_lambda must be >= 0.")
    if oea_zo_trust_q0 not in {"identity", "delta"}:
        raise ValueError("oea_zo_trust_q0 must be one of: 'identity', 'delta'.")
    if oea_zo_drift_mode not in {"none", "penalty", "hard"}:
        raise ValueError("oea_zo_drift_mode must be one of: 'none', 'penalty', 'hard'.")
    if float(oea_zo_drift_gamma) < 0.0:
        raise ValueError("oea_zo_drift_gamma must be >= 0.")
    if float(oea_zo_drift_delta) < 0.0:
        raise ValueError("oea_zo_drift_delta must be >= 0.")
    if oea_zo_selector not in {
        "objective",
        "dev",
        "evidence",
        "probe_mixup",
        "probe_mixup_hard",
        "iwcv",
        "iwcv_ucb",
        "calibrated_ridge",
        "calibrated_guard",
        "calibrated_ridge_guard",
        "calibrated_stack_ridge",
        "calibrated_stack_ridge_guard",
        "calibrated_stack_ridge_guard_borda",
        "calibrated_stack_ridge_guard_borda3",
        "calibrated_stack_bandit_guard",
        "prefer_fbcsp",
        "oracle",
    }:
        raise ValueError(
            "oea_zo_selector must be one of: "
            "'objective', 'dev', 'evidence', 'probe_mixup', 'probe_mixup_hard', 'iwcv', 'iwcv_ucb', "
            "'calibrated_ridge', 'calibrated_guard', 'calibrated_ridge_guard', "
            "'calibrated_stack_ridge', 'calibrated_stack_ridge_guard', 'calibrated_stack_ridge_guard_borda', "
            "'calibrated_stack_ridge_guard_borda3', "
            "'calibrated_stack_bandit_guard', 'prefer_fbcsp', 'oracle'."
        )
    if float(oea_zo_iwcv_kappa) < 0.0:
        raise ValueError("oea_zo_iwcv_kappa must be >= 0.")
    if float(oea_zo_calib_ridge_alpha) <= 0.0:
        raise ValueError("oea_zo_calib_ridge_alpha must be > 0.")
    if int(oea_zo_calib_max_subjects) < 0:
        raise ValueError("oea_zo_calib_max_subjects must be >= 0.")
    if float(oea_zo_calib_guard_c) <= 0.0:
        raise ValueError("oea_zo_calib_guard_c must be > 0.")
    if not (0.0 <= float(oea_zo_calib_guard_threshold) <= 1.0):
        raise ValueError("oea_zo_calib_guard_threshold must be in [0,1].")
    if float(oea_zo_calib_guard_margin) < 0.0:
        raise ValueError("oea_zo_calib_guard_margin must be >= 0.")
    if oea_zo_marginal_mode not in {
        "none",
        "l2_uniform",
        "kl_uniform",
        "hinge_uniform",
        "hard_min",
        "kl_prior",
    }:
        raise ValueError(
            "oea_zo_marginal_mode must be one of: "
            "'none', 'l2_uniform', 'kl_uniform', 'hinge_uniform', 'hard_min', 'kl_prior'."
        )
    if float(oea_zo_marginal_beta) < 0.0:
        raise ValueError("oea_zo_marginal_beta must be >= 0.")
    if not (0.0 <= float(oea_zo_marginal_tau) <= 1.0):
        raise ValueError("oea_zo_marginal_tau must be in [0,1].")
    if oea_zo_marginal_prior not in {"uniform", "source", "anchor_pred"}:
        raise ValueError("oea_zo_marginal_prior must be one of: 'uniform', 'source', 'anchor_pred'.")
    if not (0.0 <= float(oea_zo_marginal_prior_mix) <= 1.0):
        raise ValueError("oea_zo_marginal_prior_mix must be in [0,1].")
    if int(oea_zo_bilevel_iters) < 0:
        raise ValueError("oea_zo_bilevel_iters must be >= 0.")
    if float(oea_zo_bilevel_temp) <= 0.0:
        raise ValueError("oea_zo_bilevel_temp must be > 0.")
    if float(oea_zo_bilevel_step) < 0.0:
        raise ValueError("oea_zo_bilevel_step must be >= 0.")
    if not (0.0 < float(oea_zo_bilevel_coverage_target) <= 1.0):
        raise ValueError("oea_zo_bilevel_coverage_target must be in (0,1].")
    if float(oea_zo_bilevel_coverage_power) < 0.0:
        raise ValueError("oea_zo_bilevel_coverage_power must be >= 0.")
    if float(oea_zo_min_improvement) < 0.0:
        raise ValueError("oea_zo_min_improvement must be >= 0.")
    if not (0.0 <= float(oea_zo_holdout_fraction) < 1.0):
        raise ValueError("oea_zo_holdout_fraction must be in [0,1).")
    if oea_zo_warm_start not in {"none", "delta"}:
        raise ValueError("oea_zo_warm_start must be one of: 'none', 'delta'")
    if int(oea_zo_warm_iters) < 0:
        raise ValueError("oea_zo_warm_iters must be >= 0.")
    if float(oea_zo_fallback_min_marginal_entropy) < 0.0:
        raise ValueError("oea_zo_fallback_min_marginal_entropy must be >= 0.")
    if int(oea_zo_iters) < 0:
        raise ValueError("oea_zo_iters must be >= 0.")
    if float(oea_zo_lr) <= 0.0:
        raise ValueError("oea_zo_lr must be > 0.")
    if float(oea_zo_mu) <= 0.0:
        raise ValueError("oea_zo_mu must be > 0.")
    if int(oea_zo_k) < 1:
        raise ValueError("oea_zo_k must be >= 1.")
    if int(oea_zo_localmix_neighbors) < 0:
        raise ValueError("oea_zo_localmix_neighbors must be >= 0.")
    if float(oea_zo_localmix_self_bias) < 0.0:
        raise ValueError("oea_zo_localmix_self_bias must be >= 0.")
    if float(oea_zo_l2) < 0.0:
        raise ValueError("oea_zo_l2 must be >= 0.")
    if float(mm_safe_mdm_guard_threshold) >= 0.0 and not (0.0 <= float(mm_safe_mdm_guard_threshold) <= 1.0):
        raise ValueError("mm_safe_mdm_guard_threshold must be in [0,1] (or <0 to disable).")
    if float(mm_safe_mdm_min_pred_improve) < 0.0:
        raise ValueError("mm_safe_mdm_min_pred_improve must be >= 0.")
    if float(mm_safe_mdm_drift_delta) < 0.0:
        raise ValueError("mm_safe_mdm_drift_delta must be >= 0.")
    if float(stack_safe_fbcsp_guard_threshold) >= 0.0 and not (0.0 <= float(stack_safe_fbcsp_guard_threshold) <= 1.0):
        raise ValueError("stack_safe_fbcsp_guard_threshold must be in [0,1] (or <0 to disable).")
    if float(stack_safe_fbcsp_min_pred_improve) < 0.0:
        raise ValueError("stack_safe_fbcsp_min_pred_improve must be >= 0.")
    if float(stack_safe_fbcsp_drift_delta) < 0.0:
        raise ValueError("stack_safe_fbcsp_drift_delta must be >= 0.")
    if float(stack_safe_fbcsp_max_pred_disagree) < -1.0:
        raise ValueError("stack_safe_fbcsp_max_pred_disagree must be -1 (disable) or in [0,1].")
    if float(stack_safe_fbcsp_max_pred_disagree) >= 0.0 and not (
        0.0 <= float(stack_safe_fbcsp_max_pred_disagree) <= 1.0
    ):
        raise ValueError("stack_safe_fbcsp_max_pred_disagree must be -1 (disable) or in [0,1].")
    if float(stack_safe_tsa_guard_threshold) >= 0.0 and not (0.0 <= float(stack_safe_tsa_guard_threshold) <= 1.0):
        raise ValueError("stack_safe_tsa_guard_threshold must be in [0,1] (or <0 to disable).")
    if float(stack_safe_tsa_min_pred_improve) < 0.0:
        raise ValueError("stack_safe_tsa_min_pred_improve must be >= 0.")
    if float(stack_safe_tsa_drift_delta) < 0.0:
        raise ValueError("stack_safe_tsa_drift_delta must be >= 0.")
    if float(stack_safe_anchor_guard_delta) < 0.0:
        raise ValueError("stack_safe_anchor_guard_delta must be >= 0.")
    if float(stack_safe_anchor_probe_hard_worsen) < -1.0:
        raise ValueError("stack_safe_anchor_probe_hard_worsen must be -1 (disable) or > -1.")
    if float(stack_safe_min_pred_improve) < 0.0:
        raise ValueError("stack_safe_min_pred_improve must be >= 0.")
    if not isinstance(stack_calib_per_family, (bool, np.bool_)):
        raise ValueError("stack_calib_per_family must be a bool.")
    if str(stack_calib_per_family_mode) not in {"hard", "blend"}:
        raise ValueError("stack_calib_per_family_mode must be one of: 'hard', 'blend'.")
    if float(stack_calib_per_family_shrinkage) < 0.0:
        raise ValueError("stack_calib_per_family_shrinkage must be >= 0.")
    if str(stack_feature_set) not in {"base", "base_delta", "stacked", "stacked_delta"}:
        raise ValueError("stack_feature_set must be one of: 'base', 'base_delta', 'stacked', 'stacked_delta'.")
    stack_fams = {str(f).strip().lower() for f in stack_candidate_families if str(f).strip()}
    if not stack_fams:
        stack_fams = {"ea"}
    allowed_stack_fams = {"ea", "fbcsp", "rpa", "tsa", "chan", "ts_svc", "tsa_ts_svc", "fgmdm"}
    if not stack_fams.issubset(allowed_stack_fams):
        raise ValueError(f"stack_candidate_families must be subset of {sorted(allowed_stack_fams)}; got {sorted(stack_fams)}")
    stack_fams.add("ea")
    if "tsa" in stack_fams and "rpa" not in stack_fams:
        raise ValueError("stack_candidate_families: 'tsa' requires 'rpa'.")
    if float(si_subject_lambda) < 0.0:
        raise ValueError("si_subject_lambda must be >= 0.")
    if float(si_ridge) <= 0.0:
        raise ValueError("si_ridge must be > 0.")
    if int(si_proj_dim) < 0:
        raise ValueError("si_proj_dim must be >= 0 (0 means keep full dim).")
    if any(int(r) < 0 for r in si_chan_candidate_ranks):
        raise ValueError("si_chan_candidate_ranks must be all >= 0.")
    if any(float(lam) < 0.0 for lam in si_chan_candidate_lambdas):
        raise ValueError("si_chan_candidate_lambdas must be all >= 0.")

    diag_subjects_set = {int(s) for s in diagnostics_subjects} if diagnostics_subjects else set()

    # Optional: extra per-subject diagnostics for specific alignments.
    extra_rows: list[dict] | None = (
        []
        if alignment
        in {"ea_si_chan_safe", "ea_si_chan_multi_safe", "ea_si_chan_spsa_safe", "ea_mm_safe", "ea_stack_multi_safe"}
        else None
    )

    def _rankdata(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(x.size, dtype=np.float64)
        return ranks

    def _trace_normalize_covs(covs: np.ndarray) -> np.ndarray:
        covs = np.asarray(covs, dtype=np.float64)
        tr = np.trace(covs, axis1=1, axis2=2)
        tr = np.maximum(tr, 1e-12)
        return covs / tr[:, None, None]

    def _covs_for_riemann(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_c = X - np.mean(X, axis=2, keepdims=True)
        covs = covariances_from_epochs(X_c, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
        return _trace_normalize_covs(covs)

    def _tsa_tangent_vectors_from_covs(covs: np.ndarray) -> np.ndarray:
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.mean import mean_riemann
        from pyriemann.utils.tangentspace import tangent_space

        covs = np.asarray(covs, dtype=np.float64)
        if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
            raise ValueError("Expected covs with shape (n,C,C).")
        c = int(covs.shape[1])
        m = mean_riemann(covs)
        w = invsqrtm(m)
        cov_rec = np.einsum("ij,njk,kl->nil", w, covs, w.T, optimize=True)
        v = tangent_space(cov_rec, np.eye(c, dtype=np.float64))
        v = np.asarray(v, dtype=np.float64)
        norms = np.linalg.norm(v, axis=1)
        scale = float(np.mean(norms)) if norms.size else 1.0
        scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        return v / scale

    # Fast path: subject-wise EA can be precomputed once.
    if alignment in {
        "ea",
        "ea_fbcsp",
        "ea_ts_lr",
        "ea_si",
        "ea_si_chan",
        "ea_si_chan_safe",
        "ea_si_chan_multi_safe",
        "ea_si_chan_spsa_safe",
        "ea_mm_safe",
        "ea_zo",
        "ea_si_zo",
    }:
        if alignment == "ea_fbcsp" and bool(inplace_pre_align):
            # IMPORTANT (memory): EA+FBCSP is particularly memory-hungry on large datasets
            # because it later casts the concatenated training tensor to float64 and allocates
            # additional filterbank buffers. To keep peak RSS bounded, optionally EA-align
            # epochs *in-place* (mutating the input dict) so we don't keep both raw and aligned
            # copies in memory at once.
            for s in list(subject_data.keys()):
                sd = subject_data[int(s)]
                X_aligned = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
                subject_data[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
            subject_data_raw = subject_data
        else:
            aligned: Dict[int, SubjectData] = {}
            for s, sd in subject_data.items():
                X_aligned = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
                aligned[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
            subject_data = aligned
    elif alignment in {"rpa", "tsa"}:
        aligned = {}
        for s, sd in subject_data.items():
            X_aligned = LogEuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
        subject_data = aligned
    elif alignment == "ea_stack_multi_safe":
        # For stacked candidate selection we always need EA(anchor). The LEA/RPA view is only required
        # if a candidate family needs it (rpa/tsa).
        need_rpa_view = ("rpa" in stack_fams) or ("tsa" in stack_fams)
        aligned_ea: Dict[int, SubjectData] = {}
        aligned_rpa: Dict[int, SubjectData] = {}
        for s, sd in subject_data.items():
            X_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned_ea[int(s)] = SubjectData(subject=int(s), X=X_ea, y=sd.y)
            if need_rpa_view:
                X_rpa = LogEuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
                aligned_rpa[int(s)] = SubjectData(subject=int(s), X=X_rpa, y=sd.y)
        subject_data = aligned_ea
        subject_data_rpa = aligned_rpa if need_rpa_view else None
        # IMPORTANT (memory): for very large datasets, keeping three full copies of the epochs
        # (raw + EA + RPA) can trigger OOM during calibration/model fitting. The stacked methods
        # operate entirely on the aligned views, so we can drop the original raw copy here.
        subject_data_raw = subject_data

    # Precompute per-subject SPD covariances for Riemannian/TS baselines to avoid O(n_folds)
    # recomputation of trial-wise covariances (HGD becomes prohibitively slow otherwise).
    #
    # Note: caches are built after alignment preprocessing so they reflect the final view used in the fold loop.
    cov_raw_by_subject: dict[int, np.ndarray] | None = None
    cov_centered_by_subject: dict[int, np.ndarray] | None = None
    cov_riemann_by_subject: dict[int, np.ndarray] | None = None

    need_cov_raw = alignment in {"ts_lr", "ea_ts_lr", "rpa_ts_lr"}
    need_cov_centered = alignment in {"riemann_mdm", "rpa_mdm", "rpa_rot_mdm"}
    need_cov_riemann = alignment in {"ts_svc", "tsa_ts_svc", "fgmdm"} or (
        alignment == "ea_stack_multi_safe" and bool({"ts_svc", "tsa_ts_svc", "fgmdm"} & stack_fams)
    )

    if need_cov_raw:
        cov_raw_by_subject = {}
    if need_cov_centered or need_cov_riemann:
        cov_centered_by_subject = {}
    if need_cov_riemann:
        cov_riemann_by_subject = {}

    if need_cov_raw or need_cov_centered or need_cov_riemann:
        for s, sd in subject_data.items():
            s = int(s)
            if cov_raw_by_subject is not None:
                cov_raw_by_subject[s] = covariances_from_epochs(
                    sd.X, eps=float(oea_eps), shrinkage=float(oea_shrinkage)
                )

            if cov_centered_by_subject is not None:
                X_c = sd.X - np.mean(sd.X, axis=2, keepdims=True)
                cov_c = covariances_from_epochs(X_c, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
                cov_centered_by_subject[s] = cov_c

            if cov_riemann_by_subject is not None:
                # Center + trace-normalize covariances (equivalent to _covs_for_riemann on this subject).
                cov_riemann_by_subject[s] = _trace_normalize_covs(cov_centered_by_subject[s])

    # Cache for expensive per-train-set computations (used by ea_si_chan_multi_safe).
    chan_bundle_cache: dict[tuple[int, ...], dict] = {}
    chan_candidate_grid: list[tuple[int, float]] = []
    if alignment in {"ea_si_chan_multi_safe", "ea_si_chan_spsa_safe", "ea_mm_safe"}:
        ranks = [int(r) for r in (list(si_chan_candidate_ranks) or [int(si_proj_dim)])]
        lambdas = [float(l) for l in (list(si_chan_candidate_lambdas) or [float(si_subject_lambda)])]
        seen: set[tuple[int, float]] = set()
        for r in ranks:
            for lam in lambdas:
                key = (int(r), float(lam))
                if key in seen:
                    continue
                seen.add(key)
                chan_candidate_grid.append(key)

    def _get_chan_bundle(train_subjects_subset: Sequence[int]) -> dict:
        """Return cached models for a given train-subject subset (used by ea_si_chan_multi_safe)."""

        key = tuple(sorted(int(s) for s in train_subjects_subset))
        if key in chan_bundle_cache:
            return chan_bundle_cache[key]

        X_train_parts = [subject_data[int(s)].X for s in key]
        y_train_parts = [subject_data[int(s)].y for s in key]
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        subj_train = np.concatenate(
            [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in key],
            axis=0,
        )

        bundle: dict = {"model_id": fit_csp_lda(X_train, y_train, n_components=n_components), "candidates": {}}
        n_channels = int(X_train.shape[1])

        # Candidate channel projectors + per-candidate CSP+LDA models.
        for r, lam in chan_candidate_grid:
            r = int(r)
            lam = float(lam)
            if r <= 0 or r >= n_channels:
                continue
            chan_params = ChannelProjectorParams(subject_lambda=float(lam), ridge=float(si_ridge), n_components=int(r))
            A = learn_subject_invariant_channel_projector(
                X=X_train,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )
            if np.allclose(A, np.eye(n_channels, dtype=np.float64), atol=1e-10):
                continue
            X_train_A = apply_spatial_transform(A, X_train)
            model_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)
            bundle["candidates"][(int(r), float(lam))] = {"A": A, "model": model_A, "rank": int(r), "lambda": float(lam)}

        chan_bundle_cache[key] = bundle
        return bundle

    # Cache for expensive per-train-set computations (used by ea_stack_multi_safe).
    stack_bundle_cache: dict[tuple[int, ...], dict] = {}
    stack_chan_candidate_grid: list[tuple[int, float]] = []
    if alignment == "ea_stack_multi_safe":
        ranks = [int(r) for r in (list(si_chan_candidate_ranks) or [int(si_proj_dim)])]
        lambdas = [float(l) for l in (list(si_chan_candidate_lambdas) or [float(si_subject_lambda)])]
        seen: set[tuple[int, float]] = set()
        for r in ranks:
            for lam in lambdas:
                key = (int(r), float(lam))
                if key in seen:
                    continue
                seen.add(key)
                stack_chan_candidate_grid.append(key)

    def _get_stack_bundle(train_subjects_subset: Sequence[int]) -> dict:
        """Return cached models for a given train-subject subset (used by ea_stack_multi_safe)."""

        need_rpa_view = ("rpa" in stack_fams) or ("tsa" in stack_fams)
        if need_rpa_view and subject_data_rpa is None:
            raise RuntimeError("ea_stack_multi_safe requires precomputed subject_data_rpa when using rpa/tsa.")

        key = tuple(sorted(int(s) for s in train_subjects_subset))
        if key in stack_bundle_cache:
            return stack_bundle_cache[key]

        # Anchor (EA) view.
        X_train_ea = np.concatenate([subject_data[int(s)].X for s in key], axis=0)
        y_train = np.concatenate([subject_data[int(s)].y for s in key], axis=0)
        subj_train = np.concatenate(
            [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in key],
            axis=0,
        )
        model_ea = fit_csp_lda(X_train_ea, y_train, n_components=n_components)

        bundle: dict = {
            "subjects": key,
            "ea": {"model": model_ea},
            "chan": {"candidates": {}},
        }

        if "fbcsp" in stack_fams:
            # EA-FBCSP view (EA time series + filterbank CSP+LDA).
            # Keep consistent with the default fbcsp/ea_fbcsp branch in this file.
            bands = [
                (8.0, 12.0),
                (10.0, 14.0),
                (12.0, 16.0),
                (14.0, 18.0),
                (16.0, 20.0),
                (18.0, 22.0),
                (20.0, 24.0),
                (22.0, 26.0),
                (24.0, 28.0),
                (26.0, 30.0),
            ]
            fb_n_components = max(2, min(4, int(n_components)))
            model_fbcsp = fit_fbcsp_lda(
                X_train_ea,
                y_train,
                bands=bands,
                sfreq=float(sfreq),
                n_components=fb_n_components,
                filter_order=4,
                multiclass_strategy=str(fbcsp_multiclass_strategy),
                select_k=24,
            )
            bundle["fbcsp"] = {"model": model_fbcsp}

        if "ts_svc" in stack_fams:
            try:
                from pyriemann.tangentspace import TangentSpace
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC

                if cov_riemann_by_subject is not None:
                    cov_tr = np.concatenate([cov_riemann_by_subject[int(s)] for s in key], axis=0)
                else:
                    cov_tr = _covs_for_riemann(X_train_ea)
                ts = TangentSpace(metric="riemann")
                x_tr = ts.fit_transform(cov_tr)
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    SVC(kernel="linear", probability=True, random_state=0),
                )
                clf.fit(x_tr, y_train)
                ev = _compute_gaussian_evidence_params_from_feats(
                    feats=x_tr,
                    y_train=y_train,
                    class_order=tuple([str(c) for c in class_order]),
                    ridge=float(si_ridge),
                )
                bundle["ts_svc"] = {"ts": ts, "clf": clf, "evidence": ev}
            except Exception:
                pass

        if "tsa_ts_svc" in stack_fams:
            try:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC

                if cov_riemann_by_subject is not None:
                    cov_tr = np.concatenate([cov_riemann_by_subject[int(s)] for s in key], axis=0)
                else:
                    cov_tr = _covs_for_riemann(X_train_ea)
                v_tr = _tsa_tangent_vectors_from_covs(cov_tr)
                mu_s = np.zeros((len(class_order), v_tr.shape[1]), dtype=np.float64)
                for i, c in enumerate([str(x) for x in class_order]):
                    mask = np.asarray(y_train == c)
                    if np.any(mask):
                        mu_s[int(i)] = np.mean(v_tr[mask], axis=0)

                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    SVC(kernel="linear", probability=True, random_state=0),
                )
                clf.fit(v_tr, y_train)
                ev = _compute_gaussian_evidence_params_from_feats(
                    feats=v_tr,
                    y_train=y_train,
                    class_order=tuple([str(c) for c in class_order]),
                    ridge=float(si_ridge),
                )
                bundle["tsa_ts_svc"] = {"clf": clf, "mu_s": mu_s, "evidence": ev}
            except Exception:
                pass

        if "fgmdm" in stack_fams:
            try:
                from pyriemann.classification import FgMDM

                if cov_riemann_by_subject is not None:
                    cov_tr = np.concatenate([cov_riemann_by_subject[int(s)] for s in key], axis=0)
                else:
                    cov_tr = _covs_for_riemann(X_train_ea)
                fg = FgMDM(metric="riemann")
                fg.fit(cov_tr, y_train)
                bundle["fgmdm"] = {"model": fg}
            except Exception:
                pass

        # Channel projector candidates (learned on EA view).
        if "chan" in stack_fams:
            n_channels = int(X_train_ea.shape[1])
            # IMPORTANT (efficiency/memory): the expensive scatter computation does not
            # depend on (lambda, rank). Compute it once per train-subject set, then
            # re-solve A for each hyper-parameter pair.
            scatter_chan = compute_channel_projector_scatter(
                X=X_train_ea,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
            )
            for r, lam in stack_chan_candidate_grid:
                r = int(r)
                lam = float(lam)
                if r <= 0 or r >= n_channels:
                    continue
                if scatter_chan is None:
                    continue
                A = solve_channel_projector_from_scatter(
                    scatter_chan,
                    subject_lambda=float(lam),
                    ridge=float(si_ridge),
                    n_components=int(r),
                    eps=float(oea_eps),
                )
                if np.allclose(A, np.eye(n_channels, dtype=np.float64), atol=1e-10):
                    continue
                X_train_A = apply_spatial_transform(A, X_train_ea)
                model_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)
                bundle["chan"]["candidates"][(int(r), float(lam))] = {
                    "A": A,
                    "model": model_A,
                    "rank": int(r),
                    "lambda": float(lam),
                }
                try:
                    del X_train_A
                except Exception:
                    pass
                # IMPORTANT (memory): the chan candidate loop repeatedly allocates very large
                # (n_trials,C,T) tensors. Explicitly collect + trim here to avoid allocator RSS
                # growth across candidates / folds on large datasets (e.g., PhysionetMI).
                gc.collect()
                _maybe_malloc_trim()

        # IMPORTANT (memory): `X_train_ea` is a very large array on high-subject datasets.
        # We free it before constructing the (optional) RPA view to avoid holding two full
        # time-series training tensors at once.
        try:
            del X_train_ea
        except Exception:
            pass
        gc.collect()
        _maybe_malloc_trim()

        if need_rpa_view:
            X_train_rpa = np.concatenate([subject_data_rpa[int(s)].X for s in key], axis=0)
            model_rpa = fit_csp_lda(X_train_rpa, y_train, n_components=n_components)
            bundle["rpa"] = {"model": model_rpa}
            try:
                del X_train_rpa
            except Exception:
                pass
            gc.collect()
            _maybe_malloc_trim()

        stack_bundle_cache[key] = bundle
        gc.collect()
        _maybe_malloc_trim()
        return bundle

    for test_subject in subjects_test:
        model: TrainedModel | None = None
        train_subjects = [s for s in subjects_all if s != test_subject]
        do_diag = diagnostics_dir is not None and int(test_subject) in diag_subjects_set
        zo_diag: dict | None = None
        z_test_base: np.ndarray | None = None

        # Build per-fold aligned train/test data if needed.
        if alignment in {"none", "ea", "rpa", "fbcsp", "ea_fbcsp"}:
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
        elif alignment in {"riemann_mdm", "rpa_mdm", "rpa_rot_mdm"}:
            # pyRiemann baselines on SPD covariances (Riemannian TL / Procrustes family).
            #
            # Note: this branch operates on covariance matrices directly (not on time series),
            # so `X_test` is replaced by (n_trials,C,C) SPD matrices.
            from pyriemann.classification import MDM
            from pyriemann.transfer import TLCenter, TLRotate, encode_domains

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            y_train_parts = []
            dom_train_parts = []
            cov_train_parts: list[np.ndarray] | None = [] if cov_centered_by_subject is not None else None
            for s in train_subjects:
                sd = subject_data[int(s)]
                y_train_parts.append(sd.y)
                dom_train_parts.append(np.full(sd.y.shape[0], f"src_{int(s)}", dtype=object))
                if cov_train_parts is not None:
                    cov_train_parts.append(cov_centered_by_subject[int(s)])
            y_train = np.concatenate(y_train_parts, axis=0)
            dom_train = np.concatenate(dom_train_parts, axis=0)

            dom_test = np.full(y_test.shape[0], "target", dtype=object)

            if cov_train_parts is not None and cov_centered_by_subject is not None:
                cov_train = np.concatenate(cov_train_parts, axis=0)
                cov_test = cov_centered_by_subject[int(test_subject)]
            else:
                # Use centered covariances (subtract per-channel mean) for tangent-space features.
                X_train_parts = [subject_data[int(s)].X for s in train_subjects]
                X_train = np.concatenate(X_train_parts, axis=0)
                X_train_c = X_train - np.mean(X_train, axis=2, keepdims=True)
                X_test_c = X_test_raw - np.mean(X_test_raw, axis=2, keepdims=True)
                cov_train = covariances_from_epochs(X_train_c, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
                cov_test = covariances_from_epochs(X_test_c, eps=float(oea_eps), shrinkage=float(oea_shrinkage))

            if alignment == "riemann_mdm":
                model = MDM(metric="riemann")
                model.fit(cov_train, y_train)
                X_test = cov_test
            else:
                try:
                    from pyriemann.transfer import TLStretch
                except Exception:
                    try:
                        from pyriemann.transfer._estimators import TLStretch
                    except Exception as exc:
                        raise ImportError(
                            "Missing pyriemann.transfer.TLStretch required for rpa-mdm / rpa-rot-mdm baselines."
                        ) from exc
                # Center + stretch (RPA without rotation) using both source and target (unlabeled) covariances.
                y_dummy = np.full(y_test.shape[0], str(class_order[0]), dtype=object)
                cov_all = np.concatenate([cov_train, cov_test], axis=0)
                y_all = np.concatenate([y_train, y_dummy], axis=0)
                dom_all = np.concatenate([dom_train, dom_test], axis=0)
                _, y_enc = encode_domains(cov_all, y_all, dom_all)

                center = TLCenter(target_domain="target", metric="riemann")
                cov_centered = center.fit_transform(cov_all, y_enc)

                stretch = TLStretch(target_domain="target", centered_data=True, metric="riemann")
                cov_stretched = stretch.fit_transform(cov_centered, y_enc)

                cov_src = cov_stretched[: cov_train.shape[0]]
                cov_tgt = cov_stretched[cov_train.shape[0] :]

                if alignment == "rpa_rot_mdm":
                    # One-step pseudo-label rotation (RPA full): predict pseudo labels on target then rotate sources.
                    base = MDM(metric="riemann")
                    base.fit(cov_src, y_train)
                    y_pseudo = np.asarray(base.predict(cov_tgt))
                    y_all2 = np.concatenate([y_train, y_pseudo], axis=0)
                    _, y_enc2 = encode_domains(cov_stretched, y_all2, dom_all)

                    rotate = TLRotate(target_domain="target", metric="euclid", n_jobs=1)
                    cov_rot = rotate.fit_transform(cov_stretched, y_enc2)
                    cov_src = cov_rot[: cov_train.shape[0]]
                    cov_tgt = cov_rot[cov_train.shape[0] :]

                model = MDM(metric="riemann")
                model.fit(cov_src, y_train)
                X_test = cov_tgt
        elif alignment in {"ts_lr", "ea_ts_lr"}:
            # Tangent-space classifier: TangentSpace(metric='riemann') + LogisticRegression on SPD covariances.
            from pyriemann.tangentspace import TangentSpace
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            y_train_parts = [subject_data[s].y for s in train_subjects]
            y_train = np.concatenate(y_train_parts, axis=0)

            if cov_raw_by_subject is not None:
                cov_train = np.concatenate([cov_raw_by_subject[int(s)] for s in train_subjects], axis=0)
                cov_test = cov_raw_by_subject[int(test_subject)]
            else:
                X_train_parts = [subject_data[s].X for s in train_subjects]
                X_train = np.concatenate(X_train_parts, axis=0)
                cov_train = covariances_from_epochs(X_train, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
                cov_test = covariances_from_epochs(X_test_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))

            # Riemannian pipelines commonly apply trace-normalization to reduce per-trial power scale effects
            # (keeps SPD and improves cross-subject robustness).
            def _trace_normalize(covs: np.ndarray) -> np.ndarray:
                covs = np.asarray(covs, dtype=np.float64)
                tr = np.trace(covs, axis1=1, axis2=2)
                tr = np.maximum(tr, 1e-12)
                return covs / tr[:, None, None]

            cov_train = _trace_normalize(cov_train)
            cov_test = _trace_normalize(cov_test)

            model = make_pipeline(
                TangentSpace(metric="riemann"),
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    n_jobs=1,
                ),
            )
            model.fit(cov_train, y_train)
            X_test = cov_test
        elif alignment == "ts_svc":
            # Tangent-space classifier: TangentSpace(metric='riemann') + linear SVC on SPD covariances.
            from pyriemann.tangentspace import TangentSpace
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            y_train_parts = [subject_data[s].y for s in train_subjects]
            y_train = np.concatenate(y_train_parts, axis=0)

            if cov_riemann_by_subject is not None:
                cov_train = np.concatenate([cov_riemann_by_subject[int(s)] for s in train_subjects], axis=0)
                cov_test = cov_riemann_by_subject[int(test_subject)]
            else:
                X_train_parts = [subject_data[s].X for s in train_subjects]
                X_train = np.concatenate(X_train_parts, axis=0)
                cov_train = _covs_for_riemann(X_train)
                cov_test = _covs_for_riemann(X_test_raw)

            model = make_pipeline(
                TangentSpace(metric="riemann"),
                StandardScaler(with_mean=True, with_std=True),
                SVC(kernel="linear", probability=True, random_state=0),
            )
            model.fit(cov_train, y_train)
            X_test = cov_test
        elif alignment == "tsa_ts_svc":
            # TSA-style alignment in tangent space (recenter+rescale per domain + pseudo-label Procrustes rotation),
            # then linear SVC on tangent vectors.
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            y_train_parts = [subject_data[s].y for s in train_subjects]
            y_train = np.concatenate(y_train_parts, axis=0)

            if cov_riemann_by_subject is not None:
                cov_train = np.concatenate([cov_riemann_by_subject[int(s)] for s in train_subjects], axis=0)
                cov_test = cov_riemann_by_subject[int(test_subject)]
            else:
                X_train_parts = [subject_data[s].X for s in train_subjects]
                X_train = np.concatenate(X_train_parts, axis=0)
                cov_train = _covs_for_riemann(X_train)
                cov_test = _covs_for_riemann(X_test_raw)

            v_train = _tsa_tangent_vectors_from_covs(cov_train)
            v_test = _tsa_tangent_vectors_from_covs(cov_test)

            model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                SVC(kernel="linear", probability=True, random_state=0),
            )
            model.fit(v_train, y_train)

            # Pseudo labels on target to build Procrustes rotation.
            y_pseudo = np.asarray(model.predict(v_test), dtype=object)
            mu_s = np.zeros((len(class_order), v_train.shape[1]), dtype=np.float64)
            mu_t = np.full_like(mu_s, np.nan)
            for i, c in enumerate([str(x) for x in class_order]):
                m_s = y_train == c
                if np.any(m_s):
                    mu_s[int(i)] = np.mean(v_train[m_s], axis=0)
                m_t = y_pseudo == c
                if np.any(m_t):
                    mu_t[int(i)] = np.mean(v_test[m_t], axis=0)

            valid = np.all(np.isfinite(mu_t), axis=1) & np.all(np.isfinite(mu_s), axis=1)
            r = np.eye(int(v_test.shape[1]), dtype=np.float64)
            if int(np.sum(valid)) >= 2:
                a = mu_t[valid]
                b = mu_s[valid]
                u, _s, vt = np.linalg.svd(a.T @ b, full_matrices=False)
                r = u @ vt
            X_test = v_test @ r
        elif alignment == "fgmdm":
            # FgMDM baseline on SPD covariances.
            from pyriemann.classification import FgMDM

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            y_train_parts = [subject_data[s].y for s in train_subjects]
            y_train = np.concatenate(y_train_parts, axis=0)

            if cov_riemann_by_subject is not None:
                cov_train = np.concatenate([cov_riemann_by_subject[int(s)] for s in train_subjects], axis=0)
                cov_test = cov_riemann_by_subject[int(test_subject)]
            else:
                X_train_parts = [subject_data[s].X for s in train_subjects]
                X_train = np.concatenate(X_train_parts, axis=0)
                cov_train = _covs_for_riemann(X_train)
                cov_test = _covs_for_riemann(X_test_raw)

            model = FgMDM(metric="riemann")
            model.fit(cov_train, y_train)
            X_test = cov_test
        elif alignment == "rpa_ts_lr":
            # RPA-style tangent-space classifier: TLCenter+TLStretch (unlabeled target) then
            # TangentSpace(metric='riemann') + LogisticRegression on covariances.
            from pyriemann.tangentspace import TangentSpace
            from pyriemann.transfer import TLCenter, encode_domains
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            try:
                from pyriemann.transfer import TLStretch
            except Exception:
                try:
                    from pyriemann.transfer._estimators import TLStretch
                except Exception as exc:
                    raise ImportError(
                        "Missing pyriemann.transfer.TLStretch required for rpa-ts-lr baseline."
                    ) from exc

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            y_train_parts = []
            dom_train_parts = []
            cov_train_parts: list[np.ndarray] | None = [] if cov_raw_by_subject is not None else None
            for s in train_subjects:
                sd = subject_data[int(s)]
                y_train_parts.append(sd.y)
                dom_train_parts.append(np.full(sd.y.shape[0], f"src_{int(s)}", dtype=object))
                if cov_train_parts is not None:
                    cov_train_parts.append(cov_raw_by_subject[int(s)])
            y_train = np.concatenate(y_train_parts, axis=0)
            dom_train = np.concatenate(dom_train_parts, axis=0)

            dom_test = np.full(y_test.shape[0], "target", dtype=object)

            if cov_train_parts is not None and cov_raw_by_subject is not None:
                cov_train = np.concatenate(cov_train_parts, axis=0)
                cov_test = cov_raw_by_subject[int(test_subject)]
            else:
                X_train_parts = [subject_data[int(s)].X for s in train_subjects]
                X_train = np.concatenate(X_train_parts, axis=0)
                cov_train = covariances_from_epochs(X_train, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
                cov_test = covariances_from_epochs(X_test_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))

            # Center + stretch (RPA without rotation) using both source and target (unlabeled) covariances.
            y_dummy = np.full(y_test.shape[0], str(class_order[0]), dtype=object)
            cov_all = np.concatenate([cov_train, cov_test], axis=0)
            y_all = np.concatenate([y_train, y_dummy], axis=0)
            dom_all = np.concatenate([dom_train, dom_test], axis=0)
            _, y_enc = encode_domains(cov_all, y_all, dom_all)

            center = TLCenter(target_domain="target", metric="riemann")
            cov_centered = center.fit_transform(cov_all, y_enc)

            stretch = TLStretch(target_domain="target", centered_data=True, metric="riemann")
            cov_stretched = stretch.fit_transform(cov_centered, y_enc)

            cov_src = cov_stretched[: cov_train.shape[0]]
            cov_tgt = cov_stretched[cov_train.shape[0] :]

            model = make_pipeline(
                TangentSpace(metric="riemann"),
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    n_jobs=1,
                ),
            )
            model.fit(cov_src, y_train)
            X_test = cov_tgt
        elif alignment == "tsa":
            # Tangent-space alignment (TSA) using pseudo-label anchors in the LEA/RPA-whitened space.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
            q_tsa = _compute_tsa_target_rotation(
                z_train=X_train,
                y_train=y_train,
                z_target=X_test,
                model=model,
                class_order=tuple([str(c) for c in class_order]),
                pseudo_mode=str(oea_pseudo_mode),
                pseudo_iters=int(max(0, oea_pseudo_iters)),
                q_blend=float(oea_q_blend),
                pseudo_confidence=float(oea_pseudo_confidence),
                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                pseudo_balance=bool(oea_pseudo_balance),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
            )
            X_test = apply_spatial_transform(q_tsa, X_test)
        elif alignment == "ea_stack_multi_safe":
            # Stacked multi-candidate selection with safe fallback to the EA anchor.
            #
            # Candidate families (per fold):
            # - EA anchor (A=I, using EA-whitened data)
            # - RPA (LEA-whitened data)
            # - TSA (LEA-whitened + TSA target rotation)
            # - EA-SI-CHAN (rank-deficient channel projectors on EA-whitened data)
            # - EA-FBCSP (EA-whitened time series + filterbank CSP+LDA)
            need_rpa_view = ("rpa" in stack_fams) or ("tsa" in stack_fams)
            if need_rpa_view and subject_data_rpa is None:
                raise RuntimeError("ea_stack_multi_safe requires subject_data_rpa when using rpa/tsa candidates.")
            include_fbcsp = "fbcsp" in stack_fams
            include_rpa = "rpa" in stack_fams
            include_tsa = "tsa" in stack_fams
            include_chan = "chan" in stack_fams
            include_ts_svc = "ts_svc" in stack_fams
            include_tsa_ts_svc = "tsa_ts_svc" in stack_fams
            include_fgmdm = "fgmdm" in stack_fams

            class_labels = tuple([str(c) for c in class_order])
            selector = str(oea_zo_selector)
            use_stack = selector in {
                "calibrated_stack_ridge",
                "calibrated_stack_ridge_guard",
                "calibrated_stack_ridge_guard_borda",
                "calibrated_stack_ridge_guard_borda3",
                "calibrated_stack_bandit_guard",
            }
            use_ridge = selector in {
                "calibrated_ridge",
                "calibrated_ridge_guard",
                "calibrated_stack_ridge",
                "calibrated_stack_ridge_guard",
                "calibrated_stack_ridge_guard_borda",
                "calibrated_stack_ridge_guard_borda3",
                "calibrated_stack_bandit_guard",
            }
            use_guard = selector in {
                "calibrated_guard",
                "calibrated_ridge_guard",
                "calibrated_stack_ridge_guard",
                "calibrated_stack_ridge_guard_borda",
                "calibrated_stack_ridge_guard_borda3",
                "calibrated_stack_bandit_guard",
            }
            use_bandit = selector in {"calibrated_stack_bandit_guard"}

            outer_bundle = _get_stack_bundle(train_subjects)
            model_ea = outer_bundle["ea"]["model"]
            model_rpa = (outer_bundle.get("rpa", {}).get("model") if need_rpa_view else None)
            chan_outer: dict = dict(outer_bundle.get("chan", {}).get("candidates", {})) if include_chan else {}
            # For reporting only (n_train) and consistency with other branches.
            y_train = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
            X_train = np.empty((0,) + tuple(subject_data[int(train_subjects[0])].X.shape[1:]), dtype=np.float64)

            # Per-fold calibration on pseudo-target subjects (source-only).
            cert = None
            guard = None
            bandit_policy = None
            cert_by_family: dict[str, RidgeCertificate] = {}
            guard_by_family: dict[str, LogisticGuard] = {}
            family_counts: dict[str, int] = {}
            ridge_train_spearman = float("nan")
            ridge_train_pearson = float("nan")
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(
                *,
                p_id: np.ndarray,
                p_c: np.ndarray,
                feats_c: np.ndarray | None = None,
                lda_c=None,
                lda_ev: dict | None = None,
                seed_local: int = 0,
            ) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))
                mean_conf = float(np.mean(np.max(np.clip(p_c, 1e-12, 1.0), axis=1)))

                d = _drift_vec(p_id, p_c)
                try:
                    y_id = np.asarray(np.argmax(p_id, axis=1), dtype=int).reshape(-1)
                    y_c = np.asarray(np.argmax(p_c, axis=1), dtype=int).reshape(-1)
                    pred_disagree = float(np.mean(y_id != y_c)) if y_id.shape == y_c.shape else 0.0
                except Exception:
                    pred_disagree = 0.0
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "mean_confidence": float(mean_conf),
                    "pred_disagree": float(pred_disagree),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])

                if use_stack and feats_c is not None and lda_c is not None:
                    try:
                        keep = _select_keep_indices(
                            p_c,
                            class_order=class_labels,
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                        )
                        rec["n_keep"] = int(keep.size)
                    except Exception:
                        rec["n_keep"] = 0
                    rec["n_best_total"] = int(p_c.shape[0])
                    rec["n_full_total"] = int(p_c.shape[0])

                    try:
                        ev = _evidence_nll(
                            proba=p_c,
                            feats=np.asarray(feats_c, dtype=np.float64),
                            lda_evidence=lda_ev,
                            class_order=class_labels,
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                            reliable_metric=str(oea_zo_reliable_metric),
                            reliable_threshold=float(oea_zo_reliable_threshold),
                            reliable_alpha=float(oea_zo_reliable_alpha),
                        )
                        rec["evidence_nll_best"] = float(ev)
                        rec["evidence_nll_full"] = float(ev)
                    except Exception:
                        rec["evidence_nll_best"] = float("nan")
                        rec["evidence_nll_full"] = float("nan")

                    try:
                        pm, pm_stats = _probe_mixup(
                            proba=p_c,
                            feats=np.asarray(feats_c, dtype=np.float64),
                            lda=lda_c,
                            class_order=class_labels,
                            seed_local=int(seed_local),
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                        )
                        rec["probe_mixup_best"] = float(pm)
                        rec["probe_mixup_full"] = float(pm)
                        rec["probe_mixup_pairs_best"] = int(pm_stats.n_pairs)
                        rec["probe_mixup_pairs_full"] = int(pm_stats.n_pairs)
                        rec["probe_mixup_keep_best"] = int(pm_stats.n_keep)
                        rec["probe_mixup_keep_full"] = int(pm_stats.n_keep)
                        rec["probe_mixup_frac_intra_best"] = float(pm_stats.frac_intra)
                        rec["probe_mixup_frac_intra_full"] = float(pm_stats.frac_intra)
                    except Exception:
                        rec["probe_mixup_best"] = float("nan")
                        rec["probe_mixup_full"] = float("nan")

                    try:
                        pmh, pmh_stats = _probe_mixup(
                            proba=p_c,
                            feats=np.asarray(feats_c, dtype=np.float64),
                            lda=lda_c,
                            class_order=class_labels,
                            seed_local=int(seed_local),
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                            mode="hard_major",
                            beta_alpha=0.4,
                        )
                        rec["probe_mixup_hard_best"] = float(pmh)
                        rec["probe_mixup_hard_full"] = float(pmh)
                        rec["probe_mixup_hard_pairs_best"] = int(pmh_stats.n_pairs)
                        rec["probe_mixup_hard_pairs_full"] = int(pmh_stats.n_pairs)
                        rec["probe_mixup_hard_keep_best"] = int(pmh_stats.n_keep)
                        rec["probe_mixup_hard_keep_full"] = int(pmh_stats.n_keep)
                        rec["probe_mixup_hard_frac_intra_best"] = float(pmh_stats.frac_intra)
                        rec["probe_mixup_hard_frac_intra_full"] = float(pmh_stats.frac_intra)
                    except Exception:
                        rec["probe_mixup_hard_best"] = float("nan")
                        rec["probe_mixup_hard_full"] = float("nan")

                return rec

            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_ridge_rows: List[np.ndarray] = []
                y_ridge_rows: List[float] = []
                X_guard_rows: List[np.ndarray] = []
                y_guard_rows: List[int] = []
                improve_guard_rows: List[float] = []
                X_ridge_by_family: dict[str, list[np.ndarray]] = {}
                y_ridge_by_family: dict[str, list[float]] = {}
                X_guard_by_family: dict[str, list[np.ndarray]] = {}
                y_guard_by_family: dict[str, list[int]] = {}
                feat_names: tuple[str, ...] | None = None
                X_bandit_rows: list[np.ndarray] = []
                r_bandit_rows: list[float] = []
                g_bandit_rows: list[int] = []

                def _add_calib_sample(*, fam: str, feats: np.ndarray, improve: float) -> None:
                    fam = str(fam).strip().lower() or "unknown"
                    improve = float(improve)
                    if use_ridge:
                        X_ridge_rows.append(feats)
                        y_ridge_rows.append(improve)
                        if bool(stack_calib_per_family):
                            X_ridge_by_family.setdefault(fam, []).append(feats)
                            y_ridge_by_family.setdefault(fam, []).append(improve)
                    if use_guard:
                        yb = 1 if improve >= float(oea_zo_calib_guard_margin) else 0
                        X_guard_rows.append(feats)
                        y_guard_rows.append(int(yb))
                        improve_guard_rows.append(improve)
                        if bool(stack_calib_per_family):
                            X_guard_by_family.setdefault(fam, []).append(feats)
                            y_guard_by_family.setdefault(fam, []).append(int(yb))

                def _add_bandit_sample(*, gid: int, feats: np.ndarray, reward: float) -> None:
                    if not use_bandit:
                        return
                    X_bandit_rows.append(np.asarray(feats, dtype=np.float64))
                    r_bandit_rows.append(float(reward))
                    g_bandit_rows.append(int(gid))

                # Calibration can be very expensive on large datasets if we refit models for each pseudo-target
                # (leave-one-out over subjects). When `oea_zo_calib_max_subjects>0`, we instead use a single
                # held-out calibration set of subjects and fit the calibration models once per fold.
                calib_max = int(oea_zo_calib_max_subjects)
                use_holdout_calib = calib_max > 0 and calib_max < len(train_subjects)
                need_inner_train_arrays = bool(use_stack or include_tsa)
                holdout_inner_train: list[int] | None = None
                holdout_inner_bundle: dict | None = None
                holdout_z_tr_ea: np.ndarray | None = None
                holdout_y_tr: np.ndarray | None = None
                holdout_z_tr_rpa: np.ndarray | None = None
                if use_holdout_calib:
                    holdout_inner_train = [int(s) for s in train_subjects if int(s) not in set(calib_subjects)]
                    if len(holdout_inner_train) >= 2:
                        holdout_inner_bundle = _get_stack_bundle(holdout_inner_train)
                        if need_inner_train_arrays:
                            holdout_z_tr_ea = np.concatenate([subject_data[int(s)].X for s in holdout_inner_train], axis=0)
                            holdout_y_tr = np.concatenate([subject_data[int(s)].y for s in holdout_inner_train], axis=0)
                            holdout_z_tr_rpa = (
                                np.concatenate([subject_data_rpa[int(s)].X for s in holdout_inner_train], axis=0)
                                if need_rpa_view and subject_data_rpa is not None
                                else None
                            )
                    else:
                        use_holdout_calib = False

                for pseudo_t in calib_subjects:
                    if use_holdout_calib:
                        if holdout_inner_train is None or holdout_inner_bundle is None or holdout_z_tr_ea is None or holdout_y_tr is None:
                            if need_inner_train_arrays:
                                continue
                        inner_train = holdout_inner_train
                        inner_bundle = holdout_inner_bundle
                        if need_inner_train_arrays:
                            z_tr_ea_inner = holdout_z_tr_ea
                            y_tr_inner = holdout_y_tr
                            z_tr_rpa_inner = holdout_z_tr_rpa
                        else:
                            z_tr_ea_inner = None
                            y_tr_inner = None
                            z_tr_rpa_inner = None
                    else:
                        inner_train = [s for s in train_subjects if s != pseudo_t]
                        if len(inner_train) < 2:
                            continue
                        inner_bundle = _get_stack_bundle(inner_train)
                        if need_inner_train_arrays:
                            z_tr_ea_inner = np.concatenate([subject_data[int(s)].X for s in inner_train], axis=0)
                            y_tr_inner = np.concatenate([subject_data[int(s)].y for s in inner_train], axis=0)
                            z_tr_rpa_inner = (
                                np.concatenate([subject_data_rpa[int(s)].X for s in inner_train], axis=0)
                                if need_rpa_view and subject_data_rpa is not None
                                else None
                            )
                        else:
                            z_tr_ea_inner = None
                            y_tr_inner = None
                            z_tr_rpa_inner = None
                    seed_pseudo_base = int(oea_zo_calib_seed) + int(test_subject) * 997 + int(pseudo_t) * 10007

                    # Anchor (EA) predictions on the pseudo-target.
                    z_p_ea = subject_data[int(pseudo_t)].X
                    y_p = subject_data[int(pseudo_t)].y
                    m_ea = inner_bundle["ea"]["model"]
                    m_fbcsp = (inner_bundle.get("fbcsp", {}).get("model") if include_fbcsp else None)
                    m_rpa = (inner_bundle.get("rpa", {}).get("model") if need_rpa_view else None)

                    ev_ea_inner = None
                    ev_rpa_inner = None
                    ev_fbcsp_inner = None
                    if use_stack:
                        try:
                            ev_ea_inner = _compute_lda_evidence_params(
                                model=m_ea,
                                X_train=z_tr_ea_inner,
                                y_train=y_tr_inner,
                                class_order=class_labels,
                                ridge=float(si_ridge),
                            )
                        except Exception:
                            ev_ea_inner = None
                        if m_fbcsp is not None:
                            try:
                                ev_fbcsp_inner = _compute_lda_evidence_params(
                                    model=m_fbcsp,
                                    X_train=z_tr_ea_inner,
                                    y_train=y_tr_inner,
                                    class_order=class_labels,
                                    ridge=float(si_ridge),
                                )
                            except Exception:
                                ev_fbcsp_inner = None
                        if m_rpa is not None and z_tr_rpa_inner is not None:
                            try:
                                ev_rpa_inner = _compute_lda_evidence_params(
                                    model=m_rpa,
                                    X_train=z_tr_rpa_inner,
                                    y_train=y_tr_inner,
                                    class_order=class_labels,
                                    ridge=float(si_ridge),
                                )
                            except Exception:
                                ev_rpa_inner = None
                    p_id = _reorder_proba_columns(m_ea.predict_proba(z_p_ea), m_ea.classes_, list(class_labels))
                    acc_id = float(accuracy_score(y_p, np.asarray(m_ea.predict(z_p_ea))))

                    # Identity (EA anchor) record: used as the reference for anchor-relative (delta) features.
                    rec_id_c = None
                    if use_stack or use_bandit or str(stack_feature_set) in {"stacked_delta", "base_delta"}:
                        try:
                            feats_id_c = _features_before_lda(model=m_ea, X=z_p_ea) if use_stack else None
                            rec_id_c = _record_for_candidate(
                                p_id=p_id,
                                p_c=p_id,
                                feats_c=feats_id_c,
                                lda_c=(m_ea.pipeline.named_steps["lda"] if use_stack else None),
                                lda_ev=ev_ea_inner,
                                seed_local=seed_pseudo_base + 1,
                            )
                            rec_id_c["kind"] = "identity"
                            rec_id_c["cand_family"] = "ea"
                            if use_stack and z_tr_ea_inner is not None and y_tr_inner is not None:
                                try:
                                    rec_id_c.update(
                                        _iwcv_ucb_stats_for_model(
                                            model=m_ea,
                                            X_source=z_tr_ea_inner,
                                            y_source=y_tr_inner,
                                            X_target=z_p_ea,
                                            class_order=class_labels,
                                            seed=seed_pseudo_base + 101,
                                            kappa=float(oea_zo_iwcv_kappa),
                                        )
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            rec_id_c = None

                    def _feats_for_stack(rec: dict) -> tuple[np.ndarray, tuple[str, ...]]:
                        if not use_stack:
                            return candidate_features_from_record(rec, n_classes=len(class_labels), include_pbar=True)
                        if str(stack_feature_set) == "base":
                            return candidate_features_from_record(rec, n_classes=len(class_labels), include_pbar=True)
                        if str(stack_feature_set) == "base_delta":
                            if rec_id_c is None:
                                raise RuntimeError("stack_feature_set=base_delta requires a valid identity record.")
                            return candidate_features_delta_from_records(
                                rec,
                                anchor=rec_id_c,
                                n_classes=len(class_labels),
                                include_pbar=True,
                            )
                        if str(stack_feature_set) == "stacked_delta":
                            if rec_id_c is None:
                                raise RuntimeError("stack_feature_set=stacked_delta requires a valid identity record.")
                            return stacked_candidate_features_delta_from_records(
                                rec,
                                anchor=rec_id_c,
                                n_classes=len(class_labels),
                                include_pbar=True,
                            )
                        return stacked_candidate_features_from_record(rec, n_classes=len(class_labels), include_pbar=True)

                    # Identity action for bandit training (reward=0).
                    if use_bandit and rec_id_c is not None:
                        try:
                            feats_id, names = _feats_for_stack(rec_id_c)
                            if feat_names is None:
                                feat_names = names
                            _add_bandit_sample(gid=int(pseudo_t), feats=feats_id, reward=0.0)
                        except Exception:
                            pass

                    # EA-FBCSP candidate (EA view).
                    if m_fbcsp is not None:
                        try:
                            p_fbcsp = _reorder_proba_columns(
                                m_fbcsp.predict_proba(z_p_ea), m_fbcsp.classes_, list(class_labels)
                            )
                            acc_fbcsp = float(accuracy_score(y_p, np.asarray(m_fbcsp.predict(z_p_ea))))
                            improve_fbcsp = float(acc_fbcsp - acc_id)
                            feats_fbcsp_c = _features_before_lda(model=m_fbcsp, X=z_p_ea) if use_stack else None
                            rec_fbcsp = _record_for_candidate(
                                p_id=p_id,
                                p_c=p_fbcsp,
                                feats_c=feats_fbcsp_c,
                                lda_c=(m_fbcsp.pipeline.named_steps["lda"] if use_stack else None),
                                lda_ev=ev_fbcsp_inner,
                                seed_local=seed_pseudo_base + 11,
                            )
                            rec_fbcsp["cand_family"] = "fbcsp"
                            if use_stack and z_tr_ea_inner is not None and y_tr_inner is not None:
                                try:
                                    rec_fbcsp.update(
                                        _iwcv_ucb_stats_for_model(
                                            model=m_fbcsp,
                                            X_source=z_tr_ea_inner,
                                            y_source=y_tr_inner,
                                            X_target=z_p_ea,
                                            class_order=class_labels,
                                            seed=seed_pseudo_base + 111,
                                            kappa=float(oea_zo_iwcv_kappa),
                                        )
                                    )
                                except Exception:
                                    pass
                            feats_fbcsp, names = _feats_for_stack(rec_fbcsp)
                            if feat_names is None:
                                feat_names = names
                            _add_calib_sample(fam="fbcsp", feats=feats_fbcsp, improve=improve_fbcsp)
                            _add_bandit_sample(gid=int(pseudo_t), feats=feats_fbcsp, reward=improve_fbcsp)
                        except Exception:
                            pass

                    # RPA candidate.
                    if include_rpa and m_rpa is not None and subject_data_rpa is not None:
                        z_p_rpa = subject_data_rpa[int(pseudo_t)].X
                        p_rpa = _reorder_proba_columns(m_rpa.predict_proba(z_p_rpa), m_rpa.classes_, list(class_labels))
                        acc_rpa = float(accuracy_score(y_p, np.asarray(m_rpa.predict(z_p_rpa))))
                        improve_rpa = float(acc_rpa - acc_id)
                        feats_rpa_c = _features_before_lda(model=m_rpa, X=z_p_rpa) if use_stack else None
                        rec_rpa = _record_for_candidate(
                            p_id=p_id,
                            p_c=p_rpa,
                            feats_c=feats_rpa_c,
                            lda_c=(m_rpa.pipeline.named_steps["lda"] if use_stack else None),
                            lda_ev=ev_rpa_inner,
                            seed_local=seed_pseudo_base + 22,
                        )
                        rec_rpa["cand_family"] = "rpa"
                        if use_stack and z_tr_rpa_inner is not None and y_tr_inner is not None:
                            try:
                                rec_rpa.update(
                                    _iwcv_ucb_stats_for_model(
                                        model=m_rpa,
                                        X_source=z_tr_rpa_inner,
                                        y_source=y_tr_inner,
                                        X_target=z_p_rpa,
                                        class_order=class_labels,
                                        seed=seed_pseudo_base + 122,
                                        kappa=float(oea_zo_iwcv_kappa),
                                    )
                                )
                            except Exception:
                                pass
                        feats_rpa, names = _feats_for_stack(rec_rpa)
                        if feat_names is None:
                            feat_names = names
                        _add_calib_sample(fam="rpa", feats=feats_rpa, improve=improve_rpa)
                        _add_bandit_sample(gid=int(pseudo_t), feats=feats_rpa, reward=improve_rpa)

                        # TSA candidate (built on the RPA/LEA view).
                        if include_tsa and z_tr_rpa_inner is not None:
                            try:
                                q_tsa = _compute_tsa_target_rotation(
                                    z_train=z_tr_rpa_inner,
                                    y_train=y_tr_inner,
                                    z_target=z_p_rpa,
                                    model=m_rpa,
                                    class_order=class_labels,
                                    pseudo_mode=str(oea_pseudo_mode),
                                    pseudo_iters=int(max(0, oea_pseudo_iters)),
                                    q_blend=float(oea_q_blend),
                                    pseudo_confidence=float(oea_pseudo_confidence),
                                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                                    pseudo_balance=bool(oea_pseudo_balance),
                                    eps=float(oea_eps),
                                    shrinkage=float(oea_shrinkage),
                                )
                                z_p_tsa = apply_spatial_transform(q_tsa, z_p_rpa)
                                p_tsa = _reorder_proba_columns(
                                    m_rpa.predict_proba(z_p_tsa), m_rpa.classes_, list(class_labels)
                                )
                                acc_tsa = float(accuracy_score(y_p, np.asarray(m_rpa.predict(z_p_tsa))))
                                improve_tsa = float(acc_tsa - acc_id)
                                feats_tsa_c = _features_before_lda(model=m_rpa, X=z_p_tsa) if use_stack else None
                                rec_tsa = _record_for_candidate(
                                    p_id=p_id,
                                    p_c=p_tsa,
                                    feats_c=feats_tsa_c,
                                    lda_c=(m_rpa.pipeline.named_steps["lda"] if use_stack else None),
                                    lda_ev=ev_rpa_inner,
                                    seed_local=seed_pseudo_base + 33,
                                )
                                rec_tsa["cand_family"] = "tsa"
                                if use_stack and z_tr_rpa_inner is not None and y_tr_inner is not None:
                                    try:
                                        rec_tsa.update(
                                            _iwcv_ucb_stats_for_model(
                                                model=m_rpa,
                                                X_source=z_tr_rpa_inner,
                                                y_source=y_tr_inner,
                                                X_target=z_p_tsa,
                                                class_order=class_labels,
                                                seed=seed_pseudo_base + 133,
                                                kappa=float(oea_zo_iwcv_kappa),
                                            )
                                        )
                                    except Exception:
                                        pass
                                feats_tsa, names = _feats_for_stack(rec_tsa)
                                if feat_names is None:
                                    feat_names = names
                                _add_calib_sample(fam="tsa", feats=feats_tsa, improve=improve_tsa)
                                _add_bandit_sample(gid=int(pseudo_t), feats=feats_tsa, reward=improve_tsa)
                            except Exception:
                                pass

                    # Tangent-space SVC candidate (EA view -> cov -> TS -> linear SVC).
                    if include_ts_svc:
                        try:
                            info = dict(inner_bundle.get("ts_svc", {}))
                            ts = info.get("ts", None)
                            clf = info.get("clf", None)
                            ev = info.get("evidence", None)
                            if ts is not None and clf is not None:
                                cov_p = _covs_for_riemann(z_p_ea)
                                x_p = np.asarray(ts.transform(cov_p), dtype=np.float64)
                                p_ts = _reorder_proba_columns(
                                    clf.predict_proba(x_p),
                                    getattr(clf, "classes_", np.asarray(class_labels, dtype=object)),
                                    list(class_labels),
                                )
                                acc_ts = float(accuracy_score(y_p, np.asarray(clf.predict(x_p))))
                                improve_ts = float(acc_ts - acc_id)
                                rec_ts = _record_for_candidate(
                                    p_id=p_id,
                                    p_c=p_ts,
                                    feats_c=(x_p if use_stack else None),
                                    lda_c=(clf if use_stack else None),
                                    lda_ev=(ev if use_stack else None),
                                    seed_local=seed_pseudo_base + 44,
                                )
                                rec_ts["cand_family"] = "ts_svc"
                                feats_ts, names = _feats_for_stack(rec_ts)
                                if feat_names is None:
                                    feat_names = names
                                _add_calib_sample(fam="ts_svc", feats=feats_ts, improve=improve_ts)
                                _add_bandit_sample(gid=int(pseudo_t), feats=feats_ts, reward=improve_ts)
                        except Exception:
                            pass

                    # Tangent-space alignment + SVC candidate (TSA in tangent space; pseudo-label rotation).
                    if include_tsa_ts_svc:
                        try:
                            info = dict(inner_bundle.get("tsa_ts_svc", {}))
                            clf = info.get("clf", None)
                            mu_s = np.asarray(info.get("mu_s"), dtype=np.float64)
                            ev = info.get("evidence", None)
                            if clf is not None and mu_s.ndim == 2 and mu_s.shape[0] == len(class_labels):
                                cov_p = _covs_for_riemann(z_p_ea)
                                v_t = _tsa_tangent_vectors_from_covs(cov_p)
                                y_pseudo = np.asarray(clf.predict(v_t), dtype=object)

                                mu_t = np.full_like(mu_s, np.nan)
                                for i, c in enumerate([str(x) for x in class_labels]):
                                    m = y_pseudo == c
                                    if np.any(m):
                                        mu_t[int(i)] = np.mean(v_t[m], axis=0)
                                valid = np.all(np.isfinite(mu_t), axis=1) & np.all(np.isfinite(mu_s), axis=1)

                                r = np.eye(int(v_t.shape[1]), dtype=np.float64)
                                if int(np.sum(valid)) >= 2:
                                    a = mu_t[valid]
                                    b = mu_s[valid]
                                    u, _s, vt = np.linalg.svd(a.T @ b, full_matrices=False)
                                    r = u @ vt

                                v_t_rot = v_t @ r
                                p_tsa_ts = _reorder_proba_columns(
                                    clf.predict_proba(v_t_rot),
                                    getattr(clf, "classes_", np.asarray(class_labels, dtype=object)),
                                    list(class_labels),
                                )
                                acc_tsa_ts = float(accuracy_score(y_p, np.asarray(clf.predict(v_t_rot))))
                                improve_tsa_ts = float(acc_tsa_ts - acc_id)
                                rec_tsa_ts = _record_for_candidate(
                                    p_id=p_id,
                                    p_c=p_tsa_ts,
                                    feats_c=(v_t_rot if use_stack else None),
                                    lda_c=(clf if use_stack else None),
                                    lda_ev=(ev if use_stack else None),
                                    seed_local=seed_pseudo_base + 55,
                                )
                                rec_tsa_ts["cand_family"] = "tsa_ts_svc"
                                feats_tsa_ts, names = _feats_for_stack(rec_tsa_ts)
                                if feat_names is None:
                                    feat_names = names
                                _add_calib_sample(fam="tsa_ts_svc", feats=feats_tsa_ts, improve=improve_tsa_ts)
                                _add_bandit_sample(gid=int(pseudo_t), feats=feats_tsa_ts, reward=improve_tsa_ts)
                        except Exception:
                            pass

                    # FgMDM candidate on covariances (no probe/evidence features yet).
                    if include_fgmdm:
                        try:
                            fg = inner_bundle.get("fgmdm", {}).get("model", None)
                            if fg is not None:
                                cov_p = _covs_for_riemann(z_p_ea)
                                p_fg = _reorder_proba_columns(
                                    fg.predict_proba(cov_p),
                                    getattr(fg, "classes_", np.asarray(class_labels, dtype=object)),
                                    list(class_labels),
                                )
                                acc_fg = float(accuracy_score(y_p, np.asarray(fg.predict(cov_p))))
                                improve_fg = float(acc_fg - acc_id)
                                rec_fg = _record_for_candidate(
                                    p_id=p_id,
                                    p_c=p_fg,
                                    feats_c=None,
                                    lda_c=None,
                                    lda_ev=None,
                                    seed_local=seed_pseudo_base + 66,
                                )
                                rec_fg["cand_family"] = "fgmdm"
                                feats_fg, names = _feats_for_stack(rec_fg)
                                if feat_names is None:
                                    feat_names = names
                                _add_calib_sample(fam="fgmdm", feats=feats_fg, improve=improve_fg)
                                _add_bandit_sample(gid=int(pseudo_t), feats=feats_fg, reward=improve_fg)
                        except Exception:
                            pass

                    # Channel projector candidates (EA view).
                    if include_chan:
                        cand_inner_chan: dict = dict(inner_bundle.get("chan", {}).get("candidates", {}))
                        for cand_key, info in cand_inner_chan.items():
                            A = info["A"]
                            m_A = info["model"]
                            z_p_A = apply_spatial_transform(A, z_p_ea)
                            p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, list(class_labels))
                            acc_A = float(accuracy_score(y_p, np.asarray(m_A.predict(z_p_A))))
                            improve_A = float(acc_A - acc_id)
                            z_tr_A_inner = None
                            ev_A_inner = None
                            if use_stack:
                                try:
                                    z_tr_A_inner = apply_spatial_transform(A, z_tr_ea_inner)
                                    ev_A_inner = _compute_lda_evidence_params(
                                        model=m_A,
                                        X_train=z_tr_A_inner,
                                        y_train=y_tr_inner,
                                        class_order=class_labels,
                                        ridge=float(si_ridge),
                                    )
                                except Exception:
                                    z_tr_A_inner = None
                                    ev_A_inner = None

                            rank_val = int(info.get("rank", 0))
                            lam_val = float(info.get("lambda", 0.0))
                            lam_bin = int(np.round(lam_val * 1000.0))
                            feats_A_c = _features_before_lda(model=m_A, X=z_p_A) if use_stack else None
                            rec_A = _record_for_candidate(
                                p_id=p_id,
                                p_c=p_A,
                                feats_c=feats_A_c,
                                lda_c=(m_A.pipeline.named_steps["lda"] if use_stack else None),
                                lda_ev=ev_A_inner,
                                seed_local=seed_pseudo_base + 1000 + 31 * rank_val + lam_bin,
                            )
                            rec_A["cand_family"] = "chan"
                            rec_A["cand_key"] = cand_key
                            rec_A["cand_rank"] = float(info.get("rank", float("nan")))
                            rec_A["cand_lambda"] = float(info.get("lambda", float("nan")))
                            if use_stack and z_tr_A_inner is not None and y_tr_inner is not None:
                                try:
                                    rec_A.update(
                                        _iwcv_ucb_stats_for_model(
                                            model=m_A,
                                            X_source=z_tr_A_inner,
                                            y_source=y_tr_inner,
                                            X_target=z_p_A,
                                            class_order=class_labels,
                                            seed=seed_pseudo_base + 1000 + 31 * rank_val + lam_bin,
                                            kappa=float(oea_zo_iwcv_kappa),
                                        )
                                    )
                                except Exception:
                                    pass
                            feats_A, names = _feats_for_stack(rec_A)
                            if feat_names is None:
                                feat_names = names
                            _add_calib_sample(fam="chan", feats=feats_A, improve=improve_A)
                            _add_bandit_sample(gid=int(pseudo_t), feats=feats_A, reward=improve_A)

                if use_ridge and X_ridge_rows and feat_names is not None:
                    X_ridge = np.vstack(X_ridge_rows)
                    y_ridge = np.asarray(y_ridge_rows, dtype=np.float64)
                    cert = train_ridge_certificate(
                        X_ridge,
                        y_ridge,
                        feature_names=feat_names,
                        alpha=float(oea_zo_calib_ridge_alpha),
                    )
                    try:
                        pred = np.asarray(cert.predict_accuracy(X_ridge), dtype=np.float64).reshape(-1)
                        if y_ridge.size >= 2:
                            ridge_train_pearson = float(np.corrcoef(pred, y_ridge)[0, 1])
                            ridge_train_spearman = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                    except Exception:
                        pass

                if use_guard and X_guard_rows and feat_names is not None:
                    X_guard = np.vstack(X_guard_rows)
                    y_guard = np.asarray(y_guard_rows, dtype=int)
                    if len(np.unique(y_guard)) >= 2:
                        guard = train_logistic_guard(
                            X_guard,
                            y_guard,
                            feature_names=feat_names,
                            c=float(oea_zo_calib_guard_c),
                        )
                        try:
                            p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                            improve_train = np.asarray(improve_guard_rows, dtype=np.float64).reshape(-1)
                            guard_train_auc = float(roc_auc_score(y_guard, p_train))
                            if improve_train.size == p_train.size and improve_train.size >= 2:
                                guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                                guard_train_spearman = float(
                                    np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1]
                                )
                        except Exception:
                            pass

                # Optional: per-family calibrated models (fallback to global cert/guard if missing).
                if bool(stack_calib_per_family) and feat_names is not None:
                    if use_ridge:
                        for fam, X_list in X_ridge_by_family.items():
                            if not X_list:
                                continue
                            try:
                                X_f = np.vstack(X_list)
                                y_f = np.asarray(y_ridge_by_family.get(fam, []), dtype=np.float64)
                                if X_f.shape[0] != y_f.shape[0] or X_f.shape[0] < 2:
                                    continue
                                family_counts[str(fam)] = int(X_f.shape[0])
                                cert_by_family[str(fam)] = train_ridge_certificate(
                                    X_f,
                                    y_f,
                                    feature_names=feat_names,
                                    alpha=float(oea_zo_calib_ridge_alpha),
                                )
                            except Exception:
                                continue
                    if use_guard:
                        for fam, X_list in X_guard_by_family.items():
                            if not X_list:
                                continue
                            try:
                                X_f = np.vstack(X_list)
                                y_f = np.asarray(y_guard_by_family.get(fam, []), dtype=int)
                                if X_f.shape[0] != y_f.shape[0] or X_f.shape[0] < 4:
                                    continue
                                if np.unique(y_f).size < 2:
                                    continue
                                guard_by_family[str(fam)] = train_logistic_guard(
                                    X_f,
                                    y_f,
                                    feature_names=feat_names,
                                    c=float(oea_zo_calib_guard_c),
                                )
                            except Exception:
                                continue

                # Optional: bandit policy on stacked features (full-information pseudo-target rewards).
                if use_bandit and X_bandit_rows and feat_names is not None:
                    try:
                        X_b = np.vstack(X_bandit_rows)
                        r_b = np.asarray(r_bandit_rows, dtype=np.float64).reshape(-1)
                        g_b = np.asarray(g_bandit_rows, dtype=int).reshape(-1)
                        bandit_policy = train_softmax_bandit_policy(
                            X_b,
                            rewards=r_b,
                            group_ids=g_b,
                            feature_names=feat_names,
                            l2=float(oea_zo_calib_ridge_alpha),
                            lr=0.1,
                            iters=300,
                            seed=int(oea_zo_calib_seed) + int(test_subject) * 997 + 17,
                        )
                    except Exception:
                        bandit_policy = None

            # Build target-subject candidate records (unlabeled).
            X_test_ea = subject_data[int(test_subject)].X
            y_test = subject_data[int(test_subject)].y
            X_test_rpa = (
                subject_data_rpa[int(test_subject)].X if need_rpa_view and subject_data_rpa is not None else None
            )

            z_tr_ea_outer = np.concatenate([subject_data[int(s)].X for s in train_subjects], axis=0)
            y_tr_outer = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
            z_tr_rpa_outer = (
                np.concatenate([subject_data_rpa[int(s)].X for s in train_subjects], axis=0)
                if need_rpa_view and subject_data_rpa is not None
                else None
            )

            ev_ea_outer = None
            ev_rpa_outer = None
            ev_fbcsp_outer = None
            if use_stack:
                try:
                    ev_ea_outer = _compute_lda_evidence_params(
                        model=model_ea,
                        X_train=z_tr_ea_outer,
                        y_train=y_tr_outer,
                        class_order=class_labels,
                        ridge=float(si_ridge),
                    )
                except Exception:
                    ev_ea_outer = None
                if model_rpa is not None and z_tr_rpa_outer is not None:
                    try:
                        ev_rpa_outer = _compute_lda_evidence_params(
                            model=model_rpa,
                            X_train=z_tr_rpa_outer,
                            y_train=y_tr_outer,
                            class_order=class_labels,
                            ridge=float(si_ridge),
                        )
                    except Exception:
                        ev_rpa_outer = None

            p_id_t = _reorder_proba_columns(model_ea.predict_proba(X_test_ea), model_ea.classes_, list(class_labels))
            feats_id_t = _features_before_lda(model=model_ea, X=X_test_ea) if use_stack else None
            rec_id = _record_for_candidate(
                p_id=p_id_t,
                p_c=p_id_t,
                feats_c=feats_id_t,
                lda_c=(model_ea.pipeline.named_steps["lda"] if use_stack else None),
                lda_ev=ev_ea_outer,
                seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 1,
            )
            rec_id["kind"] = "identity"
            rec_id["cand_family"] = "ea"
            if use_stack:
                try:
                    rec_id.update(
                        _iwcv_ucb_stats_for_model(
                            model=model_ea,
                            X_source=z_tr_ea_outer,
                            y_source=y_tr_outer,
                            X_target=X_test_ea,
                            class_order=class_labels,
                            seed=int(oea_zo_seed) + int(test_subject) * 997 + 101,
                            kappa=float(oea_zo_iwcv_kappa),
                        )
                    )
                except Exception:
                    pass
            if bool(do_diag):
                y_hat = np.asarray([class_labels[int(i)] for i in np.argmax(p_id_t, axis=1)], dtype=object)
                rec_id["accuracy"] = float(accuracy_score(y_test, y_hat))
            records: list[dict] = [rec_id]

            # EA-FBCSP candidate.
            model_fbcsp = None
            if include_fbcsp:
                try:
                    model_fbcsp = outer_bundle["fbcsp"]["model"]
                    if use_stack and ev_fbcsp_outer is None:
                        try:
                            ev_fbcsp_outer = _compute_lda_evidence_params(
                                model=model_fbcsp,
                                X_train=z_tr_ea_outer,
                                y_train=y_tr_outer,
                                class_order=class_labels,
                                ridge=float(si_ridge),
                            )
                        except Exception:
                            ev_fbcsp_outer = None
                    p_fbcsp_t = _reorder_proba_columns(
                        model_fbcsp.predict_proba(X_test_ea), model_fbcsp.classes_, list(class_labels)
                    )
                    feats_fbcsp_t = _features_before_lda(model=model_fbcsp, X=X_test_ea) if use_stack else None
                    rec_fbcsp_t = _record_for_candidate(
                        p_id=p_id_t,
                        p_c=p_fbcsp_t,
                        feats_c=feats_fbcsp_t,
                        lda_c=(model_fbcsp.pipeline.named_steps["lda"] if use_stack else None),
                        lda_ev=ev_fbcsp_outer,
                        seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 2,
                    )
                    rec_fbcsp_t["cand_family"] = "fbcsp"
                    if use_stack:
                        try:
                            rec_fbcsp_t.update(
                                _iwcv_ucb_stats_for_model(
                                    model=model_fbcsp,
                                    X_source=z_tr_ea_outer,
                                    y_source=y_tr_outer,
                                    X_target=X_test_ea,
                                    class_order=class_labels,
                                    seed=int(oea_zo_seed) + int(test_subject) * 997 + 111,
                                    kappa=float(oea_zo_iwcv_kappa),
                                )
                            )
                        except Exception:
                            pass
                    if bool(do_diag):
                        y_hat = np.asarray([class_labels[int(i)] for i in np.argmax(p_fbcsp_t, axis=1)], dtype=object)
                        rec_fbcsp_t["accuracy"] = float(accuracy_score(y_test, y_hat))
                    records.append(rec_fbcsp_t)
                except Exception:
                    model_fbcsp = None

            # RPA candidate.
            if include_rpa and model_rpa is not None and X_test_rpa is not None:
                p_rpa_t = _reorder_proba_columns(
                    model_rpa.predict_proba(X_test_rpa), model_rpa.classes_, list(class_labels)
                )
                feats_rpa_t = _features_before_lda(model=model_rpa, X=X_test_rpa) if use_stack else None
                rec_rpa_t = _record_for_candidate(
                    p_id=p_id_t,
                    p_c=p_rpa_t,
                    feats_c=feats_rpa_t,
                    lda_c=(model_rpa.pipeline.named_steps["lda"] if use_stack else None),
                    lda_ev=ev_rpa_outer,
                    seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 3,
                )
                rec_rpa_t["cand_family"] = "rpa"
                if use_stack and z_tr_rpa_outer is not None:
                    try:
                        rec_rpa_t.update(
                            _iwcv_ucb_stats_for_model(
                                model=model_rpa,
                                X_source=z_tr_rpa_outer,
                                y_source=y_tr_outer,
                                X_target=X_test_rpa,
                                class_order=class_labels,
                                seed=int(oea_zo_seed) + int(test_subject) * 997 + 122,
                                kappa=float(oea_zo_iwcv_kappa),
                            )
                        )
                    except Exception:
                        pass
                if bool(do_diag):
                    y_hat = np.asarray([class_labels[int(i)] for i in np.argmax(p_rpa_t, axis=1)], dtype=object)
                    rec_rpa_t["accuracy"] = float(accuracy_score(y_test, y_hat))
                records.append(rec_rpa_t)

            # TSA candidate.
            X_test_tsa = None
            if (
                include_tsa
                and model_rpa is not None
                and X_test_rpa is not None
                and z_tr_rpa_outer is not None
            ):
                try:
                    q_tsa = _compute_tsa_target_rotation(
                        z_train=z_tr_rpa_outer,
                        y_train=y_tr_outer,
                        z_target=X_test_rpa,
                        model=model_rpa,
                        class_order=class_labels,
                        pseudo_mode=str(oea_pseudo_mode),
                        pseudo_iters=int(max(0, oea_pseudo_iters)),
                        q_blend=float(oea_q_blend),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                    )
                    X_test_tsa = apply_spatial_transform(q_tsa, X_test_rpa)
                    p_tsa_t = _reorder_proba_columns(
                        model_rpa.predict_proba(X_test_tsa), model_rpa.classes_, list(class_labels)
                    )
                    feats_tsa_t = _features_before_lda(model=model_rpa, X=X_test_tsa) if use_stack else None
                    rec_tsa_t = _record_for_candidate(
                        p_id=p_id_t,
                        p_c=p_tsa_t,
                        feats_c=feats_tsa_t,
                        lda_c=(model_rpa.pipeline.named_steps["lda"] if use_stack else None),
                        lda_ev=ev_rpa_outer,
                        seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 4,
                    )
                    rec_tsa_t["cand_family"] = "tsa"
                    rec_tsa_t["tsa_q_blend"] = float(oea_q_blend)
                    if use_stack and z_tr_rpa_outer is not None:
                        try:
                            rec_tsa_t.update(
                                _iwcv_ucb_stats_for_model(
                                    model=model_rpa,
                                    X_source=z_tr_rpa_outer,
                                    y_source=y_tr_outer,
                                    X_target=X_test_tsa,
                                    class_order=class_labels,
                                    seed=int(oea_zo_seed) + int(test_subject) * 997 + 133,
                                    kappa=float(oea_zo_iwcv_kappa),
                                )
                            )
                        except Exception:
                            pass
                    if bool(do_diag):
                        y_hat = np.asarray([class_labels[int(i)] for i in np.argmax(p_tsa_t, axis=1)], dtype=object)
                        rec_tsa_t["accuracy"] = float(accuracy_score(y_test, y_hat))
                    records.append(rec_tsa_t)
                except Exception:
                    X_test_tsa = None

            # Tangent-space SVC candidate (EA view -> cov -> TS -> linear SVC).
            if include_ts_svc:
                try:
                    info = dict(outer_bundle.get("ts_svc", {}))
                    ts = info.get("ts", None)
                    clf = info.get("clf", None)
                    ev = info.get("evidence", None)
                    if ts is not None and clf is not None:
                        cov_t = _covs_for_riemann(X_test_ea)
                        x_t = np.asarray(ts.transform(cov_t), dtype=np.float64)
                        p_ts_t = _reorder_proba_columns(
                            clf.predict_proba(x_t),
                            getattr(clf, "classes_", np.asarray(class_labels, dtype=object)),
                            list(class_labels),
                        )
                        rec_ts_t = _record_for_candidate(
                            p_id=p_id_t,
                            p_c=p_ts_t,
                            feats_c=(x_t if use_stack else None),
                            lda_c=(clf if use_stack else None),
                            lda_ev=(ev if use_stack else None),
                            seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 5,
                        )
                        rec_ts_t["cand_family"] = "ts_svc"
                        if bool(do_diag):
                            y_hat = np.asarray(clf.predict(x_t), dtype=object)
                            rec_ts_t["accuracy"] = float(accuracy_score(y_test, y_hat))
                        records.append(rec_ts_t)
                except Exception:
                    pass

            # Tangent-space alignment + SVC candidate (TSA in tangent space; pseudo-label rotation).
            if include_tsa_ts_svc:
                try:
                    info = dict(outer_bundle.get("tsa_ts_svc", {}))
                    clf = info.get("clf", None)
                    mu_s = np.asarray(info.get("mu_s"), dtype=np.float64)
                    ev = info.get("evidence", None)
                    if clf is not None and mu_s.ndim == 2 and mu_s.shape[0] == len(class_labels):
                        cov_t = _covs_for_riemann(X_test_ea)
                        v_t = _tsa_tangent_vectors_from_covs(cov_t)
                        y_pseudo = np.asarray(clf.predict(v_t), dtype=object)

                        mu_t = np.full_like(mu_s, np.nan)
                        for i, c in enumerate([str(x) for x in class_labels]):
                            m = y_pseudo == c
                            if np.any(m):
                                mu_t[int(i)] = np.mean(v_t[m], axis=0)
                        valid = np.all(np.isfinite(mu_t), axis=1) & np.all(np.isfinite(mu_s), axis=1)

                        r = np.eye(int(v_t.shape[1]), dtype=np.float64)
                        if int(np.sum(valid)) >= 2:
                            a = mu_t[valid]
                            b = mu_s[valid]
                            u, _s, vt = np.linalg.svd(a.T @ b, full_matrices=False)
                            r = u @ vt

                        v_t_rot = v_t @ r
                        p_tsa_ts_t = _reorder_proba_columns(
                            clf.predict_proba(v_t_rot),
                            getattr(clf, "classes_", np.asarray(class_labels, dtype=object)),
                            list(class_labels),
                        )
                        rec_tsa_ts_t = _record_for_candidate(
                            p_id=p_id_t,
                            p_c=p_tsa_ts_t,
                            feats_c=(v_t_rot if use_stack else None),
                            lda_c=(clf if use_stack else None),
                            lda_ev=(ev if use_stack else None),
                            seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 6,
                        )
                        rec_tsa_ts_t["cand_family"] = "tsa_ts_svc"
                        if bool(do_diag):
                            y_hat = np.asarray(clf.predict(v_t_rot), dtype=object)
                            rec_tsa_ts_t["accuracy"] = float(accuracy_score(y_test, y_hat))
                        records.append(rec_tsa_ts_t)
                except Exception:
                    pass

            # FgMDM candidate on covariances (no probe/evidence features yet).
            if include_fgmdm:
                try:
                    fg = outer_bundle.get("fgmdm", {}).get("model", None)
                    if fg is not None:
                        cov_t = _covs_for_riemann(X_test_ea)
                        p_fg_t = _reorder_proba_columns(
                            fg.predict_proba(cov_t),
                            getattr(fg, "classes_", np.asarray(class_labels, dtype=object)),
                            list(class_labels),
                        )
                        rec_fg_t = _record_for_candidate(
                            p_id=p_id_t,
                            p_c=p_fg_t,
                            feats_c=None,
                            lda_c=None,
                            lda_ev=None,
                            seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 7,
                        )
                        rec_fg_t["cand_family"] = "fgmdm"
                        if bool(do_diag):
                            y_hat = np.asarray(fg.predict(cov_t), dtype=object)
                            rec_fg_t["accuracy"] = float(accuracy_score(y_test, y_hat))
                        records.append(rec_fg_t)
                except Exception:
                    pass

            # Channel projector candidates.
            ev_chan_outer: dict = {}
            for cand_key, info in (chan_outer.items() if include_chan else []):
                A = info["A"]
                m_A = info["model"]
                X_test_A = apply_spatial_transform(A, X_test_ea)
                p_A_t = _reorder_proba_columns(m_A.predict_proba(X_test_A), m_A.classes_, list(class_labels))
                z_tr_A_outer = None
                ev_A_outer = None
                if use_stack:
                    if cand_key in ev_chan_outer:
                        ev_A_outer = ev_chan_outer[cand_key]
                    else:
                        try:
                            z_tr_A_outer = apply_spatial_transform(A, z_tr_ea_outer)
                            ev_A_outer = _compute_lda_evidence_params(
                                model=m_A,
                                X_train=z_tr_A_outer,
                                y_train=y_tr_outer,
                                class_order=class_labels,
                                ridge=float(si_ridge),
                            )
                        except Exception:
                            z_tr_A_outer = None
                            ev_A_outer = None
                        ev_chan_outer[cand_key] = ev_A_outer

                rank_val = int(info.get("rank", 0))
                lam_val = float(info.get("lambda", 0.0))
                lam_bin = int(np.round(lam_val * 1000.0))
                feats_A_t = _features_before_lda(model=m_A, X=X_test_A) if use_stack else None
                rec = _record_for_candidate(
                    p_id=p_id_t,
                    p_c=p_A_t,
                    feats_c=feats_A_t,
                    lda_c=(m_A.pipeline.named_steps["lda"] if use_stack else None),
                    lda_ev=ev_A_outer,
                    seed_local=int(oea_zo_seed) + int(test_subject) * 997 + 1000 + 31 * rank_val + lam_bin,
                )
                rec["cand_family"] = "chan"
                rec["cand_key"] = cand_key
                rec["cand_rank"] = float(info.get("rank", float("nan")))
                rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                if use_stack and y_tr_outer is not None:
                    try:
                        if z_tr_A_outer is None:
                            z_tr_A_outer = apply_spatial_transform(A, z_tr_ea_outer)
                        rec.update(
                            _iwcv_ucb_stats_for_model(
                                model=m_A,
                                X_source=z_tr_A_outer,
                                y_source=y_tr_outer,
                                X_target=X_test_A,
                                class_order=class_labels,
                                seed=int(oea_zo_seed) + int(test_subject) * 997 + 1000 + 31 * rank_val + lam_bin,
                                kappa=float(oea_zo_iwcv_kappa),
                            )
                        )
                    except Exception:
                        pass
                if bool(do_diag):
                    y_hat = np.asarray([class_labels[int(i)] for i in np.argmax(p_A_t, axis=1)], dtype=object)
                    rec["accuracy"] = float(accuracy_score(y_test, y_hat))
                records.append(rec)

                selected = rec_id
            if selector == "calibrated_stack_bandit_guard" and bandit_policy is not None and guard is not None:
                # For diagnostics/safety gates we still compute ridge-predicted improvements when available.
                if cert is not None:
                    try:
                        for rec in records:
                            if str(stack_feature_set) == "stacked_delta":
                                feats, _ = stacked_candidate_features_delta_from_records(
                                    rec,
                                    anchor=rec_id,
                                    n_classes=len(class_labels),
                                    include_pbar=True,
                                )
                            elif str(stack_feature_set) == "base_delta":
                                feats, _ = candidate_features_delta_from_records(
                                    rec,
                                    anchor=rec_id,
                                    n_classes=len(class_labels),
                                    include_pbar=True,
                                )
                            elif str(stack_feature_set) == "base":
                                feats, _ = candidate_features_from_record(
                                    rec,
                                    n_classes=len(class_labels),
                                    include_pbar=True,
                                )
                            else:
                                feats, _ = stacked_candidate_features_from_record(
                                    rec, n_classes=len(class_labels), include_pbar=True
                                )
                            rec["ridge_pred_improve"] = float(cert.predict_accuracy(feats)[0])
                    except Exception:
                        pass

                selected = select_by_guarded_bandit_policy(
                    records,
                    policy=bandit_policy,
                    guard=guard,
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    anchor_guard_delta=float(stack_safe_anchor_guard_delta),
                    anchor_probe_hard_worsen=float(stack_safe_anchor_probe_hard_worsen),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set=str(stack_feature_set),
                )
            elif selector == "calibrated_stack_ridge_guard" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    cert_by_family=(cert_by_family if bool(stack_calib_per_family) else None),
                    guard_by_family=(guard_by_family if bool(stack_calib_per_family) else None),
                    family_counts=(family_counts if bool(stack_calib_per_family) else None),
                    family_blend_mode=str(stack_calib_per_family_mode),
                    family_shrinkage=float(stack_calib_per_family_shrinkage),
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    anchor_guard_delta=float(stack_safe_anchor_guard_delta),
                    anchor_probe_hard_worsen=float(stack_safe_anchor_probe_hard_worsen),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set=str(stack_feature_set),
                )
            elif selector == "calibrated_stack_ridge_guard_borda" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    cert_by_family=(cert_by_family if bool(stack_calib_per_family) else None),
                    guard_by_family=(guard_by_family if bool(stack_calib_per_family) else None),
                    family_counts=(family_counts if bool(stack_calib_per_family) else None),
                    family_blend_mode=str(stack_calib_per_family_mode),
                    family_shrinkage=float(stack_calib_per_family_shrinkage),
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    anchor_guard_delta=float(stack_safe_anchor_guard_delta),
                    anchor_probe_hard_worsen=float(stack_safe_anchor_probe_hard_worsen),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set=str(stack_feature_set),
                    score_mode="borda_ridge_probe",
                )
            elif selector == "calibrated_stack_ridge_guard_borda3" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    cert_by_family=(cert_by_family if bool(stack_calib_per_family) else None),
                    guard_by_family=(guard_by_family if bool(stack_calib_per_family) else None),
                    family_counts=(family_counts if bool(stack_calib_per_family) else None),
                    family_blend_mode=str(stack_calib_per_family_mode),
                    family_shrinkage=float(stack_calib_per_family_shrinkage),
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    anchor_guard_delta=float(stack_safe_anchor_guard_delta),
                    anchor_probe_hard_worsen=float(stack_safe_anchor_probe_hard_worsen),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set=str(stack_feature_set),
                    score_mode="borda_ridge_probe_iwcv",
                )
            elif selector == "calibrated_ridge_guard" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    cert_by_family=(cert_by_family if bool(stack_calib_per_family) else None),
                    guard_by_family=(guard_by_family if bool(stack_calib_per_family) else None),
                    family_counts=(family_counts if bool(stack_calib_per_family) else None),
                    family_blend_mode=str(stack_calib_per_family_mode),
                    family_shrinkage=float(stack_calib_per_family_shrinkage),
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    anchor_guard_delta=float(stack_safe_anchor_guard_delta),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "calibrated_stack_ridge" and cert is not None:
                selected = select_by_predicted_improvement(
                    records,
                    cert=cert,
                    n_classes=len(class_labels),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set=str(stack_feature_set),
                )
            elif selector == "calibrated_ridge" and cert is not None:
                selected = select_by_predicted_improvement(
                    records,
                    cert=cert,
                    n_classes=len(class_labels),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set="base",
                )
            elif selector == "calibrated_guard" and guard is not None:
                selected = select_by_guarded_objective(
                    records,
                    guard=guard,
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "prefer_fbcsp":
                # Lightweight policy: prefer the (single) FBCSP candidate when available,
                # then let the family-specific safety gates decide accept/fallback.
                fbcsp_recs = [
                    r
                    for r in records
                    if str(r.get("cand_family", "")).lower() == "fbcsp" and str(r.get("kind", "")) != "identity"
                ]
                selected = fbcsp_recs[0] if fbcsp_recs else rec_id
            elif selector == "objective":
                best = min(records, key=lambda r: float(r.get("score", r.get("objective_base", 0.0))))
                selected = best

            # Family-specific high-risk gate: treat FBCSP as risky and enforce stricter acceptance rules.
            pre_family = str(selected.get("cand_family", "ea"))
            pre_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            pre_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            pre_drift = float(selected.get("drift_best", float("nan")))
            pre_pred_disagree = float(selected.get("pred_disagree", float("nan")))
            anchor_guard_pos = float(rec_id.get("guard_p_pos", float("nan")))
            anchor_probe_hard = float(rec_id.get("probe_mixup_hard_best", float("nan")))
            base_thr = float(oea_zo_calib_guard_threshold)
            anchor_thr = float(base_thr)
            if float(stack_safe_anchor_guard_delta) > 0.0 and np.isfinite(anchor_guard_pos):
                anchor_thr = max(float(anchor_thr), float(anchor_guard_pos) + float(stack_safe_anchor_guard_delta))
            probe_thr = float("nan")
            if float(stack_safe_anchor_probe_hard_worsen) > -1.0 and np.isfinite(anchor_probe_hard):
                # Allow both modes:
                # - eps >= 0: do-not-worsen   => h(c) <= h(EA) + eps
                # - eps <  0: min-improve     => h(c) <= h(EA) + eps  (i.e., h(c) <= h(EA) - |eps|)
                probe_thr = float(anchor_probe_hard) + float(stack_safe_anchor_probe_hard_worsen)
            fbcsp_blocked = 0
            fbcsp_block_reason = ""
            tsa_blocked = 0
            tsa_block_reason = ""
            min_pred_blocked = 0
            min_pred_block_reason = ""
            fbcsp_gate_active = (
                float(stack_safe_fbcsp_guard_threshold) >= 0.0
                or float(stack_safe_fbcsp_min_pred_improve) > 0.0
                or float(stack_safe_fbcsp_drift_delta) > 0.0
                or float(stack_safe_fbcsp_max_pred_disagree) >= 0.0
            )
            apply_fbcsp_gate = selector in {
                "calibrated_ridge_guard",
                "calibrated_stack_ridge_guard",
                "calibrated_stack_ridge_guard_borda",
                "calibrated_stack_ridge_guard_borda3",
                "calibrated_stack_bandit_guard",
                "prefer_fbcsp",
            }
            if (
                fbcsp_gate_active
                and pre_family == "fbcsp"
                and str(selected.get("kind", "")) != "identity"
                and apply_fbcsp_gate
            ):
                base_thr = float(oea_zo_calib_guard_threshold)
                fbcsp_thr = (
                    max(base_thr, float(stack_safe_fbcsp_guard_threshold))
                    if float(stack_safe_fbcsp_guard_threshold) >= 0.0
                    else base_thr
                )
                reasons: list[str] = []
                if np.isfinite(pre_guard_pos) and float(pre_guard_pos) < float(fbcsp_thr):
                    reasons.append("guard")
                if float(stack_safe_fbcsp_min_pred_improve) > 0.0 and (
                    not np.isfinite(pre_ridge_pred)
                    or float(pre_ridge_pred) < float(stack_safe_fbcsp_min_pred_improve)
                ):
                    reasons.append("min_pred")
                if float(stack_safe_fbcsp_drift_delta) > 0.0 and (
                    not np.isfinite(pre_drift) or float(pre_drift) > float(stack_safe_fbcsp_drift_delta)
                ):
                    reasons.append("drift")
                if float(stack_safe_fbcsp_max_pred_disagree) >= 0.0 and (
                    not np.isfinite(pre_pred_disagree)
                    or float(pre_pred_disagree) > float(stack_safe_fbcsp_max_pred_disagree)
                ):
                    reasons.append("pred_disagree")

                if reasons:
                    fbcsp_blocked = 1
                    fbcsp_block_reason = ",".join(reasons)

                    if selector == "prefer_fbcsp":
                        # With the lightweight policy we simply fall back to EA when FBCSP is rejected.
                        selected = rec_id
                    else:
                        # Re-select among the remaining candidates using already-computed ridge/guard scores.
                        best_alt: dict | None = None
                        best_alt_score = -float("inf")
                        for rec in records:
                            if str(rec.get("kind", "")) == "identity":
                                continue
                            p_pos = float(rec.get("guard_p_pos", float("nan")))
                            if not np.isfinite(p_pos) or float(p_pos) < float(anchor_thr):
                                continue
                            if np.isfinite(probe_thr):
                                probe = float(rec.get("probe_mixup_hard_best", float("nan")))
                                if not np.isfinite(probe) or float(probe) > float(probe_thr):
                                    continue

                            fam = str(rec.get("cand_family", ""))
                            if fam == "fbcsp":
                                if float(stack_safe_fbcsp_guard_threshold) >= 0.0 and float(p_pos) < float(fbcsp_thr):
                                    continue
                                pred = float(rec.get("ridge_pred_improve", float("nan")))
                                if float(stack_safe_fbcsp_min_pred_improve) > 0.0 and (
                                    not np.isfinite(pred) or float(pred) < float(stack_safe_fbcsp_min_pred_improve)
                                ):
                                    continue
                                drift = float(rec.get("drift_best", 0.0))
                                if float(stack_safe_fbcsp_drift_delta) > 0.0 and float(drift) > float(
                                    stack_safe_fbcsp_drift_delta
                                ):
                                    continue
                                if float(stack_safe_fbcsp_max_pred_disagree) >= 0.0:
                                    disagree = float(rec.get("pred_disagree", float("nan")))
                                    if not np.isfinite(disagree) or float(disagree) > float(stack_safe_fbcsp_max_pred_disagree):
                                        continue

                            drift = float(rec.get("drift_best", 0.0))
                            if str(oea_zo_drift_mode) == "hard" and float(oea_zo_drift_delta) > 0.0 and float(
                                drift
                            ) > float(oea_zo_drift_delta):
                                continue

                            pred = float(rec.get("ridge_pred_improve", float("nan")))
                            if not np.isfinite(pred):
                                continue
                            score = float(pred)
                            if str(oea_zo_drift_mode) == "penalty" and float(oea_zo_drift_gamma) > 0.0:
                                score = float(score) - float(oea_zo_drift_gamma) * float(drift)

                            if float(score) > float(best_alt_score):
                                best_alt_score = float(score)
                                best_alt = rec

                        if best_alt is not None and float(best_alt_score) > 0.0:
                            selected = best_alt
                        else:
                            selected = rec_id

            # Family-specific high-risk gate: treat TSA as risky and enforce stricter acceptance rules.
            tsa_gate_active = (
                float(stack_safe_tsa_guard_threshold) >= 0.0
                or float(stack_safe_tsa_min_pred_improve) > 0.0
                or float(stack_safe_tsa_drift_delta) > 0.0
            )
            if (
                tsa_gate_active
                and str(selected.get("cand_family", "")) == "tsa"
                and str(selected.get("kind", "")) != "identity"
                and selector in {
                    "calibrated_ridge_guard",
                    "calibrated_stack_ridge_guard",
                    "calibrated_stack_ridge_guard_borda",
                    "calibrated_stack_ridge_guard_borda3",
                    "calibrated_stack_bandit_guard",
                }
            ):
                base_thr = float(oea_zo_calib_guard_threshold)
                tsa_thr = (
                    max(base_thr, float(stack_safe_tsa_guard_threshold))
                    if float(stack_safe_tsa_guard_threshold) >= 0.0
                    else base_thr
                )
                reasons: list[str] = []
                cur_guard_pos = float(selected.get("guard_p_pos", float("nan")))
                cur_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
                cur_drift = float(selected.get("drift_best", float("nan")))
                if np.isfinite(cur_guard_pos) and float(cur_guard_pos) < float(tsa_thr):
                    reasons.append("guard")
                if float(stack_safe_tsa_min_pred_improve) > 0.0 and (
                    not np.isfinite(cur_ridge_pred) or float(cur_ridge_pred) < float(stack_safe_tsa_min_pred_improve)
                ):
                    reasons.append("min_pred")
                if float(stack_safe_tsa_drift_delta) > 0.0 and (
                    not np.isfinite(cur_drift) or float(cur_drift) > float(stack_safe_tsa_drift_delta)
                ):
                    reasons.append("drift")

                if reasons:
                    tsa_blocked = 1
                    tsa_block_reason = ",".join(reasons)

                    # Re-select among the remaining candidates using already-computed ridge/guard scores.
                    best_alt: dict | None = None
                    best_alt_score = -float("inf")
                    for rec in records:
                        if str(rec.get("kind", "")) == "identity":
                            continue
                        p_pos = float(rec.get("guard_p_pos", float("nan")))
                        if not np.isfinite(p_pos) or float(p_pos) < float(anchor_thr):
                            continue
                        if np.isfinite(probe_thr):
                            probe = float(rec.get("probe_mixup_hard_best", float("nan")))
                            if not np.isfinite(probe) or float(probe) > float(probe_thr):
                                continue

                        fam = str(rec.get("cand_family", ""))
                        if fam == "fbcsp":
                            # Re-apply FBCSP high-risk gate for alternatives.
                            if fbcsp_gate_active:
                                fbcsp_thr = (
                                    max(base_thr, float(stack_safe_fbcsp_guard_threshold))
                                    if float(stack_safe_fbcsp_guard_threshold) >= 0.0
                                    else base_thr
                                )
                                if float(stack_safe_fbcsp_guard_threshold) >= 0.0 and float(p_pos) < float(fbcsp_thr):
                                    continue
                                pred = float(rec.get("ridge_pred_improve", float("nan")))
                                if float(stack_safe_fbcsp_min_pred_improve) > 0.0 and (
                                    not np.isfinite(pred) or float(pred) < float(stack_safe_fbcsp_min_pred_improve)
                                ):
                                    continue
                                drift = float(rec.get("drift_best", 0.0))
                                if float(stack_safe_fbcsp_drift_delta) > 0.0 and float(drift) > float(
                                    stack_safe_fbcsp_drift_delta
                                ):
                                    continue
                                if float(stack_safe_fbcsp_max_pred_disagree) >= 0.0:
                                    disagree = float(rec.get("pred_disagree", float("nan")))
                                    if not np.isfinite(disagree) or float(disagree) > float(stack_safe_fbcsp_max_pred_disagree):
                                        continue

                        if fam == "tsa" and tsa_gate_active:
                            if float(stack_safe_tsa_guard_threshold) >= 0.0 and float(p_pos) < float(tsa_thr):
                                continue
                            pred = float(rec.get("ridge_pred_improve", float("nan")))
                            if float(stack_safe_tsa_min_pred_improve) > 0.0 and (
                                not np.isfinite(pred) or float(pred) < float(stack_safe_tsa_min_pred_improve)
                            ):
                                continue
                            drift = float(rec.get("drift_best", 0.0))
                            if float(stack_safe_tsa_drift_delta) > 0.0 and float(drift) > float(stack_safe_tsa_drift_delta):
                                continue

                        drift = float(rec.get("drift_best", 0.0))
                        if str(oea_zo_drift_mode) == "hard" and float(oea_zo_drift_delta) > 0.0 and float(
                            drift
                        ) > float(oea_zo_drift_delta):
                            continue

                        pred = float(rec.get("ridge_pred_improve", float("nan")))
                        if not np.isfinite(pred):
                            continue
                        score = float(pred)
                        if str(oea_zo_drift_mode) == "penalty" and float(oea_zo_drift_gamma) > 0.0:
                            score = float(score) - float(oea_zo_drift_gamma) * float(drift)

                        if float(score) > float(best_alt_score):
                            best_alt_score = float(score)
                            best_alt = rec

                    if best_alt is not None and float(best_alt_score) > 0.0:
                        selected = best_alt
                    else:
                        selected = rec_id

            # Global gate: require a minimum (blended) ridge-predicted improvement for *all* non-identity candidates.
            # This is a lightweight multiple-testing correction: if the selector can't predict a sufficiently large gain,
            # we fall back to the EA(anchor) solution.
            if (
                float(stack_safe_min_pred_improve) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and selector
                in {
                    "calibrated_ridge_guard",
                    "calibrated_stack_ridge_guard",
                    "calibrated_stack_ridge_guard_borda",
                    "calibrated_stack_ridge_guard_borda3",
                    "calibrated_stack_bandit_guard",
                    "calibrated_stack_ridge",
                }
            ):
                cur_pred = float(selected.get("ridge_pred_improve", float("nan")))
                if not np.isfinite(cur_pred) or float(cur_pred) < float(stack_safe_min_pred_improve):
                    min_pred_blocked = 1
                    min_pred_block_reason = "min_pred"

                    best_alt: dict | None = None
                    best_alt_score = -float("inf")
                    for rec in records:
                        if str(rec.get("kind", "")) == "identity":
                            continue

                        p_pos = float(rec.get("guard_p_pos", float("nan")))
                        if not np.isfinite(p_pos) or float(p_pos) < float(anchor_thr):
                            continue

                        if np.isfinite(probe_thr):
                            probe = float(rec.get("probe_mixup_hard_best", float("nan")))
                            if not np.isfinite(probe) or float(probe) > float(probe_thr):
                                continue

                        pred = float(rec.get("ridge_pred_improve", float("nan")))
                        if not np.isfinite(pred) or float(pred) < float(stack_safe_min_pred_improve):
                            continue

                        fam = str(rec.get("cand_family", ""))
                        if fam == "fbcsp" and fbcsp_gate_active:
                            fbcsp_thr = (
                                max(base_thr, float(stack_safe_fbcsp_guard_threshold))
                                if float(stack_safe_fbcsp_guard_threshold) >= 0.0
                                else base_thr
                            )
                            if float(stack_safe_fbcsp_guard_threshold) >= 0.0 and float(p_pos) < float(fbcsp_thr):
                                continue
                            if float(stack_safe_fbcsp_min_pred_improve) > 0.0 and float(pred) < float(
                                stack_safe_fbcsp_min_pred_improve
                            ):
                                continue
                            if float(stack_safe_fbcsp_max_pred_disagree) >= 0.0:
                                disagree = float(rec.get("pred_disagree", float("nan")))
                                if not np.isfinite(disagree) or float(disagree) > float(stack_safe_fbcsp_max_pred_disagree):
                                    continue

                        if fam == "tsa" and tsa_gate_active:
                            tsa_thr = (
                                max(base_thr, float(stack_safe_tsa_guard_threshold))
                                if float(stack_safe_tsa_guard_threshold) >= 0.0
                                else base_thr
                            )
                            if float(stack_safe_tsa_guard_threshold) >= 0.0 and float(p_pos) < float(tsa_thr):
                                continue
                            if float(stack_safe_tsa_min_pred_improve) > 0.0 and float(pred) < float(
                                stack_safe_tsa_min_pred_improve
                            ):
                                continue

                        drift = float(rec.get("drift_best", 0.0))
                        if fam == "fbcsp" and fbcsp_gate_active and float(stack_safe_fbcsp_drift_delta) > 0.0 and float(
                            drift
                        ) > float(stack_safe_fbcsp_drift_delta):
                            continue
                        if fam == "tsa" and tsa_gate_active and float(stack_safe_tsa_drift_delta) > 0.0 and float(
                            drift
                        ) > float(stack_safe_tsa_drift_delta):
                            continue

                        if str(oea_zo_drift_mode) == "hard" and float(oea_zo_drift_delta) > 0.0 and float(
                            drift
                        ) > float(oea_zo_drift_delta):
                            continue

                        score = float(pred)
                        if str(oea_zo_drift_mode) == "penalty" and float(oea_zo_drift_gamma) > 0.0:
                            score = float(score) - float(oea_zo_drift_gamma) * float(drift)

                        if float(score) > float(best_alt_score):
                            best_alt_score = float(score)
                            best_alt = rec

                    if best_alt is not None and float(best_alt_score) > 0.0:
                        selected = best_alt
                    else:
                        selected = rec_id

            if (
                float(oea_zo_fallback_min_marginal_entropy) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and float(selected.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
            ):
                selected = rec_id

            accept = str(selected.get("kind", "")) != "identity"
            sel_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            sel_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            sel_family = str(selected.get("cand_family", "ea"))
            sel_rank = float(selected.get("cand_rank", float("nan")))
            sel_lam = float(selected.get("cand_lambda", float("nan")))

            # Apply the selected candidate.
            if not accept or sel_family == "ea":
                model = model_ea
                X_test = X_test_ea
            elif sel_family == "fbcsp" and model_fbcsp is not None:
                model = model_fbcsp
                X_test = X_test_ea
            elif sel_family == "rpa":
                model = model_rpa
                X_test = X_test_rpa
            elif sel_family == "tsa" and X_test_tsa is not None:
                model = model_rpa
                X_test = X_test_tsa
            elif sel_family == "chan":
                sel_key = selected.get("cand_key", None)
                if sel_key in chan_outer:
                    A_sel = chan_outer[sel_key]["A"]
                    model = chan_outer[sel_key]["model"]
                    X_test = apply_spatial_transform(A_sel, X_test_ea)
                else:
                    model = model_ea
                    X_test = X_test_ea
            else:
                model = model_ea
                X_test = X_test_ea

            # Analysis-only: compute true improvement for the selected candidate (not used in selection).
            try:
                acc_id_t = float(accuracy_score(y_test, np.asarray(model_ea.predict(X_test_ea))))
            except Exception:
                acc_id_t = float("nan")
            try:
                acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
            except Exception:
                acc_sel_t = float("nan")
            improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

            if extra_rows is not None:
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "stack_multi_accept": int(bool(accept)),
                        "stack_multi_family": str(sel_family),
                        "stack_multi_guard_pos": float(sel_guard_pos),
                        "stack_multi_ridge_pred_improve": float(sel_ridge_pred),
                        "stack_multi_pre_family": str(pre_family),
                        "stack_multi_pre_guard_pos": float(pre_guard_pos),
                        "stack_multi_pre_ridge_pred_improve": float(pre_ridge_pred),
                        "stack_multi_pre_drift_best": float(pre_drift),
                        "stack_multi_fbcsp_blocked": int(fbcsp_blocked),
                        "stack_multi_fbcsp_block_reason": str(fbcsp_block_reason),
                        "stack_multi_tsa_blocked": int(tsa_blocked),
                        "stack_multi_tsa_block_reason": str(tsa_block_reason),
                        "stack_multi_min_pred_blocked": int(min_pred_blocked),
                        "stack_multi_min_pred_block_reason": str(min_pred_block_reason),
                        "stack_multi_acc_anchor": float(acc_id_t),
                        "stack_multi_acc_selected": float(acc_sel_t),
                        "stack_multi_improve": float(improve_t),
                        "stack_multi_sel_rank": float(sel_rank),
                        "stack_multi_sel_lambda": float(sel_lam),
                        "stack_multi_ridge_train_spearman": float(ridge_train_spearman),
                        "stack_multi_ridge_train_pearson": float(ridge_train_pearson),
                        "stack_multi_guard_train_auc": float(guard_train_auc),
                        "stack_multi_guard_train_spearman": float(guard_train_spearman),
                        "stack_multi_guard_train_pearson": float(guard_train_pearson),
                    }
                )

            # Candidate-level diagnostics (analysis-only; uses test labels).
            # Mirrors the EA-ZO diagnostics layout so we can reuse analysis scripts.
            if bool(do_diag) and diagnostics_dir is not None:
                diag_dir = (
                    Path(diagnostics_dir)
                    / "diagnostics"
                    / str(diagnostics_tag)
                    / f"subject_{int(test_subject):02d}"
                )
                diag_dir.mkdir(parents=True, exist_ok=True)
                try:
                    sel_idx = int(records.index(selected))
                except Exception:
                    sel_idx = -1

                rows = []
                for idx, rec in enumerate(records):
                    row = {
                        "idx": int(idx),
                        "is_selected": int(idx == sel_idx),
                        "kind": str(rec.get("kind", "")),
                        "cand_family": str(rec.get("cand_family", "")),
                        "cand_key": str(rec.get("cand_key", "")),
                        "cand_rank": float(rec.get("cand_rank", float("nan"))),
                        "cand_lambda": float(rec.get("cand_lambda", float("nan"))),
                        "objective": float(rec.get("objective", float("nan"))),
                        "score": float(rec.get("score", float("nan"))),
                        "objective_base": float(rec.get("objective_base", float("nan"))),
                        "mean_entropy": float(rec.get("mean_entropy", float("nan"))),
                        "mean_confidence": float(rec.get("mean_confidence", float("nan"))),
                        "entropy_bar": float(rec.get("entropy_bar", float("nan"))),
                        "pred_disagree": float(rec.get("pred_disagree", float("nan"))),
                        "drift_best": float(rec.get("drift_best", float("nan"))),
                        "drift_best_std": float(rec.get("drift_best_std", float("nan"))),
                        "drift_best_q90": float(rec.get("drift_best_q90", float("nan"))),
                        "drift_best_q95": float(rec.get("drift_best_q95", float("nan"))),
                        "drift_best_max": float(rec.get("drift_best_max", float("nan"))),
                        "drift_best_tail_frac": float(rec.get("drift_best_tail_frac", float("nan"))),
                        "evidence_nll_best": float(rec.get("evidence_nll_best", float("nan"))),
                        "evidence_nll_full": float(rec.get("evidence_nll_full", float("nan"))),
                        "iwcv_nll": float(rec.get("iwcv_nll", float("nan"))),
                        "iwcv_ucb": float(rec.get("iwcv_ucb", float("nan"))),
                        "iwcv_eff_n": float(rec.get("iwcv_eff_n", float("nan"))),
                        "iwcv_var": float(rec.get("iwcv_var", float("nan"))),
                        "iwcv_se": float(rec.get("iwcv_se", float("nan"))),
                        "probe_mixup_best": float(rec.get("probe_mixup_best", float("nan"))),
                        "probe_mixup_full": float(rec.get("probe_mixup_full", float("nan"))),
                        "probe_mixup_hard_best": float(rec.get("probe_mixup_hard_best", float("nan"))),
                        "probe_mixup_hard_full": float(rec.get("probe_mixup_hard_full", float("nan"))),
                        "ridge_pred_improve": float(rec.get("ridge_pred_improve", float("nan"))),
                        "guard_p_pos": float(rec.get("guard_p_pos", float("nan"))),
                        "bandit_score": float(rec.get("bandit_score", float("nan"))),
                        "ridge_pred_improve_global": float(rec.get("ridge_pred_improve_global", float("nan"))),
                        "guard_p_pos_global": float(rec.get("guard_p_pos_global", float("nan"))),
                        "ridge_pred_improve_family": float(rec.get("ridge_pred_improve_family", float("nan"))),
                        "guard_p_pos_family": float(rec.get("guard_p_pos_family", float("nan"))),
                        "family_blend_w": float(rec.get("family_blend_w", float("nan"))),
                        "accuracy": float(rec.get("accuracy", float("nan"))),
                    }

                    p_bar = np.asarray(rec.get("p_bar_full", []), dtype=np.float64).reshape(-1)
                    for k, name in enumerate(class_labels):
                        row[f"pbar_{name}"] = float(p_bar[k]) if k < p_bar.shape[0] else float("nan")
                    q_bar = np.asarray(rec.get("q_bar", []), dtype=np.float64).reshape(-1)
                    for k, name in enumerate(class_labels):
                        row[f"qbar_{name}"] = float(q_bar[k]) if k < q_bar.shape[0] else float("nan")

                    rows.append(row)

                pd.DataFrame(rows).to_csv(diag_dir / "candidates.csv", index=False)
        elif alignment == "ea_si":
            # Train on EA-whitened data with a subject-invariant feature projector (Route B),
            # then evaluate directly (no test-time Q_t optimization).
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )
            # Fit CSP once to define feature space, learn projection, then train LDA on projected features.
            from mne.decoding import CSP  # local import to keep module import cost low

            csp = CSP(n_components=int(n_components))
            csp.fit(X_train, y_train)
            feats = np.asarray(csp.transform(X_train), dtype=np.float64)
            proj_params = HSICProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            mean_f, W = learn_hsic_subject_invariant_projector(
                X=feats,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                params=proj_params,
            )
            projector = CenteredLinearProjector(mean=mean_f, W=W)
            model = fit_csp_projected_lda(
                X_train=X_train,
                y_train=y_train,
                projector=projector,
                csp=csp,
                n_components=n_components,
            )
        elif alignment == "ea_si_chan":
            # Channel-space subject-invariant projection (pre-CSP): learn a low-rank projector A (C×C),
            # apply it to both train/test, then train a standard CSP+LDA model.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            chan_params = ChannelProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            A = learn_subject_invariant_channel_projector(
                X=X_train,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )
            X_train = apply_spatial_transform(A, X_train)
            X_test = apply_spatial_transform(A, X_test)
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
        elif alignment == "ea_si_chan_safe":
            # EA anchor vs. channel projector candidate (binary choice). Use a fold-local calibrated guard
            # trained on pseudo-target subjects; otherwise fallback to EA (identity).
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            chan_params = ChannelProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            A = learn_subject_invariant_channel_projector(
                X=X_train,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )

            # Train both models on the same source fold (EA anchor vs projected).
            model_id = fit_csp_lda(X_train, y_train, n_components=n_components)
            X_train_A = apply_spatial_transform(A, X_train)
            model_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)

            # Calibrate a per-fold guard using pseudo-targets from the source subjects.
            rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
            calib_subjects = list(train_subjects)
            if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                rng.shuffle(calib_subjects)
                calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

            X_guard_rows: List[np.ndarray] = []
            y_guard_rows: List[int] = []
            improve_rows: List[float] = []
            feat_names: tuple[str, ...] | None = None

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                return rec

            for pseudo_t in calib_subjects:
                inner_train = [s for s in train_subjects if s != pseudo_t]
                if len(inner_train) < 2:
                    continue

                X_inner = np.concatenate([subject_data[s].X for s in inner_train], axis=0)
                y_inner = np.concatenate([subject_data[s].y for s in inner_train], axis=0)
                subj_inner = np.concatenate(
                    [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in inner_train],
                    axis=0,
                )

                A_inner = learn_subject_invariant_channel_projector(
                    X=X_inner,
                    y=y_inner,
                    subjects=subj_inner,
                    class_order=tuple([str(c) for c in class_order]),
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    params=chan_params,
                )
                m_id = fit_csp_lda(X_inner, y_inner, n_components=n_components)
                X_inner_A = apply_spatial_transform(A_inner, X_inner)
                m_A = fit_csp_lda(X_inner_A, y_inner, n_components=n_components)

                z_p = subject_data[int(pseudo_t)].X
                y_p = subject_data[int(pseudo_t)].y
                p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, list(class_order))
                p_A = _reorder_proba_columns(
                    m_A.predict_proba(apply_spatial_transform(A_inner, z_p)),
                    m_A.classes_,
                    list(class_order),
                )

                yp_id = np.asarray(m_id.predict(z_p))
                yp_A = np.asarray(m_A.predict(apply_spatial_transform(A_inner, z_p)))
                acc_id = float(accuracy_score(y_p, yp_id))
                acc_A = float(accuracy_score(y_p, yp_A))
                improve = float(acc_A - acc_id)

                rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                feats_vec, names = candidate_features_from_record(rec, n_classes=len(class_order), include_pbar=True)
                if feat_names is None:
                    feat_names = names
                X_guard_rows.append(feats_vec)
                y_guard_rows.append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                improve_rows.append(float(improve))

            guard = None
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")
            if X_guard_rows and feat_names is not None:
                X_guard = np.vstack(X_guard_rows)
                y_guard = np.asarray(y_guard_rows, dtype=int)
                # Need both classes to train.
                if len(np.unique(y_guard)) >= 2:
                    guard = train_logistic_guard(
                        X_guard,
                        y_guard,
                        feature_names=feat_names,
                        c=float(oea_zo_calib_guard_c),
                    )
                    try:
                        p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                        improve_train = np.asarray(improve_rows, dtype=np.float64).reshape(-1)
                        guard_train_auc = float(roc_auc_score(y_guard, p_train))
                        if improve_train.size >= 2:
                            guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                            guard_train_spearman = float(
                                np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1]
                            )
                    except Exception:
                        pass

            # Decide whether to accept the projected candidate on the target subject.
            p_id_t = _reorder_proba_columns(model_id.predict_proba(X_test), model_id.classes_, list(class_order))
            X_test_A = apply_spatial_transform(A, X_test)
            p_A_t = _reorder_proba_columns(model_A.predict_proba(X_test_A), model_A.classes_, list(class_order))

            accept = False
            pos = float("nan")
            acc_id_t = float("nan")
            acc_A_t = float("nan")
            improve_t = float("nan")
            if guard is not None:
                rec_t = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                feats_t, _names = candidate_features_from_record(rec_t, n_classes=len(class_order), include_pbar=True)
                pos = float(guard.predict_pos_proba(feats_t)[0])
                accept = pos >= float(oea_zo_calib_guard_threshold)

                # Optional hard drift guard.
                if str(oea_zo_drift_mode) == "hard" and float(oea_zo_drift_delta) > 0.0:
                    drift_mean = float(np.mean(_drift_vec(p_id_t, p_A_t)))
                    if drift_mean > float(oea_zo_drift_delta):
                        accept = False

                # Optional marginal-entropy fallback.
                if float(oea_zo_fallback_min_marginal_entropy) > 0.0 and accept:
                    p_bar = np.mean(np.clip(p_A_t, 1e-12, 1.0), axis=0)
                    p_bar = p_bar / float(np.sum(p_bar))
                    ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))
                    if ent_bar < float(oea_zo_fallback_min_marginal_entropy):
                        accept = False

            # Analysis-only: compute true improvement on the target subject (not used in selection).
            try:
                yp_id_t = np.asarray(model_id.predict(X_test))
                yp_A_t = np.asarray(model_A.predict(X_test_A))
                acc_id_t = float(accuracy_score(y_test, yp_id_t))
                acc_A_t = float(accuracy_score(y_test, yp_A_t))
                improve_t = float(acc_A_t - acc_id_t)
            except Exception:
                pass

            if extra_rows is not None:
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "chan_safe_accept": int(bool(accept)),
                        "chan_safe_guard_pos": float(pos),
                        "chan_safe_acc_anchor": float(acc_id_t),
                        "chan_safe_acc_candidate": float(acc_A_t),
                        "chan_safe_improve": float(improve_t),
                        "chan_safe_guard_train_auc": float(guard_train_auc),
                        "chan_safe_guard_train_spearman": float(guard_train_spearman),
                        "chan_safe_guard_train_pearson": float(guard_train_pearson),
                    }
                )

            # Fallback to EA unless confidently accepted.
            if accept:
                model = model_A
                X_test = X_test_A
            else:
                model = model_id
                X_test = X_test
        elif alignment == "ea_si_chan_multi_safe":
            # Multi-candidate EA-SI-CHAN with calibrated selection (ridge/guard) and safe fallback to EA anchor.
            #
            # Candidate set includes:
            # - identity anchor (A=I)
            # - multiple channel projectors A=QQᵀ learned with different (rank, λ) on the source subjects
            #
            # Selection is performed on the target subject without using target labels.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y
            X_test_raw = X_test

            # For reporting only (n_train) and consistency with other branches.
            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            selector = str(oea_zo_selector)
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}

            outer_bundle = _get_chan_bundle(train_subjects)
            model_id = outer_bundle["model_id"]
            candidates_outer: dict = dict(outer_bundle.get("candidates", {}))

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                # Convenience aliases for selectors that expect `score`/`objective`.
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])
                return rec

            # Calibrate ridge/guard on pseudo-target subjects (source-only; per outer fold).
            cert = None
            guard = None
            ridge_train_spearman = float("nan")
            ridge_train_pearson = float("nan")
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")

            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_ridge_rows: List[np.ndarray] = []
                y_ridge_rows: List[float] = []
                X_guard_rows: List[np.ndarray] = []
                y_guard_rows: List[int] = []
                improve_guard_rows: List[float] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    inner_bundle = _get_chan_bundle(inner_train)
                    m_id = inner_bundle["model_id"]
                    cand_inner: dict = dict(inner_bundle.get("candidates", {}))
                    if not cand_inner:
                        continue

                    z_p = subject_data[int(pseudo_t)].X
                    y_p = subject_data[int(pseudo_t)].y
                    p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, list(class_order))
                    yp_id = np.asarray(m_id.predict(z_p))
                    acc_id = float(accuracy_score(y_p, yp_id))

                    for cand_key, info in cand_inner.items():
                        A = info["A"]
                        m_A = info["model"]
                        z_p_A = apply_spatial_transform(A, z_p)
                        p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, list(class_order))
                        yp_A = np.asarray(m_A.predict(z_p_A))
                        acc_A = float(accuracy_score(y_p, yp_A))
                        improve = float(acc_A - acc_id)

                        rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                        feats_vec, names = candidate_features_from_record(
                            rec, n_classes=len(class_order), include_pbar=True
                        )
                        if feat_names is None:
                            feat_names = names
                        if use_ridge:
                            X_ridge_rows.append(feats_vec)
                            y_ridge_rows.append(float(improve))
                        if use_guard:
                            X_guard_rows.append(feats_vec)
                            y_guard_rows.append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows.append(float(improve))

                if use_ridge and X_ridge_rows and feat_names is not None:
                    X_ridge = np.vstack(X_ridge_rows)
                    y_ridge = np.asarray(y_ridge_rows, dtype=np.float64)
                    cert = train_ridge_certificate(
                        X_ridge,
                        y_ridge,
                        feature_names=feat_names,
                        alpha=float(oea_zo_calib_ridge_alpha),
                    )
                    try:
                        pred = np.asarray(cert.predict_accuracy(X_ridge), dtype=np.float64).reshape(-1)
                        if y_ridge.size >= 2:
                            ridge_train_pearson = float(np.corrcoef(pred, y_ridge)[0, 1])
                            ridge_train_spearman = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                    except Exception:
                        pass

                if use_guard and X_guard_rows and feat_names is not None:
                    X_guard = np.vstack(X_guard_rows)
                    y_guard = np.asarray(y_guard_rows, dtype=int)
                    if len(np.unique(y_guard)) >= 2:
                        guard = train_logistic_guard(
                            X_guard,
                            y_guard,
                            feature_names=feat_names,
                            c=float(oea_zo_calib_guard_c),
                        )
                        try:
                            p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                            improve_train = np.asarray(improve_guard_rows, dtype=np.float64).reshape(-1)
                            guard_train_auc = float(roc_auc_score(y_guard, p_train))
                            if improve_train.size == p_train.size and improve_train.size >= 2:
                                guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                                guard_train_spearman = float(
                                    np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1]
                                )
                        except Exception:
                            pass

            # Build candidate records on the target subject (unlabeled).
            p_id_t = _reorder_proba_columns(model_id.predict_proba(X_test_raw), model_id.classes_, list(class_order))
            rec_id = _record_for_candidate(p_id=p_id_t, p_c=p_id_t)
            rec_id["kind"] = "identity"
            rec_id["cand_key"] = None
            records: list[dict] = [rec_id]

            for cand_key, info in candidates_outer.items():
                A = info["A"]
                m_A = info["model"]
                X_test_A = apply_spatial_transform(A, X_test_raw)
                p_A_t = _reorder_proba_columns(m_A.predict_proba(X_test_A), m_A.classes_, list(class_order))
                rec = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                rec["kind"] = "candidate"
                rec["cand_key"] = cand_key
                rec["cand_rank"] = float(info.get("rank", float("nan")))
                rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                records.append(rec)

            selected = rec_id
            if selector == "calibrated_ridge_guard" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    n_classes=len(class_order),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "calibrated_ridge" and cert is not None:
                selected = select_by_predicted_improvement(
                    records,
                    cert=cert,
                    n_classes=len(class_order),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set="base",
                )
            elif selector == "calibrated_guard" and guard is not None:
                selected = select_by_guarded_objective(
                    records,
                    guard=guard,
                    n_classes=len(class_order),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "objective":
                best = min(records, key=lambda r: float(r.get("score", r.get("objective_base", 0.0))))
                selected = best

            # Optional marginal-entropy fallback (unlabeled safety valve).
            if (
                float(oea_zo_fallback_min_marginal_entropy) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and float(selected.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
            ):
                selected = rec_id

            # Apply selection.
            accept = str(selected.get("kind", "")) != "identity"
            sel_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            sel_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            sel_key = selected.get("cand_key", None)
            sel_rank = float(selected.get("cand_rank", float("nan")))
            sel_lam = float(selected.get("cand_lambda", float("nan")))

            if accept and sel_key in candidates_outer:
                A_sel = candidates_outer[sel_key]["A"]
                model = candidates_outer[sel_key]["model"]
                X_test = apply_spatial_transform(A_sel, X_test_raw)
            else:
                model = model_id
                X_test = X_test_raw

            # Analysis-only: compute true improvement for the selected transform (not used in selection).
            try:
                acc_id_t = float(accuracy_score(y_test, np.asarray(model_id.predict(X_test_raw))))
            except Exception:
                acc_id_t = float("nan")
            try:
                acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
            except Exception:
                acc_sel_t = float("nan")
            improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

            if extra_rows is not None:
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "chan_multi_accept": int(bool(accept)),
                        "chan_multi_guard_pos": float(sel_guard_pos),
                        "chan_multi_ridge_pred_improve": float(sel_ridge_pred),
                        "chan_multi_acc_anchor": float(acc_id_t),
                        "chan_multi_acc_selected": float(acc_sel_t),
                        "chan_multi_improve": float(improve_t),
                        "chan_multi_sel_rank": float(sel_rank),
                        "chan_multi_sel_lambda": float(sel_lam),
                        "chan_multi_ridge_train_spearman": float(ridge_train_spearman),
                        "chan_multi_ridge_train_pearson": float(ridge_train_pearson),
                        "chan_multi_guard_train_auc": float(guard_train_auc),
                        "chan_multi_guard_train_spearman": float(guard_train_spearman),
                        "chan_multi_guard_train_pearson": float(guard_train_pearson),
                    }
                )
        elif alignment == "ea_si_chan_spsa_safe":
            # Semi-coupled bilevel (continuous lower-level, calibrated safe upper-level):
            #
            # - Lower-level: channel projector solution map A(λ) for continuous λ (rank-deficient projector),
            #   solved from precomputed scatter matrices (B,S).
            # - Upper-level: keep the calibrated ridge/guard selection rule, but optimize λ directly on the
            #   target subject via SPSA using the *predicted improvement* as the objective (unlabeled).
            #
            # This is a "half-linked" variant of the existing multi-candidate selection: instead of a discrete
            # grid over λ, we search λ continuously.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y
            X_test_raw = X_test

            # For reporting only (n_train) and consistency with other branches.
            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            selector = str(oea_zo_selector)
            if selector not in {"calibrated_ridge", "calibrated_ridge_guard"}:
                raise ValueError("ea_si_chan_spsa_safe requires --oea-zo-selector calibrated_ridge(_guard).")

            use_ridge = True
            use_guard = selector == "calibrated_ridge_guard"

            # Anchor model (EA) and discrete candidates (only used for calibration).
            outer_bundle = _get_chan_bundle(train_subjects)
            model_id = outer_bundle["model_id"]

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                # Convenience aliases for selectors that expect `score`/`objective`.
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])
                return rec

            # Calibrate ridge/guard on pseudo-target subjects (source-only; per outer fold).
            cert = None
            guard = None
            ridge_train_spearman = float("nan")
            ridge_train_pearson = float("nan")
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")

            rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
            calib_subjects = list(train_subjects)
            if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                rng.shuffle(calib_subjects)
                calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

            X_ridge_rows: List[np.ndarray] = []
            y_ridge_rows: List[float] = []
            X_guard_rows: List[np.ndarray] = []
            y_guard_rows: List[int] = []
            improve_guard_rows: List[float] = []
            feat_names: tuple[str, ...] | None = None

            for pseudo_t in calib_subjects:
                inner_train = [s for s in train_subjects if s != pseudo_t]
                if len(inner_train) < 2:
                    continue
                inner_bundle = _get_chan_bundle(inner_train)
                m_id = inner_bundle["model_id"]
                cand_inner: dict = dict(inner_bundle.get("candidates", {}))
                if not cand_inner:
                    continue

                z_p = subject_data[int(pseudo_t)].X
                y_p = subject_data[int(pseudo_t)].y
                p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, list(class_order))
                yp_id = np.asarray(m_id.predict(z_p))
                acc_id = float(accuracy_score(y_p, yp_id))

                for _cand_key, info in cand_inner.items():
                    A = info["A"]
                    m_A = info["model"]
                    z_p_A = apply_spatial_transform(A, z_p)
                    p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, list(class_order))
                    yp_A = np.asarray(m_A.predict(z_p_A))
                    acc_A = float(accuracy_score(y_p, yp_A))
                    improve = float(acc_A - acc_id)

                    rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                    rec["cand_rank"] = float(info.get("rank", float("nan")))
                    rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                    feats_vec, names = candidate_features_from_record(rec, n_classes=len(class_order), include_pbar=True)
                    if feat_names is None:
                        feat_names = names

                    X_ridge_rows.append(feats_vec)
                    y_ridge_rows.append(float(improve))

                    if use_guard:
                        X_guard_rows.append(feats_vec)
                        y_guard_rows.append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                        improve_guard_rows.append(float(improve))

            if X_ridge_rows and feat_names is not None:
                X_ridge = np.vstack(X_ridge_rows)
                y_ridge = np.asarray(y_ridge_rows, dtype=np.float64)
                cert = train_ridge_certificate(
                    X_ridge,
                    y_ridge,
                    feature_names=feat_names,
                    alpha=float(oea_zo_calib_ridge_alpha),
                )
                try:
                    pred = np.asarray(cert.predict_accuracy(X_ridge), dtype=np.float64).reshape(-1)
                    if y_ridge.size >= 2:
                        ridge_train_pearson = float(np.corrcoef(pred, y_ridge)[0, 1])
                        ridge_train_spearman = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                except Exception:
                    pass

            if use_guard and X_guard_rows and feat_names is not None:
                X_guard = np.vstack(X_guard_rows)
                y_guard = np.asarray(y_guard_rows, dtype=int)
                if len(np.unique(y_guard)) >= 2:
                    guard = train_logistic_guard(
                        X_guard,
                        y_guard,
                        feature_names=feat_names,
                        c=float(oea_zo_calib_guard_c),
                    )
                    try:
                        p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                        improve_train = np.asarray(improve_guard_rows, dtype=np.float64).reshape(-1)
                        guard_train_auc = float(roc_auc_score(y_guard, p_train))
                        if improve_train.size == p_train.size and improve_train.size >= 2:
                            guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                            guard_train_spearman = float(np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1])
                    except Exception:
                        pass

            # If calibration failed, fall back to EA anchor.
            if cert is None or (use_guard and guard is None):
                model = model_id
                X_test = X_test_raw
                try:
                    acc_id_t = float(accuracy_score(y_test, np.asarray(model_id.predict(X_test_raw))))
                except Exception:
                    acc_id_t = float("nan")
                if extra_rows is not None:
                    extra_rows.append(
                        {
                            "subject": int(test_subject),
                            "chan_spsa_accept": 0,
                            "chan_spsa_guard_pos": float("nan"),
                            "chan_spsa_ridge_pred_improve": float("nan"),
                            "chan_spsa_acc_anchor": float(acc_id_t),
                            "chan_spsa_acc_selected": float(acc_id_t),
                            "chan_spsa_improve": 0.0,
                            "chan_spsa_sel_rank": float(int(si_proj_dim) if int(si_proj_dim) > 0 else float("nan")),
                            "chan_spsa_sel_lambda": float("nan"),
                            "chan_spsa_ridge_train_spearman": float(ridge_train_spearman),
                            "chan_spsa_ridge_train_pearson": float(ridge_train_pearson),
                            "chan_spsa_guard_train_auc": float(guard_train_auc),
                            "chan_spsa_guard_train_spearman": float(guard_train_spearman),
                            "chan_spsa_guard_train_pearson": float(guard_train_pearson),
                        }
                    )
            else:
                # Precompute scatter for fast re-solving A(λ) and anchor probabilities on target subject.
                scatter = compute_channel_projector_scatter(
                    X=X_train,
                    y=y_train,
                    subjects=subj_train,
                    class_order=tuple([str(c) for c in class_order]),
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                )
                p_id_t = _reorder_proba_columns(
                    model_id.predict_proba(X_test_raw), model_id.classes_, list(class_order)
                )

                cont_rank = int(si_proj_dim) if int(si_proj_dim) > 0 else int(X_train.shape[1]) - 1
                cont_rank = max(1, min(int(X_train.shape[1]) - 1, cont_rank))

                drift_mode = str(oea_zo_drift_mode)
                drift_gamma = float(oea_zo_drift_gamma)
                drift_delta = float(oea_zo_drift_delta)
                thr = float(oea_zo_calib_guard_threshold)

                rng_opt = np.random.RandomState(int(oea_zo_seed) + int(test_subject) * 997)
                iters = max(1, int(oea_zo_iters))
                lr = float(oea_zo_lr)
                mu = max(1e-8, float(oea_zo_mu))

                # Optimize log(λ) for multiplicative stability.
                #
                # NOTE: In early experiments, an overly wide λ range could produce out-of-distribution
                # projectors that the calibrated selector (trained on a small λ grid) cannot reliably
                # score, causing negative transfer. We therefore:
                #   1) Set a λ trust region based on the configured candidate grid.
                #   2) Warm-start from the best grid candidate (by predicted improvement) when possible.
                lambda_grid = [float(l) for l in (list(si_chan_candidate_lambdas) or [float(si_subject_lambda)])]
                lambda_grid = [l for l in lambda_grid if np.isfinite(l) and l > 0.0]
                if lambda_grid:
                    lam_min = float(np.min(np.asarray(lambda_grid, dtype=np.float64)))
                    lam_max = float(np.max(np.asarray(lambda_grid, dtype=np.float64)))
                else:
                    lam_min = 1e-3
                    lam_max = 10.0

                phi = float(np.log(max(float(si_subject_lambda), 1e-8)))
                # Allow modest exploration beyond the discrete grid while avoiding extreme extrapolation.
                phi_lo = float(np.log(max(lam_min / 2.0, 1e-3)))
                phi_hi = float(np.log(min(lam_max * 2.0, 10.0)))
                if not np.isfinite(phi_lo) or not np.isfinite(phi_hi) or phi_lo >= phi_hi:
                    phi_lo = float(np.log(1e-3))
                    phi_hi = float(np.log(10.0))

                best_pred = -float("inf")
                best_model: TrainedModel | None = None
                best_A: np.ndarray | None = None
                best_rec: dict | None = None
                best_lam = float("nan")

                def _eval_phi(phi_val: float):
                    if scatter is None:
                        return 1.0, False, -float("inf"), float("nan"), float("nan"), None, None, None
                    lam = float(np.exp(float(np.clip(phi_val, phi_lo, phi_hi))))
                    A = solve_channel_projector_from_scatter(
                        scatter,
                        subject_lambda=float(lam),
                        ridge=float(si_ridge),
                        n_components=int(cont_rank),
                        eps=float(oea_eps),
                    )
                    X_train_A = apply_spatial_transform(A, X_train)
                    m_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)
                    X_test_A = apply_spatial_transform(A, X_test_raw)
                    p_A_t = _reorder_proba_columns(
                        m_A.predict_proba(X_test_A), m_A.classes_, list(class_order)
                    )
                    rec = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                    rec["cand_rank"] = float(cont_rank)
                    rec["cand_lambda"] = float(lam)

                    feats, _names = candidate_features_from_record(rec, n_classes=len(class_order), include_pbar=True)
                    pred_improve = float(cert.predict_accuracy(feats)[0])
                    p_pos = float(guard.predict_pos_proba(feats)[0]) if use_guard and guard is not None else 1.0
                    rec["ridge_pred_improve"] = float(pred_improve)
                    rec["guard_p_pos"] = float(p_pos)

                    drift = float(rec.get("drift_best", 0.0))
                    pred_eff = float(pred_improve)
                    if drift_mode == "hard" and drift_delta > 0.0 and drift > drift_delta:
                        ok = False
                    else:
                        ok = True
                    if use_guard and p_pos < thr:
                        ok = False
                    if drift_mode == "penalty" and drift_gamma > 0.0:
                        pred_eff = float(pred_eff) - float(drift_gamma) * float(drift)

                    obj = -float(pred_eff)
                    if not ok:
                        obj = 1.0
                    return obj, ok, pred_eff, p_pos, lam, rec, A, m_A

                # Evaluate the discrete grid first (so SPSA never does worse than the best grid candidate
                # under the same calibrated selector).
                for lam0 in lambda_grid:
                    obj0, ok0, pred0, _ppos0, lam_eval, rec0, A0, m0 = _eval_phi(float(np.log(float(lam0))))
                    if ok0 and pred0 > best_pred:
                        best_pred = float(pred0)
                        best_model = m0
                        best_A = A0
                        best_rec = rec0
                        best_lam = float(lam_eval)
                        phi = float(np.log(max(best_lam, 1e-8)))

                for _t in range(iters):
                    u = 1.0 if rng_opt.rand() < 0.5 else -1.0
                    obj_p, ok_p, pred_p, ppos_p, lam_p, rec_p, A_p, m_p = _eval_phi(phi + mu * u)
                    obj_m, ok_m, pred_m, ppos_m, lam_m, rec_m, A_m, m_m = _eval_phi(phi - mu * u)
                    g = float(obj_p - obj_m) / float(2.0 * mu) * float(u)
                    phi = float(np.clip(phi - lr * g, phi_lo, phi_hi))

                    if ok_p and pred_p > best_pred:
                        best_pred = float(pred_p)
                        best_model = m_p
                        best_A = A_p
                        best_rec = rec_p
                        best_lam = float(lam_p)
                    if ok_m and pred_m > best_pred:
                        best_pred = float(pred_m)
                        best_model = m_m
                        best_A = A_m
                        best_rec = rec_m
                        best_lam = float(lam_m)

                # Final accept/fallback decision.
                accept = best_model is not None and best_A is not None and best_rec is not None and best_pred > 0.0
                if (
                    accept
                    and float(oea_zo_fallback_min_marginal_entropy) > 0.0
                    and float(best_rec.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
                ):
                    accept = False

                if accept:
                    model = best_model
                    X_test = apply_spatial_transform(best_A, X_test_raw)
                else:
                    model = model_id
                    X_test = X_test_raw

                # Analysis-only: compute true improvement for the selected transform (not used in selection).
                try:
                    acc_id_t = float(accuracy_score(y_test, np.asarray(model_id.predict(X_test_raw))))
                except Exception:
                    acc_id_t = float("nan")
                try:
                    acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
                except Exception:
                    acc_sel_t = float("nan")
                improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

                if extra_rows is not None:
                    extra_rows.append(
                        {
                            "subject": int(test_subject),
                            "chan_spsa_accept": int(bool(accept)),
                            "chan_spsa_guard_pos": float(best_rec.get("guard_p_pos", float("nan"))) if best_rec else float("nan"),
                            "chan_spsa_ridge_pred_improve": float(best_rec.get("ridge_pred_improve", float("nan"))) if best_rec else float("nan"),
                            "chan_spsa_acc_anchor": float(acc_id_t),
                            "chan_spsa_acc_selected": float(acc_sel_t),
                            "chan_spsa_improve": float(improve_t),
                            "chan_spsa_sel_rank": float(cont_rank),
                            "chan_spsa_sel_lambda": float(best_lam),
                            "chan_spsa_ridge_train_spearman": float(ridge_train_spearman),
                            "chan_spsa_ridge_train_pearson": float(ridge_train_pearson),
                            "chan_spsa_guard_train_auc": float(guard_train_auc),
                            "chan_spsa_guard_train_spearman": float(guard_train_spearman),
                            "chan_spsa_guard_train_pearson": float(guard_train_pearson),
                        }
                    )
        elif alignment == "ea_mm_safe":
            # Cross-model multi-candidate selection with calibrated selection (ridge/guard)
            # and safe fallback to the EA anchor.
            #
            # Candidate set includes:
            # - EA anchor (CSP+LDA on EA-whitened time series)
            # - EA-SI-CHAN candidates (rank-deficient channel projectors + CSP+LDA)
            # - MDM(RPA) candidate (TLCenter+TLStretch on SPD covariances + MDM classifier)
            #
            # Selection is performed on the target subject without using target labels.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y
            X_test_raw = X_test

            # For reporting only (n_train) and consistency with other branches.
            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            selector = str(oea_zo_selector)
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}

            outer_bundle = _get_chan_bundle(train_subjects)
            model_id = outer_bundle["model_id"]
            candidates_outer: dict = dict(outer_bundle.get("candidates", {}))

            class_labels = list([str(c) for c in class_order])

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _safe_float_local(x, default: float = 0.0) -> float:
                try:
                    v = float(x)
                except Exception:
                    return float(default)
                if not np.isfinite(v):
                    return float(default)
                return float(v)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                # Convenience aliases for selectors that expect `score`/`objective`.
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])
                return rec

            def _fit_rpa_mdm(
                *,
                inner_train_subjects: Sequence[int],
                target_subject: int,
            ):
                from pyriemann.classification import MDM
                from pyriemann.transfer import TLCenter, encode_domains

                try:
                    from pyriemann.transfer import TLStretch
                except Exception:
                    try:
                        from pyriemann.transfer._estimators import TLStretch
                    except Exception as exc:
                        raise ImportError(
                            "Missing pyriemann.transfer.TLStretch required for MDM(RPA) candidate."
                        ) from exc

                X_train_parts_raw: list[np.ndarray] = []
                y_train_parts_raw: list[np.ndarray] = []
                dom_train_parts: list[np.ndarray] = []
                for s in inner_train_subjects:
                    sd = subject_data_raw[int(s)]
                    X_train_parts_raw.append(sd.X)
                    y_train_parts_raw.append(sd.y)
                    dom_train_parts.append(np.full(sd.y.shape[0], f"src_{int(s)}", dtype=object))
                X_tr_raw = np.concatenate(X_train_parts_raw, axis=0)
                y_tr = np.concatenate(y_train_parts_raw, axis=0)
                dom_train = np.concatenate(dom_train_parts, axis=0)

                z_t_raw = subject_data_raw[int(target_subject)].X

                cov_train = covariances_from_epochs(X_tr_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
                cov_test = covariances_from_epochs(z_t_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))

                dom_test = np.full(cov_test.shape[0], "target", dtype=object)
                y_dummy = np.full(cov_test.shape[0], str(class_order[0]), dtype=object)
                cov_all = np.concatenate([cov_train, cov_test], axis=0)
                y_all = np.concatenate([y_tr, y_dummy], axis=0)
                dom_all = np.concatenate([dom_train, dom_test], axis=0)
                _, y_enc = encode_domains(cov_all, y_all, dom_all)

                center = TLCenter(target_domain="target", metric="riemann")
                cov_centered = center.fit_transform(cov_all, y_enc)

                stretch = TLStretch(target_domain="target", centered_data=True, metric="riemann")
                cov_stretched = stretch.fit_transform(cov_centered, y_enc)

                cov_src = cov_stretched[: cov_train.shape[0]]
                cov_tgt = cov_stretched[cov_train.shape[0] :]

                model_mdm = MDM(metric="riemann")
                model_mdm.fit(cov_src, y_tr)
                return model_mdm, cov_tgt

            # Calibrate ridge/guard per candidate family (chan vs mdm) on pseudo-target subjects.
            cert_by_family: dict[str, RidgeCertificate | None] = {"chan": None, "mdm": None}
            guard_by_family: dict[str, LogisticGuard | None] = {"chan": None, "mdm": None}

            ridge_train_spearman_chan = float("nan")
            ridge_train_pearson_chan = float("nan")
            ridge_train_spearman_mdm = float("nan")
            ridge_train_pearson_mdm = float("nan")
            guard_train_auc_chan = float("nan")
            guard_train_spearman_chan = float("nan")
            guard_train_pearson_chan = float("nan")
            guard_train_auc_mdm = float("nan")
            guard_train_spearman_mdm = float("nan")
            guard_train_pearson_mdm = float("nan")

            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_ridge_rows: dict[str, List[np.ndarray]] = {"chan": [], "mdm": []}
                y_ridge_rows: dict[str, List[float]] = {"chan": [], "mdm": []}
                X_guard_rows: dict[str, List[np.ndarray]] = {"chan": [], "mdm": []}
                y_guard_rows: dict[str, List[int]] = {"chan": [], "mdm": []}
                improve_guard_rows: dict[str, List[float]] = {"chan": [], "mdm": []}
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    inner_bundle = _get_chan_bundle(inner_train)
                    m_id = inner_bundle["model_id"]
                    cand_inner: dict = dict(inner_bundle.get("candidates", {}))

                    z_p = subject_data[int(pseudo_t)].X
                    y_p = subject_data[int(pseudo_t)].y
                    p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, class_labels)
                    yp_id = np.asarray(m_id.predict(z_p))
                    acc_id = float(accuracy_score(y_p, yp_id))

                    # Channel candidates.
                    for cand_key, info in cand_inner.items():
                        A = info["A"]
                        m_A = info["model"]
                        z_p_A = apply_spatial_transform(A, z_p)
                        p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, class_labels)
                        yp_A = np.asarray(m_A.predict(z_p_A))
                        acc_A = float(accuracy_score(y_p, yp_A))
                        improve = float(acc_A - acc_id)

                        rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                        rec["cand_family"] = "chan"
                        rec["cand_key"] = cand_key
                        rec["cand_rank"] = float(info.get("rank", float("nan")))
                        rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                        feats_vec, names = candidate_features_from_record(
                            rec, n_classes=len(class_order), include_pbar=True
                        )
                        if feat_names is None:
                            feat_names = names
                        if use_ridge:
                            X_ridge_rows["chan"].append(feats_vec)
                            y_ridge_rows["chan"].append(float(improve))
                        if use_guard:
                            X_guard_rows["chan"].append(feats_vec)
                            y_guard_rows["chan"].append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows["chan"].append(float(improve))

                    # MDM(RPA) candidate.
                    try:
                        mdm_inner, cov_tgt_inner = _fit_rpa_mdm(
                            inner_train_subjects=inner_train,
                            target_subject=int(pseudo_t),
                        )
                        p_mdm = _reorder_proba_columns(
                            mdm_inner.predict_proba(cov_tgt_inner), mdm_inner.classes_, class_labels
                        )
                        yp_mdm = np.asarray(mdm_inner.predict(cov_tgt_inner))
                        acc_mdm = float(accuracy_score(y_p, yp_mdm))
                        improve_mdm = float(acc_mdm - acc_id)

                        rec_mdm = _record_for_candidate(p_id=p_id, p_c=p_mdm)
                        rec_mdm["cand_family"] = "mdm"
                        rec_mdm["cand_key"] = "mdm_rpa"
                        rec_mdm["cand_rank"] = float("nan")
                        rec_mdm["cand_lambda"] = float("nan")
                        feats_mdm, names_mdm = candidate_features_from_record(
                            rec_mdm, n_classes=len(class_order), include_pbar=True
                        )
                        if feat_names is None:
                            feat_names = names_mdm
                        if use_ridge:
                            X_ridge_rows["mdm"].append(feats_mdm)
                            y_ridge_rows["mdm"].append(float(improve_mdm))
                        if use_guard:
                            X_guard_rows["mdm"].append(feats_mdm)
                            y_guard_rows["mdm"].append(1 if improve_mdm >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows["mdm"].append(float(improve_mdm))
                    except Exception:
                        pass

                # Train per-family models.
                for fam in ("chan", "mdm"):
                    if use_ridge and X_ridge_rows[fam] and feat_names is not None:
                        X_ridge = np.vstack(X_ridge_rows[fam])
                        y_ridge = np.asarray(y_ridge_rows[fam], dtype=np.float64)
                        cert_by_family[fam] = train_ridge_certificate(
                            X_ridge,
                            y_ridge,
                            feature_names=feat_names,
                            alpha=float(oea_zo_calib_ridge_alpha),
                        )
                        try:
                            pred = np.asarray(
                                cert_by_family[fam].predict_accuracy(X_ridge), dtype=np.float64
                            ).reshape(-1)
                            if y_ridge.size >= 2:
                                pear = float(np.corrcoef(pred, y_ridge)[0, 1])
                                spear = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                                if fam == "chan":
                                    ridge_train_pearson_chan = pear
                                    ridge_train_spearman_chan = spear
                                else:
                                    ridge_train_pearson_mdm = pear
                                    ridge_train_spearman_mdm = spear
                        except Exception:
                            pass

                    if use_guard and X_guard_rows[fam] and feat_names is not None:
                        X_guard = np.vstack(X_guard_rows[fam])
                        y_guard = np.asarray(y_guard_rows[fam], dtype=int)
                        if len(np.unique(y_guard)) >= 2:
                            guard_by_family[fam] = train_logistic_guard(
                                X_guard,
                                y_guard,
                                feature_names=feat_names,
                                c=float(oea_zo_calib_guard_c),
                            )
                            try:
                                p_train = np.asarray(
                                    guard_by_family[fam].predict_pos_proba(X_guard), dtype=np.float64
                                ).reshape(-1)
                                improve_train = np.asarray(improve_guard_rows[fam], dtype=np.float64).reshape(-1)
                                auc = float(roc_auc_score(y_guard, p_train))
                                pear = float("nan")
                                spear = float("nan")
                                if improve_train.size == p_train.size and improve_train.size >= 2:
                                    pear = float(np.corrcoef(p_train, improve_train)[0, 1])
                                    spear = float(np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1])
                                if fam == "chan":
                                    guard_train_auc_chan = auc
                                    guard_train_pearson_chan = pear
                                    guard_train_spearman_chan = spear
                                else:
                                    guard_train_auc_mdm = auc
                                    guard_train_pearson_mdm = pear
                                    guard_train_spearman_mdm = spear
                            except Exception:
                                pass

            # Build candidate records on the target subject (unlabeled).
            p_id_t = _reorder_proba_columns(model_id.predict_proba(X_test_raw), model_id.classes_, class_labels)
            rec_id = _record_for_candidate(p_id=p_id_t, p_c=p_id_t)
            rec_id["kind"] = "identity"
            rec_id["cand_key"] = None
            rec_id["cand_family"] = "ea"
            rec_id["cand_rank"] = float("nan")
            rec_id["cand_lambda"] = float("nan")
            records: list[dict] = [rec_id]

            # Channel candidates.
            for cand_key, info in candidates_outer.items():
                A = info["A"]
                m_A = info["model"]
                X_test_A = apply_spatial_transform(A, X_test_raw)
                p_A_t = _reorder_proba_columns(m_A.predict_proba(X_test_A), m_A.classes_, class_labels)
                rec = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                rec["kind"] = "candidate"
                rec["cand_key"] = cand_key
                rec["cand_family"] = "chan"
                rec["cand_rank"] = float(info.get("rank", float("nan")))
                rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                records.append(rec)

            # MDM(RPA) candidate.
            mdm_outer = None
            cov_tgt_outer = None
            try:
                mdm_outer, cov_tgt_outer = _fit_rpa_mdm(inner_train_subjects=train_subjects, target_subject=int(test_subject))
                p_mdm_t = _reorder_proba_columns(
                    mdm_outer.predict_proba(cov_tgt_outer), mdm_outer.classes_, class_labels
                )
                rec_mdm = _record_for_candidate(p_id=p_id_t, p_c=p_mdm_t)
                rec_mdm["kind"] = "candidate"
                rec_mdm["cand_key"] = "mdm_rpa"
                rec_mdm["cand_family"] = "mdm"
                rec_mdm["cand_rank"] = float("nan")
                rec_mdm["cand_lambda"] = float("nan")
                records.append(rec_mdm)
            except Exception:
                mdm_outer = None
                cov_tgt_outer = None

            # Per-family selection (avoid mixing CSP+LDA vs MDM probability statistics).
            selected = rec_id
            drift_mode = str(oea_zo_drift_mode)
            drift_gamma = float(oea_zo_drift_gamma)
            drift_delta = float(oea_zo_drift_delta)

            if selector in {"calibrated_ridge_guard", "calibrated_ridge", "calibrated_guard"}:
                best_pred = -float("inf")
                best_score = float("inf")
                best_rec: dict | None = None

                for rec in records:
                    if str(rec.get("kind", "")) == "identity":
                        continue
                    fam = str(rec.get("cand_family", "")).strip().lower()
                    if fam not in {"chan", "mdm"}:
                        continue

                    feats, _names = candidate_features_from_record(
                        rec, n_classes=len(class_order), include_pbar=True
                    )

                    # Guard probability (if required).
                    if selector in {"calibrated_ridge_guard", "calibrated_guard"}:
                        g = guard_by_family.get(fam, None)
                        if g is None:
                            continue
                        p_pos = float(g.predict_pos_proba(feats)[0])
                        rec["guard_p_pos"] = float(p_pos)
                        thr = float(oea_zo_calib_guard_threshold)
                        if fam == "mdm" and float(mm_safe_mdm_guard_threshold) >= 0.0:
                            thr = max(float(thr), float(mm_safe_mdm_guard_threshold))
                        if p_pos < float(thr):
                            continue

                    drift = _safe_float_local(rec.get("drift_best", 0.0))
                    if fam == "mdm" and float(mm_safe_mdm_drift_delta) > 0.0 and drift > float(mm_safe_mdm_drift_delta):
                        continue
                    if drift_mode == "hard" and drift_delta > 0.0 and drift > drift_delta:
                        continue

                    if selector in {"calibrated_ridge_guard", "calibrated_ridge"}:
                        c = cert_by_family.get(fam, None)
                        if c is None:
                            continue
                        pred_improve = float(c.predict_accuracy(feats)[0])
                        rec["ridge_pred_improve"] = float(pred_improve)
                        if fam == "mdm" and float(mm_safe_mdm_min_pred_improve) > 0.0 and float(pred_improve) < float(
                            mm_safe_mdm_min_pred_improve
                        ):
                            continue
                        if drift_mode == "penalty" and drift_gamma > 0.0:
                            pred_improve = float(pred_improve) - drift_gamma * float(drift)
                        if pred_improve > best_pred:
                            best_pred = float(pred_improve)
                            best_rec = rec
                    else:
                        # Guard-only selector: pick by objective among accepted candidates.
                        score = _safe_float_local(rec.get("score", rec.get("objective", 0.0)))
                        if drift_mode == "penalty" and drift_gamma > 0.0:
                            score = float(score) + drift_gamma * float(drift)
                        if score < best_score:
                            best_score = float(score)
                            best_rec = rec

                if selector == "calibrated_guard":
                    selected = best_rec if best_rec is not None else rec_id
                else:
                    if best_rec is None or not np.isfinite(best_pred) or float(best_pred) <= 0.0:
                        selected = rec_id
                    else:
                        selected = best_rec
            elif selector == "objective":
                selected = min(records, key=lambda r: float(r.get("score", r.get("objective_base", 0.0))))

            # Optional marginal-entropy fallback (unlabeled safety valve).
            if (
                float(oea_zo_fallback_min_marginal_entropy) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and float(selected.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
            ):
                selected = rec_id

            # Apply selection.
            accept = str(selected.get("kind", "")) != "identity"
            sel_family = str(selected.get("cand_family", "ea"))
            sel_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            sel_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            sel_key = selected.get("cand_key", None)

            if accept and sel_family == "chan" and sel_key in candidates_outer:
                A_sel = candidates_outer[sel_key]["A"]
                model = candidates_outer[sel_key]["model"]
                X_test = apply_spatial_transform(A_sel, X_test_raw)
            elif accept and sel_family == "mdm" and mdm_outer is not None and cov_tgt_outer is not None:
                model = mdm_outer
                X_test = cov_tgt_outer
            else:
                model = model_id
                X_test = X_test_raw
                accept = False
                sel_family = "ea"

            # Analysis-only: compute true improvement for the selected candidate (not used in selection).
            try:
                acc_id_t = float(accuracy_score(y_test, np.asarray(model_id.predict(X_test_raw))))
            except Exception:
                acc_id_t = float("nan")
            try:
                acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
            except Exception:
                acc_sel_t = float("nan")
            improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

            if extra_rows is not None:
                # Aggregate training stats (compat with existing columns).
                mm_ridge_train_spearman = float(
                    np.nanmean(np.asarray([ridge_train_spearman_chan, ridge_train_spearman_mdm], dtype=np.float64))
                )
                mm_ridge_train_pearson = float(
                    np.nanmean(np.asarray([ridge_train_pearson_chan, ridge_train_pearson_mdm], dtype=np.float64))
                )
                mm_guard_train_auc = float(
                    np.nanmean(np.asarray([guard_train_auc_chan, guard_train_auc_mdm], dtype=np.float64))
                )
                mm_guard_train_spearman = float(
                    np.nanmean(np.asarray([guard_train_spearman_chan, guard_train_spearman_mdm], dtype=np.float64))
                )
                mm_guard_train_pearson = float(
                    np.nanmean(np.asarray([guard_train_pearson_chan, guard_train_pearson_mdm], dtype=np.float64))
                )
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "mm_safe_accept": int(bool(accept)),
                        "mm_safe_family": str(sel_family),
                        "mm_safe_guard_pos": float(sel_guard_pos),
                        "mm_safe_ridge_pred_improve": float(sel_ridge_pred),
                        "mm_safe_acc_anchor": float(acc_id_t),
                        "mm_safe_acc_selected": float(acc_sel_t),
                        "mm_safe_improve": float(improve_t),
                        "mm_safe_ridge_train_spearman": float(mm_ridge_train_spearman),
                        "mm_safe_ridge_train_pearson": float(mm_ridge_train_pearson),
                        "mm_safe_guard_train_auc": float(mm_guard_train_auc),
                        "mm_safe_guard_train_spearman": float(mm_guard_train_spearman),
                        "mm_safe_guard_train_pearson": float(mm_guard_train_pearson),
                        "mm_safe_chan_ridge_train_spearman": float(ridge_train_spearman_chan),
                        "mm_safe_chan_ridge_train_pearson": float(ridge_train_pearson_chan),
                        "mm_safe_mdm_ridge_train_spearman": float(ridge_train_spearman_mdm),
                        "mm_safe_mdm_ridge_train_pearson": float(ridge_train_pearson_mdm),
                        "mm_safe_chan_guard_train_auc": float(guard_train_auc_chan),
                        "mm_safe_chan_guard_train_spearman": float(guard_train_spearman_chan),
                        "mm_safe_chan_guard_train_pearson": float(guard_train_pearson_chan),
                        "mm_safe_mdm_guard_train_auc": float(guard_train_auc_mdm),
                        "mm_safe_mdm_guard_train_spearman": float(guard_train_spearman_mdm),
                        "mm_safe_mdm_guard_train_pearson": float(guard_train_pearson_mdm),
                    }
                )
        elif alignment == "ea_si_zo":
            # Train on EA-whitened source data with subject-invariant projection, then
            # adapt only Q_t at test time via ZO (upper-level).
            class_labels = tuple([str(c) for c in class_order])

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            from mne.decoding import CSP  # local import to keep module import cost low

            csp = CSP(n_components=int(n_components))
            csp.fit(X_train, y_train)
            feats = np.asarray(csp.transform(X_train), dtype=np.float64)
            proj_params = HSICProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            mean_f, W = learn_hsic_subject_invariant_projector(
                X=feats,
                y=y_train,
                subjects=subj_train,
                class_order=class_labels,
                params=proj_params,
            )
            projector = CenteredLinearProjector(mean=mean_f, W=W)
            model = fit_csp_projected_lda(
                X_train=X_train,
                y_train=y_train,
                projector=projector,
                csp=csp,
                n_components=n_components,
            )

            y_test = subject_data[int(test_subject)].y

            # Optional: offline calibrated certificate / guard (trained only on source subjects in this fold).
            selector = str(oea_zo_selector)
            use_stack = selector == "calibrated_stack_ridge"
            use_ridge_guard = selector == "calibrated_ridge_guard"
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}
            use_evidence = selector == "evidence"
            use_probe_mixup = selector == "probe_mixup"
            use_probe_mixup_hard = selector == "probe_mixup_hard"
            use_iwcv = selector == "iwcv"
            use_iwcv_ucb = selector == "iwcv_ucb"
            use_dev = selector == "dev"
            use_oracle = selector == "oracle"
            cert = None
            guard = None
            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_calib_rows: List[np.ndarray] = []
                y_calib_rows: List[float] = []
                y_guard_rows: List[int] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    X_inner = np.concatenate([subject_data[s].X for s in inner_train], axis=0)
                    y_inner = np.concatenate([subject_data[s].y for s in inner_train], axis=0)

                    subj_inner = np.concatenate(
                        [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in inner_train],
                        axis=0,
                    )
                    csp_inner = CSP(n_components=int(n_components))
                    csp_inner.fit(X_inner, y_inner)
                    feats_inner = np.asarray(csp_inner.transform(X_inner), dtype=np.float64)
                    mean_i, W_i = learn_hsic_subject_invariant_projector(
                        X=feats_inner,
                        y=y_inner,
                        subjects=subj_inner,
                        class_order=class_labels,
                        params=proj_params,
                    )
                    projector_i = CenteredLinearProjector(mean=mean_i, W=W_i)
                    model_inner = fit_csp_projected_lda(
                        X_train=X_inner,
                        y_train=y_inner,
                        projector=projector_i,
                        csp=csp_inner,
                        n_components=n_components,
                    )

                    diffs_inner = []
                    for s in inner_train:
                        diffs_inner.append(
                            class_cov_diff(
                                subject_data[int(s)].X,
                                subject_data[int(s)].y,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        )
                    d_ref_inner = np.mean(np.stack(diffs_inner, axis=0), axis=0)

                    z_pseudo = subject_data[int(pseudo_t)].X
                    y_pseudo = subject_data[int(pseudo_t)].y

                    marginal_prior_inner: np.ndarray | None = None
                    if oea_zo_marginal_mode == "kl_prior":
                        if oea_zo_marginal_prior == "uniform":
                            marginal_prior_inner = np.ones(len(class_labels), dtype=np.float64) / float(
                                len(class_labels)
                            )
                        elif oea_zo_marginal_prior == "source":
                            counts = np.array([(y_inner == c).sum() for c in class_labels], dtype=np.float64)
                            marginal_prior_inner = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                        else:
                            proba_id = model_inner.predict_proba(z_pseudo)
                            proba_id = _reorder_proba_columns(proba_id, model_inner.classes_, list(class_order))
                            marginal_prior_inner = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))
                        mix = float(oea_zo_marginal_prior_mix)
                        if mix > 0.0 and marginal_prior_inner is not None:
                            u = np.ones_like(marginal_prior_inner) / float(marginal_prior_inner.shape[0])
                            marginal_prior_inner = (1.0 - mix) * marginal_prior_inner + mix * u
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))

                    lda_ev_inner = None
                    if str(oea_zo_objective) == "lda_nll" or use_stack:
                        lda_ev_inner = _compute_lda_evidence_params(
                            model=model_inner,
                            X_train=X_inner,
                            y_train=y_inner,
                            class_order=class_labels,
                        )

                    _q_sel, diag_inner = _optimize_qt_oea_zo(
                        z_t=z_pseudo,
                        model=model_inner,
                        class_order=class_labels,
                        d_ref=d_ref_inner,
                        lda_evidence=lda_ev_inner,
                        channel_names=channel_names,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
                        localmix_neighbors=int(oea_zo_localmix_neighbors),
                        localmix_self_bias=float(oea_zo_localmix_self_bias),
                        infomax_lambda=float(oea_zo_infomax_lambda),
                        reliable_metric=str(oea_zo_reliable_metric),
                        reliable_threshold=float(oea_zo_reliable_threshold),
                        reliable_alpha=float(oea_zo_reliable_alpha),
                        trust_lambda=float(oea_zo_trust_lambda),
                        trust_q0=str(oea_zo_trust_q0),
                        marginal_mode=str(oea_zo_marginal_mode),
                        marginal_beta=float(oea_zo_marginal_beta),
                        marginal_tau=float(oea_zo_marginal_tau),
                        marginal_prior=marginal_prior_inner,
                        bilevel_iters=int(oea_zo_bilevel_iters),
                        bilevel_temp=float(oea_zo_bilevel_temp),
                        bilevel_step=float(oea_zo_bilevel_step),
                        bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                        bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        holdout_fraction=float(oea_zo_holdout_fraction),
                        fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                        iters=int(oea_zo_iters),
                        lr=float(oea_zo_lr),
                        mu=float(oea_zo_mu),
                        n_rotations=int(oea_zo_k),
                        seed=int(oea_zo_seed) + int(pseudo_t) * 997,
                        l2=float(oea_zo_l2),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        return_diagnostics=True,
                    )

                    recs = list(diag_inner.get("records", []))
                    if not recs:
                        continue
                    feats_list: List[np.ndarray] = []
                    acc_list: List[float] = []
                    acc_id: float | None = None
                    for rec in recs:
                        if use_stack:
                            feats_vec, names = stacked_candidate_features_from_record(
                                rec, n_classes=len(class_labels)
                            )
                        else:
                            feats_vec, names = candidate_features_from_record(rec, n_classes=len(class_labels))
                        if feat_names is None:
                            feat_names = names
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        yp = model_inner.predict(apply_spatial_transform(Q, z_pseudo))
                        acc = float(accuracy_score(y_pseudo, yp))
                        if str(rec.get("kind", "")) == "identity":
                            acc_id = acc
                        feats_list.append(feats_vec)
                        acc_list.append(acc)
                    if acc_id is None:
                        continue
                    for feats_vec, acc in zip(feats_list, acc_list):
                        improve = float(acc - float(acc_id))
                        y_calib_rows.append(float(improve))
                        y_guard_rows.append(1 if float(improve) >= float(oea_zo_calib_guard_margin) else 0)
                        X_calib_rows.append(feats_vec)

                if X_calib_rows and feat_names is not None:
                    X_calib = np.vstack(X_calib_rows)
                    y_calib = np.asarray(y_calib_rows, dtype=np.float64)
                    y_guard = np.asarray(y_guard_rows, dtype=int)
                    if use_ridge:
                        cert = train_ridge_certificate(
                            X=X_calib,
                            y=y_calib,
                            feature_names=feat_names,
                            alpha=float(oea_zo_calib_ridge_alpha),
                        )
                    if use_guard:
                        guard = train_logistic_guard(
                            X=X_calib,
                            y=y_guard,
                            feature_names=feat_names,
                            C=float(oea_zo_calib_guard_c),
                        )

            z_t = subject_data[int(test_subject)].X
            d_ref = np.mean(
                np.stack(
                    [
                        class_cov_diff(
                            subject_data[int(s)].X,
                            subject_data[int(s)].y,
                            class_order=class_labels,
                            eps=oea_eps,
                            shrinkage=oea_shrinkage,
                        )
                        for s in train_subjects
                    ],
                    axis=0,
                ),
                axis=0,
            )

            want_diag = (
                bool(do_diag)
                or (use_ridge and cert is not None)
                or (use_guard and guard is not None)
                or use_evidence
                or use_probe_mixup
                or use_probe_mixup_hard
                or use_iwcv
                or use_iwcv_ucb
                or use_dev
                or use_oracle
            )
            if use_oracle:
                want_diag = True

            lda_ev = None
            if str(oea_zo_objective) == "lda_nll" or use_evidence or use_stack or bool(do_diag):
                lda_ev = _compute_lda_evidence_params(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    class_order=class_labels,
                )

            opt_res = _optimize_qt_oea_zo(
                z_t=z_t,
                model=model,
                class_order=class_labels,
                d_ref=d_ref,
                lda_evidence=lda_ev,
                channel_names=channel_names,
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                pseudo_mode=str(oea_pseudo_mode),
                warm_start=str(oea_zo_warm_start),
                warm_iters=int(oea_zo_warm_iters),
                q_blend=float(oea_q_blend),
                objective=str(oea_zo_objective),
                transform=str(oea_zo_transform),
                localmix_neighbors=int(oea_zo_localmix_neighbors),
                localmix_self_bias=float(oea_zo_localmix_self_bias),
                infomax_lambda=float(oea_zo_infomax_lambda),
                reliable_metric=str(oea_zo_reliable_metric),
                reliable_threshold=float(oea_zo_reliable_threshold),
                reliable_alpha=float(oea_zo_reliable_alpha),
                trust_lambda=float(oea_zo_trust_lambda),
                trust_q0=str(oea_zo_trust_q0),
                marginal_mode=str(oea_zo_marginal_mode),
                marginal_beta=float(oea_zo_marginal_beta),
                marginal_tau=float(oea_zo_marginal_tau),
                marginal_prior=None,
                bilevel_iters=int(oea_zo_bilevel_iters),
                bilevel_temp=float(oea_zo_bilevel_temp),
                bilevel_step=float(oea_zo_bilevel_step),
                bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                drift_mode=str(oea_zo_drift_mode),
                drift_gamma=float(oea_zo_drift_gamma),
                drift_delta=float(oea_zo_drift_delta),
                min_improvement=float(oea_zo_min_improvement),
                holdout_fraction=float(oea_zo_holdout_fraction),
                fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                iters=int(oea_zo_iters),
                lr=float(oea_zo_lr),
                mu=float(oea_zo_mu),
                n_rotations=int(oea_zo_k),
                seed=int(oea_zo_seed) + int(test_subject) * 997,
                l2=float(oea_zo_l2),
                pseudo_confidence=float(oea_pseudo_confidence),
                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                pseudo_balance=bool(oea_pseudo_balance),
                return_diagnostics=bool(want_diag),
            )
            if want_diag:
                q_t, zo_diag = opt_res
            else:
                q_t = opt_res

            if zo_diag is not None:
                selected: dict | None = None
                if use_oracle:
                    best_rec = None
                    best_acc = -1.0
                    for rec in zo_diag.get("records", []):
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        yp = model.predict(apply_spatial_transform(Q, z_t))
                        acc = float(accuracy_score(y_test, yp))
                        if acc > best_acc:
                            best_acc = acc
                            best_rec = rec
                    selected = best_rec
                elif use_evidence:
                    selected = select_by_evidence_nll(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup:
                    selected = select_by_probe_mixup(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup_hard:
                    selected = select_by_probe_mixup_hard(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_iwcv:
                    selected = select_by_iwcv_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_iwcv_ucb:
                    selected = select_by_iwcv_ucb(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        kappa=float(oea_zo_iwcv_kappa),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_dev:
                    selected = select_by_dev_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_ridge_guard and cert is not None and guard is not None:
                    selected = select_by_guarded_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        feature_set="stacked" if use_stack else "base",
                    )
                elif use_ridge and cert is not None:
                    selected = select_by_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        n_classes=len(class_labels),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        feature_set="stacked" if use_stack else "base",
                    )
                elif use_guard and guard is not None:
                    selected = select_by_guarded_objective(
                        zo_diag.get("records", []),
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                    )
                if selected is not None:
                    q_t = np.asarray(selected.get("Q"), dtype=np.float64)

            X_test = apply_spatial_transform(q_t, z_t)
        elif alignment in {"ea_zo", "raw_zo"}:
            # Train on the current channel space, then adapt only Q_t at test time.
            # - ea_zo: the current space is EA-whitened (per-subject).
            # - raw_zo: the current space is the raw (preprocessed) channel space (no whitening).
            class_labels = tuple([str(c) for c in class_order])
            use_post_ea = str(oea_zo_transform) in {"local_mix_then_ea", "local_affine_then_ea"}
            if use_post_ea and alignment != "ea_zo":
                raise ValueError(
                    "oea_zo_transform in {'local_mix_then_ea','local_affine_then_ea'} "
                    "is only supported with alignment='ea_zo'."
                )

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
            y_test = subject_data[int(test_subject)].y

            # Optional: offline calibrated certificate / guard (trained only on source subjects in this fold).
            selector = str(oea_zo_selector)
            use_stack = selector == "calibrated_stack_ridge"
            use_ridge_guard = selector == "calibrated_ridge_guard"
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}
            use_evidence = selector == "evidence"
            use_probe_mixup = selector == "probe_mixup"
            use_probe_mixup_hard = selector == "probe_mixup_hard"
            use_iwcv = selector == "iwcv"
            use_iwcv_ucb = selector == "iwcv_ucb"
            use_dev = selector == "dev"
            use_oracle = selector == "oracle"
            cert = None
            guard = None
            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_calib_rows: List[np.ndarray] = []
                y_calib_rows: List[float] = []
                y_guard_rows: List[int] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    X_inner = np.concatenate([subject_data[s].X for s in inner_train], axis=0)
                    y_inner = np.concatenate([subject_data[s].y for s in inner_train], axis=0)
                    model_inner = fit_csp_lda(X_inner, y_inner, n_components=n_components)

                    diffs_inner = []
                    for s in inner_train:
                        diffs_inner.append(
                            class_cov_diff(
                                subject_data[int(s)].X,
                                subject_data[int(s)].y,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        )
                    d_ref_inner = np.mean(np.stack(diffs_inner, axis=0), axis=0)

                    z_pseudo = (
                        subject_data_raw[int(pseudo_t)].X if use_post_ea else subject_data[int(pseudo_t)].X
                    )
                    y_pseudo = subject_data[int(pseudo_t)].y

                    marginal_prior_inner: np.ndarray | None = None
                    if oea_zo_marginal_mode == "kl_prior":
                        if oea_zo_marginal_prior == "uniform":
                            marginal_prior_inner = np.ones(len(class_labels), dtype=np.float64) / float(
                                len(class_labels)
                            )
                        elif oea_zo_marginal_prior == "source":
                            counts = np.array([(y_inner == c).sum() for c in class_labels], dtype=np.float64)
                            marginal_prior_inner = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                        else:
                            if use_post_ea:
                                z_pseudo_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(
                                    z_pseudo
                                )
                                proba_id = model_inner.predict_proba(z_pseudo_ea)
                            else:
                                proba_id = model_inner.predict_proba(z_pseudo)
                            proba_id = _reorder_proba_columns(
                                proba_id, model_inner.classes_, list(class_order)
                            )
                            marginal_prior_inner = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))
                        mix = float(oea_zo_marginal_prior_mix)
                        if mix > 0.0 and marginal_prior_inner is not None:
                            u = np.ones_like(marginal_prior_inner) / float(marginal_prior_inner.shape[0])
                            marginal_prior_inner = (1.0 - mix) * marginal_prior_inner + mix * u
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))

                    lda_ev_inner = None
                    if str(oea_zo_objective) == "lda_nll" or use_stack:
                        lda_ev_inner = _compute_lda_evidence_params(
                            model=model_inner,
                            X_train=X_inner,
                            y_train=y_inner,
                            class_order=class_labels,
                        )

                    _qt_inner, diag_inner = _optimize_qt_oea_zo(
                        z_t=z_pseudo,
                        model=model_inner,
                        class_order=class_labels,
                        d_ref=d_ref_inner,
                        lda_evidence=lda_ev_inner,
                        channel_names=channel_names,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
                        localmix_neighbors=int(oea_zo_localmix_neighbors),
                        localmix_self_bias=float(oea_zo_localmix_self_bias),
                        infomax_lambda=float(oea_zo_infomax_lambda),
                        reliable_metric=str(oea_zo_reliable_metric),
                        reliable_threshold=float(oea_zo_reliable_threshold),
                        reliable_alpha=float(oea_zo_reliable_alpha),
                        trust_lambda=float(oea_zo_trust_lambda),
                        trust_q0=str(oea_zo_trust_q0),
                        marginal_mode=str(oea_zo_marginal_mode),
                        marginal_beta=float(oea_zo_marginal_beta),
                        marginal_tau=float(oea_zo_marginal_tau),
                        marginal_prior=marginal_prior_inner,
                        bilevel_iters=int(oea_zo_bilevel_iters),
                        bilevel_temp=float(oea_zo_bilevel_temp),
                        bilevel_step=float(oea_zo_bilevel_step),
                        bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                        bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        holdout_fraction=float(oea_zo_holdout_fraction),
                        fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                        iters=int(oea_zo_iters),
                        lr=float(oea_zo_lr),
                        mu=float(oea_zo_mu),
                        n_rotations=int(oea_zo_k),
                        seed=int(oea_zo_seed) + int(pseudo_t) * 997,
                        l2=float(oea_zo_l2),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        return_diagnostics=True,
                    )
                    recs = list(diag_inner.get("records", []))
                    if not recs:
                        continue
                    feats_list: List[np.ndarray] = []
                    acc_list: List[float] = []
                    acc_id: float | None = None
                    for rec in recs:
                        if use_stack:
                            feats, names = stacked_candidate_features_from_record(rec, n_classes=len(class_labels))
                        else:
                            feats, names = candidate_features_from_record(rec, n_classes=len(class_labels))
                        if feat_names is None:
                            feat_names = names
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        Xp = apply_spatial_transform(Q, z_pseudo)
                        yp = model_inner.predict(Xp)
                        acc = float(accuracy_score(y_pseudo, yp))
                        if str(rec.get("kind", "")) == "identity":
                            acc_id = acc
                        feats_list.append(feats)
                        acc_list.append(acc)
                    if acc_id is None:
                        continue
                    for feats, acc in zip(feats_list, acc_list):
                        improve = float(acc - float(acc_id))
                        y_calib_rows.append(float(improve))
                        y_guard_rows.append(1 if float(improve) >= float(oea_zo_calib_guard_margin) else 0)
                        X_calib_rows.append(feats)

                if X_calib_rows and feat_names is not None:
                    X_cal = np.stack(X_calib_rows, axis=0)
                    if use_ridge:
                        cert = train_ridge_certificate(
                            X_cal,
                            np.asarray(y_calib_rows, dtype=np.float64),
                            feature_names=feat_names,
                            alpha=float(oea_zo_calib_ridge_alpha),
                        )
                    if use_guard:
                        y_guard = np.asarray(y_guard_rows, dtype=int).reshape(-1)
                        if np.unique(y_guard).size >= 2:
                            guard = train_logistic_guard(
                                X_cal,
                                y_guard,
                                feature_names=feat_names,
                                c=float(oea_zo_calib_guard_c),
                            )
                        else:
                            guard = None
                else:
                    cert = None
                    guard = None

            diffs_train = []
            for s in train_subjects:
                diffs_train.append(
                    class_cov_diff(
                        subject_data[int(s)].X,
                        subject_data[int(s)].y,
                        class_order=class_labels,
                        eps=oea_eps,
                        shrinkage=oea_shrinkage,
                    )
                )
            d_ref = np.mean(np.stack(diffs_train, axis=0), axis=0)

            z_t = subject_data_raw[int(test_subject)].X if use_post_ea else subject_data[int(test_subject)].X
            z_test_base = z_t
            marginal_prior_vec: np.ndarray | None = None
            if oea_zo_marginal_mode == "kl_prior":
                if oea_zo_marginal_prior == "uniform":
                    marginal_prior_vec = np.ones(len(class_labels), dtype=np.float64) / float(len(class_labels))
                elif oea_zo_marginal_prior == "source":
                    counts = np.array([(y_train == c).sum() for c in class_labels], dtype=np.float64)
                    marginal_prior_vec = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                else:
                    # anchor_pred: use target predicted marginal at Q=I (EA), fixed during optimization.
                    if use_post_ea:
                        z_t_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(z_t)
                        proba_id = model.predict_proba(z_t_ea)
                    else:
                        proba_id = model.predict_proba(z_t)
                    proba_id = _reorder_proba_columns(proba_id, model.classes_, list(class_order))
                    marginal_prior_vec = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                    marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))
                mix = float(oea_zo_marginal_prior_mix)
                if mix > 0.0 and marginal_prior_vec is not None:
                    u = np.ones_like(marginal_prior_vec) / float(marginal_prior_vec.shape[0])
                    marginal_prior_vec = (1.0 - mix) * marginal_prior_vec + mix * u
                    marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

            want_diag = (
                bool(do_diag)
                or (use_ridge and cert is not None)
                or (use_guard and guard is not None)
                or use_evidence
                or use_probe_mixup
                or use_probe_mixup_hard
                or use_iwcv
                or use_iwcv_ucb
                or use_oracle
            )
            lda_ev = None
            if str(oea_zo_objective) == "lda_nll" or use_evidence or use_stack or bool(do_diag):
                lda_ev = _compute_lda_evidence_params(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    class_order=class_labels,
                )
            opt_res = _optimize_qt_oea_zo(
                z_t=z_t,
                model=model,
                class_order=class_labels,
                d_ref=d_ref,
                lda_evidence=lda_ev,
                channel_names=channel_names,
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                pseudo_mode=str(oea_pseudo_mode),
                warm_start=str(oea_zo_warm_start),
                warm_iters=int(oea_zo_warm_iters),
                q_blend=float(oea_q_blend),
                objective=str(oea_zo_objective),
                transform=str(oea_zo_transform),
                localmix_neighbors=int(oea_zo_localmix_neighbors),
                localmix_self_bias=float(oea_zo_localmix_self_bias),
                infomax_lambda=float(oea_zo_infomax_lambda),
                reliable_metric=str(oea_zo_reliable_metric),
                reliable_threshold=float(oea_zo_reliable_threshold),
                reliable_alpha=float(oea_zo_reliable_alpha),
                trust_lambda=float(oea_zo_trust_lambda),
                trust_q0=str(oea_zo_trust_q0),
                marginal_mode=str(oea_zo_marginal_mode),
                marginal_beta=float(oea_zo_marginal_beta),
                marginal_tau=float(oea_zo_marginal_tau),
                marginal_prior=marginal_prior_vec,
                bilevel_iters=int(oea_zo_bilevel_iters),
                bilevel_temp=float(oea_zo_bilevel_temp),
                bilevel_step=float(oea_zo_bilevel_step),
                bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                drift_mode=str(oea_zo_drift_mode),
                drift_gamma=float(oea_zo_drift_gamma),
                drift_delta=float(oea_zo_drift_delta),
                min_improvement=float(oea_zo_min_improvement),
                holdout_fraction=float(oea_zo_holdout_fraction),
                fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                iters=int(oea_zo_iters),
                lr=float(oea_zo_lr),
                mu=float(oea_zo_mu),
                n_rotations=int(oea_zo_k),
                seed=int(oea_zo_seed) + int(test_subject) * 997,
                l2=float(oea_zo_l2),
                pseudo_confidence=float(oea_pseudo_confidence),
                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                pseudo_balance=bool(oea_pseudo_balance),
                return_diagnostics=bool(want_diag),
            )
            if want_diag:
                q_t, zo_diag = opt_res
            else:
                q_t = opt_res

            if zo_diag is not None:
                selected: dict | None = None
                if use_oracle:
                    best_rec = None
                    best_acc = -1.0
                    for rec in zo_diag.get("records", []):
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        yp = model.predict(apply_spatial_transform(Q, z_t))
                        acc = float(accuracy_score(y_test, yp))
                        if acc > best_acc:
                            best_acc = acc
                            best_rec = rec
                    selected = best_rec
                elif use_evidence:
                    selected = select_by_evidence_nll(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup:
                    selected = select_by_probe_mixup(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup_hard:
                    selected = select_by_probe_mixup_hard(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_iwcv:
                    selected = select_by_iwcv_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_iwcv_ucb:
                    selected = select_by_iwcv_ucb(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        kappa=float(oea_zo_iwcv_kappa),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_dev:
                    selected = select_by_dev_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_ridge_guard and cert is not None and guard is not None:
                    selected = select_by_guarded_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                    )
                elif use_ridge and cert is not None:
                    selected = select_by_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        n_classes=len(class_labels),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        feature_set="stacked" if use_stack else "base",
                    )
                elif use_guard and guard is not None:
                    selected = select_by_guarded_objective(
                        zo_diag.get("records", []),
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                    )
                if selected is not None:
                    q_t = np.asarray(selected.get("Q"), dtype=np.float64)
            X_test = apply_spatial_transform(q_t, z_t)
        elif alignment == "oea_cov":
            # OEA (cov-eig) selection: pick Q_s = U_ref U_sᵀ, where U_s is eigenbasis of C_s
            # and U_ref from the average covariance of the training subjects.
            ea_by_subject: Dict[int, EuclideanAligner] = {}
            covs_train: List[np.ndarray] = []
            for s in train_subjects:
                ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(subject_data[s].X)
                ea_by_subject[int(s)] = ea
                covs_train.append(ea.cov_)
            c_ref = np.mean(np.stack(covs_train, axis=0), axis=0)
            _evals_ref, u_ref = sorted_eigh(c_ref)

            def _align_one(subj: int) -> np.ndarray:
                ea = ea_by_subject.get(int(subj))
                if ea is None:
                    ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(
                        subject_data[int(subj)].X
                    )
                    ea_by_subject[int(subj)] = ea
                z = ea.transform(subject_data[int(subj)].X)
                q = u_ref @ ea.eigvecs_.T
                q = blend_with_identity(q, oea_q_blend)
                return apply_spatial_transform(q, z)

            X_train = np.concatenate([_align_one(s) for s in train_subjects], axis=0)
            y_train = np.concatenate([subject_data[s].y for s in train_subjects], axis=0)
            X_test = _align_one(test_subject)
            y_test = subject_data[test_subject].y
        else:
            # alignment in {"oea","oea_zo"}: optimistic selection based on a discriminative
            # covariance signature (binary: Δ=Cov(c1)-Cov(c0); multiclass: between-class scatter).
            class_labels = tuple([str(c) for c in class_order])

            # 1) EA whitening for each subject (no Q yet).
            ea_by_subject: Dict[int, EuclideanAligner] = {}
            z_by_subject: Dict[int, np.ndarray] = {}
            for s in subjects:
                ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(subject_data[s].X)
                ea_by_subject[int(s)] = ea
                z_by_subject[int(s)] = ea.transform(subject_data[s].X)

            # 2) Build training reference Δ_ref from labeled source subjects.
            diffs_train = []
            for s in train_subjects:
                diffs_train.append(
                    class_cov_diff(
                        z_by_subject[int(s)],
                        subject_data[int(s)].y,
                        class_order=class_labels,
                        eps=oea_eps,
                        shrinkage=oea_shrinkage,
                    )
                )
            d_ref = np.mean(np.stack(diffs_train, axis=0), axis=0)

            # 3) Align each training subject by choosing Q_s that best matches Δ_ref.
            def _align_train_subject(s: int) -> np.ndarray:
                d_s = class_cov_diff(
                    z_by_subject[int(s)],
                    subject_data[int(s)].y,
                    class_order=class_labels,
                    eps=oea_eps,
                    shrinkage=oea_shrinkage,
                )
                q_s = orthogonal_align_symmetric(d_s, d_ref)
                q_s = blend_with_identity(q_s, oea_q_blend)
                return apply_spatial_transform(q_s, z_by_subject[int(s)])

            X_train = np.concatenate([_align_train_subject(s) for s in train_subjects], axis=0)
            y_train = np.concatenate([subject_data[s].y for s in train_subjects], axis=0)

            # 4) Train the classifier once (frozen after this).
            model = fit_csp_lda(X_train, y_train, n_components=n_components)

            # 5) Target subject: select Q_t using *unlabeled* target data (no classifier update).
            z_t = z_by_subject[int(test_subject)]
            if alignment == "oea":
                q_t = np.eye(z_t.shape[1], dtype=np.float64)
                for _ in range(int(max(0, oea_pseudo_iters))):
                    X_t_cur = apply_spatial_transform(q_t, z_t)
                    proba = model.predict_proba(X_t_cur)
                    proba = _reorder_proba_columns(proba, model.classes_, list(class_labels))

                    if oea_pseudo_mode == "soft":
                        d_t = _soft_class_cov_diff(
                            z_t,
                            proba=proba,
                            class_order=class_labels,
                            eps=oea_eps,
                            shrinkage=oea_shrinkage,
                        )
                    else:
                        y_pseudo = np.asarray(model.predict(X_t_cur))
                        keep = _select_pseudo_indices(
                            y_pseudo=y_pseudo,
                            proba=proba,
                            class_order=class_labels,
                            confidence=float(oea_pseudo_confidence),
                            topk_per_class=int(oea_pseudo_topk_per_class),
                            balance=bool(oea_pseudo_balance),
                        )
                        if keep.size == 0:
                            break
                        d_t = class_cov_diff(
                            z_t[keep],
                            y_pseudo[keep],
                            class_order=class_labels,
                            eps=oea_eps,
                            shrinkage=oea_shrinkage,
                        )
                    q_t = orthogonal_align_symmetric(d_t, d_ref)
                    q_t = blend_with_identity(q_t, oea_q_blend)
            else:
                # If using KL(π||p̄) in the ZO objective, build π per fold.
                marginal_prior_vec: np.ndarray | None = None
                if oea_zo_marginal_mode == "kl_prior":
                    if oea_zo_marginal_prior == "uniform":
                        marginal_prior_vec = np.ones(len(class_labels), dtype=np.float64) / float(
                            len(class_labels)
                        )
                    elif oea_zo_marginal_prior == "source":
                        counts = np.array([(y_train == c).sum() for c in class_labels], dtype=np.float64)
                        marginal_prior_vec = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                else:
                    if use_post_ea:
                        z_t_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(z_t)
                        proba_id = model.predict_proba(z_t_ea)
                    else:
                        proba_id = model.predict_proba(z_t)
                    proba_id = _reorder_proba_columns(proba_id, model.classes_, list(class_order))
                    marginal_prior_vec = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                    marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))
                    mix = float(oea_zo_marginal_prior_mix)
                    if mix > 0.0 and marginal_prior_vec is not None:
                        u = np.ones_like(marginal_prior_vec) / float(marginal_prior_vec.shape[0])
                        marginal_prior_vec = (1.0 - mix) * marginal_prior_vec + mix * u
                        marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

                selector = str(oea_zo_selector)
                use_evidence = selector == "evidence"
                use_probe_mixup = selector == "probe_mixup"
                use_probe_mixup_hard = selector == "probe_mixup_hard"
                use_iwcv = selector == "iwcv"
                use_iwcv_ucb = selector == "iwcv_ucb"
                use_dev = selector == "dev"
                use_oracle = selector == "oracle"
                want_diag = (
                    bool(do_diag)
                    or use_evidence
                    or use_probe_mixup
                    or use_probe_mixup_hard
                    or use_iwcv
                    or use_iwcv_ucb
                    or use_dev
                    or use_oracle
                )
                lda_ev = None
                if str(oea_zo_objective) == "lda_nll" or use_evidence or bool(do_diag):
                    lda_ev = _compute_lda_evidence_params(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        class_order=class_labels,
                    )
                opt_res = _optimize_qt_oea_zo(
                    z_t=z_t,
                    model=model,
                    class_order=class_labels,
                    d_ref=d_ref,
                    lda_evidence=lda_ev,
                    channel_names=channel_names,
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    pseudo_mode=str(oea_pseudo_mode),
                    warm_start=str(oea_zo_warm_start),
                    warm_iters=int(oea_zo_warm_iters),
                    q_blend=float(oea_q_blend),
                    objective=str(oea_zo_objective),
                    transform=str(oea_zo_transform),
                    localmix_neighbors=int(oea_zo_localmix_neighbors),
                    localmix_self_bias=float(oea_zo_localmix_self_bias),
                    infomax_lambda=float(oea_zo_infomax_lambda),
                    reliable_metric=str(oea_zo_reliable_metric),
                    reliable_threshold=float(oea_zo_reliable_threshold),
                    reliable_alpha=float(oea_zo_reliable_alpha),
                    trust_lambda=float(oea_zo_trust_lambda),
                    trust_q0=str(oea_zo_trust_q0),
                    marginal_mode=str(oea_zo_marginal_mode),
                    marginal_beta=float(oea_zo_marginal_beta),
                    marginal_tau=float(oea_zo_marginal_tau),
                    marginal_prior=marginal_prior_vec,
                    bilevel_iters=int(oea_zo_bilevel_iters),
                    bilevel_temp=float(oea_zo_bilevel_temp),
                    bilevel_step=float(oea_zo_bilevel_step),
                    bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                    bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    min_improvement=float(oea_zo_min_improvement),
                    holdout_fraction=float(oea_zo_holdout_fraction),
                    fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                    iters=int(oea_zo_iters),
                    lr=float(oea_zo_lr),
                    mu=float(oea_zo_mu),
                    n_rotations=int(oea_zo_k),
                    seed=int(oea_zo_seed) + int(test_subject) * 997,
                    l2=float(oea_zo_l2),
                    pseudo_confidence=float(oea_pseudo_confidence),
                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                    pseudo_balance=bool(oea_pseudo_balance),
                    return_diagnostics=bool(want_diag),
                )
                if want_diag:
                    q_t, zo_diag = opt_res
                    if use_oracle:
                        y_test = subject_data[test_subject].y
                        best_rec = None
                        best_acc = -1.0
                        for rec in zo_diag.get("records", []):
                            Q = np.asarray(rec.get("Q"), dtype=np.float64)
                            yp = model.predict(apply_spatial_transform(Q, z_t))
                            acc = float(accuracy_score(y_test, yp))
                            if acc > best_acc:
                                best_acc = acc
                                best_rec = rec
                        if best_rec is not None:
                            q_t = np.asarray(best_rec.get("Q"), dtype=np.float64)
                    elif use_evidence:
                        sel = select_by_evidence_nll(
                            zo_diag.get("records", []),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_probe_mixup:
                        sel = select_by_probe_mixup(
                            zo_diag.get("records", []),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_probe_mixup_hard:
                        sel = select_by_probe_mixup_hard(
                            zo_diag.get("records", []),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_iwcv:
                        sel = select_by_iwcv_nll(
                            zo_diag.get("records", []),
                            model=model,
                            z_source=X_train,
                            y_source=y_train,
                            z_target=z_t,
                            class_order=class_labels,
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                            seed=int(oea_zo_seed) + int(test_subject) * 997,
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_iwcv_ucb:
                        sel = select_by_iwcv_ucb(
                            zo_diag.get("records", []),
                            model=model,
                            z_source=X_train,
                            y_source=y_train,
                            z_target=z_t,
                            class_order=class_labels,
                            kappa=float(oea_zo_iwcv_kappa),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                            seed=int(oea_zo_seed) + int(test_subject) * 997,
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_dev:
                        sel = select_by_dev_nll(
                            zo_diag.get("records", []),
                            model=model,
                            z_source=X_train,
                            y_source=y_train,
                            z_target=z_t,
                            class_order=class_labels,
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                            seed=int(oea_zo_seed) + int(test_subject) * 997,
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                else:
                    q_t = opt_res
                z_test_base = z_t

            X_test = apply_spatial_transform(q_t, z_t)
            y_test = subject_data[test_subject].y

        if model is None:
            if alignment in {"fbcsp", "ea_fbcsp"}:
                # Default filterbank within the base paper_fir 8–30 Hz protocol band.
                bands = [
                    (8.0, 12.0),
                    (10.0, 14.0),
                    (12.0, 16.0),
                    (14.0, 18.0),
                    (16.0, 20.0),
                    (18.0, 22.0),
                    (20.0, 24.0),
                    (22.0, 26.0),
                    (24.0, 28.0),
                    (26.0, 30.0),
                ]
                fb_n_components = max(2, min(4, int(n_components)))
                model = fit_fbcsp_lda(
                    X_train,
                    y_train,
                    bands=bands,
                    sfreq=float(sfreq),
                    n_components=fb_n_components,
                    filter_order=4,
                    multiclass_strategy=str(fbcsp_multiclass_strategy),
                    select_k=24,
                )
            else:
                model = fit_csp_lda(X_train, y_train, n_components=n_components)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_proba = _reorder_proba_columns(y_proba, model.classes_, class_order)

        if do_diag and zo_diag is not None:
            _write_zo_diagnostics(
                zo_diag,
                out_dir=Path(diagnostics_dir),
                tag=str(diagnostics_tag),
                subject=int(test_subject),
                model=model,
                z_t=z_test_base if z_test_base is not None else subject_data[int(test_subject)].X,
                y_true=y_test,
                class_order=class_order,
            )

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )

        fold_rows.append(
            FoldResult(
                subject=int(test_subject),
                n_train=int(len(y_train)),
                n_test=int(len(y_test)),
                **metrics,
            )
        )
        models_by_subject[int(test_subject)] = model
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(test_subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))

        # IMPORTANT (memory): for alignment='ea_stack_multi_safe', the per-train-set cache is
        # typically *not* reusable across LOSO folds (each fold has a different train-subject set).
        # Keeping all fold bundles can therefore grow RSS linearly with #subjects and trigger OOM
        # on large datasets (e.g., PhysionetMI). Clearing here keeps memory bounded while
        # preserving within-fold caching (holdout / inner folds).
        if alignment == "ea_stack_multi_safe":
            stack_bundle_cache.clear()

        # Heuristic memory pressure relief for large datasets (e.g., high-density MI):
        # after each fold, force collection and (optionally) trim allocator arenas.
        if int(len(y_train)) >= 5000 or int(subject_data_raw[int(test_subject)].X.shape[1]) >= 64:
            gc.collect()
            _maybe_malloc_trim()

    results_df = pd.DataFrame([asdict(r) for r in fold_rows]).sort_values("subject")
    if extra_rows is not None and extra_rows:
        extra_df = pd.DataFrame(extra_rows).sort_values("subject")
        results_df = results_df.merge(extra_df, on="subject", how="left")
    y_true_cat = np.concatenate(y_true_all, axis=0)
    y_pred_cat = np.concatenate(y_pred_all, axis=0)
    y_proba_cat = np.concatenate(y_proba_all, axis=0)
    subj_cat = np.concatenate(subj_all, axis=0)
    trial_cat = np.concatenate(trial_all, axis=0)

    pred_df = pd.DataFrame(
        {
            "subject": subj_cat,
            "trial": trial_cat,
            "y_true": y_true_cat,
            "y_pred": y_pred_cat,
        }
    )
    for i, c in enumerate(list(class_order)):
        pred_df[f"proba_{c}"] = y_proba_cat[:, int(i)]

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )


def cross_session_within_subject_evaluation(
    subject_session_data: Dict[int, Dict[str, SubjectData]],
    *,
    train_sessions: Sequence[str],
    test_sessions: Sequence[str],
    class_order: Sequence[str],
    channel_names: Sequence[str] | None = None,
    n_components: int = 4,
    average: str = "macro",
    alignment: str = "ea",
    oea_eps: float = 1e-10,
    oea_shrinkage: float = 0.0,
    oea_pseudo_iters: int = 2,
    oea_q_blend: float = 1.0,
    oea_pseudo_mode: str = "hard",
    oea_pseudo_confidence: float = 0.0,
    oea_pseudo_topk_per_class: int = 0,
    oea_pseudo_balance: bool = False,
    oea_zo_objective: str = "entropy",
    oea_zo_transform: str = "orthogonal",
    oea_zo_localmix_neighbors: int = 4,
    oea_zo_localmix_self_bias: float = 3.0,
    oea_zo_infomax_lambda: float = 1.0,
    oea_zo_reliable_metric: str = "none",
    oea_zo_reliable_threshold: float = 0.0,
    oea_zo_reliable_alpha: float = 10.0,
    oea_zo_trust_lambda: float = 0.0,
    oea_zo_trust_q0: str = "identity",
    oea_zo_marginal_mode: str = "none",
    oea_zo_marginal_beta: float = 0.0,
    oea_zo_marginal_tau: float = 0.05,
    oea_zo_marginal_prior: str = "uniform",
    oea_zo_marginal_prior_mix: float = 0.0,
    oea_zo_bilevel_iters: int = 5,
    oea_zo_bilevel_temp: float = 1.0,
    oea_zo_bilevel_step: float = 1.0,
    oea_zo_bilevel_coverage_target: float = 0.5,
    oea_zo_bilevel_coverage_power: float = 1.0,
    oea_zo_drift_mode: str = "none",
    oea_zo_drift_gamma: float = 0.0,
    oea_zo_drift_delta: float = 0.0,
    oea_zo_selector: str = "objective",
    oea_zo_iwcv_kappa: float = 1.0,
    oea_zo_calib_ridge_alpha: float = 1.0,
    oea_zo_calib_max_subjects: int = 0,
    oea_zo_calib_seed: int = 0,
    oea_zo_calib_guard_c: float = 1.0,
    oea_zo_calib_guard_threshold: float = 0.5,
    oea_zo_calib_guard_margin: float = 0.0,
    oea_zo_min_improvement: float = 0.0,
    oea_zo_holdout_fraction: float = 0.0,
    oea_zo_warm_start: str = "none",
    oea_zo_warm_iters: int = 1,
    oea_zo_fallback_min_marginal_entropy: float = 0.0,
    oea_zo_iters: int = 30,
    oea_zo_lr: float = 0.5,
    oea_zo_mu: float = 0.1,
    oea_zo_k: int = 50,
    oea_zo_seed: int = 0,
    oea_zo_l2: float = 0.0,
    diagnostics_dir: Path | None = None,
    diagnostics_subjects: Sequence[int] = (),
    diagnostics_tag: str = "",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[int, TrainedModel],
]:
    """Within-subject cross-session evaluation.

    For each subject, train on `train_sessions` and test on `test_sessions`.
    This is useful for single-subject cross-session domain shift (often smaller than cross-subject).
    """

    train_sessions = [str(s) for s in train_sessions]
    test_sessions = [str(s) for s in test_sessions]
    class_order = [str(c) for c in class_order]
    if alignment not in {"none", "ea", "rpa", "ea_zo", "rpa_zo", "tsa", "tsa_zo", "oea_cov", "oea", "oea_zo"}:
        raise ValueError(
            "alignment must be one of: "
            "'none', 'ea', 'rpa', 'ea_zo', 'rpa_zo', 'tsa', 'tsa_zo', 'oea_cov', 'oea', 'oea_zo'"
        )
    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")
    if int(oea_zo_localmix_neighbors) < 0:
        raise ValueError("oea_zo_localmix_neighbors must be >= 0.")
    if float(oea_zo_localmix_self_bias) < 0.0:
        raise ValueError("oea_zo_localmix_self_bias must be >= 0.")

    use_post_ea = str(oea_zo_transform) in {"local_mix_then_ea", "local_affine_then_ea"}
    if use_post_ea and alignment != "ea_zo":
        raise ValueError(
            "oea_zo_transform in {'local_mix_then_ea','local_affine_then_ea'} "
            "is only supported with alignment='ea_zo'."
        )

    subjects = sorted(subject_session_data.keys())
    if not subjects:
        raise ValueError("Empty subject_session_data.")

    diag_subjects_set = {int(s) for s in diagnostics_subjects} if diagnostics_subjects else set()

    fold_rows: List[FoldResult] = []
    models_by_subject: Dict[int, TrainedModel] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []
    train_sess_all: List[np.ndarray] = []
    test_sess_all: List[np.ndarray] = []

    class_labels = tuple(class_order)

    for subject in subjects:
        sess_map = subject_session_data[int(subject)]
        available = sorted(sess_map.keys())
        missing_train = [s for s in train_sessions if s not in sess_map]
        missing_test = [s for s in test_sessions if s not in sess_map]
        if missing_train or missing_test:
            raise ValueError(
                f"Subject {int(subject)} missing requested sessions. "
                f"train_missing={missing_train}, test_missing={missing_test}, available={available}"
            )

        X_train = np.concatenate([sess_map[s].X for s in train_sessions], axis=0)
        y_train = np.concatenate([sess_map[s].y for s in train_sessions], axis=0)
        X_test_raw = np.concatenate([sess_map[s].X for s in test_sessions], axis=0)
        y_test = np.concatenate([sess_map[s].y for s in test_sessions], axis=0)

        do_diag = diagnostics_dir is not None and int(subject) in diag_subjects_set
        zo_diag: dict | None = None
        z_test_base: np.ndarray | None = None

        if alignment == "none":
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
            X_test = X_test_raw
        else:
            base_aligner_cls = (
                LogEuclideanAligner
                if alignment in {"rpa", "rpa_zo", "tsa", "tsa_zo"}
                else EuclideanAligner
            )
            base_train = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_train)
            base_test = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_test_raw)
            z_train = base_train.transform(X_train)
            z_test = base_test.transform(X_test_raw)
            z_test_base = X_test_raw if use_post_ea else z_test

            if alignment in {"ea", "rpa"}:
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                X_test = z_test
            elif alignment == "oea_cov":
                # Session-wise cov-eig alignment: align test eigen-basis to train eigen-basis.
                _evals_ref, u_ref = sorted_eigh(base_train.cov_)
                q_t = u_ref @ base_test.eigvecs_.T
                q_t = blend_with_identity(q_t, oea_q_blend)
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                X_test = apply_spatial_transform(q_t, z_test)
            elif alignment == "tsa":
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                q_tsa = _compute_tsa_target_rotation(
                    z_train=z_train,
                    y_train=y_train,
                    z_target=z_test,
                    model=model,
                    class_order=class_labels,
                    pseudo_mode=str(oea_pseudo_mode),
                    pseudo_iters=int(max(0, oea_pseudo_iters)),
                    q_blend=float(oea_q_blend),
                    pseudo_confidence=float(oea_pseudo_confidence),
                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                    pseudo_balance=bool(oea_pseudo_balance),
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                )
                X_test = apply_spatial_transform(q_tsa, z_test)
            else:
                # Discriminative signature reference from labeled train session(s).
                d_ref = class_cov_diff(
                    z_train,
                    y_train,
                    class_order=class_labels,
                    eps=oea_eps,
                    shrinkage=oea_shrinkage,
                )

                if alignment == "oea":
                    model = fit_csp_lda(z_train, y_train, n_components=n_components)
                    q_t = np.eye(z_test.shape[1], dtype=np.float64)
                    for _ in range(int(max(0, oea_pseudo_iters))):
                        X_t_cur = apply_spatial_transform(q_t, z_test)
                        proba = model.predict_proba(X_t_cur)
                        proba = _reorder_proba_columns(proba, model.classes_, list(class_labels))

                        if oea_pseudo_mode == "soft":
                            d_t = _soft_class_cov_diff(
                                z_test,
                                proba=proba,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        else:
                            y_pseudo = np.asarray(model.predict(X_t_cur))
                            keep = _select_pseudo_indices(
                                y_pseudo=y_pseudo,
                                proba=proba,
                                class_order=class_labels,
                                confidence=float(oea_pseudo_confidence),
                                topk_per_class=int(oea_pseudo_topk_per_class),
                                balance=bool(oea_pseudo_balance),
                            )
                            if keep.size == 0:
                                break
                            d_t = class_cov_diff(
                                z_test[keep],
                                y_pseudo[keep],
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        q_t = orthogonal_align_symmetric(d_t, d_ref)
                        q_t = blend_with_identity(q_t, oea_q_blend)
                    X_test = apply_spatial_transform(q_t, z_test)
                else:
                    # alignment == "ea_zo" or "oea_zo": freeze classifier, optimize Q_t on unlabeled target.
                    model = fit_csp_lda(z_train, y_train, n_components=n_components)
                    if alignment == "tsa_zo":
                        q_base = _compute_tsa_target_rotation(
                            z_train=z_train,
                            y_train=y_train,
                            z_target=z_test,
                            model=model,
                            class_order=class_labels,
                            pseudo_mode=str(oea_pseudo_mode),
                            pseudo_iters=int(max(0, oea_pseudo_iters)),
                            q_blend=float(oea_q_blend),
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                            eps=float(oea_eps),
                            shrinkage=float(oea_shrinkage),
                        )
                        z_test_base = apply_spatial_transform(q_base, z_test)
                    else:
                        z_test_base = z_test

                    selector = str(oea_zo_selector)
                    use_stack = selector == "calibrated_stack_ridge"
                    use_ridge_guard = selector == "calibrated_ridge_guard"
                    use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
                    use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}
                    use_evidence = selector == "evidence"
                    use_probe_mixup = selector == "probe_mixup"
                    use_probe_mixup_hard = selector == "probe_mixup_hard"
                    use_iwcv = selector == "iwcv"
                    use_iwcv_ucb = selector == "iwcv_ucb"
                    use_dev = selector == "dev"
                    use_oracle = selector == "oracle"
                    cert = None
                    guard = None

                    if use_ridge or use_guard:
                        rng = np.random.RandomState(int(oea_zo_calib_seed) + int(subject) * 997)
                        calib_subjects = [s for s in subjects if s != int(subject)]
                        if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(
                            calib_subjects
                        ):
                            rng.shuffle(calib_subjects)
                            calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                        X_calib_rows: List[np.ndarray] = []
                        y_calib_rows: List[float] = []
                        y_guard_rows: List[int] = []
                        feat_names: tuple[str, ...] | None = None

                        for pseudo_t in calib_subjects:
                            pseudo_map = subject_session_data[int(pseudo_t)]
                            X_tr_p = np.concatenate([pseudo_map[s].X for s in train_sessions], axis=0)
                            y_tr_p = np.concatenate([pseudo_map[s].y for s in train_sessions], axis=0)
                            X_te_p = np.concatenate([pseudo_map[s].X for s in test_sessions], axis=0)
                            y_te_p = np.concatenate([pseudo_map[s].y for s in test_sessions], axis=0)

                            base_tr_p = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_tr_p)
                            base_te_p = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_te_p)
                            z_tr_p = base_tr_p.transform(X_tr_p)
                            z_te_p = base_te_p.transform(X_te_p)

                            model_p = fit_csp_lda(z_tr_p, y_tr_p, n_components=n_components)
                            d_ref_p = class_cov_diff(
                                z_tr_p,
                                y_tr_p,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )

                            z_te_p_base = X_te_p if use_post_ea else z_te_p
                            if alignment == "tsa_zo":
                                q_base_p = _compute_tsa_target_rotation(
                                    z_train=z_tr_p,
                                    y_train=y_tr_p,
                                    z_target=z_te_p,
                                    model=model_p,
                                    class_order=class_labels,
                                    pseudo_mode=str(oea_pseudo_mode),
                                    pseudo_iters=int(max(0, oea_pseudo_iters)),
                                    q_blend=float(oea_q_blend),
                                    pseudo_confidence=float(oea_pseudo_confidence),
                                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                                    pseudo_balance=bool(oea_pseudo_balance),
                                    eps=float(oea_eps),
                                    shrinkage=float(oea_shrinkage),
                                )
                                z_te_p_base = apply_spatial_transform(q_base_p, z_te_p)

                            marginal_prior_p: np.ndarray | None = None
                            if oea_zo_marginal_mode == "kl_prior":
                                if oea_zo_marginal_prior == "uniform":
                                    marginal_prior_p = np.ones(len(class_labels), dtype=np.float64) / float(
                                        len(class_labels)
                                    )
                                elif oea_zo_marginal_prior == "source":
                                    counts = np.array([(y_tr_p == c).sum() for c in class_labels], dtype=np.float64)
                                    marginal_prior_p = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                                else:
                                    proba_id = model_p.predict_proba(z_te_p if use_post_ea else z_te_p_base)
                                    proba_id = _reorder_proba_columns(
                                        proba_id, model_p.classes_, list(class_labels)
                                    )
                                    marginal_prior_p = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                                    marginal_prior_p = marginal_prior_p / float(np.sum(marginal_prior_p))
                                mix = float(oea_zo_marginal_prior_mix)
                                if mix > 0.0 and marginal_prior_p is not None:
                                    u = np.ones_like(marginal_prior_p) / float(marginal_prior_p.shape[0])
                                    marginal_prior_p = (1.0 - mix) * marginal_prior_p + mix * u
                                    marginal_prior_p = marginal_prior_p / float(np.sum(marginal_prior_p))

                            lda_ev_p = None
                            if str(oea_zo_objective) == "lda_nll" or use_stack:
                                lda_ev_p = _compute_lda_evidence_params(
                                    model=model_p,
                                    X_train=z_tr_p,
                                    y_train=y_tr_p,
                                    class_order=class_labels,
                                )

                            _q_sel, diag_p = _optimize_qt_oea_zo(
                                z_t=z_te_p_base,
                                model=model_p,
                                class_order=class_labels,
                                d_ref=d_ref_p,
                                lda_evidence=lda_ev_p,
                                channel_names=channel_names,
                                eps=float(oea_eps),
                                shrinkage=float(oea_shrinkage),
                                pseudo_mode=str(oea_pseudo_mode),
                                warm_start=str(oea_zo_warm_start),
                                warm_iters=int(oea_zo_warm_iters),
                                q_blend=float(oea_q_blend),
                                objective=str(oea_zo_objective),
                                transform=str(oea_zo_transform),
                                localmix_neighbors=int(oea_zo_localmix_neighbors),
                                localmix_self_bias=float(oea_zo_localmix_self_bias),
                                infomax_lambda=float(oea_zo_infomax_lambda),
                                reliable_metric=str(oea_zo_reliable_metric),
                                reliable_threshold=float(oea_zo_reliable_threshold),
                                reliable_alpha=float(oea_zo_reliable_alpha),
                                trust_lambda=float(oea_zo_trust_lambda),
                                trust_q0=str(oea_zo_trust_q0),
                                marginal_mode=str(oea_zo_marginal_mode),
                                marginal_beta=float(oea_zo_marginal_beta),
                                marginal_tau=float(oea_zo_marginal_tau),
                                marginal_prior=marginal_prior_p,
                                bilevel_iters=int(oea_zo_bilevel_iters),
                                bilevel_temp=float(oea_zo_bilevel_temp),
                                bilevel_step=float(oea_zo_bilevel_step),
                                bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                                bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                holdout_fraction=float(oea_zo_holdout_fraction),
                                fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                                iters=int(oea_zo_iters),
                                lr=float(oea_zo_lr),
                                mu=float(oea_zo_mu),
                                n_rotations=int(oea_zo_k),
                                seed=int(oea_zo_seed) + int(pseudo_t) * 997,
                                l2=float(oea_zo_l2),
                                pseudo_confidence=float(oea_pseudo_confidence),
                                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                                pseudo_balance=bool(oea_pseudo_balance),
                                return_diagnostics=True,
                            )

                            recs = list(diag_p.get("records", []))
                            if not recs:
                                continue
                            feats_list: List[np.ndarray] = []
                            acc_list: List[float] = []
                            acc_id: float | None = None
                            for rec in recs:
                                if use_stack:
                                    feats, names = stacked_candidate_features_from_record(
                                        rec, n_classes=len(class_labels)
                                    )
                                else:
                                    feats, names = candidate_features_from_record(rec, n_classes=len(class_labels))
                                if feat_names is None:
                                    feat_names = names
                                Q = np.asarray(rec.get("Q"), dtype=np.float64)
                                yp = model_p.predict(apply_spatial_transform(Q, z_te_p_base))
                                acc = float(accuracy_score(y_te_p, yp))
                                if str(rec.get("kind", "")) == "identity":
                                    acc_id = acc
                                feats_list.append(feats)
                                acc_list.append(acc)
                            if acc_id is None:
                                continue
                            for feats, acc in zip(feats_list, acc_list):
                                improve = float(acc - float(acc_id))
                                y_calib_rows.append(float(improve))
                                y_guard_rows.append(
                                    1 if float(improve) >= float(oea_zo_calib_guard_margin) else 0
                                )
                                X_calib_rows.append(feats)

                        if X_calib_rows and feat_names is not None:
                            X_cal = np.stack(X_calib_rows, axis=0)
                            if use_ridge:
                                cert = train_ridge_certificate(
                                    X_cal,
                                    np.asarray(y_calib_rows, dtype=np.float64),
                                    feature_names=feat_names,
                                    alpha=float(oea_zo_calib_ridge_alpha),
                                )
                            if use_guard:
                                y_guard = np.asarray(y_guard_rows, dtype=int).reshape(-1)
                                if np.unique(y_guard).size >= 2:
                                    guard = train_logistic_guard(
                                        X_cal,
                                        y_guard,
                                        feature_names=feat_names,
                                        c=float(oea_zo_calib_guard_c),
                                    )
                                else:
                                    guard = None

                    marginal_prior_vec: np.ndarray | None = None
                    if oea_zo_marginal_mode == "kl_prior":
                        if oea_zo_marginal_prior == "uniform":
                            marginal_prior_vec = np.ones(len(class_labels), dtype=np.float64) / float(
                                len(class_labels)
                            )
                        elif oea_zo_marginal_prior == "source":
                            counts = np.array([(y_train == c).sum() for c in class_labels], dtype=np.float64)
                            marginal_prior_vec = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                        else:
                            proba_id = model.predict_proba(z_test)
                            proba_id = _reorder_proba_columns(proba_id, model.classes_, list(class_labels))
                            marginal_prior_vec = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                            marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))
                        mix = float(oea_zo_marginal_prior_mix)
                        if mix > 0.0 and marginal_prior_vec is not None:
                            u = np.ones_like(marginal_prior_vec) / float(marginal_prior_vec.shape[0])
                            marginal_prior_vec = (1.0 - mix) * marginal_prior_vec + mix * u
                            marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

                    want_diag = (
                        bool(do_diag)
                        or (use_ridge and cert is not None)
                        or (use_guard and guard is not None)
                        or use_evidence
                        or use_probe_mixup
                        or use_probe_mixup_hard
                        or use_iwcv
                        or use_iwcv_ucb
                        or use_dev
                        or use_oracle
                    )
                    if use_oracle:
                        want_diag = True
                    lda_ev = None
                    if str(oea_zo_objective) == "lda_nll" or use_evidence or use_stack or bool(do_diag):
                        lda_ev = _compute_lda_evidence_params(
                            model=model,
                            X_train=z_train,
                            y_train=y_train,
                            class_order=class_labels,
                        )
                    opt_res = _optimize_qt_oea_zo(
                        z_t=z_test_base if z_test_base is not None else z_test,
                        model=model,
                        class_order=class_labels,
                        d_ref=d_ref,
                        lda_evidence=lda_ev,
                        channel_names=channel_names,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
                        localmix_neighbors=int(oea_zo_localmix_neighbors),
                        localmix_self_bias=float(oea_zo_localmix_self_bias),
                        infomax_lambda=float(oea_zo_infomax_lambda),
                        reliable_metric=str(oea_zo_reliable_metric),
                        reliable_threshold=float(oea_zo_reliable_threshold),
                        reliable_alpha=float(oea_zo_reliable_alpha),
                        trust_lambda=float(oea_zo_trust_lambda),
                        trust_q0=str(oea_zo_trust_q0),
                        marginal_mode=str(oea_zo_marginal_mode),
                        marginal_beta=float(oea_zo_marginal_beta),
                        marginal_tau=float(oea_zo_marginal_tau),
                        marginal_prior=marginal_prior_vec,
                        bilevel_iters=int(oea_zo_bilevel_iters),
                        bilevel_temp=float(oea_zo_bilevel_temp),
                        bilevel_step=float(oea_zo_bilevel_step),
                        bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                        bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        holdout_fraction=float(oea_zo_holdout_fraction),
                        fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                        iters=int(oea_zo_iters),
                        lr=float(oea_zo_lr),
                        mu=float(oea_zo_mu),
                        n_rotations=int(oea_zo_k),
                        seed=int(oea_zo_seed) + int(subject) * 997,
                        l2=float(oea_zo_l2),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        return_diagnostics=bool(want_diag),
                    )
                    if want_diag:
                        q_t, zo_diag = opt_res
                    else:
                        q_t = opt_res

                    if zo_diag is not None:
                        selected: dict | None = None
                        if use_oracle:
                            best_rec = None
                            best_acc = -1.0
                            for rec in zo_diag.get("records", []):
                                Q = np.asarray(rec.get("Q"), dtype=np.float64)
                                yp = model.predict(
                                    apply_spatial_transform(Q, z_test_base if z_test_base is not None else z_test)
                                )
                                acc = float(accuracy_score(y_test, yp))
                                if acc > best_acc:
                                    best_acc = acc
                                    best_rec = rec
                            selected = best_rec
                        elif use_evidence:
                            selected = select_by_evidence_nll(
                                zo_diag.get("records", []),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                            )
                        elif use_probe_mixup:
                            selected = select_by_probe_mixup(
                                zo_diag.get("records", []),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                            )
                        elif use_probe_mixup_hard:
                            selected = select_by_probe_mixup_hard(
                                zo_diag.get("records", []),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                            )
                        elif use_iwcv:
                            selected = select_by_iwcv_nll(
                                zo_diag.get("records", []),
                                model=model,
                                z_source=z_train,
                                y_source=y_train,
                                z_target=z_test_base if z_test_base is not None else z_test,
                                class_order=class_labels,
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                seed=int(oea_zo_seed) + int(subject) * 997,
                            )
                        elif use_iwcv_ucb:
                            selected = select_by_iwcv_ucb(
                                zo_diag.get("records", []),
                                model=model,
                                z_source=z_train,
                                y_source=y_train,
                                z_target=z_test_base if z_test_base is not None else z_test,
                                class_order=class_labels,
                                kappa=float(oea_zo_iwcv_kappa),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                seed=int(oea_zo_seed) + int(subject) * 997,
                            )
                        elif use_dev:
                            selected = select_by_dev_nll(
                                zo_diag.get("records", []),
                                model=model,
                                z_source=z_train,
                                y_source=y_train,
                                z_target=z_test_base if z_test_base is not None else z_test,
                                class_order=class_labels,
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                seed=int(oea_zo_seed) + int(subject) * 997,
                            )
                        elif use_ridge_guard and cert is not None and guard is not None:
                            selected = select_by_guarded_predicted_improvement(
                                zo_diag.get("records", []),
                                cert=cert,
                                guard=guard,
                                n_classes=len(class_labels),
                                threshold=float(oea_zo_calib_guard_threshold),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                            )
                        elif use_ridge and cert is not None:
                            selected = select_by_predicted_improvement(
                                zo_diag.get("records", []),
                                cert=cert,
                                n_classes=len(class_labels),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                feature_set="stacked" if use_stack else "base",
                            )
                        elif use_guard and guard is not None:
                            selected = select_by_guarded_objective(
                                zo_diag.get("records", []),
                                guard=guard,
                                n_classes=len(class_labels),
                                threshold=float(oea_zo_calib_guard_threshold),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                            )
                        if selected is not None:
                            q_t = np.asarray(selected.get("Q"), dtype=np.float64)

                    X_test = apply_spatial_transform(q_t, z_test_base if z_test_base is not None else z_test)

        y_pred = np.asarray(model.predict(X_test))
        y_proba = np.asarray(model.predict_proba(X_test))
        y_proba = _reorder_proba_columns(y_proba, model.classes_, list(class_order))

        if zo_diag is not None and do_diag and diagnostics_dir is not None:
            _write_zo_diagnostics(
                zo_diag,
                out_dir=Path(diagnostics_dir),
                tag=str(diagnostics_tag),
                subject=int(subject),
                model=model,
                z_t=z_test_base if z_test_base is not None else X_test_raw,
                y_true=y_test,
                class_order=class_order,
            )

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )

        fold_rows.append(
            FoldResult(
                subject=int(subject),
                n_train=int(len(y_train)),
                n_test=int(len(y_test)),
                **metrics,
            )
        )
        models_by_subject[int(subject)] = model
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))
        train_sess_all.append(np.full(shape=(int(len(y_test)),), fill_value=",".join(train_sessions), dtype=object))
        test_sess_all.append(np.full(shape=(int(len(y_test)),), fill_value=",".join(test_sessions), dtype=object))

    results_df = pd.DataFrame([asdict(r) for r in fold_rows]).sort_values("subject")
    y_true_cat = np.concatenate(y_true_all, axis=0)
    y_pred_cat = np.concatenate(y_pred_all, axis=0)
    y_proba_cat = np.concatenate(y_proba_all, axis=0)
    subj_cat = np.concatenate(subj_all, axis=0)
    trial_cat = np.concatenate(trial_all, axis=0)
    tr_sess_cat = np.concatenate(train_sess_all, axis=0)
    te_sess_cat = np.concatenate(test_sess_all, axis=0)

    pred_df = pd.DataFrame(
        {
            "subject": subj_cat,
            "train_sessions": tr_sess_cat,
            "test_sessions": te_sess_cat,
            "trial": trial_cat,
            "y_true": y_true_cat,
            "y_pred": y_pred_cat,
        }
    )
    for i, c in enumerate(list(class_order)):
        pred_df[f"proba_{c}"] = y_proba_cat[:, int(i)]

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )
