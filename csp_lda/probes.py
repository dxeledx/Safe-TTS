from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .proba import reorder_proba_columns as _reorder_proba_columns
from .zo import _select_pseudo_indices


def _row_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def select_keep_indices(
    proba: np.ndarray,
    *,
    class_order: Sequence[str],
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
) -> np.ndarray:
    """Select a reliable subset (indices) using the same pseudo-label filtering logic as ZO."""

    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D.")

    if float(pseudo_confidence) <= 0.0 and int(pseudo_topk_per_class) == 0 and not bool(pseudo_balance):
        return np.arange(proba.shape[0], dtype=int)

    class_order = [str(c) for c in class_order]
    pred_idx = np.argmax(proba, axis=1)
    classes_arr = np.asarray(class_order, dtype=object)
    y_pseudo = classes_arr[pred_idx]
    return _select_pseudo_indices(
        y_pseudo=y_pseudo,
        proba=proba,
        class_order=class_order,
        confidence=float(pseudo_confidence),
        topk_per_class=int(pseudo_topk_per_class),
        balance=bool(pseudo_balance),
    )


def reliability_weights(
    proba: np.ndarray,
    *,
    reliable_metric: str,
    reliable_threshold: float,
    reliable_alpha: float,
) -> np.ndarray:
    """Continuous reliability weights w_i in [0,1] based on proba (for certificates)."""

    reliable_metric = str(reliable_metric)
    if reliable_metric == "none":
        return np.ones(int(np.asarray(proba).shape[0]), dtype=np.float64)

    p = np.asarray(proba, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)

    ent = _row_entropy(p)
    conf = np.max(p, axis=1)
    if reliable_metric == "confidence":
        return _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
    if reliable_metric == "entropy":
        return _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
    raise ValueError("reliable_metric must be one of: 'none', 'confidence', 'entropy'.")


def evidence_nll(
    *,
    proba: np.ndarray,
    feats: np.ndarray,
    lda_evidence: dict | None,
    class_order: Sequence[str],
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
    reliable_metric: str,
    reliable_threshold: float,
    reliable_alpha: float,
) -> float:
    """Compute -log p(z) under the frozen (source) Gaussian mixture in feature space."""

    if lda_evidence is None:
        return float("nan")

    proba = np.asarray(proba, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float64)
    keep = select_keep_indices(
        proba,
        class_order=class_order,
        pseudo_confidence=float(pseudo_confidence),
        pseudo_topk_per_class=int(pseudo_topk_per_class),
        pseudo_balance=bool(pseudo_balance),
    )
    if keep.size == 0:
        return float("nan")

    proba = proba[keep]
    feats = feats[keep]
    if feats.ndim != 2:
        return float("nan")

    mu = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
    priors = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
    cov_inv = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
    logdet = float(lda_evidence.get("logdet", 0.0))
    if mu.ndim != 2 or cov_inv.ndim != 2:
        return float("nan")
    if feats.shape[1] != int(mu.shape[1]):
        return float("nan")

    diff = feats[:, None, :] - mu[None, :, :]
    qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv, diff, optimize=True)
    log_norm = float(mu.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet)
    log_gauss = -0.5 * (log_norm + qf)
    log_pr = np.log(np.clip(priors, 1e-12, 1.0)).reshape(1, -1)
    log_joint = log_pr + log_gauss
    m = np.max(log_joint, axis=1, keepdims=True)
    log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

    w = reliability_weights(
        proba,
        reliable_metric=str(reliable_metric),
        reliable_threshold=float(reliable_threshold),
        reliable_alpha=float(reliable_alpha),
    )
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return float("nan")
    return float(-np.sum(w * log_p) / w_sum)


@dataclass(frozen=True)
class ProbeStats:
    n_keep: int
    n_pairs: int
    frac_intra: float


def probe_mixup(
    *,
    proba: np.ndarray,
    feats: np.ndarray,
    lda,
    class_order: Sequence[str],
    seed_local: int,
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
    n_pairs: int = 200,
    lam: float = 0.5,
    mode: str = "soft",
    beta_alpha: float = 0.0,
) -> tuple[float, ProbeStats]:
    """MixUp-style probe score (label-free) in the *LDA feature space*.

    Lower is better. Uses pseudo labels from `proba` after pseudo filtering.
    """

    proba = np.asarray(proba, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float64)
    keep = select_keep_indices(
        proba,
        class_order=class_order,
        pseudo_confidence=float(pseudo_confidence),
        pseudo_topk_per_class=int(pseudo_topk_per_class),
        pseudo_balance=bool(pseudo_balance),
    )
    if keep.size == 0:
        return float("nan"), ProbeStats(n_keep=0, n_pairs=0, frac_intra=float("nan"))

    p = np.asarray(proba[keep], dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    f = np.asarray(feats[keep], dtype=np.float64)

    y_idx = np.argmax(p, axis=1).astype(int)
    classes = np.unique(y_idx).tolist()
    idx_by_class = {c: np.flatnonzero(y_idx == c) for c in classes}
    classes_intra = [c for c in classes if int(idx_by_class[c].size) >= 2]
    has_inter = len(classes) >= 2
    if not classes_intra and not has_inter:
        return float("nan"), ProbeStats(n_keep=int(keep.size), n_pairs=0, frac_intra=float("nan"))

    rng = np.random.RandomState(int(seed_local))
    n_pairs = int(max(0, n_pairs))
    if n_pairs == 0:
        return float("nan"), ProbeStats(n_keep=int(keep.size), n_pairs=0, frac_intra=float("nan"))
    n_pairs = min(n_pairs, 10_000)

    mode = str(mode)
    if mode not in {"soft", "hard_major"}:
        raise ValueError("probe_mixup mode must be 'soft' or 'hard_major'.")
    beta_alpha = float(beta_alpha)
    if beta_alpha < 0.0:
        raise ValueError("beta_alpha must be >= 0.")

    i_list: list[int] = []
    j_list: list[int] = []
    ki_list: list[int] = []
    kj_list: list[int] = []
    lam_list: list[float] = []
    same_list: list[bool] = []

    for _ in range(n_pairs):
        use_intra = bool(classes_intra) and (not has_inter or rng.rand() < 0.5)
        if use_intra:
            c = int(rng.choice(classes_intra))
            idxs = idx_by_class[c]
            a, b = rng.choice(idxs, size=2, replace=False).tolist()
            lam_val = float(lam)
            if beta_alpha > 0.0:
                lam_val = float(rng.beta(beta_alpha, beta_alpha))
                lam_val = float(np.clip(lam_val, 1e-6, 1.0 - 1e-6))
            i_list.append(int(a))
            j_list.append(int(b))
            ki_list.append(int(c))
            kj_list.append(int(c))
            lam_list.append(float(lam_val))
            same_list.append(True)
            continue

        if not has_inter:
            continue
        c1, c2 = rng.choice(classes, size=2, replace=False).tolist()
        a = int(rng.choice(idx_by_class[int(c1)]))
        b = int(rng.choice(idx_by_class[int(c2)]))
        lam_val = float(lam)
        if beta_alpha > 0.0:
            lam_val = float(rng.beta(beta_alpha, beta_alpha))
            lam_val = float(np.clip(lam_val, 1e-6, 1.0 - 1e-6))
        # MixVal-style: when λ>0.5 use the hard pseudo label of the dominant sample.
        # Implement by folding λ to [0.5,1] and swapping (i,j) when needed.
        if mode == "hard_major" and lam_val < 0.5:
            a, b = b, a
            c1, c2 = c2, c1
            lam_val = 1.0 - lam_val
        i_list.append(int(a))
        j_list.append(int(b))
        ki_list.append(int(c1))
        kj_list.append(int(c2))
        lam_list.append(float(lam_val))
        same_list.append(False)

    if not i_list:
        return float("nan"), ProbeStats(n_keep=int(keep.size), n_pairs=0, frac_intra=float("nan"))

    i_arr = np.asarray(i_list, dtype=int)
    j_arr = np.asarray(j_list, dtype=int)
    lam_arr = np.asarray(lam_list, dtype=np.float64)
    ki_arr = np.asarray(ki_list, dtype=int)
    kj_arr = np.asarray(kj_list, dtype=int)
    same_arr = np.asarray(same_list, dtype=bool)

    f_mix = lam_arr.reshape(-1, 1) * f[i_arr] + (1.0 - lam_arr).reshape(-1, 1) * f[j_arr]
    proba_mix = np.asarray(lda.predict_proba(f_mix), dtype=np.float64)
    proba_mix = _reorder_proba_columns(proba_mix, lda.classes_, list(class_order))
    proba_mix = np.clip(proba_mix, 1e-12, 1.0)
    proba_mix = proba_mix / np.sum(proba_mix, axis=1, keepdims=True)
    logp = np.log(proba_mix)

    lp_i = logp[np.arange(logp.shape[0]), ki_arr]
    lp_j = logp[np.arange(logp.shape[0]), kj_arr]
    if mode == "hard_major":
        ce = -lp_i
    else:
        ce = np.where(same_arr, -lp_i, -(lam_arr * lp_i + (1.0 - lam_arr) * lp_j))

    score = float(np.mean(ce))
    stats = ProbeStats(n_keep=int(keep.size), n_pairs=int(ce.size), frac_intra=float(np.mean(same_arr)))
    return float(score), stats

