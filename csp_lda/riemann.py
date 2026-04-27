from __future__ import annotations

import numpy as np


def covariances_from_epochs(
    X: np.ndarray,
    *,
    eps: float = 1e-10,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Compute SPD trial covariances from epochs.

    Parameters
    ----------
    X:
        Epochs array with shape (n_trials, n_channels, n_times).
    eps:
        Eigenvalue floor as eps * max_eig (per-trial) to ensure SPD.
    shrinkage:
        Optional shrinkage in [0,1): (1-a)*C + a*(tr(C)/C)I.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials,n_channels,n_times); got {X.shape}.")
    n_trials, n_channels, n_times = X.shape
    if int(n_trials) < 1:
        raise ValueError("Need at least 1 trial to compute covariances.")

    if float(shrinkage) > 0.0 and not (0.0 <= float(shrinkage) < 1.0):
        raise ValueError("shrinkage must be in [0, 1).")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0.")

    c = int(n_channels)
    t = int(n_times)
    eye = np.eye(c, dtype=np.float64)
    scale = 1.0 / float(max(1, t))
    alpha = float(shrinkage)

    # Vectorized covariance: cov[n] = (1/T) * X[n] @ X[n]^T.
    covs = scale * np.einsum("nct,ndt->ncd", X, X, optimize=True)
    covs = 0.5 * (covs + np.swapaxes(covs, 1, 2))

    if alpha > 0.0:
        tr = np.trace(covs, axis1=1, axis2=2)
        covs *= 1.0 - alpha
        diag = np.arange(c)
        covs[:, diag, diag] += (alpha * (tr / float(c)))[:, None]

    # Ensure SPD via diagonal loading (much faster than per-trial eigendecomposition).
    tr = np.trace(covs, axis1=1, axis2=2)
    floor = float(eps) * (tr / float(c))
    floor = np.where(np.isfinite(floor) & (floor > 0.0), floor, float(eps))
    diag = np.arange(c)
    covs[:, diag, diag] += floor[:, None]

    return np.asarray(covs, dtype=np.float64)
