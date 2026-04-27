from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mne.decoding import CSP
import numpy as np
from scipy.signal import butter, sosfiltfilt
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline

from .alignment import BaseAligner, NoAligner
from .subject_invariant import CenteredLinearProjector


class EnsureFloat64(BaseEstimator, TransformerMixin):
    """Cast EEG epochs array to float64 for MNE decoding utilities."""

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        return self

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        return np.asarray(X, dtype=np.float64)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filterbank CSP feature extractor.

    This transformer applies a bank of band-pass filters to epochs and fits a CSP
    model per band. At transform time it concatenates CSP log-variance features
    across bands.
    """

    def __init__(
        self,
        *,
        bands: list[tuple[float, float]],
        sfreq: float,
        n_components: int,
        filter_order: int = 4,
        filter_batch_size: int = 256,
        csp_reg: float | str | None = None,
        multiclass_strategy: str = "auto",
    ) -> None:
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.filter_order = filter_order
        self.filter_batch_size = filter_batch_size
        self.csp_reg = csp_reg
        self.multiclass_strategy = multiclass_strategy

        self._sos: list[np.ndarray] | None = None
        self._csps_by_band: list[list[CSP]] | None = None

    def _filtfilt_into(self, *, sos: np.ndarray, X: np.ndarray, out: np.ndarray) -> None:
        """Run sosfiltfilt in trial chunks to bound peak memory.

        scipy.signal.sosfiltfilt processes trials independently along axis=-1. Filtering in
        smaller batches therefore preserves results but avoids allocating very large temporary
        arrays on datasets with many trials (e.g., PhysionetMI).
        """

        X = np.asarray(X, dtype=np.float64, order="C")
        if out.shape != X.shape:
            raise ValueError(f"FilterBankCSP: out shape {out.shape} != X shape {X.shape}.")
        if out.dtype != np.float64:
            raise ValueError("FilterBankCSP: out must be float64.")

        n_trials = int(X.shape[0])
        bs = int(self.filter_batch_size)
        if bs <= 0 or bs >= n_trials:
            out[...] = sosfiltfilt(sos, X, axis=-1)
            return

        for start in range(0, n_trials, bs):
            end = min(start + bs, n_trials)
            out[start:end] = sosfiltfilt(sos, X[start:end], axis=-1)

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        X = np.asarray(X, dtype=np.float64, order="C")
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n_trials,n_channels,n_times); got {X.shape}.")
        if y is None:
            raise ValueError("y must be provided for CSP fitting.")
        y = np.asarray(y)

        sfreq = float(self.sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0.0:
            raise ValueError("sfreq must be positive and finite.")
        nyq = 0.5 * sfreq

        bands = [(float(lo), float(hi)) for lo, hi in list(self.bands)]
        if not bands:
            raise ValueError("bands must be a non-empty list of (fmin,fmax).")

        sos_list: list[np.ndarray] = []
        csps_by_band: list[list[CSP]] = []
        X_f = np.empty_like(X, dtype=np.float64, order="C")

        classes = np.unique(y)
        strategy = str(self.multiclass_strategy).strip().lower()
        if strategy == "auto":
            strategy = "multiclass" if int(classes.size) <= 2 else "ovo"
        if strategy not in {"multiclass", "ovo", "ovr"}:
            raise ValueError("multiclass_strategy must be one of: 'auto', 'multiclass', 'ovo', 'ovr'.")

        valid_bands: list[tuple[float, float]] = []
        for fmin, fmax in bands:
            if 0.0 < fmin < fmax < nyq:
                valid_bands.append((float(fmin), float(fmax)))
        if len(valid_bands) != len(bands):
            dropped = [b for b in bands if b not in valid_bands]
            warnings.warn(
                f"FilterBankCSP: dropped {len(dropped)}/{len(bands)} invalid bands for sfreq={sfreq} (nyq={nyq}): {dropped}"
            )
        if not valid_bands:
            raise ValueError(f"No valid bands for sfreq={sfreq} (nyq={nyq}); got bands={bands}.")

        for fmin, fmax in valid_bands:
            sos = butter(int(self.filter_order), [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
            self._filtfilt_into(sos=sos, X=X, out=X_f)
            sos_list.append(sos)
            csps: list[CSP] = []
            if strategy == "multiclass":
                csp = CSP(n_components=int(self.n_components), reg=self.csp_reg)
                csp.fit(X_f, y)
                csps.append(csp)
            elif strategy == "ovo":
                # One-vs-one CSP per class pair (common in multi-class FBCSP).
                from itertools import combinations

                for c1, c2 in combinations(classes.tolist(), 2):
                    mask = (y == c1) | (y == c2)
                    if int(np.sum(mask)) < 2:
                        continue
                    csp = CSP(n_components=int(self.n_components), reg=self.csp_reg)
                    csp.fit(X_f[mask], y[mask])
                    csps.append(csp)
            else:  # "ovr"
                for c in classes.tolist():
                    y_bin = (y == c).astype(int)
                    if int(np.unique(y_bin).size) < 2:
                        continue
                    csp = CSP(n_components=int(self.n_components), reg=self.csp_reg)
                    csp.fit(X_f, y_bin)
                    csps.append(csp)
            csps_by_band.append(csps)

        self._sos = sos_list
        self._csps_by_band = csps_by_band
        return self

    def fit_transform(self, X, y=None, **fit_params):  # noqa: N803  (match sklearn signature)
        """Fit CSPs per band and transform X in one pass.

        This avoids filtering the training data twice during `Pipeline.fit` (fit+transform),
        which is a major bottleneck for large datasets like Schirrmeister2017 (HGD).
        """

        X = np.asarray(X, dtype=np.float64, order="C")
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n_trials,n_channels,n_times); got {X.shape}.")
        if y is None:
            raise ValueError("y must be provided for CSP fitting.")
        y = np.asarray(y)

        sfreq = float(self.sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0.0:
            raise ValueError("sfreq must be positive and finite.")
        nyq = 0.5 * sfreq

        bands = [(float(lo), float(hi)) for lo, hi in list(self.bands)]
        if not bands:
            raise ValueError("bands must be a non-empty list of (fmin,fmax).")

        sos_list: list[np.ndarray] = []
        csps_by_band: list[list[CSP]] = []
        feats: list[np.ndarray] = []
        X_f = np.empty_like(X, dtype=np.float64, order="C")

        classes = np.unique(y)
        strategy = str(self.multiclass_strategy).strip().lower()
        if strategy == "auto":
            strategy = "multiclass" if int(classes.size) <= 2 else "ovo"
        if strategy not in {"multiclass", "ovo", "ovr"}:
            raise ValueError("multiclass_strategy must be one of: 'auto', 'multiclass', 'ovo', 'ovr'.")

        valid_bands: list[tuple[float, float]] = []
        for fmin, fmax in bands:
            if 0.0 < fmin < fmax < nyq:
                valid_bands.append((float(fmin), float(fmax)))
        if len(valid_bands) != len(bands):
            dropped = [b for b in bands if b not in valid_bands]
            warnings.warn(
                f"FilterBankCSP: dropped {len(dropped)}/{len(bands)} invalid bands for sfreq={sfreq} (nyq={nyq}): {dropped}"
            )
        if not valid_bands:
            raise ValueError(f"No valid bands for sfreq={sfreq} (nyq={nyq}); got bands={bands}.")

        for fmin, fmax in valid_bands:
            sos = butter(int(self.filter_order), [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
            self._filtfilt_into(sos=sos, X=X, out=X_f)
            sos_list.append(sos)
            csps: list[CSP] = []
            if strategy == "multiclass":
                csp = CSP(n_components=int(self.n_components), reg=self.csp_reg)
                csp.fit(X_f, y)
                csps.append(csp)
            elif strategy == "ovo":
                # One-vs-one CSP per class pair (common in multi-class FBCSP).
                from itertools import combinations

                for c1, c2 in combinations(classes.tolist(), 2):
                    mask = (y == c1) | (y == c2)
                    if int(np.sum(mask)) < 2:
                        continue
                    csp = CSP(n_components=int(self.n_components), reg=self.csp_reg)
                    csp.fit(X_f[mask], y[mask])
                    csps.append(csp)
            else:  # "ovr"
                for c in classes.tolist():
                    y_bin = (y == c).astype(int)
                    if int(np.unique(y_bin).size) < 2:
                        continue
                    csp = CSP(n_components=int(self.n_components), reg=self.csp_reg)
                    csp.fit(X_f, y_bin)
                    csps.append(csp)
            csps_by_band.append(csps)

            for csp in csps:
                feats.append(np.asarray(csp.transform(X_f), dtype=np.float64))

        self._sos = sos_list
        self._csps_by_band = csps_by_band
        return np.concatenate(feats, axis=1)

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        X = np.asarray(X, dtype=np.float64, order="C")
        if self._sos is None or self._csps_by_band is None:
            raise RuntimeError("FilterBankCSP is not fitted yet.")
        feats = []
        X_f = np.empty_like(X, dtype=np.float64, order="C")
        for sos, csps in zip(self._sos, self._csps_by_band):
            self._filtfilt_into(sos=sos, X=X, out=X_f)
            for csp in csps:
                feats.append(np.asarray(csp.transform(X_f), dtype=np.float64))
        return np.concatenate(feats, axis=1)


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Select top-k features by mutual information with labels (training only)."""

    def __init__(self, *, k: int = 24, random_state: int = 0) -> None:
        self.k = int(k)
        self.random_state = int(random_state)
        self.indices_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        X = np.asarray(X, dtype=np.float64)
        if y is None:
            raise ValueError("y must be provided for feature selection.")
        y = np.asarray(y)

        scores = mutual_info_classif(X, y, random_state=self.random_state)
        scores = np.asarray(scores, dtype=np.float64)

        k = int(self.k)
        if k <= 0 or k >= int(scores.size):
            self.indices_ = np.arange(scores.size, dtype=int)
            return self

        order = np.argsort(scores)[::-1]
        self.indices_ = np.asarray(order[:k], dtype=int)
        return self

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        X = np.asarray(X, dtype=np.float64)
        if self.indices_ is None:
            raise RuntimeError("MutualInfoSelector is not fitted yet.")
        return X[:, self.indices_]


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Pipeline

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self):
        return self.pipeline.named_steps["lda"].classes_

    @property
    def csp(self) -> CSP:
        return self.pipeline.named_steps["csp"]


def build_csp_lda_pipeline(
    n_components: int = 4,
    aligner: Optional[BaseAligner] = None,
    csp_reg: float | str | None = None,
) -> Pipeline:
    """Build CSP+LDA sklearn pipeline.

    Notes
    -----
    - CSP uses `mne.decoding.CSP` (classical CSP implementation used in many
      motor-imagery baselines). We set `n_components=4` per requirement.
    - LDA uses scikit-learn default parameters.
    """

    if aligner is None:
        aligner = NoAligner()

    return Pipeline(
        steps=[
            ("to_float64", EnsureFloat64()),
            ("align", aligner),
            ("csp", CSP(n_components=int(n_components), reg=csp_reg)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )


def fit_csp_lda(
    X_train,
    y_train,
    n_components: int = 4,
    aligner: Optional[BaseAligner] = None,
) -> TrainedModel:
    # CSP can fail on rank-deficient inputs (e.g. low-rank channel projectors) because the
    # generalized eigen-problem requires SPD covariance. We retry with covariance regularization.
    last_err: Exception | None = None
    for csp_reg in (None, 1e-6, 1e-3, 1e-1):
        try:
            pipeline = build_csp_lda_pipeline(n_components=n_components, aligner=aligner, csp_reg=csp_reg)
            pipeline.fit(X_train, y_train)
            return TrainedModel(pipeline=pipeline)
        except Exception as e:  # noqa: BLE001  (narrowing to LinAlgError is brittle across scipy/mne)
            msg = str(e)
            if "positive definite" in msg or "LinAlgError" in type(e).__name__:
                last_err = e
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("CSP+LDA fitting failed for unknown reason.")


def build_fbcsp_lda_pipeline(
    *,
    bands: list[tuple[float, float]],
    sfreq: float,
    n_components: int = 4,
    filter_order: int = 4,
    csp_reg: float | str | None = None,
    multiclass_strategy: str = "auto",
    select_k: int = 24,
    mi_random_state: int = 0,
) -> Pipeline:
    """Build FilterBank-CSP + LDA pipeline (no alignment step here).

    Notes
    -----
    - Alignment (EA/OEA/...) is handled outside this pipeline by preparing X.
    - CSP is fitted per band; features are concatenated then fed into LDA.
    """

    return Pipeline(
        steps=[
            ("to_float64", EnsureFloat64()),
            (
                "fbcsp",
                FilterBankCSP(
                    bands=list(bands),
                    sfreq=float(sfreq),
                    n_components=int(n_components),
                    filter_order=int(filter_order),
                    csp_reg=csp_reg,
                    multiclass_strategy=str(multiclass_strategy),
                ),
            ),
            ("select", MutualInfoSelector(k=int(select_k), random_state=int(mi_random_state))),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def fit_fbcsp_lda(
    X_train,
    y_train,
    *,
    bands: list[tuple[float, float]],
    sfreq: float,
    n_components: int = 4,
    filter_order: int = 4,
    multiclass_strategy: str = "auto",
    select_k: int = 24,
) -> TrainedModel:
    # Similar to `fit_csp_lda`, CSP can fail on some inputs; retry with increasing regularization.
    last_err: Exception | None = None
    for csp_reg in (None, 1e-6, 1e-3, 1e-1):
        try:
            pipeline = build_fbcsp_lda_pipeline(
                bands=bands,
                sfreq=sfreq,
                n_components=n_components,
                filter_order=filter_order,
                csp_reg=csp_reg,
                multiclass_strategy=str(multiclass_strategy),
                select_k=int(select_k),
            )
            pipeline.fit(X_train, y_train)
            return TrainedModel(pipeline=pipeline)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "positive definite" in msg or "LinAlgError" in type(e).__name__:
                last_err = e
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("FBCSP+LDA fitting failed for unknown reason.")


def fit_csp_projected_lda(
    *,
    X_train,
    y_train,
    projector: CenteredLinearProjector,
    csp: CSP | None = None,
    n_components: int = 4,
    aligner: Optional[BaseAligner] = None,
) -> TrainedModel:
    """Fit CSP then train LDA on projected CSP features.

    This is used for subject-invariant feature learning where the projector is
    learned externally (may depend on subject IDs), but we still want a standard
    sklearn Pipeline for inference and for ZO utilities that access pipeline steps.
    """

    if aligner is None:
        aligner = NoAligner()

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    if csp is None:
        csp = CSP(n_components=int(n_components))
        csp.fit(X_train, y_train)

    feats = np.asarray(csp.transform(X_train), dtype=np.float64)
    feats_proj = np.asarray(projector.transform(feats), dtype=np.float64)

    lda = LinearDiscriminantAnalysis()
    lda.fit(feats_proj, y_train)

    pipeline = Pipeline(
        steps=[
            ("to_float64", EnsureFloat64()),
            ("align", aligner),
            ("csp", csp),
            ("proj", projector),
            ("lda", lda),
        ]
    )
    return TrainedModel(pipeline=pipeline)
