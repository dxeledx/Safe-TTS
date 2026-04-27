from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from braindecode import EEGClassifier
from braindecode.models import ATCNet
from braindecode.models import Deep4Net
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

from csp_lda.data import SubjectData
from csp_lda.metrics import compute_metrics


@dataclass(frozen=True)
class Deep4NetParams:
    max_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-2
    weight_decay: float = 5e-4
    valid_split: float = 0.2
    early_stop_patience: int = 10
    seed: int = 0
    standardize: str = "train_zscore"  # "none" | "train_zscore"
    device: str = "cpu"


@dataclass(frozen=True)
class ATCNetParams:
    max_epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    valid_split: float = 0.2
    early_stop_patience: int = 15
    seed: int = 0
    standardize: str = "train_zscore"  # "none" | "train_zscore"
    device: str = "cpu"


@dataclass(frozen=True)
class TCFormerParams:
    max_epochs: int = 1000
    batch_size: int = 48
    lr: float = 9e-4
    weight_decay: float = 1e-3
    valid_split: float = 0.2
    early_stop_patience: int = 200
    seed: int = 0
    standardize: str = "train_zscore"  # "none" | "train_zscore"
    device: str = "cpu"


def _labels_to_int(y: np.ndarray, *, class_order: Sequence[str]) -> np.ndarray:
    label_to_idx = {str(c): int(i) for i, c in enumerate(list(class_order))}
    try:
        return np.asarray([label_to_idx[str(v)] for v in y], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Unknown label in y not present in class_order: {e}") from e


def _ints_to_labels(y: np.ndarray, *, class_order: Sequence[str]) -> np.ndarray:
    order = [str(c) for c in class_order]
    return np.asarray([order[int(i)] for i in y], dtype=object)


def _standardize_train_zscore(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (X_train - mean) / std, (X_test - mean) / std


def loso_deep4net_evaluation(
    subject_data: Dict[int, SubjectData],
    *,
    class_order: Sequence[str],
    test_subjects: Sequence[int] | None = None,
    average: str = "macro",
    sfreq: float = 250.0,
    params: Deep4NetParams | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, EEGClassifier]]:
    """LOSO evaluation for a Braindecode Deep4Net baseline (trialwise decoding).

    Notes
    -----
    - Uses only training subjects for fitting any standardization statistics.
    - Does NOT adapt on the target subject (strict LOSO).
    - Returns `y_true/y_pred` as class labels (strings) consistent with `class_order`.
    """

    if params is None:
        params = Deep4NetParams()

    subjects_all = sorted(subject_data.keys())
    subjects_test = subjects_all
    if test_subjects is not None and len(test_subjects) > 0:
        test_set = {int(s) for s in test_subjects}
        subjects_test = [int(s) for s in subjects_all if int(s) in test_set]
        missing = sorted(test_set - set(subjects_all))
        if missing:
            raise ValueError(f"test_subjects contains unknown subject ids: {missing}")
    n_classes = int(len(class_order))
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")

    fold_rows: List[dict] = []
    models_by_subject: Dict[int, EEGClassifier] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    for test_subject in subjects_test:
        train_subjects = [s for s in subjects_all if s != int(test_subject)]

        X_train = np.concatenate([subject_data[int(s)].X for s in train_subjects], axis=0).astype(
            np.float32, copy=False
        )
        y_train = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
        X_test = np.asarray(subject_data[int(test_subject)].X, dtype=np.float32, order="C")
        y_test = np.asarray(subject_data[int(test_subject)].y, dtype=object)

        y_train_int = _labels_to_int(y_train, class_order=class_order)

        if params.standardize == "train_zscore":
            X_train, X_test = _standardize_train_zscore(X_train, X_test)
            X_train = X_train.astype(np.float32, copy=False)
            X_test = X_test.astype(np.float32, copy=False)
        elif params.standardize == "none":
            pass
        else:
            raise ValueError("params.standardize must be one of: 'none', 'train_zscore'")

        fold_seed = int(params.seed) + int(test_subject) * 997
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        clf = EEGClassifier(
            module=Deep4Net,
            module__n_chans=int(X_train.shape[1]),
            module__n_outputs=n_classes,
            module__n_times=int(X_train.shape[2]),
            module__final_conv_length="auto",
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=float(params.lr),
            optimizer__weight_decay=float(params.weight_decay),
            batch_size=int(params.batch_size),
            max_epochs=int(params.max_epochs),
            train_split=ValidSplit(float(params.valid_split), random_state=fold_seed),
            callbacks=[
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=int(params.early_stop_patience)),
                ),
            ],
            device=str(params.device),
            classes=list(range(n_classes)),
            verbose=0,
        )
        clf.fit(X_train, y_train_int)

        # NOTE: Braindecode models still output log-softmax; `predict_proba` therefore returns log-probabilities.
        y_log_proba = np.asarray(clf.predict_proba(X_test), dtype=np.float64)
        y_proba = np.exp(y_log_proba)
        y_pred_int = y_proba.argmax(axis=1).astype(int)

        y_pred = _ints_to_labels(y_pred_int, class_order=class_order)

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )
        fold_rows.append(
            {
                "subject": int(test_subject),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **metrics,
            }
        )
        models_by_subject[int(test_subject)] = clf
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(test_subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))

    results_df = pd.DataFrame(fold_rows).sort_values("subject")
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

    # Add params snapshot for reproducibility (kept out of pred_df for compactness).
    results_df.attrs["deep4net_params"] = asdict(params)

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )


def loso_atcnet_evaluation(
    subject_data: Dict[int, SubjectData],
    *,
    class_order: Sequence[str],
    test_subjects: Sequence[int] | None = None,
    average: str = "macro",
    sfreq: float = 250.0,
    params: ATCNetParams | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, EEGClassifier]]:
    """LOSO evaluation for a Braindecode ATCNet baseline (trialwise decoding)."""

    if params is None:
        params = ATCNetParams()

    subjects_all = sorted(subject_data.keys())
    subjects_test = subjects_all
    if test_subjects is not None and len(test_subjects) > 0:
        test_set = {int(s) for s in test_subjects}
        subjects_test = [int(s) for s in subjects_all if int(s) in test_set]
        missing = sorted(test_set - set(subjects_all))
        if missing:
            raise ValueError(f"test_subjects contains unknown subject ids: {missing}")
    n_classes = int(len(class_order))
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")

    fold_rows: List[dict] = []
    models_by_subject: Dict[int, EEGClassifier] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    for test_subject in subjects_test:
        train_subjects = [s for s in subjects_all if s != int(test_subject)]

        X_train = np.concatenate([subject_data[int(s)].X for s in train_subjects], axis=0).astype(
            np.float32, copy=False
        )
        y_train = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
        X_test = np.asarray(subject_data[int(test_subject)].X, dtype=np.float32, order="C")
        y_test = np.asarray(subject_data[int(test_subject)].y, dtype=object)

        y_train_int = _labels_to_int(y_train, class_order=class_order)

        if params.standardize == "train_zscore":
            X_train, X_test = _standardize_train_zscore(X_train, X_test)
            X_train = X_train.astype(np.float32, copy=False)
            X_test = X_test.astype(np.float32, copy=False)
        elif params.standardize == "none":
            pass
        else:
            raise ValueError("params.standardize must be one of: 'none', 'train_zscore'")

        fold_seed = int(params.seed) + int(test_subject) * 997
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        n_times = int(X_train.shape[2])
        input_window_seconds = float(n_times / float(sfreq))

        clf = EEGClassifier(
            module=ATCNet,
            module__n_chans=int(X_train.shape[1]),
            module__n_outputs=n_classes,
            module__n_times=n_times,
            module__sfreq=float(sfreq),
            module__input_window_seconds=input_window_seconds,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=float(params.lr),
            optimizer__weight_decay=float(params.weight_decay),
            batch_size=int(params.batch_size),
            max_epochs=int(params.max_epochs),
            train_split=ValidSplit(float(params.valid_split), random_state=fold_seed),
            callbacks=[
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=int(params.early_stop_patience)),
                ),
            ],
            device=str(params.device),
            classes=list(range(n_classes)),
            verbose=0,
        )
        clf.fit(X_train, y_train_int)

        y_log_proba = np.asarray(clf.predict_proba(X_test), dtype=np.float64)
        y_proba = np.exp(y_log_proba)
        y_pred_int = y_proba.argmax(axis=1).astype(int)
        y_pred = _ints_to_labels(y_pred_int, class_order=class_order)

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )
        fold_rows.append(
            {
                "subject": int(test_subject),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **metrics,
            }
        )
        models_by_subject[int(test_subject)] = clf
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(test_subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))

    results_df = pd.DataFrame(fold_rows).sort_values("subject")
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

    results_df.attrs["atcnet_params"] = asdict(params)

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return ex / denom


def loso_tcformer_evaluation(
    subject_data: Dict[int, SubjectData],
    *,
    class_order: Sequence[str],
    test_subjects: Sequence[int] | None = None,
    average: str = "macro",
    sfreq: float = 250.0,
    params: TCFormerParams | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, EEGClassifier]]:
    """LOSO evaluation for a TCFormer baseline (trialwise decoding)."""

    if params is None:
        params = TCFormerParams()

    # Lazy import so that runs without tcformer don't require einops.
    from csp_lda.tcformer import TCFormerNet

    subjects_all = sorted(subject_data.keys())
    subjects_test = subjects_all
    if test_subjects is not None and len(test_subjects) > 0:
        test_set = {int(s) for s in test_subjects}
        subjects_test = [int(s) for s in subjects_all if int(s) in test_set]
        missing = sorted(test_set - set(subjects_all))
        if missing:
            raise ValueError(f"test_subjects contains unknown subject ids: {missing}")

    n_classes = int(len(class_order))
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")

    fold_rows: List[dict] = []
    models_by_subject: Dict[int, EEGClassifier] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    for test_subject in subjects_test:
        train_subjects = [s for s in subjects_all if s != int(test_subject)]

        X_train = np.concatenate([subject_data[int(s)].X for s in train_subjects], axis=0).astype(
            np.float32, copy=False
        )
        y_train = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
        X_test = np.asarray(subject_data[int(test_subject)].X, dtype=np.float32, order="C")
        y_test = np.asarray(subject_data[int(test_subject)].y, dtype=object)

        y_train_int = _labels_to_int(y_train, class_order=class_order)

        if params.standardize == "train_zscore":
            X_train, X_test = _standardize_train_zscore(X_train, X_test)
            X_train = X_train.astype(np.float32, copy=False)
            X_test = X_test.astype(np.float32, copy=False)
        elif params.standardize == "none":
            pass
        else:
            raise ValueError("params.standardize must be one of: 'none', 'train_zscore'")

        fold_seed = int(params.seed) + int(test_subject) * 997
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        clf = EEGClassifier(
            module=TCFormerNet,
            module__n_chans=int(X_train.shape[1]),
            module__n_outputs=n_classes,
            module__n_times=int(X_train.shape[2]),
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=float(params.lr),
            optimizer__weight_decay=float(params.weight_decay),
            batch_size=int(params.batch_size),
            max_epochs=int(params.max_epochs),
            train_split=ValidSplit(float(params.valid_split), random_state=fold_seed),
            callbacks=[
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=int(params.early_stop_patience)),
                ),
            ],
            device=str(params.device),
            classes=list(range(n_classes)),
            verbose=0,
        )
        clf.fit(X_train, y_train_int)

        y_score = np.asarray(clf.predict_proba(X_test), dtype=np.float64)
        # EEGClassifier may return logits/log-probs depending on predict_nonlinearity; enforce probabilities.
        if (
            (y_score.ndim != 2)
            or (y_score.shape[1] != n_classes)
            or (np.nanmin(y_score) < 0.0)
            or (np.nanmax(y_score) > 1.0)
            or (not np.allclose(np.nansum(y_score, axis=1), 1.0, atol=1e-3, rtol=0.0))
        ):
            y_proba = _softmax_np(y_score)
        else:
            y_proba = y_score

        y_pred_int = y_proba.argmax(axis=1).astype(int)
        y_pred = _ints_to_labels(y_pred_int, class_order=class_order)

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )
        fold_rows.append(
            {
                "subject": int(test_subject),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **metrics,
            }
        )
        models_by_subject[int(test_subject)] = clf
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(test_subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))

    results_df = pd.DataFrame(fold_rows).sort_values("subject")
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

    results_df.attrs["tcformer_params"] = asdict(params)

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )
