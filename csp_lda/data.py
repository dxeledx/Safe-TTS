from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import re
from moabb.paradigms import MotorImagery
from scipy.signal import firwin, lfilter


@dataclass(frozen=True)
class SubjectData:
    subject: int
    X: np.ndarray  # shape: (n_trials, n_channels, n_times)
    y: np.ndarray  # shape: (n_trials,)

def resolve_moabb_dataset(dataset: str):
    """Resolve a MOABB dataset name to a dataset instance.

    Accepts common aliases/casings, e.g.:
    - BNCI2014_001 / bnci2014_001 / BNCI2014001
    - Cho2017 / PhysionetMI / Schirrmeister2017
    """

    dataset = str(dataset).strip()
    if not dataset:
        raise ValueError("dataset must be a non-empty string.")

    import moabb.datasets as moabb_datasets

    # Fast path: exact match.
    if hasattr(moabb_datasets, dataset):
        cls = getattr(moabb_datasets, dataset)
        return cls()

    # Normalize BNCI variants: BNCI2014001 -> BNCI2014_001 when available.
    m = re.match(r"^BNCI(\d{4})_?(\d{3})$", dataset.strip().upper())
    if m:
        with_underscore = f"BNCI{m.group(1)}_{m.group(2)}"
        without_underscore = f"BNCI{m.group(1)}{m.group(2)}"
        if hasattr(moabb_datasets, with_underscore):
            return getattr(moabb_datasets, with_underscore)()
        if hasattr(moabb_datasets, without_underscore):
            return getattr(moabb_datasets, without_underscore)()

    # Case-insensitive match for other datasets.
    lower_to_name = {name.lower(): name for name in dir(moabb_datasets)}
    key = dataset.lower()
    if key in lower_to_name:
        cls = getattr(moabb_datasets, lower_to_name[key])
        return cls()

    raise ValueError(f"Unknown MOABB dataset: {dataset}")


class MoabbMotorImageryLoader:
    """Load a MOABB MotorImagery dataset using MOABB Dataset+Paradigm."""

    def __init__(
        self,
        *,
        dataset: str,
        fmin: float,
        fmax: float,
        tmin: float,
        tmax: float,
        resample: float,
        events: Sequence[str],
        sessions: Optional[Sequence[str]] = None,
        preprocess: str = "moabb",
        car: bool = False,
        paper_fir_order: int = 50,
        paper_fir_window: str = "hamming",
    ) -> None:
        self.dataset = resolve_moabb_dataset(dataset)
        self.dataset_id = str(self.dataset.__class__.__name__)
        self.sessions = tuple(sessions) if sessions is not None else None
        self.preprocess = str(preprocess)
        self.car = bool(car)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.resample = float(resample)
        self.events = tuple(events)
        self.paper_fir_order = int(paper_fir_order)
        self.paper_fir_window = str(paper_fir_window)

        if self.preprocess == "moabb":
            self.paradigm = MotorImagery(
                events=list(events),
                n_classes=len(events),
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                resample=resample,
            )
        elif self.preprocess == "paper_fir":
            # Paper-matched preprocessing is implemented manually on MOABB Raw objects.
            self.paradigm = None
        else:
            raise ValueError("preprocess must be one of: 'moabb', 'paper_fir'")

    @property
    def subject_list(self) -> List[int]:
        return list(self.dataset.subject_list)

    def load_arrays(
        self,
        subjects: Optional[Sequence[int]] = None,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Return (X, y, meta) as numpy arrays.

        - preprocess="moabb": use MOABB paradigm standard pipeline.
        - preprocess="paper_fir": use causal FIR(Hamming) bandpass, then epoch.
        """

        if subjects is None:
            subjects = self.subject_list
        subjects = list(subjects)

        if self.preprocess == "moabb":
            if self.paradigm is None:
                raise RuntimeError("MOABB paradigm is not initialized.")
            # IMPORTANT (memory): for large datasets (e.g., Schirrmeister2017 / HGD),
            # calling `get_data` for all subjects at once can transiently hold many Raw/Epochs
            # objects in memory (filtering/resampling/copying), leading to OOM kills.
            # We therefore load *per subject* and concatenate at the end.
            X_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            meta_parts: List[pd.DataFrame] = []

            for subject in subjects:
                X_s, y_s, meta_s = self.paradigm.get_data(dataset=self.dataset, subjects=[int(subject)])

                if self.sessions is not None:
                    # Most MOABB MI datasets expose a `session` column. Some (e.g., Schirrmeister2017)
                    # also expose meaningful splits in `run` (e.g., 0train/1test) while `session`
                    # stays constant. We therefore allow filtering by *either* column.
                    allowed_tokens = [str(s).strip().lower() for s in self.sessions if str(s).strip()]
                    allowed: set[str] = set()
                    for tok in allowed_tokens:
                        allowed.add(tok)
                        if tok == "0train":
                            allowed.update({"train", "session_t", "session_train"})
                        elif tok == "1test":
                            allowed.update({"test", "session_e", "session_test"})
                        elif tok == "train":
                            allowed.add("0train")
                        elif tok == "test":
                            allowed.add("1test")
                    mask = None
                    if "session" in meta_s.columns:
                        session_col = meta_s["session"].astype(str).str.strip().str.lower()
                        mask = session_col.isin(sorted(allowed)).to_numpy()
                    if "run" in meta_s.columns:
                        run_col = meta_s["run"].astype(str).str.strip().str.lower()
                        mask_run = run_col.isin(sorted(allowed)).to_numpy()
                        mask = mask_run if mask is None else (mask | mask_run)
                    if mask is None:
                        raise ValueError("MOABB metadata must contain a 'session' or 'run' column for filtering.")
                    if not bool(np.any(mask)):
                        continue
                    X_s = X_s[mask]
                    y_s = y_s[mask]
                    meta_s = meta_s.loc[mask].reset_index(drop=True)

                X_parts.append(np.asarray(X_s, dtype=dtype, order="C"))
                y_parts.append(np.asarray(y_s))
                meta_parts.append(meta_s)

            if not X_parts:
                raise RuntimeError("No trials found after MOABB preprocessing (empty subject list or session filter?).")

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            meta = pd.concat(meta_parts, axis=0, ignore_index=True)
            if self.car:
                self._apply_car_inplace(X)
            return np.asarray(X, dtype=dtype, order="C"), np.asarray(y), meta

        # paper_fir mode
        return self._load_arrays_paper_fir(subjects=subjects, dtype=dtype)

    def load_epochs_info(self, subject: Optional[int] = None):
        """Load one subject as MNE Epochs to obtain `info` for topographic plotting."""

        if subject is None:
            subject = self.subject_list[0]
        subject = int(subject)

        if self.preprocess == "moabb":
            if self.paradigm is None:
                raise RuntimeError("MOABB paradigm is not initialized.")
            epochs, _y, meta = self.paradigm.get_data(
                dataset=self.dataset, subjects=[subject], return_epochs=True
            )
            if self.sessions is not None:
                session_col = meta["session"].astype(str)
                mask = session_col.isin([str(s) for s in self.sessions]).to_numpy()
                epochs = epochs[mask]
            return epochs.info

        # paper_fir: load one raw and keep EEG info (channel positions, etc.)
        allowed_sessions: set[str] | None = None
        if self.sessions is not None:
            allowed_sessions = {str(s).strip().lower() for s in self.sessions if str(s).strip()}
        raws = self.dataset.get_data(subjects=[subject])[subject]
        for session_name, runs in raws.items():
            if allowed_sessions is not None:
                sess = str(session_name).strip().lower()
                if sess not in allowed_sessions:
                    if sess in {"session_t", "session_train", "train"} and "0train" in allowed_sessions:
                        pass
                    elif sess in {"session_e", "session_test", "test"} and "1test" in allowed_sessions:
                        pass
                    else:
                        continue
            for _run_name, raw in runs.items():
                raw = raw.copy().pick_types(eeg=True, eog=False, stim=False, misc=False)
                return raw.info
        raise RuntimeError("Could not find matching session/run to extract MNE info.")

    def _load_arrays_paper_fir(
        self,
        *,
        subjects: Sequence[int],
        dtype: np.dtype,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Paper-matched preprocessing on Raw.

        - causal FIR (order=`paper_fir_order`, Hamming by default), bandpass [fmin, fmax]
        - resample to target sfreq if needed
        - epoch tmin..tmax relative to cue (annotation onset)
        - select requested events
        """

        event_id = {e: int(self.dataset.event_id[e]) for e in self.events}
        code_to_label = {v: k for k, v in event_id.items()}

        allowed_sessions: set[str] | None = None
        if self.sessions is not None:
            allowed_sessions = {str(s).strip().lower() for s in self.sessions if str(s).strip()}

        # IMPORTANT (memory): some datasets (e.g., PhysionetMI) contain many runs per subject.
        # Accumulating one X array per run can keep a large number of intermediate arrays alive
        # until the final concatenation. We therefore aggregate per subject and only then append.
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        meta_rows: List[dict] = []

        for subject in subjects:
            X_sub_parts: List[np.ndarray] = []
            y_sub_parts: List[np.ndarray] = []
            meta_sub_rows: List[dict] = []

            sessions = self.dataset.get_data(subjects=[int(subject)])[int(subject)]
            for session_name, runs in sessions.items():
                if allowed_sessions is not None:
                    sess = str(session_name).strip().lower()
                    if sess not in allowed_sessions:
                        # Common MOABB naming variations for BCI IV 2a:
                        # 0train/1test  <->  session_T/session_E (and other train/test aliases).
                        if sess in {"session_t", "session_train", "train"} and "0train" in allowed_sessions:
                            pass
                        elif sess in {"session_e", "session_test", "test"} and "1test" in allowed_sessions:
                            pass
                        else:
                            continue

                for run_name, raw in runs.items():
                    # Keep a copy with stim channel for robust event extraction (some MOABB
                    # datasets provide events on stim and do not populate `raw.annotations`).
                    raw0 = raw.copy()

                    # 1) Events: prefer annotations, fall back to stim channel.
                    events, _ = mne.events_from_annotations(raw0, event_id=event_id, verbose="ERROR")
                    if len(events) == 0:
                        stim_picks = mne.pick_types(raw0.info, stim=True)
                        if len(stim_picks) > 0:
                            stim_chs = [raw0.ch_names[int(p)] for p in stim_picks]
                            events = mne.find_events(
                                raw0, stim_channel=stim_chs, shortest_event=1, verbose="ERROR"
                            )
                            if len(events) > 0:
                                keep_codes = np.array(sorted(set(event_id.values())), dtype=int)
                                events = events[np.isin(events[:, 2], keep_codes)]
                    if len(events) == 0:
                        continue

                    # 2) Resample with event index correction (only if needed).
                    if raw0.info["sfreq"] != self.resample:
                        raw0, events = raw0.resample(self.resample, npad="auto", events=events)

                    # 3) EEG-only copy for filtering/epoching (do NOT filter stim channels).
                    raw_eeg = raw0.copy().pick_types(eeg=True, eog=False, stim=False, misc=False)

                    self._apply_causal_fir(raw_eeg)

                    epochs = mne.Epochs(
                        raw_eeg,
                        events,
                        event_id=event_id,
                        tmin=self.tmin,
                        tmax=self.tmax,
                        baseline=None,
                        preload=True,
                        on_missing="ignore",
                        verbose="ERROR",
                    )
                    if len(epochs) == 0:
                        continue

                    X_e = epochs.get_data()
                    y_e = np.asarray([code_to_label[c] for c in epochs.events[:, 2]])

                    X_sub_parts.append(np.asarray(X_e, dtype=dtype, order="C"))
                    y_sub_parts.append(y_e)
                    meta_sub_rows.extend(
                        {
                            "subject": int(subject),
                            "session": str(session_name),
                            "run": str(run_name),
                        }
                        for _ in range(len(y_e))
                    )

            if not X_sub_parts:
                continue

            X_parts.append(np.concatenate(X_sub_parts, axis=0))
            y_parts.append(np.concatenate(y_sub_parts, axis=0))
            meta_rows.extend(meta_sub_rows)

        if not X_parts:
            raise RuntimeError("No epochs found after paper_fir preprocessing.")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        meta = pd.DataFrame(meta_rows)
        if self.car:
            self._apply_car_inplace(X)
        return X, y, meta

    @staticmethod
    def _apply_car_inplace(X: np.ndarray) -> None:
        """Common average reference (CAR), in-place.

        Subtract the per-timepoint mean across channels for each trial:
            X[t] <- X[t] - mean_c X[t, c, :]

        This is unsupervised and can reduce per-subject amplitude/reference offsets.
        """

        if X.ndim != 3:
            raise ValueError("CAR expects X with shape (n_trials, n_channels, n_times).")
        X -= X.mean(axis=1, keepdims=True)

    def _apply_causal_fir(self, raw: mne.io.BaseRaw) -> None:
        """Apply causal linear-phase FIR bandpass (Hamming window) in-place."""

        sfreq = float(raw.info["sfreq"])
        numtaps = int(self.paper_fir_order) + 1  # order 50 -> 51 taps
        b = firwin(
            numtaps=numtaps,
            cutoff=[float(self.fmin), float(self.fmax)],
            pass_zero=False,
            fs=sfreq,
            window=self.paper_fir_window,
        )
        data = raw.get_data().astype(np.float64, copy=False)
        # Causal filtering (one-pass), matching Matlab `filter` behavior.
        filtered = lfilter(b, [1.0], data, axis=-1)
        raw._data[:] = filtered


class BCIIV2aMoabbLoader(MoabbMotorImageryLoader):
    """Backward-compatible loader for MOABB BNCI2014_001 (BCI Competition IV 2a)."""

    def __init__(
        self,
        fmin: float,
        fmax: float,
        tmin: float,
        tmax: float,
        resample: float,
        events: Sequence[str],
        sessions: Optional[Sequence[str]] = None,
        preprocess: str = "moabb",
        car: bool = False,
        paper_fir_order: int = 50,
        paper_fir_window: str = "hamming",
    ) -> None:
        super().__init__(
            dataset="BNCI2014_001",
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            resample=resample,
            events=events,
            sessions=sessions,
            preprocess=preprocess,
            car=car,
            paper_fir_order=paper_fir_order,
            paper_fir_window=paper_fir_window,
        )


def split_by_subject(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> Dict[int, SubjectData]:
    """Split MOABB-returned arrays into a dict keyed by subject id."""

    if "subject" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'subject' column.")

    out: Dict[int, SubjectData] = {}
    for subject in sorted(meta["subject"].unique()):
        mask = meta["subject"].to_numpy() == subject
        out[int(subject)] = SubjectData(subject=int(subject), X=X[mask], y=y[mask])
    return out


def split_by_subject_session(
    X: np.ndarray, y: np.ndarray, meta: pd.DataFrame
) -> Dict[int, Dict[str, SubjectData]]:
    """Split MOABB-returned arrays into {subject -> {session -> SubjectData}}."""

    if "subject" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'subject' column.")
    if "session" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'session' column.")

    out: Dict[int, Dict[str, SubjectData]] = {}
    subj_col = meta["subject"].to_numpy()
    sess_col = meta["session"].astype(str).to_numpy()

    for subject in sorted(meta["subject"].unique()):
        subject = int(subject)
        out[subject] = {}
        subj_mask = subj_col == subject
        sessions = sorted(pd.Series(sess_col[subj_mask]).unique().tolist())
        for session in sessions:
            sess_mask = sess_col == str(session)
            mask = subj_mask & sess_mask
            out[subject][str(session)] = SubjectData(subject=subject, X=X[mask], y=y[mask])
    return out
