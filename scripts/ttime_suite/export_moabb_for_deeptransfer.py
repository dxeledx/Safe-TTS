from __future__ import annotations

"""
Export MOABB MotorImagery datasets to a DeepTransferEEG-friendly numpy format.

This script is meant to be executed on the *remote* runner environment (remote server)
where MOABB/MNE are available. Outputs are written to a small self-contained
folder that can be consumed by `scripts/ttime_suite/run_suite_loso.py`.

Output directory layout
-----------------------
<out_dir>/
  X.npy              float32, shape (n_trials, n_channels, n_times)
  labels.npy         int64,   shape (n_trials,)
  subject_idx.npy    int64,   shape (n_trials,)  (0..N-1)
  meta.csv           per-trial metadata
  class_order.json   list[str], fixed order used for labels/probas
  export_config.json reproducibility metadata (small)
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_CLASS_ORDER_4CLASS = ("left_hand", "right_hand", "feet", "rest")


@dataclass(frozen=True)
class _ExportConfig:
    dataset: str
    preprocess: str
    fmin: float
    fmax: float
    tmin: float
    tmax: float
    resample: float
    sessions: Sequence[str] | None
    car: bool = False
    paper_fir_order: int = 50
    paper_fir_window: str = "hamming"


def _resolve_preset(name: str) -> _ExportConfig:
    key = str(name).strip().lower()
    if key in {"bnci2014_001", "bnci2014001", "bnci", "bciiv2a", "bci_iv_2a", "bciciv2a"}:
        return _ExportConfig(
            dataset="BNCI2014_001",
            preprocess="paper_fir",
            fmin=8.0,
            fmax=30.0,
            tmin=0.5,
            tmax=3.5,
            resample=250.0,
            sessions=("0train",),
            car=False,
            paper_fir_order=50,
            paper_fir_window="hamming",
        )
    if key in {"physionetmi", "physio", "physionet"}:
        # NOTE: "paper_fir" is implemented inside SAFE_TTA (csp_lda.data).
        return _ExportConfig(
            dataset="PhysionetMI",
            preprocess="paper_fir",
            fmin=8.0,
            fmax=30.0,
            tmin=0.0,
            tmax=3.0,
            resample=160.0,
            sessions=None,  # ALL
            car=False,
            paper_fir_order=50,
            paper_fir_window="hamming",
        )
    if key in {"hgd", "schirrmeister2017", "schirrmeister"}:
        return _ExportConfig(
            dataset="Schirrmeister2017",
            preprocess="moabb",
            fmin=8.0,
            fmax=30.0,
            tmin=0.5,
            tmax=3.5,
            resample=250.0,
            sessions=("0train",),
            car=False,
            paper_fir_order=50,
            paper_fir_window="hamming",
        )
    if key in {"openbmi", "lee2019", "lee2019_mi"}:
        return _ExportConfig(
            dataset="Lee2019_MI",
            preprocess="moabb",
            fmin=8.0,
            fmax=30.0,
            tmin=0.0,
            tmax=4.0,
            resample=250.0,
            sessions=None,  # ALL labeled runs exposed by MOABB dataset class
            car=False,
            paper_fir_order=50,
            paper_fir_window="hamming",
        )
    if key in {"bnci2014_004", "bnci2014004", "bciiv2b", "bci_iv_2b", "bciciv2b"}:
        return _ExportConfig(
            dataset="BNCI2014_004",
            preprocess="moabb",
            fmin=8.0,
            fmax=30.0,
            tmin=0.0,
            tmax=4.0,
            resample=250.0,
            sessions=None,  # caller may choose ALL or a specific train/test session.
            car=False,
            paper_fir_order=50,
            paper_fir_window="hamming",
        )
    raise ValueError(
        f"Unknown preset dataset: {name!r}. Expected: bnci2014_001|physionetmi|hgd|openbmi|bnci2014_004"
    )


def _apply_cli_overrides(preset: _ExportConfig, args: argparse.Namespace) -> _ExportConfig:
    updates: dict[str, object] = {}

    if getattr(args, "preprocess", None) is not None:
        updates["preprocess"] = str(args.preprocess)
    if getattr(args, "fmin", None) is not None:
        updates["fmin"] = float(args.fmin)
    if getattr(args, "fmax", None) is not None:
        updates["fmax"] = float(args.fmax)
    if getattr(args, "tmin", None) is not None:
        updates["tmin"] = float(args.tmin)
    if getattr(args, "tmax", None) is not None:
        updates["tmax"] = float(args.tmax)
    if getattr(args, "resample", None) is not None:
        updates["resample"] = float(args.resample)
    if getattr(args, "car", None) is not None:
        updates["car"] = bool(args.car)
    if getattr(args, "fir_order", None) is not None:
        updates["paper_fir_order"] = int(args.fir_order)
    if getattr(args, "fir_window", None) is not None:
        updates["paper_fir_window"] = str(args.fir_window)

    return replace(preset, **updates) if updates else preset


def _encode_labels(
    y: np.ndarray,
    *,
    class_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_int, y_str) aligned with class_order."""

    order = [str(c) for c in class_order]
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        y_int = np.asarray(y, dtype=int).reshape(-1)
        if y_int.size == 0:
            raise ValueError("Empty y.")
        # Some pipelines use 1..K labels; normalize to 0..K-1 when plausible.
        if int(np.min(y_int)) == 1 and int(np.max(y_int)) == len(order):
            y_int = y_int - 1
        if int(np.min(y_int)) < 0 or int(np.max(y_int)) >= len(order):
            raise ValueError(
                f"Numeric labels out of range for class_order (0..{len(order)-1}): "
                f"min={int(np.min(y_int))} max={int(np.max(y_int))}"
            )
        y_str = np.asarray([order[int(i)] for i in y_int], dtype=object)
        return y_int.astype(np.int64), y_str

    y_str = np.asarray(y, dtype=str).reshape(-1)
    mapping = {c: i for i, c in enumerate(order)}
    missing = sorted({s for s in set(y_str.tolist()) if s not in mapping})
    if missing:
        raise ValueError(f"Labels not in class_order: {missing}. class_order={order}")
    y_int = np.asarray([mapping[s] for s in y_str], dtype=np.int64)
    return y_int, y_str.astype(object)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _parse_sessions(s: str) -> Sequence[str] | None:
    s = str(s).strip()
    if not s or s.upper() == "ALL":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(parts) if parts else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MOABB datasets for DeepTransferEEG TTA suite.")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Preset dataset: bnci2014_001, physionetmi, hgd (Schirrmeister2017), "
            "openbmi (Lee2019_MI), or bnci2014_004 (BCI IV 2b)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (will be created).",
    )
    p.add_argument(
        "--events",
        type=str,
        default=",".join(DEFAULT_CLASS_ORDER_4CLASS),
        help="Comma-separated class/event names (default: left_hand,right_hand,feet,rest).",
    )
    p.add_argument(
        "--sessions",
        type=str,
        default="",
        help="Comma-separated sessions/runs to include (e.g. 0train). Use ALL/empty for all.",
    )
    p.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Optional subject ids to export (dataset-native ids). Example: '1-14' or '1,2,3'. Empty means ALL.",
    )
    p.add_argument(
        "--preprocess",
        type=str,
        default=None,
        choices=("moabb", "paper_fir"),
        help="Optional override for preprocessing mode.",
    )
    p.add_argument("--fmin", type=float, default=None, help="Optional bandpass low cutoff override.")
    p.add_argument("--fmax", type=float, default=None, help="Optional bandpass high cutoff override.")
    p.add_argument("--tmin", type=float, default=None, help="Optional epoch start override.")
    p.add_argument("--tmax", type=float, default=None, help="Optional epoch end override.")
    p.add_argument("--resample", type=float, default=None, help="Optional resample rate override.")
    car_group = p.add_mutually_exclusive_group()
    car_group.add_argument("--car", dest="car", action="store_true", help="Apply common average reference.")
    car_group.add_argument("--no-car", dest="car", action="store_false", help="Disable common average reference.")
    p.set_defaults(car=None)
    p.add_argument("--fir-order", type=int, default=None, help="Optional paper_fir order override.")
    p.add_argument("--fir-window", type=str, default=None, help="Optional paper_fir window override.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    preset = _resolve_preset(args.dataset)
    preset = _apply_cli_overrides(preset, args)

    class_order = [p.strip() for p in str(args.events).split(",") if p.strip()]
    if not class_order:
        raise SystemExit("--events parsed to empty list")

    # Allow overriding the preset session filter from CLI.
    sessions = _parse_sessions(args.sessions) if str(args.sessions).strip() else preset.sessions

    subjects: list[int] | None = None
    subj_str = str(args.subjects).strip()
    if subj_str and subj_str.upper() != "ALL":
        if "-" in subj_str and "," not in subj_str:
            a, b = subj_str.split("-", 1)
            subjects = list(range(int(a), int(b) + 1))
        else:
            subjects = [int(s) for s in subj_str.split(",") if str(s).strip()]
        if not subjects:
            raise SystemExit("--subjects parsed to empty list")

    # Local import: keeps module import cheap if moabb isn't installed.
    from csp_lda.data import MoabbMotorImageryLoader

    loader = MoabbMotorImageryLoader(
        dataset=preset.dataset,
        preprocess=preset.preprocess,
        fmin=preset.fmin,
        fmax=preset.fmax,
        tmin=preset.tmin,
        tmax=preset.tmax,
        resample=preset.resample,
        events=class_order,
        sessions=sessions,
        car=bool(preset.car),
        paper_fir_order=int(preset.paper_fir_order),
        paper_fir_window=str(preset.paper_fir_window),
    )

    X, y_raw, meta = loader.load_arrays(subjects=subjects, dtype=np.float32)
    if "subject" not in meta.columns:
        raise SystemExit("MOABB meta must contain a 'subject' column.")

    y_int, y_str = _encode_labels(np.asarray(y_raw), class_order=class_order)

    subjects_orig = meta["subject"].astype(int).to_numpy()
    uniq_subjects = sorted(set(subjects_orig.tolist()))
    subj_to_idx = {int(s): int(i) for i, s in enumerate(uniq_subjects)}
    subject_idx = np.asarray([subj_to_idx[int(s)] for s in subjects_orig], dtype=np.int64)

    # trial index within each subject
    trial = (
        pd.DataFrame({"subject_idx": subject_idx})
        .groupby("subject_idx", sort=False)
        .cumcount()
        .to_numpy(dtype=np.int64)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", np.asarray(X, dtype=np.float32, order="C"))
    np.save(out_dir / "labels.npy", np.asarray(y_int, dtype=np.int64, order="C"))
    np.save(out_dir / "subject_idx.npy", np.asarray(subject_idx, dtype=np.int64, order="C"))

    meta_out = pd.DataFrame(
        {
            "subject_orig": subjects_orig,
            "subject_idx": subject_idx,
            "trial": trial,
            "y_int": y_int,
            "y_str": y_str,
        }
    )
    # Keep useful provenance columns if present.
    for c in ["session", "run"]:
        if c in meta.columns:
            meta_out[c] = meta[c].astype(str).to_numpy()
    meta_out.to_csv(out_dir / "meta.csv", index=False)

    _write_json(out_dir / "class_order.json", list(class_order))
    _write_json(
        out_dir / "export_config.json",
        {
            "preset": str(args.dataset),
            "moabb_dataset": str(loader.dataset_id),
            "preprocess": str(preset.preprocess),
            "fmin": float(preset.fmin),
            "fmax": float(preset.fmax),
            "tmin": float(preset.tmin),
            "tmax": float(preset.tmax),
            "resample": float(preset.resample),
            "car": bool(preset.car),
            "paper_fir_order": int(preset.paper_fir_order),
            "paper_fir_window": str(preset.paper_fir_window),
            "events": list(class_order),
            "sessions": None if sessions is None else list(sessions),
            "subjects": None if subjects is None else list(subjects),
            "n_trials": int(X.shape[0]),
            "n_subjects": int(len(uniq_subjects)),
            "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]),
        },
    )

    print(f"Wrote export to: {out_dir}")
    print(f"X: {X.shape}  y: {y_int.shape}  subjects: {len(uniq_subjects)}")


if __name__ == "__main__":
    main()
