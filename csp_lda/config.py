from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PreprocessingConfig:
    """MOABB preprocessing settings (via Paradigm).

    Notes
    -----
    MOABB `MotorImagery` supports `fmin`, `fmax`, `tmin`, `tmax`, `resample`
    and `events`. Session selection is handled by filtering MOABB metadata
    (e.g., `sessions=('0train',)` for BNCI2014_001).
    """

    fmin: float = 8.0
    fmax: float = 30.0
    # Match He & Wu EA-CSP-LDA paper settings for BCI IV 2a:
    # use 0.5–3.5s after cue appearance as trials.
    tmin: float = 0.5
    tmax: float = 3.5
    resample: float = 250.0
    events: Sequence[str] = ("left_hand", "right_hand")
    sessions: Sequence[str] = ("0train",)
    preprocess: str = "moabb"  # "moabb" | "paper_fir"
    car: bool = False  # common average reference after temporal filtering
    paper_fir_order: int = 50
    paper_fir_window: str = "hamming"


@dataclass(frozen=True)
class ModelConfig:
    """CSP+LDA model hyperparameters."""

    csp_n_components: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration."""

    out_dir: Path
    dataset: str = "BNCI2014_001"
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    metrics_average: str = "macro"
