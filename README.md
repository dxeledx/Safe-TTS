# Safe-TTS

Safe-TTS is a research codebase for **safe test-time strategy selection** on
motor-imagery EEG. The current main line uses strict leave-one-subject-out
(LOSO) evaluation on **PhysioNetMI 3-class** (`left_hand / right_hand / feet`):
an `EEGNet` anchor is compared with public TTA candidates, then a calibrated
selector decides whether to accept a candidate or fall back to the anchor.

The newest experimental line is a **streaming warm-up Safe-TTS** protocol:
the first `W` unlabeled target trials are served to users with anchor-only
predictions while candidate TTA methods run in the background for diagnosis.
After the warm-up prefix, Safe-TTS selects one candidate or keeps the anchor,
then monitors the selected candidate online and can fall back to the anchor.

The repository also keeps the classical CSP/LDA line used during method
development, including Euclidean alignment (EA), OEA/EA-ZO variants, diagnostic
plots, and result-summary utilities.

## Current Method

- **Name**: Safe-TTS
- **Primary task**: PhysioNetMI 3-class motor imagery
- **Protocol**: strict LOSO; target labels are not used for test-time selection
- **Anchor**: `eegnet_noea`
- **Candidate pool**: `adabn`, `shot`, `tent`, `note`, `sar`, `pl`, `delta`,
  `ttime`, `cotta`, `t3a`
- **Offline selector**: multi-view evidential deployment selector with outer
  risk calibration (`D3`)
- **Streaming selector**: warm-up prefix selector with
  `absolute_core / relative_core / koopman_temporal`
- **Current confirmed streaming config**:
  `W=8`, `risk_alpha=0.40`, `min_utility_threshold=0.010`,
  `fallback_patience=2`

In the current implementation, `D3` means:

- `calibration_protocol=paper_oof_dev_cal`
- `selector_model=evidential`
- `selector_views=stats,decision,relative`
- `selector_hidden_dim=32`
- `selector_epochs=50`
- `selector_outcome_delta=0.02`
- `guard_gray_margin=0.02`
- `risk_alpha=0.40`

## Repository Tree

```text
.
├── csp_lda/                         # CSP/LDA, EA/OEA/ZO, metrics, plots, certificates
├── docs/
│   ├── current/                     # Current Safe-TTS method notes and experiment notes
│   ├── experiments/README.md        # Lab-notebook convention
│   └── SOTA.md                      # Related-work tracking table
├── scripts/
│   ├── safe_tts/                    # Canonical Safe-TTS wrappers
│   ├── ttime_suite/                 # DeepTransferEEG/MOABB export and strict LOSO TTA suite
│   ├── paper_supplement/            # Lightweight table/figure helpers
│   ├── offline_safe_tta_multi_select_crc_from_predictions.py
│   ├── offline_safe_tta_multi_select_from_predictions.py
│   └── update_results_registry.py
├── tests/                           # Small regression tests
├── third_party/DeepTransferEEG/      # Vendored public baseline code used by the TTA suite
├── run_csp_lda_loso.py              # Classical strict LOSO CSP/LDA entry point
├── run_csp_lda_cross_session.py     # Within-subject cross-session diagnostic entry point
└── requirements.txt
```

Large generated artifacts are intentionally not versioned: `outputs/`,
`runs/`, MNE/MOABB caches, checkpoints, exported `.npy` datasets, paper-asset
packs, and local reference repositories should stay local.

## Installation

Python 3.10+ is recommended. The original experiments were usually run in a
Conda environment named `eeg`, but a virtual environment is also fine.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your driver before
running the TTA suite.

## Data Preparation

The deep TTA pipeline consumes a compact DeepTransferEEG-style export generated
from MOABB:

```bash
python3 scripts/ttime_suite/export_moabb_for_deeptransfer.py \
  --dataset physionetmi \
  --events left_hand,right_hand,feet \
  --out-dir /path/to/physionetmi3_export
```

For the OpenBMI / Lee2019_MI 2-class extension:

```bash
python3 scripts/ttime_suite/export_moabb_for_deeptransfer.py \
  --dataset openbmi \
  --events left_hand,right_hand \
  --out-dir /path/to/openbmi2_export
```

The export directory contains `X.npy`, `labels.npy`, `subject_idx.npy`,
`meta.csv`, `class_order.json`, and `export_config.json`.

## Main Usage

Generate strict-LOSO predictions for the anchor and TTA candidates:

```bash
python3 scripts/safe_tts/run_physionetmi3_tta_suite.py \
  --data-dir /path/to/physionetmi3_export \
  --out-dir outputs/physionetmi3_tta_suite
```

Run Safe-TTS selection on the merged predictions:

```bash
python3 scripts/safe_tts/run_physionetmi3_safe_tts.py \
  --preds outputs/physionetmi3_tta_suite/predictions_all_methods.csv \
  --risk-alpha 0.40 \
  --out-dir outputs/physionetmi3_safe_tts_d3
```

The selector writes calibrated selection summaries, diagnostics, and the final
Safe-TTS method comparison files under the requested output directory.

Run the streaming warm-up Safe-TTS selector:

```bash
python3 scripts/safe_tts/run_warmup_safe_tts_from_predictions.py \
  --preds outputs/physionetmi3_tta_suite/predictions_all_methods.csv \
  --out-dir outputs/physionetmi3_warmup_safe_tts \
  --anchor-method eegnet_noea \
  --warmup-trials 8 \
  --risk-alpha 0.40 \
  --min-utility-threshold 0.010 \
  --online-fallback \
  --fallback-patience 2
```

This protocol reports both suffix gain after the warm-up and end-to-end gain
including the anchor-only prefix.

## Classical CSP/LDA Baselines

The older but still useful CSP/LDA pipeline remains available for strict LOSO
baselines and OEA/EA-ZO diagnostics:

```bash
python3 run_csp_lda_loso.py
```

Common variants:

```bash
python3 run_csp_lda_loso.py \
  --preprocess paper_fir \
  --n-components 6 \
  --methods csp-lda,ea-csp-lda,ea-zo-im-csp-lda

python3 run_csp_lda_cross_session.py \
  --preprocess paper_fir \
  --n-components 6
```

Outputs follow `outputs/YYYYMMDD/<N>class/...` and include result text files,
per-trial predictions, confusion matrices, CSP patterns, and method comparison
tables.

## Tests

```bash
python3 -m pytest tests
```

The tests are intentionally small and focus on stable helper behavior. Full EEG
experiments require MOABB data downloads and are normally run as explicit
experiment jobs.

## Research Notes

- Current method notes live in `docs/current/`.
- Experiment-note conventions live in `docs/experiments/README.md`.
- Related work and comparability notes live in `docs/SOTA.md`.
- Warm-up Safe-TTS notes:
  - `docs/experiments/20260427_warmup_safe_tts_protocol.md`
  - `docs/experiments/20260428_warmup_safe_tts_param_sweep.md`
- Refresh the local result registry with:

```bash
python3 scripts/update_results_registry.py \
  --outputs-dir outputs \
  --out docs/experiments/results_registry.csv
```

## License

No project-level license has been declared yet. The vendored
`third_party/DeepTransferEEG/` files keep their upstream license and notices.
