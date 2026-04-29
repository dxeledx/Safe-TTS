# Safe-TTS

Safe-TTS is a compact Python toolkit for selecting safe test-time strategies from saved prediction CSV files. It works on trial-level prediction tables produced by EEG/BCI classifiers or adaptation methods, then writes subject-level selections, merged predictions, and summary metrics.

The repository is intentionally small: it contains the selector scripts, feature/certificate utilities, and minimal dependencies needed to run them.

## File Tree

```text
Safe-TTS/
├── README.md
├── requirements.txt
├── csp_lda/
│   ├── __init__.py
│   ├── alignment.py
│   ├── certificate.py
│   └── proba.py
└── scripts/
    ├── offline_safe_tta_multi_select_crc_from_predictions.py
    ├── offline_safe_tta_multi_select_from_predictions.py
    ├── offline_safe_tta_select_from_predictions.py
    └── safe_tts/
        ├── __init__.py
        └── run_warmup_safe_tts_from_predictions.py
```

## Requirements

- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- torch

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input Format

The main scripts expect a trial-level CSV with one row per method, subject, and trial:

```text
method,subject,trial,y_true,y_pred,proba_<class_1>,proba_<class_2>,...
```

Example columns:

```text
method,subject,trial,y_true,y_pred,proba_left_hand,proba_right_hand,proba_feet
```

Each candidate method should share the same `subject` and `trial` indexing as the anchor method.

## Main Scripts

### Multi-candidate risk-controlled selector

```bash
python scripts/offline_safe_tta_multi_select_crc_from_predictions.py \
  --preds outputs/predictions_all_methods.csv \
  --anchor-method eegnet_noea \
  --candidate-methods ALL \
  --selector-model evidential \
  --selector-views stats,decision,relative \
  --calibration-protocol paper_oof_dev_cal \
  --risk-alpha 0.40 \
  --out-dir outputs/safe_tts_run \
  --method-name safe_tts \
  --date-prefix 20260429
```

Useful selector view options:

```text
stats
decision
relative
dynamic
koopman
stochastic
absolute_core
relative_core
koopman_temporal
compact
```

The script writes:

```text
<date>_predictions_all_methods.csv
<date>_per_subject_selection.csv
<date>_method_comparison.csv
```

### Warm-up selector

```bash
python scripts/safe_tts/run_warmup_safe_tts_from_predictions.py \
  --preds outputs/predictions_all_methods.csv \
  --out-dir outputs/warmup_safe_tts \
  --anchor-method eegnet_noea \
  --candidate-methods adabn_kooptta_ref,shot_kooptta_ref,tent_kooptta_ref \
  --warmup-trials 8,16,24,32 \
  --warmup-feature-set compact \
  --compact-selector-views absolute,relative,koopman \
  --selection-policy default_veto_switch \
  --default-candidate-method adabn_kooptta_ref \
  --risk-alpha 0.40
```

This script evaluates a prefix-based selection protocol. It writes one per-subject CSV for each warm-up length and a combined method-comparison CSV.

### Single-candidate selector

```bash
python scripts/offline_safe_tta_select_from_predictions.py \
  --ea-preds outputs/anchor_predictions.csv \
  --cand-preds outputs/candidate_predictions.csv \
  --out-dir outputs/single_candidate_run
```

## Core Utilities

- `csp_lda/certificate.py`: feature construction, ridge certificates, logistic/HGB guards, and evidential selector training.
- `csp_lda/alignment.py`: alignment helpers used by the shared certificate module.
- `csp_lda/proba.py`: probability-column reordering helper.

## Notes

- The code operates on saved predictions; it does not require raw EEG data for selector evaluation.
- Large experiment outputs, checkpoints, datasets, and generated figures are intentionally not tracked in this repository.
- Keep prediction CSVs and result directories outside git, for example under `outputs/`.
