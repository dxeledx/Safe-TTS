# Safe-TTS

Safe-TTS is a research codebase for safe test-time strategy selection on
motor-imagery EEG. It works from merged per-trial prediction files produced by
an anchor classifier and multiple test-time adaptation (TTA) candidates. The
selector scores the candidates and either deploys one candidate prediction or
falls back to the anchor prediction.

The current selector code supports trial-level multi-candidate selection with a
three-view diagnostic representation:

- an absolute view for candidate reliability,
- a relative view for anchor-candidate drift,
- a temporal view for within-candidate dynamics.

Generated data, checkpoints, experiment outputs, logs, and paper assets are not
intended to be versioned in this repository.

## Repository Tree

```text
.
├── csp_lda/
│   ├── certificate.py                       # Certificate, guard, and evidential selector utilities
│   ├── alignment.py                         # EA/OEA alignment helpers
│   └── proba.py                             # Probability-column helpers
├── scripts/
│   ├── safe_tts/
│   │   ├── run_trial_safe_tts_from_predictions.py
│   │   ├── run_warmup_safe_tts_from_predictions.py
│   │   └── README.md
│   ├── offline_safe_tta_multi_select_from_predictions.py
│   ├── offline_safe_tta_multi_select_crc_from_predictions.py
│   └── offline_safe_tta_select_from_predictions.py
└── requirements.txt
```

## Installation

Python 3.10+ is recommended. The original experiments were usually run in a
Conda environment named `eeg`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

If CUDA is used, install the PyTorch build that matches the local driver before
running the TTA suite.

## Prediction File Format

The trial-level Safe-TTS selector consumes a merged CSV containing the anchor
method and all candidate methods. Required columns:

```text
method, subject, trial, y_true, y_pred, proba_<class_1>, ..., proba_<class_C>
```

The selector uses `y_true` only for source-subject training/calibration and for
writing evaluation files. Target-subject candidate selection is based on
prediction-derived diagnostics.

## Trial-level Three-view Selector

The main trial-level entry point is:

```text
scripts/safe_tts/run_trial_safe_tts_from_predictions.py
```

For each subject, trial, and candidate method, the script constructs a
label-free feature vector from anchor and candidate predictions. Candidate
features are grouped into three views.

### Absolute View

The absolute view describes whether the candidate prediction itself appears
reliable.

```text
absolute_core
candidate_confidence
candidate_margin
```

- `absolute_core`: normalized candidate certainty.
- `candidate_confidence`: maximum candidate class probability.
- `candidate_margin`: gap between the largest and second-largest candidate
  probabilities.

### Relative View

The relative view describes how the candidate prediction differs from the anchor
prediction.

```text
relative_core
js_drift
confidence_delta
entropy_delta
prediction_disagree
high_conflict
```

- `relative_core`: drift score with an optional high-confidence conflict
  weight.
- `js_drift`: normalized Jensen-Shannon divergence between anchor and candidate
  probability vectors.
- `confidence_delta`: candidate confidence minus anchor confidence.
- `entropy_delta`: candidate entropy minus anchor entropy.
- `prediction_disagree`: whether anchor and candidate predicted different
  classes.
- `high_conflict`: whether anchor and candidate disagree while both are
  high-confidence.

### Temporal View

The temporal view describes within-candidate trial-to-trial dynamics.

```text
koopman_temporal
```

The implementation forms a low-dimensional state from candidate certainty,
relative drift, and high-confidence conflict, then measures normalized
state-transition magnitude across adjacent trials of the same subject and
candidate method.

### Feature Configuration

The compact preset keeps one statistic per view:

```bash
--trial-feature-preset compact
```

The current explicit three-view feature set can be passed as:

```bash
--trial-feature-preset rich \
--trial-feature-names absolute_core,candidate_confidence,candidate_margin,relative_core,js_drift,confidence_delta,entropy_delta,prediction_disagree,high_conflict,koopman_temporal
```

## Multi-candidate Selection

The trial-level selector supports free multi-candidate selection:

```bash
python3 scripts/safe_tts/run_trial_safe_tts_from_predictions.py \
  --preds outputs/physionetmi3_tta_suite/predictions_all_methods.csv \
  --out-dir outputs/trial_safe_tts \
  --anchor-method eegnet_noea \
  --candidate-methods adabn_kooptta_ref,shot_kooptta_ref,tent_kooptta_ref \
  --selection-policy free \
  --risk-only-selection \
  --risk-alpha 0.25 \
  --trial-feature-preset rich \
  --trial-feature-names absolute_core,candidate_confidence,candidate_margin,relative_core,js_drift,confidence_delta,entropy_delta,prediction_disagree,high_conflict,koopman_temporal
```

For each trial, feasible candidates are filtered by predicted risk. Among
feasible candidates, the selector chooses the candidate with the highest
predicted utility. If no candidate passes the risk gate, the final prediction is
the anchor prediction.

The script writes:

```text
<date>_trial_method_comparison.csv      # run-level summary
<date>_trial_per_subject_summary.csv    # per-subject accuracy and selection summary
<date>_trial_per_trial_selection.csv    # per-trial selected method and final prediction
```

## Warm-up Selector

The warm-up entry point is:

```text
scripts/safe_tts/run_warmup_safe_tts_from_predictions.py
```

It uses the first `W` target trials as an unlabeled observation prefix. During
the prefix, user-visible predictions come from the anchor while candidate
methods can run in the background. After the prefix, the script chooses a
candidate or keeps the anchor according to prefix diagnostics.

Example:

```bash
python3 scripts/safe_tts/run_warmup_safe_tts_from_predictions.py \
  --preds outputs/physionetmi3_tta_suite/predictions_all_methods.csv \
  --out-dir outputs/warmup_safe_tts \
  --anchor-method eegnet_noea \
  --candidate-methods adabn_kooptta_ref,shot_kooptta_ref,tent_kooptta_ref \
  --warmup-trials 8 \
  --risk-alpha 0.40
```

## Tests

```bash
python3 -m pytest tests
```

Full EEG experiments require MOABB data, exported datasets, and trained
prediction files.

## License

No project-level license has been declared yet. Vendored or third-party code
keeps its upstream license and notices.
