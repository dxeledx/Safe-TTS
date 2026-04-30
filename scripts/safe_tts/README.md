# Safe-TTS Scripts

This directory contains command-line entry points for Safe-TTS selection from
merged prediction CSV files.

## Trial-level Three-view Selection

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

The explicit feature list defines three views:

```text
absolute_view: absolute_core, candidate_confidence, candidate_margin
relative_view: relative_core, js_drift, confidence_delta, entropy_delta, prediction_disagree, high_conflict
temporal_view: koopman_temporal
```

The script writes:

```text
<date>_trial_method_comparison.csv
<date>_trial_per_subject_summary.csv
<date>_trial_per_trial_selection.csv
```

## Warm-up Selection

```bash
python3 scripts/safe_tts/run_warmup_safe_tts_from_predictions.py \
  --preds outputs/physionetmi3_tta_suite/predictions_all_methods.csv \
  --out-dir outputs/warmup_safe_tts \
  --anchor-method eegnet_noea \
  --candidate-methods adabn_kooptta_ref,shot_kooptta_ref,tent_kooptta_ref \
  --warmup-trials 8 \
  --risk-alpha 0.40
```

The warm-up script uses a prefix of unlabeled target trials for selection
diagnostics, then writes the same family of summary and selection files under
the requested output directory.
