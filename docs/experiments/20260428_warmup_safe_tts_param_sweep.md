# 2026-04-28 Warm-up Safe-TTS parameter sweep

## Goal

Continue from `20260427_warmup_safe_tts_protocol.md` and sweep the safety/deployment parameters for the streaming warm-up Safe-TTS protocol.

Dataset and prediction source:

- dataset: PhysioNetMI 3-class (`left_hand / right_hand / feet`)
- preprocessing/model family: `paper_fir + CAR`, DeepTransferEEG-style EEGNet/TTA suite
- prediction file:
  - remote: `/home/wjx/workspace/TTA_demo/outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`
  - local: `outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`
- anchor: `eegnet_noea`
- subjects/trials: 109 subjects, 7373 trials

## Sweep design

Coarse sweep:

- script: `scripts/safe_tts/run_warmup_safe_tts_from_predictions.py`
- W fixed to `8`, because the 2026-04-27 W-grid showed W=8 was the only consistently useful warm-up length.
- fast selector budget: `--n-splits 2 --selector-epochs 5`
- online fallback enabled
- grid:
  - `risk_alpha in {0.20, 0.30, 0.40, 0.60, 0.80}`
  - `min_utility_threshold in {0.000, 0.005, 0.010, 0.015, 0.020}`
  - `fallback_patience in {1, 2, 3}`
- total: 75 settings

Remote summary:

- `/home/wjx/workspace/TTA_demo/outputs/20260428/3class/param_sweep1_summary.csv`

Local summary:

- `outputs/20260428/3class/param_sweep1_summary.csv`

## Coarse-sweep result

Top fast settings by mean e2e gain:

| tag | risk alpha | utility floor | patience | e2e gain | mean acc | accept | suffix neg-transfer | fallback among accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a060_u0005_p1` | 0.60 | 0.005 | 1 | +0.0029 | 0.5790 | 0.0459 | 0.0000 | 0.2000 |
| `a080_u0005_p1` | 0.80 | 0.005 | 1 | +0.0029 | 0.5790 | 0.0459 | 0.0000 | 0.2000 |
| `a060_u0005_p2` | 0.60 | 0.005 | 2 | +0.0022 | 0.5783 | 0.0459 | 0.0092 | 0.2000 |
| `a080_u0000_p1` | 0.80 | 0.000 | 1 | +0.0022 | 0.5783 | 0.0917 | 0.0092 | 0.6000 |
| `a020_u0005_p1` | 0.20 | 0.005 | 1 | +0.0016 | 0.5778 | 0.0275 | 0.0000 | 0.0000 |
| `a040_u0010_p1` | 0.40 | 0.010 | 1 | +0.0012 | 0.5774 | 0.0183 | 0.0000 | 0.0000 |

Fast-sweep interpretation:

- Utility floor `0.005` looked better than `0.010` and `0.020`.
- Very loose `utility=0.000` increased accept rate but reintroduced negative transfer.
- `fallback_patience=1` was safest in the fast sweep.
- Higher `risk_alpha` increased verified rate, but the high-alpha point needed confirmation because CP looseness can move risk to the test subject.

## Confirmation runs

The best and nearest safer fast points were rerun with stronger selector budget:

- `--n-splits 5`
- `--selector-epochs 20`
- W=8
- online fallback enabled

| output dir | risk alpha | utility floor | patience | mean acc | e2e gain | accept | suffix neg-transfer | fallback among accepted | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `outputs/20260427/3class/warmup_safe_tts_koopman_W8_alpha040_u001_20e5fold_fallback_v1` | 0.40 | 0.010 | 2 | 0.5787 | +0.0026 | 0.0275 | 0.0000 | 0.6667 | current best confirmed |
| `outputs/20260428/3class/param_confirm_W8_a060_u0005_p1_20e5fold_fallback_v1` | 0.60 | 0.005 | 1 | 0.5779 | +0.0018 | 0.0459 | 0.0092 | 0.6000 | rejected: nonzero negative transfer |
| `outputs/20260428/3class/param_confirm_W8_a020_u0005_p1_20e5fold_fallback_v1` | 0.20 | 0.005 | 1 | 0.5768 | +0.0007 | 0.0183 | 0.0000 | 1.0000 | safe but too conservative |
| `outputs/20260428/3class/param_confirm_W8_a040_u0005_p1_20e5fold_fallback_v1` | 0.40 | 0.005 | 1 | 0.5786 | +0.0025 | 0.0275 | 0.0000 | 0.6667 | safe, slightly below current best |

Accepted subjects for `a040_u0005_p1`:

| subject | selected | e2e gain | suffix gain | no-fallback e2e gain | fallback trial |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | `note_kooptta_ref` | +0.1940 | +0.2203 | +0.1940 | -1 |
| 41 | `note_kooptta_ref` | +0.0597 | +0.0678 | +0.0299 | 20 |
| 53 | `sar_dteeg_ref` | +0.0149 | +0.0169 | +0.1791 | 12 |

Accepted subjects for previous best `a040_u0010_p2`:

| subject | selected | e2e gain | suffix gain | no-fallback e2e gain | fallback trial |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | `note_kooptta_ref` | +0.1940 | +0.2203 | +0.1940 | -1 |
| 41 | `note_kooptta_ref` | +0.0896 | +0.1017 | +0.0299 | 31 |
| 53 | `sar_dteeg_ref` | 0.0000 | 0.0000 | +0.1791 | 13 |

## Conclusion

The parameter sweep did not find a stronger confirmed operating point than the 2026-04-27 best setting:

- `W=8`
- `risk_alpha=0.40`
- `min_utility_threshold=0.010`
- `fallback_patience=2`
- 5-fold / 20-epoch selector

Current best confirmed performance:

- mean accuracy: `0.5787`
- anchor mean accuracy: `0.5761`
- e2e gain: `+0.0026` (`+0.26 pp`)
- suffix gain: `+0.0030`
- accept rate: `0.0275`
- suffix negative transfer: `0`

Scientific interpretation:

- The safety mechanism is working: confirmed zero negative transfer is achievable under the streaming warm-up protocol.
- The current bottleneck is not CP threshold tuning; it is low accept rate and over-conservative online fallback.
- Future improvement should focus on a better fallback score or delayed fallback confirmation, not simply loosening `risk_alpha`.

Next suggested lever:

- calibrate the online fallback threshold separately from the initial selection threshold, because subject 53 repeatedly shows that fallback can veto a strongly beneficial candidate too early.
