# 2026-04-16 PhysioNetMI 3-class `paper_fir + CAR` expanded-pool Safe-TTS results

## Goal

Evaluate whether the expanded DeepTransferEEG-style TTA candidate pool improves the D3 Safe-TTS operating point on PhysioNetMI 3-class `paper_fir + CAR`, using `eegnet_noea` as the deployment anchor.

Compared against the previous narrow `NOTE-fix` pool, this run adds:

- `adabn_kooptta_ref`
- `shot_kooptta_ref`
- `sar_dteeg_ref`
- `pl_dteeg_ref`
- `delta_dteeg_ref`
- `cotta_dteeg_ref`
- `ttime_dteeg_ref`

## Artifacts

- launch note:
  - `docs/experiments/20260416_physionetmi3_pfcar_deeptransfereeg_expanded_safe_tts_launch.md`
- raw TTA predictions:
  - remote: `/home/wjx/workspace/TTA_demo/outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`
  - local: `outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`
- Safe-TTS output:
  - remote: `/home/wjx/workspace/TTA_demo/outputs/20260416/3class/physio_safe_tts_d3_pfcar_deeptransfereeg_expanded_alpha0.40_v1`
  - local: `outputs/20260416/3class/physio_safe_tts_d3_pfcar_deeptransfereeg_expanded_alpha0.40_v1`

## Final summary

Main method:

- `safe-tts-d3-evidential-physionetmi3-expanded-alpha0.40-v1`

Summary relative to `eegnet_noea`:

| method | mean acc | worst acc | mean delta vs anchor | neg-transfer | accept rate | oracle mean | oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `safe-tts-d3-evidential-physionetmi3-expanded-alpha0.40-v1` | 66.68% | 35.82% | +9.06 pts | 11.93% | 90.83% | 71.31% | 4.63 pts |
| `eegnet_noea` | 57.61% | 32.84% | +0.00 pts | 0.00% | - | - | - |

## Comparison against earlier narrow-pool Safe-TTS

Reference:

- `docs/experiments/20260415_physionetmi3_pfcar_notefix_safe_tts_results.md`

| method | mean acc | worst acc | mean delta vs anchor | neg-transfer | accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous narrow-pool `safe-tts-d3-evidential-physionetmi3-alpha0.40-v1` | 62.31% | 35.82% | +4.70 pts | 13.76% | 62.39% |
| expanded-pool `safe-tts-d3-evidential-physionetmi3-expanded-alpha0.40-v1` | 66.68% | 35.82% | +9.06 pts | 11.93% | 90.83% |

Expanded-pool Safe-TTS therefore gives:

- `+4.37 pts` higher mean subject accuracy
- same worst-subject accuracy
- `-1.83 pts` lower negative-transfer rate
- `+28.44 pts` higher accept rate

## Comparison against strongest raw TTA methods

From `docs/experiments/20260416_physionetmi3_pfcar_deeptransfereeg_expanded_results.md`:

| method | mean acc | worst acc | mean delta vs anchor | neg-transfer |
| --- | ---: | ---: | ---: | ---: |
| `adabn_kooptta_ref` | 68.49% | 39.71% | +10.88 pts | 11.01% |
| `shot_kooptta_ref` | 67.82% | 39.71% | +10.21 pts | 17.43% |
| `safe-tts-d3-evidential-physionetmi3-expanded-alpha0.40-v1` | 66.68% | 35.82% | +9.06 pts | 11.93% |

Interpretation:

- Safe-TTS now clearly beats the anchor by a large margin.
- Safe-TTS is safer than `shot_kooptta_ref` in terms of negative transfer.
- Safe-TTS still does not surpass the best single raw method `adabn_kooptta_ref` on either mean or worst-subject accuracy.

## Selection behavior

Final selected method counts across 109 subjects:

- `shot_kooptta_ref`: `39`
- `adabn_kooptta_ref`: `33`
- `note_kooptta_ref`: `11`
- `tent_kooptta_ref`: `10`
- `eegnet_noea`: `10`
- `pl_dteeg_ref`: `2`
- `delta_dteeg_ref`: `2`
- `ttime_dteeg_ref`: `1`
- `sar_dteeg_ref`: `1`
- `cotta_dteeg_ref`: `0`
- `t3a_kooptta_ref`: `0`

Per-selected-method average delta vs anchor:

- `shot_kooptta_ref`: `+11.95 pts`
- `adabn_kooptta_ref`: `+8.36 pts`
- `note_kooptta_ref`: `+12.41 pts`
- `tent_kooptta_ref`: `+9.89 pts`
- `pl_dteeg_ref`: `+5.95 pts`
- `sar_dteeg_ref`: `+4.55 pts`
- `eegnet_noea`: `+0.00 pts`
- `ttime_dteeg_ref`: `+0.00 pts`
- `delta_dteeg_ref`: `-2.94 pts`

## Negative-transfer diagnostics

Overall negative-transfer count:

- `13 / 109` subjects

Negative-transfer cases by selected method:

- `shot_kooptta_ref`: `6`
- `adabn_kooptta_ref`: `5`
- `tent_kooptta_ref`: `1`
- `delta_dteeg_ref`: `1`

Most harmful cases:

- subject `94`: `delta_dteeg_ref`, `-10.29 pts`
- subject `10`: `tent_kooptta_ref`, `-8.96 pts`
- subject `47`: `adabn_kooptta_ref`, `-8.96 pts`
- subject `40`: `adabn_kooptta_ref`, `-8.82 pts`
- subject `59`: `adabn_kooptta_ref`, `-7.35 pts`
- subject `34`: `shot_kooptta_ref`, `-7.35 pts`

Worst final-accuracy subjects:

- subject `87`: selected `ttime_dteeg_ref`, final `35.82%`, same as anchor
- subject `24`: selected `adabn_kooptta_ref`, final `39.71%`
- subject `38`: selected `shot_kooptta_ref`, final `40.30%`

## Main takeaways

1. The expanded candidate pool materially improves Safe-TTS over the earlier `tent/note/t3a`-dominated pool.
2. The selector has learned the right broad preference order:
   - mostly route to `shot/adabn`
   - occasionally use `tent/note`
   - keep anchor fallback for uncertain cases
3. The selector fully blocks the two most dangerous methods in this pool:
   - `t3a_kooptta_ref`
   - `cotta_dteeg_ref`
4. Remaining negative transfer is now concentrated in aggressive positive methods, mainly `shot` and `adabn`, rather than the previously obvious high-risk methods.
5. For the paper, this run is a better Safe-TTS result than the earlier narrow-pool D3 result, but it is still not strong enough to claim domination over the best single TTA method.
