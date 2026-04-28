# 2026-04-16 PhysioNetMI 3-class `paper_fir + CAR` expanded DeepTransferEEG-style TTA results

## Goal

Expand the previous `NOTE-fix` candidate pool on PhysioNetMI 3-class `paper_fir + CAR` by adding more TTA methods referenced from the vendored `DeepTransferEEG` code path.

Existing/reused methods:

- `eegnet_noea`
- `tent_kooptta_ref`
- `note_kooptta_ref`
- `t3a_kooptta_ref`

Additional methods evaluated in this round:

- `adabn_kooptta_ref`
- `shot_kooptta_ref`
- `pl_dteeg_ref`
- `sar_dteeg_ref`
- `delta_dteeg_ref`
- `cotta_dteeg_ref`
- `ttime_dteeg_ref`

Remote output:

- `/home/wjx/workspace/TTA_demo/outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`

Local mirror:

- [predictions_all_methods.csv](/Users/jason/workspace/code/workspace/Reserch_experiment/SAFE_TTA/outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv)

## Aggregate summary

All deltas and negative-transfer rates below are relative to `eegnet_noea`.

| method | trial acc | mean subject acc | worst subject acc | mean delta vs anchor | neg-transfer rate | improved / same / worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adabn_kooptta_ref` | 68.44 | 68.49 | 39.71 | +10.88 | 11.01 | 94 / 3 / 12 |
| `shot_kooptta_ref` | 67.76 | 67.82 | 39.71 | +10.21 | 17.43 | 88 / 2 / 19 |
| `tent_kooptta_ref` | 65.89 | 65.93 | 34.85 | +8.31 | 19.27 | 82 / 6 / 21 |
| `note_kooptta_ref` | 65.88 | 65.91 | 34.85 | +8.30 | 20.18 | 82 / 5 / 22 |
| `sar_dteeg_ref` | 65.48 | 65.50 | 34.85 | +7.89 | 20.18 | 81 / 6 / 22 |
| `pl_dteeg_ref` | 62.27 | 62.27 | 30.30 | +4.66 | 30.28 | 72 / 4 / 33 |
| `delta_dteeg_ref` | 61.17 | 61.20 | 27.27 | +3.59 | 34.86 | 65 / 6 / 38 |
| `ttime_dteeg_ref` | 61.16 | 61.17 | 25.76 | +3.56 | 35.78 | 64 / 6 / 39 |
| `eegnet_noea` | 57.55 | 57.61 | 32.84 | +0.00 | 0.00 | 0 / 109 / 0 |
| `cotta_dteeg_ref` | 50.31 | 50.30 | 26.47 | -7.31 | 72.48 | 26 / 4 / 79 |
| `t3a_kooptta_ref` | 41.72 | 41.76 | 24.64 | -15.86 | 93.58 | 4 / 3 / 102 |

## Main observations

1. `adabn_kooptta_ref` is the strongest method in this round.
   - highest mean subject accuracy
   - tied-best worst-subject accuracy with `shot`
   - lowest negative-transfer rate among the strong positive methods

2. `shot_kooptta_ref` is the second-best raw performer.
   - strong mean accuracy
   - better worst-subject result than `tent/note/sar`
   - but still clearly riskier than `adabn`

3. `tent/note/sar` form a middle tier.
   - all are clearly positive over the anchor
   - but none beats `adabn`
   - `sar` does not show a clear safety advantage here

4. `pl/delta/ttime` are usable but weaker.
   - positive mean gains exist
   - worst-subject and negative-transfer are noticeably worse than the top group

5. `cotta` and `t3a` remain high-risk.
   - `cotta` underperforms the anchor and introduces large negative transfer
   - `t3a` is still the most dangerous method in the pool

## Implication for Safe-TTS

The earlier D3 analysis was based on a narrow pool dominated by `tent/note/t3a`.

This expanded run suggests the next Safe-TTS candidate pool should prioritize:

- `adabn_kooptta_ref`
- `shot_kooptta_ref`
- `tent_kooptta_ref`
- `note_kooptta_ref`
- `sar_dteeg_ref`

Optional lower-priority candidates:

- `pl_dteeg_ref`
- `delta_dteeg_ref`
- `ttime_dteeg_ref`

Likely exclude from the main safe-selection pool:

- `cotta_dteeg_ref`
- `t3a_kooptta_ref` as a high-risk stress-test candidate only
