# 2026-04-29 Warm-up Safe-Veto + Multi-candidate Switch

## Scope

Dataset/protocol: PhysioNetMI 3-class, `paper_fir + CAR`, DeepTransferEEG/EEGNet prediction suite, 109 subjects.

Prediction source:

`outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`

Anchor: `eegnet_noea`.

This is a fast pilot for experiments A/B/C/D from the method screenshots, using `n_splits=2`, `selector_epochs=5`. It is not yet the final 5-fold/20-epoch confirmation run.

## Code Changes

- Added `--method-prior-mode onehot` to append candidate identity as a strategy-prior view.
- Added `--selection-policy default_veto_switch`, `--default-candidate-method`, and `--switch-utility-margin`.
- Added `--risk-only-selection` so early selection is gated by risk first, not by a hard utility floor.
- Added dynamic Koopman chunks and `--disable-koopman-below-w`; in this pilot Koopman is disabled for `W < 16`.
- Added `--dev-objective tail_accept` with lower-tail and accept-rate terms.
- Online fallback can now score augmented method-prior features correctly.

## Important Reference: Streaming Always-candidate

Raw full-subject TTA numbers are not the correct target once the first `W` trials are forced to use anchor output. Under the streaming protocol, always using a candidate after the warm-up prefix gives:

| W | Method | Mean Acc (%) | Gain vs Anchor (pp) | Worst Acc (%) | Worst Subject | NTR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | adabn_kooptta_ref | 66.55 | 8.94 | 34.33 | 38 | 12.84 |
| 8 | shot_kooptta_ref | 65.76 | 8.15 | 34.33 | 38 | 19.27 |
| 8 | tent_kooptta_ref | 64.52 | 6.91 | 29.85 | 38 | 21.10 |
| 8 | note_kooptta_ref | 64.51 | 6.89 | 29.85 | 38 | 22.02 |
| 8 | sar_dteeg_ref | 64.06 | 6.45 | 34.33 | 38 | 25.69 |
| 16 | adabn_kooptta_ref | 63.56 | 5.95 | 29.85 | 38 | 23.85 |
| 24 | adabn_kooptta_ref | 60.39 | 2.78 | 25.37 | 38 | 36.70 |
| 32 | adabn_kooptta_ref | 57.13 | -0.48 | 20.90 | 38 | 49.54 |

Therefore the screenshot target `68.49% / worst 39.71%` corresponds to full-subject raw adabn, not to the warm-up streaming protocol.

## Raw Full-subject Method Reference

| Method | Mean Acc (%) | Gain vs Anchor (pp) | Worst Acc (%) | Worst Subject | NTR (%) |
| --- | --- | --- | --- | --- | --- |
| adabn_kooptta_ref | 68.49 | 10.88 | 39.71 | 24 | 11.01 |
| shot_kooptta_ref | 67.82 | 10.21 | 39.71 | 24 | 17.43 |
| tent_kooptta_ref | 65.93 | 8.31 | 34.85 | 80 | 19.27 |
| note_kooptta_ref | 65.91 | 8.30 | 34.85 | 80 | 20.18 |
| sar_dteeg_ref | 65.50 | 7.89 | 34.85 | 25 | 20.18 |
| pl_dteeg_ref | 62.27 | 4.66 | 30.30 | 25 | 30.28 |
| delta_dteeg_ref | 61.20 | 3.59 | 27.27 | 25 | 34.86 |
| ttime_dteeg_ref | 61.17 | 3.56 | 25.76 | 25 | 35.78 |
| eegnet_noea | 57.61 | 0.00 | 32.84 | 5 | 0.00 |
| cotta_dteeg_ref | 50.30 | -7.31 | 26.47 | 63 | 72.48 |
| t3a_kooptta_ref | 41.76 | -15.86 | 24.64 | 75 | 93.58 |

## Experiment Matrix

Common fast-pilot arguments:

```bash
--warmup-trials 8,16,24,32
--risk-alpha 0.40
--min-utility-threshold -0.05
--min-accept-rate 0.30
--risk-only-selection
--dynamic-koopman-chunks
--disable-koopman-below-w 16
--dev-objective tail_accept
--lambda-tail 0.5
--lambda-accept 0.02
--tail-frac 0.10
--n-splits 2
--selector-epochs 5
```

Outputs:

- A: `outputs/20260429/3class/expA_adabn_veto_fast_v1/`
- B: `outputs/20260429/3class/expB_safe_pool_fast_v1/`
- C: `outputs/20260429/3class/expC_safe_pool_onehot_fast_v1/`
- D: `outputs/20260429/3class/expD_safe_pool_fallback_fast_v1/`

## Results

| Experiment | W | Mean Acc (%) | E2E Gain (pp) | Worst Acc (%) | Worst Subject | Accept (%) | E2E NTR (%) | Verified (%) | Fallback among accepted (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A adabn veto | 8 | 65.01 | 7.40 | 34.33 | 38 | 82.57 | 11.93 | 98.17 | 0.00 |
| A adabn veto | 16 | 60.40 | 2.79 | 29.85 | 38 | 47.71 | 11.93 | 55.05 | 0.00 |
| A adabn veto | 24 | 57.57 | -0.04 | 32.84 | 38 | 3.67 | 2.75 | 5.50 | 0.00 |
| A adabn veto | 32 | 57.61 | 0.00 | 32.84 | 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| B safe pool | 8 | 65.22 | 7.61 | 34.33 | 38 | 83.49 | 11.93 | 97.25 | 0.00 |
| B safe pool | 16 | 59.90 | 2.28 | 29.85 | 38 | 43.12 | 11.93 | 50.46 | 0.00 |
| B safe pool | 24 | 57.76 | 0.15 | 32.84 | 38 | 2.75 | 0.92 | 4.59 | 0.00 |
| B safe pool | 32 | 57.61 | 0.00 | 32.84 | 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| C safe pool + onehot | 8 | 64.21 | 6.60 | 32.84 | 5 | 77.06 | 11.01 | 97.25 | 0.00 |
| C safe pool + onehot | 16 | 59.70 | 2.09 | 29.85 | 38 | 43.12 | 12.84 | 53.21 | 0.00 |
| C safe pool + onehot | 24 | 57.44 | -0.17 | 32.84 | 5 | 2.75 | 2.75 | 3.67 | 0.00 |
| C safe pool + onehot | 32 | 57.61 | 0.00 | 32.84 | 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| D B + online fallback | 8 | 65.22 | 7.61 | 34.33 | 38 | 83.49 | 11.93 | 97.25 | 0.00 |
| D B + online fallback | 16 | 59.91 | 2.30 | 29.85 | 38 | 43.12 | 11.93 | 50.46 | 2.13 |
| D B + online fallback | 24 | 57.76 | 0.15 | 32.84 | 38 | 2.75 | 0.92 | 4.59 | 0.00 |
| D B + online fallback | 32 | 57.61 | 0.00 | 32.84 | 5 | 0.00 | 0.00 | 0.00 | 0.00 |

Best fast-pilot setting: B, `W=8`.

Relative to streaming always-adabn at `W=8`, B reduces NTR from 12.84% to 11.93%, but loses mean accuracy from 66.55% to 65.22%. It does not recover the raw full-subject adabn result because the streaming prefix itself changes the target.

## B W=8 Diagnostics

Selected method counts:

| Method | Count |
| --- | --- |
| adabn_kooptta_ref | 89 |
| eegnet_noea | 18 |
| sar_dteeg_ref | 2 |

Worst negative-transfer accepted subjects:

| Subject | Selected | Anchor Acc | Final Acc | E2E Gain | Suffix Gain | Risk | Utility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 47 | adabn_kooptta_ref | 52.24 | 44.78 | -7.46 | -8.47 | 0.2524 | 0.0186 |
| 107 | adabn_kooptta_ref | 55.88 | 48.53 | -7.35 | -8.33 | 0.2698 | -0.0298 |
| 102 | sar_dteeg_ref | 55.22 | 49.25 | -5.97 | -6.78 | 0.2453 | 0.0264 |
| 6 | adabn_kooptta_ref | 69.12 | 63.24 | -5.88 | -6.67 | 0.2634 | -0.0358 |
| 12 | adabn_kooptta_ref | 73.53 | 69.12 | -4.41 | -5.00 | 0.2541 | -0.0032 |

Worst final-accuracy subjects:

| Subject | Selected | Accept | Anchor Acc | Final Acc | E2E Gain | Suffix Gain |
| --- | --- | --- | --- | --- | --- | --- |
| 38 | adabn_kooptta_ref | 1 | 32.84 | 34.33 | 1.49 | 1.69 |
| 87 | eegnet_noea | 0 | 35.82 | 35.82 | 0.00 | 0.00 |
| 67 | eegnet_noea | 0 | 37.31 | 37.31 | 0.00 | 0.00 |
| 24 | adabn_kooptta_ref | 1 | 36.76 | 38.24 | 1.47 | 1.67 |
| 99 | adabn_kooptta_ref | 1 | 39.13 | 39.13 | 0.00 | 0.00 |
| 26 | adabn_kooptta_ref | 1 | 42.03 | 39.13 | -2.90 | -3.28 |

## Interpretation

1. `W=8` is the only useful warm-up length in this setup. Longer warm-ups dilute the suffix and make the streaming candidate less useful.
2. A single adabn-veto already captures most of the gain. The safe pool adds only +0.21 pp mean over A, mostly by choosing `sar_dteeg_ref` for two subjects.
3. Simple one-hot method identity is not enough; C drops mean and worst. If method prior is kept, it should be source-calibrated numeric priors rather than raw one-hot.
4. Online fallback with `fallback_risk_threshold=0.65` and `patience=4` is too loose for this scorer. It did not trigger at `W=8`, so it does not reduce negative transfer.
5. The selector still assigns low risk to clearly harmful adabn cases. The main next fix is not adding more candidates; it is improving harmful-adabn detection under `W=8`.

## Next Experiment

Do not run full confirmation yet. First run a focused `W=8` sweep:

- Increase accept objective/constraint to approach streaming always-adabn: `min_accept_rate in {0.80, 0.90, 0.95}` and `lambda_accept in {0.05, 0.10}`.
- Add source-calibrated priors: per-method streaming mean gain, NTR, and lower-tail gain computed on source subjects only.
- Try a stricter fallback diagnostic around `fallback_risk_threshold in {0.35, 0.45, 0.55}` with `patience=4`, but report it as runtime anomaly detection, not primary selector risk.

