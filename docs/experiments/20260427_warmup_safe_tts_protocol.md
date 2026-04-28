# 2026-04-27 Warm-up Safe-TTS protocol redesign

## Motivation

当前 Safe-TTS 主线仍是 subject-level offline selection：对目标被试整段无标签预测序列一次性构造三视图特征，然后选择候选 TTA 或回退 anchor。这与真实 TTA 部署场景存在两个不一致：

1. 三视图特征过多，`stats / decision / relative / dynamic` 中有大量冗余统计量，论文方法不够凝练。
2. 推理时需要目标被试全部无标签 trial，不能表达 trial stream 到达后的安全部署过程。

本轮改动目标是把方法定义为：

- 前 `W` 个 trial：用户端只输出 anchor；后台对候选 TTA 做 shadow adaptation 与无标签诊断。
- 第 `W` 个 trial 后：基于 warm-up 前缀特征选择一个候选或回退 anchor。
- 后续 suffix：执行被选候选；可选地并行保留 anchor 并做在线风险监控，风险升高时回退 anchor。

## Previous-run post-mortem

本轮改动前，先诊断最近一轮主线结果：

- raw TTA predictions:
  - `outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`
- Safe-TTS output:
  - `outputs/20260416/3class/physio_safe_tts_d3_pfcar_deeptransfereeg_expanded_alpha0.40_v1`

Method-level summary from `20260416_method_comparison.csv`:

- anchor: `eegnet_noea`
- Safe-TTS mean accuracy: `0.6668`
- anchor mean accuracy: `0.5761`
- mean delta vs anchor: `+0.0906`
- worst-subject accuracy: `0.3582` (subject `87`)
- negative-transfer rate: `13 / 109 = 0.1193`
- accept rate: `0.9083`

Selected-method counts:

- `shot_kooptta_ref`: `39`
- `adabn_kooptta_ref`: `33`
- `note_kooptta_ref`: `11`
- `tent_kooptta_ref`: `10`
- `eegnet_noea`: `10`
- `pl_dteeg_ref`: `2`
- `delta_dteeg_ref`: `2`
- `ttime_dteeg_ref`: `1`
- `sar_dteeg_ref`: `1`

Negative-transfer cases by selected method:

- `shot_kooptta_ref`: `6`
- `adabn_kooptta_ref`: `5`
- `tent_kooptta_ref`: `1`
- `delta_dteeg_ref`: `1`

Top harmful subjects:

| subject | selected | anchor acc | final acc | delta |
| ---: | --- | ---: | ---: | ---: |
| 94 | `delta_dteeg_ref` | 0.7647 | 0.6618 | -0.1029 |
| 10 | `tent_kooptta_ref` | 0.6269 | 0.5373 | -0.0896 |
| 47 | `adabn_kooptta_ref` | 0.5224 | 0.4328 | -0.0896 |
| 40 | `adabn_kooptta_ref` | 0.8971 | 0.8088 | -0.0882 |
| 59 | `adabn_kooptta_ref` | 0.6176 | 0.5441 | -0.0735 |
| 34 | `shot_kooptta_ref` | 0.8529 | 0.7794 | -0.0735 |

Worst final-accuracy subjects:

| subject | selected | anchor acc | final acc | delta |
| ---: | --- | ---: | ---: | ---: |
| 87 | `ttime_dteeg_ref` | 0.3582 | 0.3582 | 0.0000 |
| 24 | `adabn_kooptta_ref` | 0.3676 | 0.3971 | +0.0294 |
| 38 | `shot_kooptta_ref` | 0.3284 | 0.4030 | +0.0746 |
| 26 | `shot_kooptta_ref` | 0.4203 | 0.4058 | -0.0145 |
| 83 | `eegnet_noea` | 0.4118 | 0.4118 | 0.0000 |

Final selected prediction distribution and confusion structure:

- true distribution: `feet=2455`, `left_hand=2480`, `right_hand=2438`
- predicted distribution: `feet=2465`, `left_hand=2421`, `right_hand=2487`

Confusion matrix:

| true \ pred | feet | left_hand | right_hand |
| --- | ---: | ---: | ---: |
| feet | 1499 | 448 | 508 |
| left_hand | 466 | 1722 | 292 |
| right_hand | 500 | 251 | 1687 |

## Failure interpretation

上一轮 expanded-pool D3 的主要问题不是候选池没有收益，而是过度接受：

- mean gain 很大，但 `90.83%` accept rate 使剩余负迁移仍达到 `11.93%`。
- 负迁移集中在强正收益候选 `shot/adabn`，说明旧 selector 对“平均上强但个别 subject 有害”的候选缺少早期过程稳定性诊断。
- `delta_dteeg_ref` 虽只被选两次，但 subject `94` 是最坏负迁移，说明旧特征对高风险候选的局部异常仍会误判。
- 当前整段 subject-level 特征会看到完整目标序列，不适合作为真实 TTA 部署协议的主表口径。

因此本轮实验的核心假设是：

> 只用 warm-up 前缀构造 `absolute_core / relative_core / koopman_temporal` 三个核心统计量，并把监督目标改成 suffix gain，可以把 Safe-TTS 从离线 subject-level selector 改成部署一致的 early diagnosis selector；在线回退则进一步降低后期非平稳带来的残余负迁移。

## Planned first implementation

新增一个独立脚本，而不是直接覆盖旧 selector：

- input: existing merged `predictions_all_methods.csv`
- feature: first `W` trials only
  - `absolute_core`: balanced decision reliability
  - `relative_core`: high-confidence-conflict weighted normalized JS drift
  - `koopman_temporal`: DMD/Koopman residual plus spectral-radius penalty on warm-up blocks
- label/gain: suffix trials only (`t > W`)
- report:
  - suffix mean gain
  - end-to-end gain including anchor-only warm-up dilution
  - negative-transfer rate on enabled candidates
  - accept rate
  - warm-up sensitivity over `W in {8,16,24,32}`

## Implementation

Implemented script:

- `scripts/safe_tts/run_warmup_safe_tts_from_predictions.py`

Protocol details:

- exact outer leave-one-subject-out over the target subjects in the prediction CSV
- for each held-out subject, train selector only on other subjects
- selector input: three warm-up prefix features `[A, R, T]`
- selector supervision: suffix gain after the warm-up prefix
- calibration: development threshold search plus calibration-subject Clopper-Pearson UCB check
- online fallback option: after deployment, keep anchor predictions available and re-score the selected candidate on a sliding window; revert to anchor after repeated risk violations

Important implementation correction:

- Early online-fallback runs computed fallback metrics but summarized pre-fallback `acc_final/e2e_gain`.
- The script now uses fallback-adjusted outputs as the main `acc_final/e2e_gain/suffix_gain` when `--online-fallback` is enabled, while preserving `*_no_fallback` diagnostic columns.

## Runs

All runs used:

- host: `lab-internal`
- remote root: `/home/wjx/workspace/TTA_demo`
- predictions:
  - `outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv`
- anchor: `eegnet_noea`

Smoke:

```bash
/home/wjx/anaconda3/bin/conda run -n eeg python scripts/safe_tts/run_warmup_safe_tts_from_predictions.py \
  --preds outputs/20260416/3class/ttime_suite_physionetmi3_pfcar_deeptransfereeg_expanded_seed0_v1/predictions_all_methods.csv \
  --out-dir outputs/20260427/3class/warmup_safe_tts_smoke_w8_s1to8 \
  --anchor-method eegnet_noea \
  --warmup-trials 8 \
  --eval-subjects 1-8 \
  --selector-epochs 5 \
  --n-splits 2
```

Main diagnostic sweeps:

| output dir | setting | purpose |
| --- | --- | --- |
| `outputs/20260427/3class/warmup_safe_tts_koopman_wgrid_alpha040_fast5e2fold_v1` | W grid, 2-fold, 5 epochs, no utility floor | initial failure diagnostic |
| `outputs/20260427/3class/warmup_safe_tts_koopman_wgrid_alpha040_fast5e2fold_u002_v2` | W grid, utility floor 0.02, no fallback | isolate utility floor |
| `outputs/20260427/3class/warmup_safe_tts_koopman_wgrid_alpha040_fast5e2fold_u000_fallback_v2` | W grid, utility floor 0.00, online fallback | fallback with loose utility |
| `outputs/20260427/3class/warmup_safe_tts_koopman_wgrid_alpha040_fast5e2fold_u001_fallback_v2` | W grid, utility floor 0.01, online fallback | fallback with moderate utility |
| `outputs/20260427/3class/warmup_safe_tts_koopman_wgrid_alpha040_fast5e2fold_u002_fallback_v2` | W grid, utility floor 0.02, online fallback | fallback with conservative utility |
| `outputs/20260427/3class/warmup_safe_tts_koopman_W8_alpha040_u001_20e5fold_fallback_v1` | W=8, 5-fold, 20 epochs, utility floor 0.01, online fallback | stronger confirmation of best diagnostic point |

## Results

### Initial W-grid failure

Without an explicit utility floor, threshold search often selected `utility_threshold=-inf`, so the policy accepted low-utility candidates if predicted risk was low. This produced negative mean gain.

| W | mean acc | anchor mean | e2e gain | suffix gain | accept | suffix neg-transfer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.5757 | 0.5761 | -0.0004 | -0.0005 | 0.0734 | 0.0367 |
| 16 | 0.5730 | 0.5761 | -0.0031 | -0.0041 | 0.0367 | 0.0275 |
| 24 | 0.5692 | 0.5761 | -0.0070 | -0.0108 | 0.0550 | 0.0550 |
| 32 | 0.5720 | 0.5761 | -0.0041 | -0.0079 | 0.0459 | 0.0367 |

Main failure cases:

- W=8: subject `12`, `t3a_kooptta_ref`, suffix gain `-0.2500`
- W=16: subject `33`, `cotta_dteeg_ref`, suffix gain `-0.2115`
- W=24: subject `41`, `cotta_dteeg_ref`, suffix gain `-0.3953`
- W=32: subject `45`, `ttime_dteeg_ref`, suffix gain `-0.4118`

Interpretation:

- The prefix features were not enough by themselves.
- Online fallback is necessary for deployment consistency.
- The selector must not accept a candidate on risk alone; it needs a positive predicted-benefit constraint.

### Utility floor and online fallback

Fast 2-fold / 5-epoch diagnostic sweep:

| setting | W | e2e gain | mean acc | accept | suffix neg-transfer | fallback among accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| utility>=0.00 + fallback | 8 | -0.0001 | 0.5760 | 0.0459 | 0.0183 | 0.4000 |
| utility>=0.00 + fallback | 16 | -0.0004 | 0.5757 | 0.0183 | 0.0092 | 0.0000 |
| utility>=0.00 + fallback | 24 | 0.0000 | 0.5761 | 0.0000 | 0.0000 | 0.0000 |
| utility>=0.00 + fallback | 32 | -0.0031 | 0.5730 | 0.0367 | 0.0275 | 0.5000 |
| utility>=0.01 + fallback | 8 | +0.0012 | 0.5774 | 0.0183 | 0.0000 | 0.0000 |
| utility>=0.01 + fallback | 16 | 0.0000 | 0.5761 | 0.0092 | 0.0000 | 0.0000 |
| utility>=0.01 + fallback | 24 | -0.0012 | 0.5749 | 0.0092 | 0.0092 | 0.0000 |
| utility>=0.01 + fallback | 32 | -0.0013 | 0.5748 | 0.0183 | 0.0092 | 0.5000 |
| utility>=0.02 + no fallback | 8 | -0.0001 | 0.5760 | 0.0183 | 0.0092 | 0.0000 |
| utility>=0.02 + fallback | 8 | +0.0011 | 0.5772 | 0.0183 | 0.0000 | 0.5000 |

### Confirmation run

Best diagnostic point was re-run with a stronger selector budget:

- output:
  - `outputs/20260427/3class/warmup_safe_tts_koopman_W8_alpha040_u001_20e5fold_fallback_v1`
- setting:
  - W=8
  - 5-fold OOF
  - 20 epochs
  - utility floor 0.01
  - online fallback enabled

Result:

| W | mean acc | anchor mean | e2e gain | suffix gain | worst acc | worst subject | accept | suffix neg-transfer | fallback among accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.5787 | 0.5761 | +0.0026 | +0.0030 | 0.3284 | 5 | 0.0275 | 0.0000 | 0.6667 |

Accepted subjects in the confirmation run:

| subject | selected | e2e gain | suffix gain | no-fallback e2e gain | fallback trial |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | `note_kooptta_ref` | +0.1940 | +0.2203 | +0.1940 | -1 |
| 41 | `note_kooptta_ref` | +0.0896 | +0.1017 | +0.0299 | 31 |
| 53 | `sar_dteeg_ref` | 0.0000 | 0.0000 | +0.1791 | 13 |

Note:

- subject `53` shows the current online fallback is overly conservative: it vetoed a candidate that would have been beneficial without fallback.
- This is acceptable for a first safety run because negative transfer is zero, but it explains the low accept/effective-gain trade-off.

## Current conclusion

The warm-up version is directionally viable but not yet paper-strong:

- It fixes the deployment-protocol issue: only the first W unlabeled trials are used for selection; suffix labels are used only for evaluation/supervision across source subjects.
- The three compressed views are implementable and produce a clean method story.
- Online fallback is necessary; without it, the same accepted set can still produce negative suffix gain.
- The current selector is too conservative: best confirmed mean gain is `+0.26 pp`, with only `2.75%` accept rate.

Next experimental lever should be one of:

1. Calibrate fallback separately from initial selection so it does not veto beneficial trajectories like subject `53`.
2. Add a direct prefix-level monotone rule or small calibrator over `[A, R, T]` before the evidential head, because utility is currently weakly separated.
3. Run a W/utility/fallback-threshold development split sweep and report the risk-gain frontier instead of fixing `utility>=0.01` by inspection.
