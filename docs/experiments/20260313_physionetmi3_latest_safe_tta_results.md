# 2026-03-13 — 最新 Safe-TTS 方法结果汇报（PhysioNetMI 3-class TTA suite）

## 实验对象

- TTA suite 输入：`outputs/20260302/3class/ttime_suite_physionetmi_3class_lrfeet_seed0_full_v2/predictions_all_methods.csv`
- Safe-TTS 输出（历史目录名仍沿用 `safe_tta`）：`outputs/20260303/3class/physio_safe_tta_ttime_suite_3class_lrfeet_seed0_full_v2_alpha*.*/`
- 本次汇报的“最新方法”采用：`safe-tts-ttime-suite-physionetmi3-alpha0.10-v1`

说明：

- anchor 方法是 `eegnet_ea`
- 本文档只汇报结果，不展开失败案例与机制分析

---

## 最新方法主结果

| Method | Mean acc | Worst-subject acc | Mean Δ vs anchor | Neg-transfer rate | Accept rate |
|---|---:|---:|---:|---:|---:|
| `eegnet_ea` | 0.7065 | 0.4118 | +0.0000 | 0.0000 | 0.0000 |
| `safe-tts-ttime-suite-physionetmi3-alpha0.10-v1` | 0.7125 | 0.4478 | +0.0060 | 0.0367 | 0.2018 |

补充统计：

- accepted subjects: `22 / 109`
- fallback to anchor: `87 / 109`
- improved subjects: `17`
- unchanged subjects: `88`
- negative-transfer subjects: `4`

---

## 各个 TTA 方法结果表

下表统一按 `predictions_all_methods.csv` 计算 subject-wise mean / worst-subject accuracy；Safe-TTS 行来自 `20260303_method_comparison.csv`。

| rank | method | mean_accuracy | worst_accuracy | mean_delta_vs_anchor |
|---|---|---|---|---|
| 1 | safe-tts-ttime-suite-physionetmi3-alpha0.10-v1 | 0.7125 | 0.4478 | 0.0060 |
| 2 | sar | 0.7098 | 0.4030 | 0.0032 |
| 3 | pl | 0.7095 | 0.4030 | 0.0029 |
| 4 | shot | 0.7075 | 0.4118 | 0.0009 |
| 5 | eegnet_ea | 0.7065 | 0.4118 | 0.0000 |
| 6 | bn_adapt | 0.6834 | 0.4118 | -0.0231 |
| 7 | t3a | 0.6783 | 0.3768 | -0.0283 |
| 8 | ttime_ensemble | 0.6669 | 0.3529 | -0.0396 |
| 9 | delta | 0.6644 | 0.3382 | -0.0422 |
| 10 | ttime | 0.6634 | 0.3824 | -0.0431 |
| 11 | cotta | 0.6622 | 0.3382 | -0.0444 |
| 12 | tent | 0.6432 | 0.3382 | -0.0634 |

对应 CSV：

- `docs/experiments/figures/20260313_physionetmi3_latest_safe_tta_results/tta_method_summary.csv`
- `docs/experiments/figures/20260313_physionetmi3_latest_safe_tta_results/tta_and_safe_method_summary.csv`

---

## Alpha Sweep

| risk_alpha | mean_accuracy | worst_accuracy | mean_delta_vs_anchor | neg_transfer_vs_anchor | accept_rate | oracle_mean_accuracy | headroom_mean | oracle_gap_mean |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.7065 | 0.4118 | 0.0000 | 0.0000 | 0.0000 | 0.7397 | 0.0331 | 0.0331 |
| 0.0500 | 0.7060 | 0.4118 | -0.0005 | 0.0183 | 0.0183 | 0.7397 | 0.0331 | 0.0337 |
| 0.1000 | 0.7125 | 0.4478 | 0.0060 | 0.0367 | 0.2018 | 0.7397 | 0.0331 | 0.0272 |
| 0.2000 | 0.7124 | 0.4478 | 0.0058 | 0.0826 | 0.3028 | 0.7397 | 0.0331 | 0.0273 |
| 0.3000 | 0.7124 | 0.4478 | 0.0058 | 0.0826 | 0.3028 | 0.7397 | 0.0331 | 0.0273 |

对应 CSV：

- `docs/experiments/figures/20260313_physionetmi3_latest_safe_tta_results/alpha_sweep_summary.csv`

---

## 图表

![](figures/20260313_physionetmi3_latest_safe_tta_results/alpha_sweep_tradeoff.png)

*图 1. Risk alpha sweep。左图给出 mean accuracy 与 worst-subject accuracy；右图给出 neg-transfer rate 与 accept rate。虚线标出本次汇报采用的 `alpha=0.10`。*

![](figures/20260313_physionetmi3_latest_safe_tta_results/selection_method_counts.png)

*图 2. Safe-TTS 的选择结果分布。左图是全部被试的最终选择；右图只统计被接受切换的被试。*

![](figures/20260313_physionetmi3_latest_safe_tta_results/subject_delta_sorted.png)

*图 3. Subject-level delta distribution。横轴为按增益排序后的被试，纵轴为相对 anchor 的 accuracy 变化。*

![](figures/20260313_physionetmi3_latest_safe_tta_results/confusion_anchor_vs_safe.png)

*图 4. Anchor `eegnet_ea` 与 Safe-TTS `alpha=0.10` 的归一化混淆矩阵对比。*

---

## 类别 Recall

| method | class | recall | support |
|---|---|---|---|
| eegnet_ea | feet | 0.6892 | 2455 |
| eegnet_ea | left_hand | 0.7093 | 2480 |
| eegnet_ea | right_hand | 0.7199 | 2438 |
| safe_tta_alpha0.10 | feet | 0.7226 | 2455 |
| safe_tta_alpha0.10 | left_hand | 0.7004 | 2480 |
| safe_tta_alpha0.10 | right_hand | 0.7133 | 2438 |

对应 CSV：

- `docs/experiments/figures/20260313_physionetmi3_latest_safe_tta_results/class_recall_comparison.csv`

---

## 结果小结

- 论文中的方法名称记为 `Safe-TTS`。当前实验目录和脚本输出里仍沿用旧的 `safe-tta-*` 命名。
- `Safe-TTS` 的定位是安全测试时策略选择框架，而不是一种新的 TTA 方法；它是在 anchor 与多个 TTA 候选策略之间进行安全选择与回退。
- 在这组 3-class PhysioNetMI TTA suite 上，对应 `Safe-TTS` 的当前汇报配置 `alpha=0.10` 取得了最高的 mean accuracy：`0.7125`，高于 anchor `eegnet_ea` 的 `0.7065`。
- 对应 worst-subject accuracy 由 `0.4118` 提升到 `0.4478`。
- 在单个候选 TTA 方法中，`sar` 和 `pl` 的结果最接近 anchor，但都低于当前 `Safe-TTS` 的最终结果。
