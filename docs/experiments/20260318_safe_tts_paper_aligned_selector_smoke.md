# 2026-03-18 — Safe-TTS paper-aligned selector smoke（PhysioNetMI 3-class）

## 本轮代码改动

目标：把当前 `Safe-TTS` 代码对齐到论文口径，主要改两处：

1. **Guard 灰区**
   - 加入对称灰区 `guard_gray_margin=0.02`
   - 当 `|Δacc| <= 0.02` 时，该 pseudo-target 样本不参与 guard 训练

2. **风险控制阶段**
   - 加入 `paper_oof_dev_cal` 协议
   - 先 cross-fit 产生 OOF 分数
   - 再拆成 `D_dev / D_cal`
   - 在 `D_dev` 选阈值
   - 在 `D_cal` 用 Clopper–Pearson 做独立验证
   - 若验证失败，则直接 fallback 到 anchor

## Smoke 命令

```bash
./.venv/bin/python scripts/safe_tts/run_physionetmi3_safe_tts.py \
  --risk-alpha 0.10 \
  --date-prefix 20260318 \
  --out-dir outputs/20260318/3class/physio_safe_tts_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.10_v3 \
  --dump-calib-grid
```

## 输入

- `outputs/20260302/3class/ttime_suite_physionetmi_3class_lrfeet_seed0_full_v2/predictions_all_methods.csv`

## 输出

- `outputs/20260318/3class/physio_safe_tts_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.10_v3/20260318_method_comparison.csv`
- `outputs/20260318/3class/physio_safe_tts_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.10_v3/20260318_per_subject_selection.csv`
- `outputs/20260318/3class/physio_safe_tts_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.10_v3/20260318_calib_grid.csv`

## 结果

- mean acc：`0.70655`
- worst acc：`0.41176`
- mean Δ vs anchor：`0.00000`
- neg-transfer：`0.00000`
- accept rate：`0.00000`

也就是：**全部 109 个 subject 都 fallback 到 `eegnet_ea`**。

## 原因诊断

从 `20260318_calib_grid.csv` 看：

- `D_dev` 会选出有限阈值（常见是 `0.001 ~ 0.010`）
- 但这些阈值在 `D_cal` 上的 `marg_ucb` 普遍明显高于 `alpha=0.10`
- 因此 109 个 target subject 的 `cal_verify` 全部失败，触发统一 fallback

一个典型现象是：

- `D_cal` 规模约为 `27` subjects（`calib_fraction=0.25`）
- 即使 0 个负迁移，one-sided CP 上界也在 `0.10` 附近
- 而当前 `D_dev` 选中的阈值在 `D_cal` 上往往还伴随非零负迁移，因此更难通过

## 当前结论

代码已经和论文口径对齐，但：

- **当前 paper-aligned 实现比旧版 selector 明显更保守**
- 在 `alpha=0.10` 下，旧结果（mean `0.7125`）**不会自动保留**

## 下一步建议

如果要让 paper-aligned 实现既符合论文，又不至于全 fallback，优先检查：

1. `calib_fraction` 是否需要增大（让 `D_cal` 更大）
2. `risk_alpha` 是否需要重新扫
3. `D_dev` 的阈值选择是否要从“纯最大化 mean gain”改成“先排序，再找第一个通过 `D_cal` 的阈值”
