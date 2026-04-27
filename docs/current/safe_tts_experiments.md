# Safe-TTS 实验设置与当前结果

## 1. 当前唯一主实验

- 数据集：`PhysioNetMI`
- 任务：三分类 `left_hand / right_hand / feet`
- 被试数：`109`
- 协议：strict LOSO
- anchor：`eegnet_ea`
- 候选池：公开源码且已核验的 5 个 TTA：`tent / t3a / cotta / shot / coral`
- 当前规范配置：`D3`

当前不写：

- `OpenBMI`
- `HGD`
- `BNCI2014_001`
- `PhysioNetMI 4-class`

这些都属于历史线或补充线，不再是当前论文主实验。

## 2. TTA suite 输入

当前主输入文件：

- `outputs/20260330/3class/ttime_suite_physionetmi3_publicsrc_seed0_v1/predictions_all_methods.csv:1`

对应实验说明：

- `docs/experiments/20260303_physionetmi3_tta_suite_safe_select.md:1`

## 3. 历史结果与当前规范结果

旧版本历史结果（仅作背景，不再作为当前规范默认点）：

- anchor `eegnet_ea`：mean acc `0.7065`
- `Safe-TTS(alpha=0.10)`：mean acc `0.7125`
- worst-subject acc：`0.4118 -> 0.4478`
- neg-transfer：`0.0367`
- accept rate：`0.2018`

结果汇总见：

- `docs/experiments/20260313_physionetmi3_latest_safe_tta_results.md:1`

## 3.1 Paper-aligned selector smoke（2026-03-18）

当前主线代码已经切到：

- `guard_gray_margin=0.02`
- `calibration_protocol=paper_oof_dev_cal`

对应 smoke 结果见：

- `docs/experiments/20260318_safe_tts_paper_aligned_selector_smoke.md:1`

该 smoke 表明：

- 在 `risk_alpha=0.10`、`calib_fraction=0.25` 下
- 新的 paper-aligned 风险控制会变得非常保守
- 当前 smoke 结果退化为全体 fallback 到 `eegnet_ea`

## 3.2 当前规范配置（D3）

当前代码默认已经整理到 `D3`：

- `risk_alpha=0.40`
- `paper_oof_dev_cal`
- `evidential`
- `stats,decision,relative`
- `selector_hidden_dim=32`
- `selector_epochs=50`
- `selector_outcome_delta=0.02`
- `guard_gray_margin=0.02`

在我们最新一次模拟全被试对照中，`D3` 是三组里唯一有效的一版：

- `D1`: legacy + paper_oof_dev_cal + `alpha=0.10` -> 全 fallback
- `D2`: legacy + legacy_pooled + `alpha=0.10` -> 全 fallback
- `D3`: evidential + paper_oof_dev_cal + `alpha=0.40` -> 可稳定接受并取得正收益

## 3.3 当前更稳的中间配置（Stable）

在最新一轮模拟对照中，我们还验证了一版更稳的中间配置 `Stable`：

- `risk_alpha=0.35`
- `guard_threshold=0.55`
- `selector_views=stats,decision,relative,dynamic,stochastic`
- `threshold_score=pred_improve_x_guard`
- `candidate_choice_score=pred_improve_x_guard`
- `min_pred_grid=0,0.01,0.02,0.05,0.10,0.15,0.20`
- 候选池收缩为：`t3a / shot / sar / bn_adapt`

这版配置在最新模拟全被试汇总中：

- `mean accuracy = 0.7275`
- 相对 anchor：`+2.09 pp`
- `worst-subject accuracy = 0.4412`
- `accept rate = 1.0`
- `neg-transfer rate = 0.0`

对应入口脚本：

- `scripts/safe_tts/run_physionetmi3_safe_tts_stable.py:1`

## 4. 当前推荐命令

### 4.1 跑 TTA suite

```bash
python scripts/safe_tts/run_physionetmi3_tta_suite.py \
  --data-dir /path/to/deeptransfer_export \
  --out-dir outputs/20260330/3class/ttime_suite_physionetmi3_publicsrc_seed0_v1
```

### 4.2 跑 Safe-TTS

```bash
python scripts/safe_tts/run_physionetmi3_safe_tts.py \
  --preds outputs/20260330/3class/ttime_suite_physionetmi3_publicsrc_seed0_v1/predictions_all_methods.csv \
  --risk-alpha 0.40
```

默认还会带上：

- `guard_gray_margin=0.02`
- `calibration_protocol=paper_oof_dev_cal`
- `calib_fraction=0.25`
- `selector_model=evidential`
- `selector_views=stats,decision,relative`
- `selector_hidden_dim=32`
- `selector_epochs=50`
- `selector_outcome_delta=0.02`

### 4.3 跑 Stable

```bash
python scripts/safe_tts/run_physionetmi3_safe_tts_stable.py \
  --preds outputs/20260330/3class/ttime_suite_physionetmi3_publicsrc_seed0_v1/predictions_all_methods.csv
```

默认会带上：

- `candidate_methods=t3a,shot,sar,bn_adapt`
- `risk_alpha=0.35`
- `guard_threshold=0.55`
- `selector_views=stats,decision,relative,dynamic,stochastic`
- `threshold_score=pred_improve_x_guard`
- `candidate_choice_score=pred_improve_x_guard`
- `min_pred_grid=0,0.01,0.02,0.05,0.10,0.15,0.20`

## 5. 当前实验叙事

当前论文的实验叙事应写成：

1. 先构建 `PhysioNetMI 3-class` 的 strict LOSO TTA suite
2. 再在不使用 target label 的前提下做 `Safe-TTS` 安全选择
3. 比较 `Safe-TTS`、anchor、单个 TTA 方法
4. 主指标至少汇报：
   - mean accuracy
   - worst-subject accuracy
   - neg-transfer rate
   - accept rate
