# 当前主线：Safe-TTS + PhysioNetMI 3-class

这份目录是**当前论文与当前仓库主线**的唯一入口。

## 当前范围

- 方法名：`Safe-TTS`
- 数据集：`PhysioNetMI`
- 任务：三分类 `left_hand / right_hand / feet`
- 协议：strict LOSO
- 候选集：公开源码且已核验的 `tent / t3a / cotta / shot / coral`
- 当前规范配置：`D3`

## 先看这三份

- 方法说明：`docs/current/safe_tts_methods.md:1`
- 实验设置与结果：`docs/current/safe_tts_experiments.md:1`
- 论文/实现对齐清单：`docs/current/paper_alignment.md:1`

## 当前实验记录

- `20260303` 选择实验：`docs/experiments/20260303_physionetmi3_tta_suite_safe_select.md:1`
- `20260313` 最新结果汇总：`docs/experiments/20260313_physionetmi3_latest_safe_tta_results.md:1`

## 当前脚本入口

- `scripts/safe_tts/run_physionetmi3_tta_suite.py`
- `scripts/safe_tts/run_physionetmi3_safe_tts.py`

其中 `run_physionetmi3_safe_tts.py` 的默认参数已经整理到 `D3`：

- `paper_oof_dev_cal`
- `evidential`
- `stats,decision,relative`
- `hidden_dim=32`
- `epochs=50`
- `outcome_delta=0.02`
- `guard_gray_margin=0.02`
- `risk_alpha=0.40`

## 历史材料

旧的 `SAFE-TTA / 4-class / HGD / BNCI / cand2e` 资产已归档到：

- `docs/legacy/README.md:1`
