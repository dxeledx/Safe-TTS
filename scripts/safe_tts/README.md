# Safe-TTS canonical scripts

## 当前推荐入口

### 1) 生成 PhysioNetMI 3-class TTA suite 预测

```bash
python scripts/safe_tts/run_physionetmi3_tta_suite.py \
  --data-dir /path/to/deeptransfer_export
```

### 2) 运行 PhysioNetMI 3-class Safe-TTS 安全选择

```bash
python scripts/safe_tts/run_physionetmi3_safe_tts.py \
  --preds outputs/20260330/3class/ttime_suite_physionetmi3_publicsrc_seed0_v1/predictions_all_methods.csv \
  --risk-alpha 0.40
```

说明：

- wrapper 不再自动回退到旧的 `20260302` 预测文件
- 若当前 canonical `predictions_all_methods.csv` 不存在，请先跑 TTA suite，或显式传入 `--preds`

当前默认配置已经整理到 `D3`：

- `calibration_protocol=paper_oof_dev_cal`
- `selector_model=evidential`
- `selector_views=stats,decision,relative`
- `selector_hidden_dim=32`
- `selector_epochs=50`
- `selector_outcome_delta=0.02`
- `guard_gray_margin=0.02`
- `risk_alpha=0.40`

## 扩展入口

### 3) 生成 OpenBMI / Lee2019_MI 2-class TTA suite 预测

先导出 DeepTransferEEG 格式：

```bash
python scripts/ttime_suite/export_moabb_for_deeptransfer.py \
  --dataset openbmi \
  --events left_hand,right_hand \
  --out-dir /path/to/openbmi_export
```

再跑 LOSO TTA suite：

```bash
python scripts/safe_tts/run_openbmi2_tta_suite.py \
  --data-dir /path/to/openbmi_export
```

## 设计原则

- 只服务当前主线：`PhysioNetMI 3-class`
- 默认名字统一成 `Safe-TTS`
- Safe-TTS 当前规范主配置统一记作 `D3`
- 默认候选池收缩到公开源码且已核验的 `tent/t3a/cotta/shot/coral`
- OpenBMI 只作为扩展数据集，不改变当前论文主线
- 底层仍复用原始脚本，避免重写核心实现
