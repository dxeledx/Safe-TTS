# Scripts Layout

## 当前主入口

当前项目默认只看：

- `scripts/safe_tts/run_physionetmi3_tta_suite.py`
- `scripts/safe_tts/run_physionetmi3_safe_tts.py`
- `scripts/safe_tts/run_physionetmi3_safe_tts_stable.py`

它们分别对应：

1. 生成 PhysioNetMI 3-class TTA suite 预测
2. `D3` 默认 Safe-TTS 选择器
3. `Stable` 更稳的中间强度 Safe-TTS 选择器

## 底层实现

- `scripts/ttime_suite/run_suite_loso.py`：TTA suite 真正执行器
- `scripts/offline_safe_tta_multi_select_crc_from_predictions.py`：CRC 风险控制选择器
- `scripts/offline_safe_tta_multi_select_from_predictions.py`：非 CRC 的快速版本

## 其他脚本

其余大量脚本大多是旧研究阶段留下的实验/作图/补充材料工具，不再是当前主线默认入口。
