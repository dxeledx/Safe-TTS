# Safe-TTS 方法说明（当前实现口径）

## 1. 方法定位

`Safe-TTS` 不是一个新的单一 TTA 算法，而是一个**安全测试时策略选择框架**：

- anchor：`eegnet_ea`
- candidates：TTA suite 中的多个测试时自适应方法
- output：对每个 target subject，要么切换到某个候选方法，要么回退到 anchor

当前主线对应的候选集来自：

- `tent`
- `t3a`
- `cotta`
- `shot`
- `coral`

当前规范配置记作 `D3`。

代码入口：`scripts/ttime_suite/run_suite_loso.py:49`

## 2. 当前实现流程

### 2.1 先生成严格 LOSO 的候选预测

对每个 target subject：

- 基座 `EEGNet + EA` 训练一次
- 6 个公开源码方法都从同一个 checkpoint 出发
- 输出合并后的 `predictions_all_methods.csv`

### 2.2 再做离线安全选择

当前使用的选择器是：

- `scripts/offline_safe_tta_multi_select_crc_from_predictions.py:212`

核心结构：

- 选择器输入由 3 个视图组成：`stats + decision + relative`
- 使用一个 3-state evidential selector，直接建模 `harmful / neutral / helpful`
- 从同一个 deployment posterior 导出
  - risk：`r = b_H + rho * u`
  - utility：`q = (b_B - b_H) - eta * u`
- 外层仍保留 `OOF + D_dev/D_cal + CP` 的独立风险校准
- 若没有候选同时满足风险门控与 utility 阈值，就回退到 anchor

`D3` 的默认超参数为：

- `risk_alpha = 0.40`
- `delta = 0.05`
- `n_splits = 5`
- `calib_fraction = 0.25`
- `guard_gray_margin = 0.02`
- `selector_hidden_dim = 32`
- `selector_epochs = 50`
- `selector_outcome_delta = 0.02`

## 3. 当前已对齐到论文的关键点

### 3.1 Guard 灰区已加入

当前实现已按论文口径加入对称灰区，默认：

- `guard_gray_margin = 0.02`
- 当 `|Δacc| <= 0.02` 时，该伪目标样本不参与 guard 训练
- 当 `Δacc > 0.02` 时记为正样本
- 当 `Δacc < -0.02` 时记为负样本

对应代码：

- `scripts/offline_safe_tta_multi_select_from_predictions.py:296`
- `scripts/offline_safe_tta_multi_select_crc_from_predictions.py:1127`

### 3.2 选择器已升级到 multi-view evidential selector

当前主线不再把 `certificate` 和 `guard` 视为两个割裂的线性头，而是默认切到：

- `stats`：当前 hand-crafted subject statistics
- `decision`：trial-level probability / entropy / margin 的聚合视图
- `relative`：相对 anchor 的概率差 / JS-KL 漂移 / flip-conflict 视图
- 两层 MLP 产生三态 evidence
- 用三态 outcome loss + pairwise ranking loss 训练
- 从共享 posterior 同时导出 risk / utility

### 3.3 风险控制已切到 OOF + dev/cal

当前 `Safe-TTS` wrapper 默认使用：

- 先 cross-fit 出 OOF 分数
- 再单独切 `D_dev / D_cal`
- 在 `D_dev` 选阈值
- 在 `D_cal` 用 CP 做独立验证

低层脚本仍保留旧的 `legacy_pooled` 协议用于历史复现，但当前主线 wrapper 默认已切到 `paper_oof_dev_cal`。

对应代码：

- `scripts/offline_safe_tta_multi_select_crc_from_predictions.py:1658`
- `scripts/offline_safe_tta_multi_select_crc_from_predictions.py:1728`
- `scripts/offline_safe_tta_multi_select_crc_from_predictions.py:1758`

## 4. 当前论文写法建议

当前项目如果按最新实验写论文，方法章应该以**当前实现**为准：

- 名字写 `Safe-TTS`
- 数据集写 `PhysioNetMI 3-class`
- 候选池默认收缩到公开源码且已核验的 5 个 TTA：`tent/t3a/cotta/shot/coral`
- 明确说明当前 guard 使用 **`±0.02` 的灰区**
- 明确说明当前 selector 已升级为 **multi-view evidential deployment selector**
- 明确说明当前 CRC 版本使用 **OOF + `D_dev/D_cal` 二阶段校准**
- 若写“当前规范实现”，优先指向 `D3`
