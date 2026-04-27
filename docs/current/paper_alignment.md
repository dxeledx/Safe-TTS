# 论文与实现对齐清单（当前主线）

## 应该保留

- 方法名：`Safe-TTS`
- 数据集：`PhysioNetMI`
- 任务：三分类
- 候选池：公开源码且已核验的 `tent/t3a/cotta/shot/coral`
- 当前规范配置：`D3`
- 当前推荐主结果口径：`alpha=0.40`

## 已对齐的实现点

### 1. Guard 灰区

当前主线代码已实现对称灰区，默认 `guard_gray_margin=0.02`。

### 2. 风险控制阶段

当前 `Safe-TTS` wrapper 默认已切到 `paper_oof_dev_cal`：

- 先 cross-fit 产生 OOF 分数
- 再分 `D_dev / D_cal`
- 在 `D_dev` 选阈值
- 在 `D_cal` 做 CP 验证

### 3. 选择器语义

当前主线默认已升级到 multi-view evidential selector：

- 输入由 `stats + decision + relative` 三个视图组成
- 输出改为 `harmful / neutral / helpful` 的 deployment posterior
- risk 与 utility 从同一个 posterior 导出，而不是两个独立线性头
- 默认 `hidden_dim=32`
- 默认 `epochs=50`
- 默认 `outcome_delta=0.02`

## 论文里仍需要删掉或改掉的

### 1. 删掉 OpenBMI 主实验

当前仓库已经补上 `OpenBMI / Lee2019_MI` 的 LOSO 入口，但还没有形成一条可作为主表的完整结果链，所以论文主实验仍不应写它。

### 2. 不要再写 PhysioNetMI 2-class 或 4-class 作为当前主实验

当前主实验是三分类，不是旧的 left/right 二分类，也不是 earlier 4-class narrative。

## 可以在论文里保留为 future work 的

- OpenBMI 扩展实验（当前已具备 runner，待完整结果）

## 当前写论文时的建议口径

论文正文先只写**已经和代码、结果对齐**的内容。想写的方法改动，先在仓库里落地，再进入论文正文。
