# 20260303 — PhysioNetMI 3-class TTA suite — Safe-TTS safe selection

> 说明：这次实验运行时输出名仍使用历史前缀 `safe-tta-*`；论文与当前项目主线统一记为 `Safe-TTS`。

## Input (TTA suite predictions)

- Remote (lab): `~/workspace/TTA_demo/outputs/20260302/3class/ttime_suite_physionetmi_3class_lrfeet_seed0_full_v2/predictions_all_methods.csv`
- Local mirror: `outputs/20260302/3class/ttime_suite_physionetmi_3class_lrfeet_seed0_full_v2/predictions_all_methods.csv`
- Subjects: 109
- Methods: 11 (`eegnet_ea` + 10 TTA methods)

## Safe-TTS selection (ours)

### CRC-style risk-calibrated selector (recommended)

Command (remote):

```bash
cd ~/workspace/TTA_demo
./.venv/bin/python -u scripts/offline_safe_tta_multi_select_crc_from_predictions.py \
  --preds outputs/20260302/3class/ttime_suite_physionetmi_3class_lrfeet_seed0_full_v2/predictions_all_methods.csv \
  --anchor-method eegnet_ea \
  --candidate-methods ALL \
  --risk-alpha 0.20 --delta 0.05 --n-splits 5 \
  --method-name safe-tta-ttime-suite-3class-alpha0.20-v1 \
  --date-prefix 20260303 \
  --no-diagnostics \
  --out-dir outputs/20260303/3class/physio_safe_tta_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.20_v1
```

Outputs:

- Remote: `outputs/20260303/3class/physio_safe_tta_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.20_v1/`
- Local mirror: `outputs/20260303/3class/physio_safe_tta_ttime_suite_3class_lrfeet_seed0_full_v2_alpha0.20_v1/`

Summary (vs anchor `eegnet_ea`):

- mean acc: `0.712366` (Δ `+0.005816` pp)
- worst-subject acc: `0.447761` (Δ `+0.035996` pp)
- neg-transfer rate: `0.08257`
- accept rate: `0.30275`

### Risk-α sweep (CRC selector)

All runs use the same setup as above, varying only `--risk-alpha` ∈ {0.00, 0.05, 0.10, 0.20, 0.30}.

| risk_alpha | mean_acc | worst_acc | meanΔ vs anchor (pp) | negT rate | accept rate |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.706550 | 0.411765 | +0.0000 | 0.0000 | 0.0000 |
| 0.05 | 0.706002 | 0.411765 | -0.0548 | 0.0183 | 0.0183 |
| 0.10 | 0.712507 | 0.447761 | +0.5957 | 0.0367 | 0.2018 |
| 0.20 | 0.712366 | 0.447761 | +0.5816 | 0.0826 | 0.3028 |
| 0.30 | 0.712366 | 0.447761 | +0.5816 | 0.0826 | 0.3028 |

Notes:
- α≥0.20 saturates on this run (selected threshold / results identical).
- α=0.10 looks like the best trade-off among these points (higher mean, lower negT than α=0.20).

### Fast (non-CRC) selector (debug)

Command (local):

```bash
./.venv/bin/python -u scripts/offline_safe_tta_multi_select_from_predictions.py \
  --preds outputs/20260302/3class/ttime_suite_physionetmi_3class_lrfeet_seed0_full_v2/predictions_all_methods.csv \
  --anchor-method eegnet_ea \
  --candidate-methods ALL \
  --method-name safe-tta-ttime-suite-3class-v1 \
  --date-prefix 20260303 \
  --no-diagnostics \
  --out-dir outputs/20260303/3class/physio_safe_tta_ttime_suite_3class_lrfeet_seed0_full_v2_v1
```

Summary (vs anchor `eegnet_ea`):

- mean acc: `0.707887` (Δ `+0.001337` pp)
- worst-subject acc: `0.411765` (Δ `+0.000000` pp)
- neg-transfer rate: `0.01835`
- accept rate: `0.07339`
