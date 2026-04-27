# Experiment Notes (lab notebook)

## 当前主线

当前论文与仓库主线统一为：

- `Safe-TTS`
- `PhysioNetMI` 三分类
- TTA suite + 安全选择

优先看这两个实验记录：

- `docs/experiments/20260303_physionetmi3_tta_suite_safe_select.md:1`
- `docs/experiments/20260313_physionetmi3_latest_safe_tta_results.md:1`

旧的 4-class / HGD / BNCI / SAFE-TTA 叙事仍保留在本目录中，但都属于历史研究轨迹，不再作为当前论文主线。

This folder contains **date-stamped experiment notes** that document:
- the exact command/config used,
- the output directory under `outputs/`,
- and a short observation/diagnosis.

File naming convention: `YYYYMMDD_<topic>.md`

## Results registry

`results_registry.csv` is a **registered table** of past runs, built by scanning `outputs/**/**/**/**_method_comparison.csv`
and (when available) reading metadata from each `*_results.txt`.

Registry fields include `dataset` (parsed from the `Dataset:` line in `*_results.txt`) so multi-dataset benchmarking stays comparable.

To refresh it (without rerunning experiments):

```bash
python3 scripts/update_results_registry.py --outputs-dir outputs --out docs/experiments/results_registry.csv
```

## Index

- `20251222_oea_zo_ablation_summary.md`: 2-class OEA/OEA-ZO ablation notes (q_blend, objectives).
- `20251222_oea_zo_infomax_lambda_sweep.md`: Sweep of `--oea-zo-infomax-lambda` values.
- `20251222_oea_zo_stability_probe_qblend05.md`: Negative-transfer probe with more aggressive `q_blend`.
- `20251223_oea_zo_4class_ablation_summary.md`: 4-class baseline vs OEA/OEA-ZO ablations.
- `20251223_stepA_4class_sweep.md`: Step-A style sweeps for 4-class EA-ZO settings.
- `20251223_stepA_stepB_attempt.md`: Step-A/Step-B attempt notes (4-class).
- `20251223_ea_zo_hard_marginal_balance.md`: 4-class marginal-balance penalty experiments.
- `20251223_S4_certificate_diagnostics.md`: Diagnostics for the “unlabeled certificate failure” case (S4).
- `20251223_safe_certificate_drift_guard.md`: Drift-guard + calibrated selector (Ridge) attempts for safer EA-ZO.
- `20251223_bilevel_reliable_infomax.md`: Draft bilevel / reliable-weighted InfoMax idea.
- `20251223_bilevel_imr_guard_4class.md`: Continuous bilevel (w,q) + calibrated guard implementation notes and 4-class runs.
- `20251223_2c_4c_safe_runs.md`: Summary of the latest 2-class and 4-class “safe” runs after adding predictions CSV output.
- `20251224_within_subject_cross_session.md`: Within-subject cross-session (0train→1test) comparison: EA vs EA/OEA-ZO.
- `20251225_cross_session_certificate_diagall.md`: Cross-session (0train→1test) diagnostics: candidate certificates vs accuracy.
- `20251226_loso_4class_routeB_ea_si.md`: Route B (EA-SI / EA-SI-ZO) notes for LOSO 4-class.
- `20251226_loso_4class_ea_si_chan.md`: EA-SI-CHAN (channel projector before CSP) LOSO 4-class notes.
- `20251227_loso_4class_ea_si_chan_safe.md`: EA-SI-CHAN wrapped by binary calibrated guard + fallback (LOSO 4-class).
- `20251227_loso_4class_ea_si_chan_multi_safe.md`: EA-SI-CHAN multi-candidate calibrated selection + fallback (LOSO 4-class).
- `20251228_loso_4class_ea_stack_multi_safe.md`: Stacked multi-family candidates (EA/RPA/TSA/EA-SI-CHAN) under calibrated ridge+guard + fallback (LOSO 4-class).
- `20251229_loso_ea_si_chan_multi_safe_sweep.md`: Route B sweep (candidate λ grid / rank / selector params) + 2-class sanity check.
