# Scripts

The scripts consume saved prediction CSV files and write selection summaries.

- `offline_safe_tta_multi_select_crc_from_predictions.py`: multi-candidate selector with calibrated risk control.
- `offline_safe_tta_multi_select_from_predictions.py`: compact multi-candidate selector without the CRC review layer.
- `offline_safe_tta_select_from_predictions.py`: single-candidate selector.
- `safe_tts/run_warmup_safe_tts_from_predictions.py`: prefix-based selector for stream-style evaluation.
