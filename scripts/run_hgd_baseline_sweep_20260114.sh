#!/usr/bin/env bash
set -euo pipefail

DATE="20260114"
OUT_BASE="outputs/${DATE}/4class"
mkdir -p "${OUT_BASE}"

COMMON_ARGS=(
  --dataset Schirrmeister2017
  --events left_hand,right_hand,feet,rest
  --sessions 0train
  --preprocess moabb
  --resample 50
  --n-components 4
  --no-plots
)

run_one () {
  local method="$1"
  local run_name="$2"
  local log_path="${OUT_BASE}/${run_name}.nohup.log"

  echo "=== ${run_name} (${method}) ===" | tee -a "${log_path}"
  conda run -n eeg python -u run_csp_lda_loso.py \
    "${COMMON_ARGS[@]}" \
    --methods "${method}" \
    --run-name "${run_name}" 2>&1 | tee -a "${log_path}"
}

# Anchor + candidate families (CSP-family first, then Riemannian / tangent-space).
run_one "ea-csp-lda" "loso4_schirr2017_0train_rs50_ea_only_v3"
run_one "rpa-csp-lda" "loso4_schirr2017_0train_rs50_rpa_csp_lda_v1"
run_one "tsa-csp-lda" "loso4_schirr2017_0train_rs50_tsa_csp_lda_v1"

# FBCSP (now auto-drops invalid high bands under low sfreq).
run_one "fbcsp-lda" "loso4_schirr2017_0train_rs50_fbcsp_lda_v1"
run_one "ea-fbcsp-lda" "loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1"

# Riemannian / TS baselines (may be slow on HGD; keep for headroom check).
run_one "riemann-mdm" "loso4_schirr2017_0train_rs50_riemann_mdm_v1"
run_one "fgmdm" "loso4_schirr2017_0train_rs50_fgmdm_v1"
run_one "ts-svc" "loso4_schirr2017_0train_rs50_ts_svc_v1"
