#!/usr/bin/env bash
set -euo pipefail

THREADS="${THREADS:-32}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

export MNE_DONTWRITE_HOME=true
export MNE_CONFIG_DIR="$PWD/.mne_config"
export MNE_DATA="$PWD/.mne_data"
export JOBLIB_TEMP_FOLDER="$PWD/.joblib"

export PYTHONPATH="$PWD/.pydeps"

DATE=20260128
METHOD_NAME="safe-tta-offline-2arm-crc-cp"
MIN_PRED_GRID="0,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.01,0.02,0.03,0.05"

find_run_dir_with_preds() {
  local run_name="$1"
  local best=""
  local best_mtime=0
  local d f mtime

  shopt -s nullglob
  for d in outputs/*/4class/"${run_name}"*; do
    if [ ! -d "$d" ]; then
      continue
    fi
    f="$(ls "$d"/*_predictions_all_methods.csv 2>/dev/null | head -n 1 || true)"
    if [ -z "$f" ]; then
      continue
    fi
    mtime="$(stat -c %Y "$f" 2>/dev/null || echo 0)"
    if [ "$mtime" -gt "$best_mtime" ]; then
      best="$d"
      best_mtime="$mtime"
    fi
  done
  shopt -u nullglob

  if [ -n "$best" ]; then
    echo "$best"
    return 0
  fi
  return 1
}

run_bnci_baseline_if_missing() {
  local run_name="$1"
  shift
  local methods="$1"
  shift
  local extra_args=("$@")

  local found
  found="$(find_run_dir_with_preds "${run_name}" || true)"
  if [ -n "${found}" ]; then
    echo "[$(date)] [BASELINES] SKIP  ${run_name} (found ${found})" >&2
    echo "${found}"  # stdout: used by callers via command substitution
    return 0
  fi

  echo "[$(date)] [BASELINES] START ${run_name}" >&2
  python3 -u run_csp_lda_loso.py \
    --dataset BNCI2014_001 \
    --preprocess paper_fir \
    --events left_hand,right_hand,feet,tongue \
    --sessions 0train \
    --tmin 0.5 --tmax 3.5 \
    --resample 250 \
    --n-components 6 \
    --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
    --methods "${methods}" \
    --no-plots \
    --run-name "${run_name}" \
    "${extra_args[@]}" 1>&2

  found="$(find_run_dir_with_preds "${run_name}" || true)"
  if [ -z "${found}" ]; then
    echo "[$(date)] [BASELINES] ERROR ${run_name} finished but no *_predictions_all_methods.csv found under outputs/*/4class/${run_name}*" >&2
    exit 1
  fi
  echo "[$(date)] [BASELINES] DONE  ${run_name} -> ${found}" >&2
  echo "${found}"  # stdout: used by callers via command substitution
}

echo "[$(date)] === BNCI2014_001 (BCI-IV 2a) — 4 two-arm experiments (EA anchor) ==="

# (1) EA vs ea-si-chan-multi-safe-csp-lda
BASE_SICHAN_MSAFE="loso4_bnci2014_001_0train_paperfir_nc6_ea_sichan_multisafe_lam3_v1"
RUN_DIR_SICHAN_MSAFE="$(run_bnci_baseline_if_missing \
  "${BASE_SICHAN_MSAFE}" \
  "ea-csp-lda,ea-si-chan-multi-safe-csp-lda" \
  --si-chan-ranks 21 \
  --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 \
  --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
)"

# (2) EA vs ea-stack-multi-safe-csp-lda (reuse prior strong config; rerun only if missing)
BASE_STACK_MSAFE="loso4_bnci2014_001_0train_paperfir_nc6_ea_stackmss_v1"
RUN_DIR_STACK_MSAFE="$(run_bnci_baseline_if_missing \
  "${BASE_STACK_MSAFE}" \
  "ea-csp-lda,ea-stack-multi-safe-csp-lda" \
  --si-chan-ranks 21 \
  --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
  --oea-zo-selector calibrated_stack_ridge_guard_borda \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-anchor-guard-delta 0.05 \
  --stack-safe-anchor-probe-hard-worsen -0.01 \
  --stack-safe-fbcsp-guard-threshold 0.95 \
  --stack-safe-fbcsp-min-pred-improve 0.05 \
  --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 \
  --stack-safe-tsa-min-pred-improve 0.05 \
  --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family \
  --stack-calib-per-family-mode blend \
  --stack-calib-per-family-shrinkage 20 \
)"

# (3) EA vs ea-si-chan-csp-lda
BASE_SICHAN="loso4_bnci2014_001_0train_paperfir_nc6_ea_sichan_v1"
RUN_DIR_SICHAN="$(run_bnci_baseline_if_missing \
  "${BASE_SICHAN}" \
  "ea-csp-lda,ea-si-chan-csp-lda" \
)"

# (4) EA vs ea-fbcsp-lda
BASE_FBCSP="loso4_bnci2014_001_0train_paperfir_nc6_ea_fbcsp_v1"
RUN_DIR_FBCSP="$(run_bnci_baseline_if_missing \
  "${BASE_FBCSP}" \
  "ea-csp-lda,ea-fbcsp-lda" \
)"

FIGDIR="docs/experiments/figures/${DATE}_bnci2014_001_crc_cp_twoarm4_v1"
mkdir -p "${FIGDIR}"

run_twoarm_crc_sweep() {
  local tag="$1"
  local pred_all="$2"
  local cand_method="$3"
  local cand_family="$4"

  if [ -z "${pred_all}" ] || [ ! -f "${pred_all}" ]; then
    echo "[$(date)] ERROR: missing preds file for ${tag}: ${pred_all}" >&2
    exit 1
  fi

  echo "[$(date)] === CP-CRC two-arm sweep: ${tag} (cand=${cand_method}) ===" >&2

  for A in 0.35 0.40 0.50; do
    local crc_out="outputs/${DATE}/4class/offline_bnci2014_001_crc_cp_twoarm_${tag}_alpha${A}_delta0.05_splits5_v1"
    echo "[$(date)] [CRC-CP] ${tag} alpha=${A} -> ${crc_out}" >&2
    python3 scripts/offline_safe_tta_multi_select_crc_from_predictions.py \
      --preds "${pred_all}" \
      --anchor-method ea-csp-lda \
      --candidate-methods "${cand_method}" \
      --candidate-family-map "${cand_method}=${cand_family}" \
      --guard-threshold 0.5 \
      --anchor-guard-delta 0.05 \
      --risk-alpha "${A}" \
      --delta 0.05 \
      --n-splits 5 \
      --min-pred-grid "${MIN_PRED_GRID}" \
      --out-dir "${crc_out}" \
      --method-name "${METHOD_NAME}" \
      --date-prefix "${DATE}"

    local merged="outputs/${DATE}/4class/loso4_bnci2014_001_merged_plus_crc_cp_twoarm_${tag}_alpha${A}_delta0.05_splits5_v1"
    python3 scripts/merge_loso_runs.py \
      --run-dirs "$(dirname "${pred_all}")" "${crc_out}" \
      --out-run-dir "${merged}" \
      --prefer-date-prefix "${DATE}"

    local tagA
    tagA="$(echo "${A}" | sed 's/\\./p/g')"

    python3 scripts/make_main_table_and_stats.py \
      --run-dir "${merged}" \
      --baseline-method ea-csp-lda \
      --out-csv "${FIGDIR}/main_table_${tag}_alpha${tagA}.csv" \
      --out-per-subject-csv "${FIGDIR}/per_subject_${tag}_alpha${tagA}.csv" \
      --out-md "${FIGDIR}/main_table_${tag}_alpha${tagA}.md"

    python3 scripts/plot_candidate_diagnostics.py \
      --run-dir "${crc_out}" \
      --method "${METHOD_NAME}" \
      --out-dir "${FIGDIR}" \
      --prefix "bnci_twoarm_${tag}_alpha${tagA}"
  done

  python3 scripts/plot_crc_risk_coverage.py \
    --run-dirs \
      "outputs/${DATE}/4class/offline_bnci2014_001_crc_cp_twoarm_${tag}_alpha0.35_delta0.05_splits5_v1" \
      "outputs/${DATE}/4class/offline_bnci2014_001_crc_cp_twoarm_${tag}_alpha0.40_delta0.05_splits5_v1" \
      "outputs/${DATE}/4class/offline_bnci2014_001_crc_cp_twoarm_${tag}_alpha0.50_delta0.05_splits5_v1" \
    --method "${METHOD_NAME}" \
    --out-dir "${FIGDIR}" \
    --prefix "bnci_twoarm_${tag}_v1"
}

PRED_SICHAN_MSAFE="$(ls "${RUN_DIR_SICHAN_MSAFE}"/*_predictions_all_methods.csv 2>/dev/null | head -n 1 || true)"
PRED_STACK_MSAFE="$(ls "${RUN_DIR_STACK_MSAFE}"/*_predictions_all_methods.csv 2>/dev/null | head -n 1 || true)"
PRED_SICHAN="$(ls "${RUN_DIR_SICHAN}"/*_predictions_all_methods.csv 2>/dev/null | head -n 1 || true)"
PRED_FBCSP="$(ls "${RUN_DIR_FBCSP}"/*_predictions_all_methods.csv 2>/dev/null | head -n 1 || true)"

run_twoarm_crc_sweep "ea_vs_sichan_multisafe" "${PRED_SICHAN_MSAFE}" "ea-si-chan-multi-safe-csp-lda" "chan"
run_twoarm_crc_sweep "ea_vs_stack_multisafe" "${PRED_STACK_MSAFE}" "ea-stack-multi-safe-csp-lda" "ea"
run_twoarm_crc_sweep "ea_vs_sichan" "${PRED_SICHAN}" "ea-si-chan-csp-lda" "chan"
run_twoarm_crc_sweep "ea_vs_fbcsp" "${PRED_FBCSP}" "ea-fbcsp-lda" "fbcsp"

echo "[$(date)] === Update results registry ==="
python3 scripts/update_results_registry.py --outputs-dir outputs --out docs/experiments/results_registry.csv

echo "[$(date)] DONE"
