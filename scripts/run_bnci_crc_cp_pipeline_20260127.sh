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

DATE=20260127

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

echo "[$(date)] === Run strict LOSO predictions (BNCI2014_001, 4-class) ==="

BASE_RUN="loso4_bnci2014_001_0train_paperfir_nc6_ea_stackmss_v1"
FOUND="$(find_run_dir_with_preds "${BASE_RUN}" || true)"
if [ -n "${FOUND}" ]; then
  echo "[$(date)] [BASELINES] SKIP  ${BASE_RUN} (found ${FOUND})"
else
  echo "[$(date)] [BASELINES] START ${BASE_RUN}"
  python3 -u run_csp_lda_loso.py \
    --dataset BNCI2014_001 \
    --preprocess paper_fir \
    --events left_hand,right_hand,feet,tongue \
    --sessions 0train \
    --n-components 6 \
    --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
    --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
    --si-chan-ranks 21 --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
    --oea-zo-selector calibrated_stack_ridge_guard_borda \
    --oea-zo-calib-guard-threshold 0.5 \
    --stack-safe-anchor-guard-delta 0.05 \
    --stack-safe-anchor-probe-hard-worsen -0.01 \
    --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
    --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
    --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
    --no-plots \
    --run-name "${BASE_RUN}"
  FOUND="$(find_run_dir_with_preds "${BASE_RUN}" || true)"
  if [ -z "${FOUND}" ]; then
    echo "[$(date)] [BASELINES] ERROR ${BASE_RUN} finished but no *_predictions_all_methods.csv found under outputs/*/4class/${BASE_RUN}*"
    exit 1
  fi
  echo "[$(date)] [BASELINES] DONE  ${BASE_RUN} -> ${FOUND}"
fi

PRED_ALL="$(ls "${FOUND}"/*_predictions_all_methods.csv 2>/dev/null | head -n 1 || true)"
if [ -z "${PRED_ALL}" ]; then
  echo "[$(date)] ERROR: cannot find predictions_all_methods.csv in ${FOUND}"
  exit 1
fi

echo "[$(date)] === Run CP-CRC (alpha sweep) ==="

METHOD_NAME="safe-tta-offline-multiarm-crc-cp"
MIN_PRED_GRID="0,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.01,0.02,0.03,0.05,0.08,0.1"
for A in 0.35 0.40 0.50; do
  OUT="outputs/${DATE}/4class/offline_bnci2014_001_safettta_crc_cp_alpha${A}_delta0.05_splits5_v1"
  echo "[$(date)] [CRC-CP] alpha=${A} -> ${OUT}"
  python3 scripts/offline_safe_tta_multi_select_crc_from_predictions.py \
    --preds "${PRED_ALL}" \
    --anchor-method ea-csp-lda \
    --candidate-methods ea-stack-multi-safe-csp-lda \
    --candidate-family-map ea-stack-multi-safe-csp-lda=stack \
    --guard-threshold 0.5 \
    --anchor-guard-delta 0.05 \
    --risk-alpha "${A}" \
    --delta 0.05 \
    --n-splits 5 \
    --min-pred-grid "${MIN_PRED_GRID}" \
    --out-dir "${OUT}" \
    --method-name "${METHOD_NAME}" \
    --date-prefix "${DATE}"
done

echo "[$(date)] === Make tables + plots ==="

FIGDIR="docs/experiments/figures/${DATE}_bnci2014_001_crc_cp_v1"
mkdir -p "${FIGDIR}"

for A in 0.35 0.40 0.50; do
  TAG="$(echo "${A}" | sed 's/\\./p/g')"
  CRC="outputs/${DATE}/4class/offline_bnci2014_001_safettta_crc_cp_alpha${A}_delta0.05_splits5_v1"
  MERGED="outputs/${DATE}/4class/loso4_bnci2014_001_merged_plus_crc_cp_alpha${A}_delta0.05_splits5_v1"

  python3 scripts/merge_loso_runs.py \
    --run-dirs "${FOUND}" "${CRC}" \
    --out-run-dir "${MERGED}" \
    --prefer-date-prefix "${DATE}"

  python3 scripts/make_main_table_and_stats.py \
    --run-dir "${MERGED}" \
    --baseline-method ea-csp-lda \
    --out-csv "${FIGDIR}/main_table_alpha${TAG}.csv" \
    --out-per-subject-csv "${FIGDIR}/per_subject_alpha${TAG}.csv" \
    --out-md "${FIGDIR}/main_table_alpha${TAG}.md"

  python3 scripts/plot_candidate_diagnostics.py \
    --run-dir "${CRC}" \
    --method "${METHOD_NAME}" \
    --out-dir "${FIGDIR}" \
    --prefix "bnci_crc_cp_alpha${TAG}"
done

python3 scripts/plot_crc_risk_coverage.py \
  --run-dirs \
    "outputs/${DATE}/4class/offline_bnci2014_001_safettta_crc_cp_alpha0.35_delta0.05_splits5_v1" \
    "outputs/${DATE}/4class/offline_bnci2014_001_safettta_crc_cp_alpha0.40_delta0.05_splits5_v1" \
    "outputs/${DATE}/4class/offline_bnci2014_001_safettta_crc_cp_alpha0.50_delta0.05_splits5_v1" \
  --method "${METHOD_NAME}" \
  --out-dir "${FIGDIR}" \
  --prefix bnci_crc_cp_v1

echo "[$(date)] === Update results registry ==="
python3 scripts/update_results_registry.py --outputs-dir outputs --out docs/experiments/results_registry.csv

echo "[$(date)] DONE"

