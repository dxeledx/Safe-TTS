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

DATE=20260126
METHODS="ea-csp-lda,lea-csp-lda,ts-lr,riemann-mdm"

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

echo "[$(date)] === B1: Run strict LOSO baselines (PhysioNetMI, 4-class) ==="

for RANGE in 21-40 41-60 61-80 81-109; do
  START="${RANGE%-*}"
  END="${RANGE#*-}"
  TEST_SUBJECTS="$(seq -s, "${START}" "${END}")"
  RUN="loso4_physionetmi_rs160_paperfir_s${RANGE}_baselines4_v1"

  FOUND="$(find_run_dir_with_preds "${RUN}" || true)"
  if [ -n "${FOUND}" ]; then
    echo "[$(date)] [BASELINES] SKIP  ${RUN} (found ${FOUND})"
    continue
  fi

  echo "[$(date)] [BASELINES] START ${RUN}"
  python3 -u run_csp_lda_loso.py \
    --dataset PhysionetMI \
    --preprocess paper_fir \
    --events left_hand,right_hand,feet,rest \
    --sessions ALL \
    --tmin 0 --tmax 3 \
    --resample 160 \
    --n-components 6 \
    --methods "${METHODS}" \
    --no-plots \
    --test-subjects "${TEST_SUBJECTS}" \
    --run-name "${RUN}"
  FOUND="$(find_run_dir_with_preds "${RUN}" || true)"
  if [ -z "${FOUND}" ]; then
    echo "[$(date)] [BASELINES] ERROR ${RUN} finished but no *_predictions_all_methods.csv found under outputs/*/4class/${RUN}*"
    exit 1
  fi
  echo "[$(date)] [BASELINES] DONE  ${RUN} -> ${FOUND}"
done

echo "[$(date)] === Merge baselines into full S1–S109 predictions ==="

BASE_S1_20="outputs/20260124/4class/loso4_physionetmi_rs160_paperfir_s1-20_merged_baselines_v1"
RUN_S21_40="$(find_run_dir_with_preds "loso4_physionetmi_rs160_paperfir_s21-40_baselines4_v1")"
RUN_S41_60="$(find_run_dir_with_preds "loso4_physionetmi_rs160_paperfir_s41-60_baselines4_v1")"
RUN_S61_80="$(find_run_dir_with_preds "loso4_physionetmi_rs160_paperfir_s61-80_baselines4_v1")"
RUN_S81_109="$(find_run_dir_with_preds "loso4_physionetmi_rs160_paperfir_s81-109_baselines4_v1")"

MERGED_BASE="outputs/${DATE}/4class/loso4_physionetmi_rs160_paperfir_s1-109_merged_baselines4_v1"
python3 scripts/merge_loso_runs.py \
  --run-dirs \
    "${BASE_S1_20}" \
    "${RUN_S21_40}" \
    "${RUN_S41_60}" \
    "${RUN_S61_80}" \
    "${RUN_S81_109}" \
  --out-run-dir "${MERGED_BASE}" \
  --prefer-date-prefix "${DATE}"

PRED_ALL="${MERGED_BASE}/${DATE}_predictions_all_methods.csv"

echo "[$(date)] === B2: Run CP-CRC (alpha sweep) on full S1–S109 ==="

for A in 0.03 0.05 0.10 0.15; do
  OUT="outputs/${DATE}/4class/offline_physionetmi_safettta_multiarm_crc_cp_s1-109_alpha${A}_delta0.05_splits5"
  echo "[$(date)] [CRC-CP] alpha=${A} -> ${OUT}"
  python3 scripts/offline_safe_tta_multi_select_crc_from_predictions.py \
    --preds "${PRED_ALL}" \
    --anchor-method ea-csp-lda \
    --candidate-methods lea-csp-lda,ts-lr,riemann-mdm \
    --guard-threshold 0.5 \
    --anchor-guard-delta 0.05 \
    --risk-alpha "${A}" \
    --delta 0.05 \
    --n-splits 5 \
    --min-pred-grid 0,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.01 \
    --out-dir "${OUT}" \
    --method-name safe-tta-offline-multiarm-crc-cp \
    --date-prefix "${DATE}"
done

echo "[$(date)] === B3: Make tables + risk-coverage plots ==="

FIGDIR="docs/experiments/figures/${DATE}_physionetmi_crc_cp_full109"
mkdir -p "${FIGDIR}"

for A in 0.03 0.05 0.10 0.15; do
  TAG="$(echo "${A}" | sed 's/\\./p/g')"
  CRC="outputs/${DATE}/4class/offline_physionetmi_safettta_multiarm_crc_cp_s1-109_alpha${A}_delta0.05_splits5"
  MERGED="outputs/${DATE}/4class/loso4_physionetmi_s1-109_merged_plus_crc_cp_alpha${A}_delta0.05_splits5"

  python3 scripts/merge_loso_runs.py \
    --run-dirs "${MERGED_BASE}" "${CRC}" \
    --out-run-dir "${MERGED}" \
    --prefer-date-prefix "${DATE}"

  python3 scripts/make_main_table_and_stats.py \
    --run-dir "${MERGED}" \
    --baseline-method ea-csp-lda \
    --out-csv "${FIGDIR}/main_table_alpha${TAG}.csv" \
    --out-per-subject-csv "${FIGDIR}/per_subject_alpha${TAG}.csv" \
    --out-md "${FIGDIR}/main_table_alpha${TAG}.md"
done

python3 scripts/plot_crc_risk_coverage.py \
  --run-dirs \
    "outputs/${DATE}/4class/offline_physionetmi_safettta_multiarm_crc_cp_s1-109_alpha0.03_delta0.05_splits5" \
    "outputs/${DATE}/4class/offline_physionetmi_safettta_multiarm_crc_cp_s1-109_alpha0.05_delta0.05_splits5" \
    "outputs/${DATE}/4class/offline_physionetmi_safettta_multiarm_crc_cp_s1-109_alpha0.10_delta0.05_splits5" \
    "outputs/${DATE}/4class/offline_physionetmi_safettta_multiarm_crc_cp_s1-109_alpha0.15_delta0.05_splits5" \
  --method safe-tta-offline-multiarm-crc-cp \
  --out-dir "${FIGDIR}" \
  --prefix physio_crc_cp_full109

echo "[$(date)] === Update results registry ==="
python3 scripts/update_results_registry.py --outputs-dir outputs --out docs/experiments/results_registry.csv

echo "[$(date)] DONE"
