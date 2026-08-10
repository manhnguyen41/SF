#!/usr/bin/env bash

# Standalone test-only version of the supplied Full VIFOS command.
# It evaluates the five existing checkpoints for seeds 52, 62, 72, 82, and 92.
#
# Usage:
#   bash test_full_vifos.sh
#
# Optional overrides:
#   GPU_ID=1 bash test_full_vifos.sh
#   NAME=full_vifos RESULTS_DIR=experiment_results bash test_full_vifos.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-$SCRIPT_DIR}"

NAME="${NAME:-full_vifos}"
MODEL_NAME="${MODEL_NAME:-strans-v6}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-experiment_results}"
THRESHOLDS_FILE="${THRESHOLDS_FILE:-${RESULTS_DIR}/thresholds/train_rainfall_percentiles.json}"
GROUP_NAME="${GROUP_NAME:-data3-r1-test-vit-tiny-all-weekly}"
DRY_RUN="${DRY_RUN:-0}"

SEEDS=(52 62 72 82 92)

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
DROPOUT_RATE="${DROPOUT_RATE:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_VIT_BLOCKS="${NUM_VIT_BLOCKS:-2}"
GSMAP_TIME_STEPS="${GSMAP_TIME_STEPS:-7}"
ECMWF_TIME_STEPS="${ECMWF_TIME_STEPS:-7}"
PATCH_SIZE="${PATCH_SIZE:-3}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"

CONFIG_PATH="${CONFIG_PATH:-config/default.yaml}"
DATA_IDX_ROOT="${DATA_IDX_ROOT:-/mnt/disk3/tunm/Subseasonal_Forecasting/data3}"
GAUGE_DATA_PATH="${GAUGE_DATA_PATH:-/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv}"
NPYARR_DIR="${NPYARR_DIR:-/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h}"
PROCESSED_ECMWF_DIR="${PROCESSED_ECMWF_DIR:-/mnt/disk3/longnd/env_data/grid_base/data3_reg_1_new_all}"
ESP_DATA_PATH="${ESP_DATA_PATH:-/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_1}"

common_args=(
  --cfg "$CONFIG_PATH"
  --name "$MODEL_NAME"
  --gsmap_time_step "$GSMAP_TIME_STEPS"
  --ecmwf_time_step "$ECMWF_TIME_STEPS"
  --in_channel 13
  --adding_type 0
  --dropout "$DROPOUT_RATE"
  --height 25
  --width 25
  --gauge_data_path "$GAUGE_DATA_PATH"
  --npyarr_dir "$NPYARR_DIR"
  --processed_ecmwf_dir "$PROCESSED_ECMWF_DIR"
  --esp_data_path "$ESP_DATA_PATH"
  --lat_start 23.25
  --lon_start 102.25
  --height_esp 30
  --width_esp 30
  --lat_esp_start 23.25
  --lon_esp_start 102.25
  --use_layer_norm
  --loss_func weightedmse
  --lr "$LEARNING_RATE"
  --use_lrscheduler
  --scheduler_type ReduceLROnPlateau
  --plateau_patience 3
  --plateau_min_lr 1e-9
  --plateau_factor 0.5
  --plateau_verbose
  --num_vit_blocks "$NUM_VIT_BLOCKS"
  --group_name "$GROUP_NAME"
  --batch_size "$BATCH_SIZE"
  --num_epochs "$NUM_EPOCHS"
  --patch_size "$PATCH_SIZE"
  --output_norm
)

for seed in "${SEEDS[@]}"; do
  data_idx_dir="${DATA_IDX_ROOT}/data6789_reg_1_seed${seed}_new_all"
  log_dir="${RESULTS_DIR}/run_logs/${NAME}"
  log_file="${log_dir}/test_seed_${seed}.log"

  command=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    "VIFOS_EXPERIMENT_NAME=${NAME}"
    "VIFOS_RESULTS_DIR=${RESULTS_DIR}"
    "VIFOS_THRESHOLDS_FILE=${THRESHOLDS_FILE}"
    "VIFOS_TEST_ONLY=1"
    "$PYTHON_BIN" main.py
    "${common_args[@]}"
    --seed "$seed"
    --data_idx_dir "$data_idx_dir"
  )

  echo "[test-only] name=$NAME model=$MODEL_NAME seed=$seed data_split_seed=$seed"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi

  mkdir -p "$log_dir"
  "${command[@]}" 2>&1 | tee "$log_file"
done

