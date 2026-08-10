#!/usr/bin/env bash

# Defaults below are intentionally identical to script/round1/*.sh because the
# evaluation will run on the same machine used for training. No path edit is
# required there. This file only centralizes optional overrides.

# Python environment containing torch, pandas, scikit-learn, scipy, peft, etc.
export PYTHON_BIN="${PYTHON_BIN:-python}"

# GPU and local output locations.
export GPU_ID="${GPU_ID:-0}"
export RESULTS_DIR="${RESULTS_DIR:-experiment_results}"
export THRESHOLDS_FILE="${THRESHOLDS_FILE:-${RESULTS_DIR}/thresholds/train_rainfall_percentiles.json}"

# Directory containing data6789_reg_1_seed{52,62,72,82,92}_new_all.
export DATA_IDX_ROOT="${DATA_IDX_ROOT:-/mnt/disk3/tunm/Subseasonal_Forecasting/data3}"

# Gauge CSV used for ground truth and train-only rainfall percentiles.
export GAUGE_DATA_PATH="${GAUGE_DATA_PATH:-/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv}"

# Raw ECMWF and GSMaP directories. Indexed CSV basenames are resolved under
# these roots, so the old absolute prefixes stored in the CSV may differ.
export NPYARR_DIR="${NPYARR_DIR:-/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h}"
export ESP_DATA_PATH="${ESP_DATA_PATH:-/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_1}"

# Writable cache root for preprocessed ECMWF/GSMaP arrays.
export PROCESSED_ECMWF_DIR="${PROCESSED_ECMWF_DIR:-/mnt/disk3/longnd/env_data/grid_base/data3_reg_1_new_all}"

# Leave unset when checkpoints are under the repository's
# saved_checkpoints/<group>[/checkpoint] directory. Otherwise uncomment one:
# export VIFOS_CHECKPOINT_PATH="/path/to/checkpoints/{session}.pt"
# export VIFOS_CHECKPOINT_PATH="/path/to/checkpoints"  # existing directory
