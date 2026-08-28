#!/usr/bin/env bash
set -euo pipefail

DATA_IDX_ROOT="${DATA_IDX_ROOT:-/mnt/disk3/tunm/Subseasonal_Forecasting/data3}"
TRAIN_INDEX="${TRAIN_INDEX:-${DATA_IDX_ROOT}/data6789_reg_1_seed52_new_all/train.csv}"
GAUGE_CSV="${GAUGE_CSV:-/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv}"
ECMWF_ROOT="${ECMWF_ROOT:-/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h}"
QUANTILE_ROOT="${QUANTILE_ROOT:-quantile_distributions/train_only}"

python script/quantile/fit_train_only_distributions.py \
  --train-index "$TRAIN_INDEX" \
  --gauge-csv "$GAUGE_CSV" \
  --ecmwf-root "$ECMWF_ROOT" \
  --output-root "$QUANTILE_ROOT" \
  --overwrite

export DATA_IDX_ROOT
export QUANTILE_ROOT
export VIFOS_QUANTILE_GRID_DIR="$QUANTILE_ROOT/s2s"
export VIFOS_QUANTILE_GAUGE_DIR="$QUANTILE_ROOT/gauge"
export VIFOS_EXPERIMENT_NAME="quantile_mapping_train_only"

bash script/round1/quantitle.sh
