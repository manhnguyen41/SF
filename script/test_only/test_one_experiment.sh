#!/usr/bin/env bash

# Run one already-trained experiment in test-only mode.
#
# Usage:
#   bash test_one_experiment.sh <experiment_name> <training_bash> [compare_with]
#
# Example:
#   bash test_one_experiment.sh full_vifos ./fullvifos.sh
#   bash test_one_experiment.sh cnn_lstm ./cnn_lstm.sh full_vifos
#
# The original training bash is deliberately reused so that model architecture,
# ablation flags, data paths, learning rate, group name, and generated checkpoint
# session name are identical to the training run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${VIFOS_ENV_FILE:-$SCRIPT_DIR/evaluation_paths.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

if (($# < 2 || $# > 3)); then
  echo "Usage: bash $0 <experiment_name> <training_bash> [compare_with]" >&2
  exit 2
fi

EXPERIMENT_NAME="$1"
TRAINING_BASH="$2"
COMPARE_EXPERIMENT="${3:-}"

if [[ ! -f "$TRAINING_BASH" ]]; then
  echo "ERROR: Training bash was not found: $TRAINING_BASH" >&2
  exit 2
fi
TRAINING_BASH="$(cd -- "$(dirname -- "$TRAINING_BASH")" && pwd)/$(basename -- "$TRAINING_BASH")"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

case "$EXPERIMENT_NAME" in
  *[!a-zA-Z0-9._-]*|'')
    echo "ERROR: experiment_name may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 2
    ;;
esac

if [[ -n "$COMPARE_EXPERIMENT" ]]; then
  case "$COMPARE_EXPERIMENT" in
    *[!a-zA-Z0-9._-]*)
      echo "ERROR: compare_with contains unsupported characters." >&2
      exit 2
      ;;
  esac
fi

# These are the five seeds already trained by the user. A training bash that
# supports SEEDS_OVERRIDE will use exactly this list. A bash with its own fixed
# loop must already contain the same five seeds.
export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-52 62 72 82 92}"

# NAME/EXPERIMENT_NAME are user-facing experiment labels. MODEL.NAME/--name in
# Python may still be an architecture identifier such as strans-v6 or cnn-lstm.
export NAME="$EXPERIMENT_NAME"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export VIFOS_EXPERIMENT_NAME="$EXPERIMENT_NAME"

# Both switches are set intentionally. RUN_MODE is understood by the revised
# bash files; VIFOS_TEST_ONLY is enforced by main.py and prevents training even
# when an older training bash does not have a RUN_MODE switch.
export RUN_MODE="test"
export VIFOS_TEST_ONLY="1"
export VIFOS_DRY_RUN="${DRY_RUN:-0}"
export VIFOS_PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
export VIFOS_RESULTS_DIR="${RESULTS_DIR:-experiment_results}"
export VIFOS_THRESHOLDS_FILE="${THRESHOLDS_FILE:-${VIFOS_RESULTS_DIR}/thresholds/train_rainfall_percentiles.json}"
export GPU_ID="${GPU_ID:-0}"
unset VIFOS_COMPUTE_THRESHOLDS_ONLY

if [[ "$VIFOS_DRY_RUN" != "1" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable or on PATH: $PYTHON_BIN" >&2
    exit 2
  fi
  for required_dir_var in DATA_IDX_ROOT NPYARR_DIR ESP_DATA_PATH; do
    required_dir="${!required_dir_var}"
    if [[ ! -d "$required_dir" ]]; then
      echo "ERROR: Missing $required_dir_var directory: $required_dir" >&2
      exit 2
    fi
  done
  if [[ ! -f "$GAUGE_DATA_PATH" ]]; then
    echo "ERROR: Missing GAUGE_DATA_PATH: $GAUGE_DATA_PATH" >&2
    exit 2
  fi
  mkdir -p "$PROCESSED_ECMWF_DIR"
  if [[ ! -w "$PROCESSED_ECMWF_DIR" ]]; then
    echo "ERROR: PROCESSED_ECMWF_DIR is not writable: $PROCESSED_ECMWF_DIR" >&2
    exit 2
  fi
  read -r -a requested_seeds <<<"$SEEDS_OVERRIDE"
  for seed in "${requested_seeds[@]}"; do
    seed_dir="$DATA_IDX_ROOT/data6789_reg_1_seed${seed}_new_all"
    for filename in train.csv valid.csv test.csv scalers1.pkl output_scaler1.pkl esp_scalers1.pkl; do
      if [[ ! -f "$seed_dir/$filename" ]]; then
        echo "ERROR: Missing seed-$seed data file: $seed_dir/$filename" >&2
        exit 2
      fi
    done
  done
fi

if [[ -n "$COMPARE_EXPERIMENT" ]]; then
  export COMPARE_WITH="$COMPARE_EXPERIMENT"
  export VIFOS_COMPARE_WITH="$COMPARE_EXPERIMENT"
else
  unset COMPARE_WITH || true
  unset VIFOS_COMPARE_WITH || true
fi

echo "============================================================"
echo "Test-only experiment : $EXPERIMENT_NAME"
echo "Training bash        : $TRAINING_BASH"
echo "Seeds                : $SEEDS_OVERRIDE"
if [[ -n "$COMPARE_EXPERIMENT" ]]; then
  echo "Paired comparison    : $COMPARE_EXPERIMENT"
fi
echo "============================================================"

if [[ "$VIFOS_DRY_RUN" == "1" ]]; then
  bash -x "$TRAINING_BASH"
else
  log_dir="${VIFOS_RESULTS_DIR}/run_logs/${EXPERIMENT_NAME}"
  mkdir -p "$log_dir"
  bash "$TRAINING_BASH" 2>&1 | tee "$log_dir/test_only.log"
fi

