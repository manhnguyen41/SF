#!/usr/bin/env bash

# Compute P90/P95/P99 once from the training split used by seed 92.
#
# Usage:
#   bash compute_thresholds.sh <full_vifos_training_bash>
#
# Example:
#   bash compute_thresholds.sh ./fullvifos.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${VIFOS_ENV_FILE:-$SCRIPT_DIR/evaluation_paths.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

if (($# != 1)); then
  echo "Usage: bash $0 <full_vifos_training_bash>" >&2
  exit 2
fi

FULL_VIFOS_BASH="$1"
if [[ ! -f "$FULL_VIFOS_BASH" ]]; then
  echo "ERROR: Full VIFOS bash was not found: $FULL_VIFOS_BASH" >&2
  exit 2
fi
FULL_VIFOS_BASH="$(cd -- "$(dirname -- "$FULL_VIFOS_BASH")" && pwd)/$(basename -- "$FULL_VIFOS_BASH")"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Use one fixed training split to define the event thresholds for every model.
# Do not compute percentiles separately for every experiment or from test data.
export RUN_MODE="thresholds"
export FORCE_THRESHOLDS="${FORCE_THRESHOLDS:-0}"
export SEEDS_OVERRIDE="92"
export THRESHOLD_MODEL_SEED="92"
export THRESHOLD_DATA_SEED="92"
export VIFOS_COMPUTE_THRESHOLDS_ONLY="1"
export VIFOS_RESULTS_DIR="${RESULTS_DIR:-experiment_results}"
export VIFOS_THRESHOLDS_FILE="${THRESHOLDS_FILE:-${VIFOS_RESULTS_DIR}/thresholds/train_rainfall_percentiles.json}"
export VIFOS_DRY_RUN="${DRY_RUN:-0}"
unset VIFOS_TEST_ONLY || true
unset VIFOS_COMPARE_WITH || true
unset COMPARE_WITH || true

if [[ -f "$VIFOS_THRESHOLDS_FILE" && "$FORCE_THRESHOLDS" != "1" ]]; then
  echo "Threshold file already exists; skipping: $VIFOS_THRESHOLDS_FILE"
  exit 0
fi

if [[ "$VIFOS_DRY_RUN" != "1" ]]; then
  if [[ ! -f "$GAUGE_DATA_PATH" ]]; then
    echo "ERROR: Missing GAUGE_DATA_PATH: $GAUGE_DATA_PATH" >&2
    exit 2
  fi
  threshold_index="$DATA_IDX_ROOT/data6789_reg_1_seed92_new_all/train.csv"
  if [[ ! -f "$threshold_index" ]]; then
    echo "ERROR: Missing reference train index: $threshold_index" >&2
    exit 2
  fi
fi

echo "Computing train-derived P90/P95/P99 with the common train split (reference seed 92)..."
if [[ "$VIFOS_DRY_RUN" == "1" ]]; then
  bash -x "$FULL_VIFOS_BASH"
else
  bash "$FULL_VIFOS_BASH"
fi

