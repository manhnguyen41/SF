#!/usr/bin/env bash

# Compute P90/P95/P99 once from the training split used by seed 92.
#
# Usage:
#   bash compute_thresholds.sh <full_vifos_training_bash>
#
# Example:
#   bash compute_thresholds.sh ./fullvifos.sh

set -Eeuo pipefail

if (($# != 1)); then
  echo "Usage: bash $0 <full_vifos_training_bash>" >&2
  exit 2
fi

FULL_VIFOS_BASH="$1"
if [[ ! -f "$FULL_VIFOS_BASH" ]]; then
  echo "ERROR: Full VIFOS bash was not found: $FULL_VIFOS_BASH" >&2
  exit 2
fi

# Use one fixed training split to define the event thresholds for every model.
# Do not compute percentiles separately for every experiment or from test data.
export RUN_MODE="thresholds"
export FORCE_THRESHOLDS="${FORCE_THRESHOLDS:-0}"
export SEEDS_OVERRIDE="92"
export THRESHOLD_MODEL_SEED="92"
export THRESHOLD_DATA_SEED="92"
export VIFOS_COMPUTE_THRESHOLDS_ONLY="1"
unset VIFOS_TEST_ONLY || true
unset VIFOS_COMPARE_WITH || true
unset COMPARE_WITH || true

echo "Computing train-derived P90/P95/P99 with data split seed 92..."
bash "$FULL_VIFOS_BASH"

