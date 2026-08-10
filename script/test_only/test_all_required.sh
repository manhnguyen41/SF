#!/usr/bin/env bash

# Run reviewer-required tests for all already-trained experiments.
#
# Default expected training bash filenames:
#   fullvifos.sh
#   cnn_lstm.sh
#   without_gsmap.sh
#   without_lead_time_embedding.sh
#   spatial_temporal_embedding.sh
#   no_pretrain.sh
#
# Run with defaults from the directory containing those training bash files:
#   bash /path/to/test_all_required.sh
#
# Or provide the actual mapping without editing this file:
#   bash test_all_required.sh \
#     full_vifos=./fullvifos.sh \
#     cnn_lstm=./train_cnn.sh \
#     without_gsmap=./train_no_gsmap.sh \
#     without_lead_time_embedding=./train_no_lt.sh \
#     spatial_temporal_embedding=./train_spatiotemporal.sh \
#     no_pretrain=./train_scratch.sh
#
# Optional already-trained baselines can be appended in the same form, e.g.:
#   conv_lstm=./conv_lstm.sh
#   unet=./unet.sh
#   quantile_mapping=./quantile_mapping.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$SCRIPT_DIR/test_one_experiment.sh"
THRESHOLD_RUNNER="$SCRIPT_DIR/compute_thresholds.sh"

if [[ ! -f "$RUN_ONE" || ! -f "$THRESHOLD_RUNNER" ]]; then
  echo "ERROR: Keep test_all_required.sh, test_one_experiment.sh, and compute_thresholds.sh in the same directory." >&2
  exit 2
fi

if (($# == 0)); then
  SPECS=(
    "full_vifos=./fullvifos.sh"
    "cnn_lstm=./cnn_lstm.sh"
    "without_gsmap=./without_gsmap.sh"
    "without_lead_time_embedding=./without_lead_time_embedding.sh"
    "spatial_temporal_embedding=./spatial_temporal_embedding.sh"
    "no_pretrain=./no_pretrain.sh"
  )
else
  SPECS=("$@")
fi

declare -a EXPERIMENT_NAMES=()
declare -a TRAINING_SCRIPTS=()
FULL_VIFOS_BASH=""

for spec in "${SPECS[@]}"; do
  if [[ "$spec" != *=* ]]; then
    echo "ERROR: Invalid mapping '$spec'. Expected experiment_name=/path/to/train.sh" >&2
    exit 2
  fi
  experiment_name="${spec%%=*}"
  training_script="${spec#*=}"
  if [[ -z "$experiment_name" || -z "$training_script" ]]; then
    echo "ERROR: Invalid empty experiment name or script path in '$spec'." >&2
    exit 2
  fi
  EXPERIMENT_NAMES+=("$experiment_name")
  TRAINING_SCRIPTS+=("$training_script")
  if [[ "$experiment_name" == "full_vifos" ]]; then
    FULL_VIFOS_BASH="$training_script"
  fi
done

if [[ -z "$FULL_VIFOS_BASH" ]]; then
  echo "ERROR: One mapping must use the exact experiment name 'full_vifos'." >&2
  exit 2
fi

# Preflight every path before running anything, so a typo cannot leave a
# confusing partial result set.
missing=0
for training_script in "${TRAINING_SCRIPTS[@]}"; do
  if [[ ! -f "$training_script" ]]; then
    echo "ERROR: Missing training bash: $training_script" >&2
    missing=1
  fi
done
if ((missing)); then
  echo "Pass the real mappings as command-line arguments; no tests were started." >&2
  exit 2
fi

if [[ "${SKIP_THRESHOLDS:-0}" != "1" ]]; then
  bash "$THRESHOLD_RUNNER" "$FULL_VIFOS_BASH"
fi

for index in "${!EXPERIMENT_NAMES[@]}"; do
  experiment_name="${EXPERIMENT_NAMES[$index]}"
  training_script="${TRAINING_SCRIPTS[$index]}"
  compare_with=""

  # The main paired statistical comparison requested by the reviewer.
  if [[ "$experiment_name" == "cnn_lstm" ]]; then
    compare_with="full_vifos"
  fi

  bash "$RUN_ONE" "$experiment_name" "$training_script" "$compare_with"
done

echo "All requested test-only runs completed."

