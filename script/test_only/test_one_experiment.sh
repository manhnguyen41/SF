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
export SEEDS_OVERRIDE="52 62 72 82 92"

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
unset VIFOS_COMPUTE_THRESHOLDS_ONLY

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

bash "$TRAINING_BASH"

