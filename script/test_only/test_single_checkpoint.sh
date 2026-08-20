#!/usr/bin/env bash

# Evaluate one checkpoint without editing a round1 training script.
#
# Usage:
#   bash script/test_only/test_single_checkpoint.sh \
#     <checkpoint-path-or-session-name> [experiment-name]
#
# A session name may be given with or without the .pt suffix. When it is not a
# path, the script searches below saved_checkpoints. Hyperparameters encoded in
# the session name are reconstructed for inference.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${VIFOS_ENV_FILE:-$SCRIPT_DIR/evaluation_paths.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

usage() {
  cat >&2 <<'EOF'
Usage:
  bash script/test_only/test_single_checkpoint.sh \
    <checkpoint-path-or-session-name> [experiment-name]

Optional environment variables:
  MODEL_NAME_OVERRIDE   Required for ambiguous legacy Modelv1 checkpoints.
  VIFOS_FORCE=1         Replace an existing result with the same output name.
  PRINT_ONLY=1          Parse and print the command without running Python.
EOF
}

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

cd "$PROJECT_ROOT"
checkpoint_input="$1"
requested_experiment_name="${2:-}"

checkpoint_filename="$(basename -- "$checkpoint_input")"
if [[ "$checkpoint_filename" != *.pt ]]; then
  checkpoint_filename="${checkpoint_filename}.pt"
fi
session_name="${checkpoint_filename%.pt}"

model_label=""
model_name=""
case "$session_name" in
  *"_CNN-LSTM-SE_PS-"*) model_label="CNN-LSTM-SE"; model_name="cnn-lstm-se" ;;
  *"_CNN-LSTM_PS-"*) model_label="CNN-LSTM"; model_name="cnn-lstm" ;;
  *"_Conv-LSTM_PS-"*) model_label="Conv-LSTM"; model_name="conv-lstm" ;;
  *"_Strans-V6_PS-"*) model_label="Strans-V6"; model_name="strans-v6" ;;
  *"_Strans-V5_PS-"*) model_label="Strans-V5"; model_name="strans-v5" ;;
  *"_Strans-V4b_PS-"*) model_label="Strans-V4b"; model_name="strans-v4b" ;;
  *"_Strans-V4B_PS-"*) model_label="Strans-V4B"; model_name="strans-v4b" ;;
  *"_Strans-V4_PS-"*) model_label="Strans-V4"; model_name="strans-v4" ;;
  *"_Strans-V3_PS-"*) model_label="Strans-V3"; model_name="strans-v3" ;;
  *"_Strans-V2_PS-"*) model_label="Strans-V2"; model_name="strans-v2" ;;
  *"_STrans_PS-"*) model_label="STrans"; model_name="strans" ;;
  *"_ModelV2_PS-"*) model_label="ModelV2"; model_name="model_v2" ;;
  *"_VIT_2Head_PS-"*) model_label="VIT_2Head"; model_name="vit-2head" ;;
  *"_Modelv1_PS-"*)
    model_label="Modelv1"
    if [[ -z "${MODEL_NAME_OVERRIDE:-}" ]]; then
      echo "ERROR: Modelv1 is a legacy ambiguous label." >&2
      echo "Set MODEL_NAME_OVERRIDE to the actual architecture, for example strans-v4b." >&2
      exit 2
    fi
    model_name="$MODEL_NAME_OVERRIDE"
    ;;
  *)
    echo "ERROR: Could not identify the model architecture from: $session_name" >&2
    echo "Use MODEL_NAME_OVERRIDE only if the filename contains the legacy Modelv1 label." >&2
    exit 2
    ;;
esac
model_name="${MODEL_NAME_OVERRIDE:-$model_name}"

header="${session_name%%_${model_label}_PS-*}"
if [[ ! "$header" =~ ^BATCH_([0-9]+)_Add_type_([0-9]+)(.+)$ ]]; then
  echo "ERROR: Could not parse batch size, adding type, and group from: $session_name" >&2
  exit 2
fi
trained_batch_size="${BASH_REMATCH[1]}"
adding_type="${BASH_REMATCH[2]}"
group_name="${BASH_REMATCH[3]}"

extract_required() {
  local pattern="$1"
  local description="$2"
  if [[ "$session_name" =~ $pattern ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse $description from: $session_name" >&2
    exit 2
  fi
}

patch_size="$(extract_required '_PS-([^_]+)_Lr-' 'patch size')"
learning_rate="$(extract_required '_Lr-([^_]+)_LF-' 'learning rate')"
dropout="$(extract_required '_DR-([^_]+)_LN-' 'dropout')"
layer_norm="$(extract_required '_LN-(True|False)-ST_' 'layer norm flag')"
spatial_type="$(extract_required '-ST_([0-9]+)_[0-9]+-ON_' 'spatial type')"
spatial_layers="$(extract_required '-ST_[0-9]+_([0-9]+)-ON_' 'spatial layers')"
output_norm="$(extract_required '-ON_(True|False)_Seed-' 'output norm flag')"
seed="$(extract_required '_Seed-([0-9]+)_LRS-' 'seed')"
use_lrs="$(extract_required '_LRS-(True|False)GSMAP_time_step' 'scheduler flag')"
gsmap_steps="$(extract_required 'GSMAP_time_step([0-9]+)ECMWF_time_step' 'GSMAP time steps')"
ecmwf_steps="$(extract_required 'ECMWF_time_step([0-9]+)_' 'ECMWF time steps')"
vit_blocks="$(extract_required 'VIT_Blocks_([0-9]+)$' 'VIT block count')"

weight_func=""
if [[ "$session_name" =~ _LF-weightedmse-wfn_([^_]+)_DR- ]]; then
  loss_name="weightedmse"
  weight_func="${BASH_REMATCH[1]}"
elif [[ "$session_name" =~ _LF-([^_]+)_DR- ]]; then
  loss_name="${BASH_REMATCH[1]}"
else
  echo "ERROR: Could not parse loss function from: $session_name" >&2
  exit 2
fi

scheduler_args=()
if [[ "$use_lrs" == "True" ]]; then
  if [[ "$session_name" =~ _ReduceLROnPlateau-(min|max)-([^_-]+)-([0-9]+)VIT_Blocks_ ]]; then
    scheduler_args=(
      --use_lrscheduler
      --scheduler_type ReduceLROnPlateau
      --plateau_mode "${BASH_REMATCH[1]}"
      --plateau_factor "${BASH_REMATCH[2]}"
      --plateau_patience "${BASH_REMATCH[3]}"
    )
  elif [[ "$session_name" =~ _CosineAnnealingLR-([^_-]+)-([^_-]+)VIT_Blocks_ ]]; then
    scheduler_args=(
      --use_lrscheduler
      --scheduler_type CosineAnnealingLR
      --cosine_t_max "${BASH_REMATCH[1]}"
      --cosine_eta_min "${BASH_REMATCH[2]}"
    )
  else
    echo "ERROR: Scheduler is enabled but its configuration could not be parsed." >&2
    exit 2
  fi
fi

checkpoint_path=""
if [[ -f "$checkpoint_input" ]]; then
  checkpoint_path="$(cd -- "$(dirname -- "$checkpoint_input")" && pwd -P)/$(basename -- "$checkpoint_input")"
elif [[ -f "${checkpoint_input}.pt" ]]; then
  checkpoint_path="$(cd -- "$(dirname -- "${checkpoint_input}.pt")" && pwd -P)/$(basename -- "${checkpoint_input}.pt")"
else
  mapfile -t checkpoint_matches < <(
    find "$PROJECT_ROOT/saved_checkpoints" -type f -name "$checkpoint_filename" -print 2>/dev/null
  )
  if ((${#checkpoint_matches[@]} == 1)); then
    checkpoint_path="${checkpoint_matches[0]}"
  elif ((${#checkpoint_matches[@]} > 1)); then
    echo "ERROR: More than one checkpoint has this filename; pass the full path:" >&2
    printf '  %s\n' "${checkpoint_matches[@]}" >&2
    exit 2
  elif [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    checkpoint_path="$checkpoint_input"
  else
    echo "ERROR: Checkpoint not found: $checkpoint_input" >&2
    exit 2
  fi
fi

safe_model_name="${model_name//-/_}"
safe_learning_rate="${learning_rate//./p}"
experiment_name="${requested_experiment_name:-single_${safe_model_name}_seed_${seed}_lr_${safe_learning_rate}}"
case "$experiment_name" in
  *[!a-zA-Z0-9._-]*|'')
    echo "ERROR: experiment-name may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 2
    ;;
esac

export VIFOS_TEST_ONLY=1
export VIFOS_CHECKPOINT_PATH="$checkpoint_path"
export VIFOS_SESSION_NAME_OVERRIDE="$session_name"
export VIFOS_EXPERIMENT_NAME="$experiment_name"
export VIFOS_RESULTS_DIR="${RESULTS_DIR:-experiment_results}"
export VIFOS_THRESHOLDS_FILE="${THRESHOLDS_FILE:-${VIFOS_RESULTS_DIR}/thresholds/train_rainfall_percentiles.json}"
export VIFOS_EVAL_BATCH_SIZE="${VIFOS_EVAL_BATCH_SIZE:-2}"
export VIFOS_SKIP_EXISTING_RESULTS="${VIFOS_SKIP_EXISTING_RESULTS:-1}"
export GPU_ID="${GPU_ID:-0}"
unset VIFOS_COMPUTE_THRESHOLDS_ONLY || true

python_args=(
  main.py
  --cfg config/default.yaml
  --name "$model_name"
  --seed "$seed"
  --gsmap_time_step "$gsmap_steps"
  --ecmwf_time_step "$ecmwf_steps"
  --in_channel "${IN_CHANNEL:-13}"
  --adding_type "$adding_type"
  --dropout "$dropout"
  --height "${GRID_HEIGHT:-25}"
  --width "${GRID_WIDTH:-25}"
  --data_idx_dir "${DATA_IDX_ROOT}/data6789_reg_1_seed${seed}_new_all"
  --gauge_data_path "$GAUGE_DATA_PATH"
  --npyarr_dir "$NPYARR_DIR"
  --processed_ecmwf_dir "$PROCESSED_ECMWF_DIR"
  --esp_data_path "$ESP_DATA_PATH"
  --lat_start "${LAT_START:-23.25}"
  --lon_start "${LON_START:-102.25}"
  --height_esp "${ESP_HEIGHT:-30}"
  --width_esp "${ESP_WIDTH:-30}"
  --lat_esp_start "${ESP_LAT_START:-23.25}"
  --lon_esp_start "${ESP_LON_START:-102.25}"
  --spatial_type "$spatial_type"
  --num_cnn_layers "$spatial_layers"
  --loss_func "$loss_name"
  --lr "$learning_rate"
  --num_vit_blocks "$vit_blocks"
  --group_name "$group_name"
  --batch_size "$trained_batch_size"
  --patch_size "$patch_size"
  "${scheduler_args[@]}"
)
if [[ "$layer_norm" == "True" ]]; then
  python_args+=(--use_layer_norm)
fi
if [[ "$output_norm" == "True" ]]; then
  python_args+=(--output_norm)
fi
if [[ -n "$weight_func" ]]; then
  python_args+=(--weight_func "$weight_func")
fi

echo "============================================================"
echo "Single-checkpoint evaluation"
echo "Checkpoint : $checkpoint_path"
echo "Output     : ${VIFOS_RESULTS_DIR}/${group_name}/${experiment_name}/seed_${seed}"
echo "Model      : $model_name"
echo "Seed       : $seed"
echo "Train LR   : $learning_rate"
echo "Train batch: $trained_batch_size"
echo "Eval batch : $VIFOS_EVAL_BATCH_SIZE"
echo "============================================================"

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
  printf 'VIFOS_CHECKPOINT_PATH=%q VIFOS_SESSION_NAME_OVERRIDE=%q ' "$VIFOS_CHECKPOINT_PATH" "$VIFOS_SESSION_NAME_OVERRIDE"
  printf '%q ' "${PYTHON_BIN:-python}" "${python_args[@]}"
  printf '\n'
  exit 0
fi

for required_dir in "$DATA_IDX_ROOT/data6789_reg_1_seed${seed}_new_all" "$NPYARR_DIR" "$ESP_DATA_PATH"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "ERROR: Missing required directory: $required_dir" >&2
    exit 2
  fi
done
if [[ ! -f "$GAUGE_DATA_PATH" ]]; then
  echo "ERROR: Missing gauge file: $GAUGE_DATA_PATH" >&2
  exit 2
fi
if [[ ! -f "$VIFOS_THRESHOLDS_FILE" ]]; then
  echo "ERROR: Missing thresholds file: $VIFOS_THRESHOLDS_FILE" >&2
  echo "Run script/test_only/compute_thresholds.sh first." >&2
  exit 2
fi
mkdir -p "$PROCESSED_ECMWF_DIR" "${VIFOS_RESULTS_DIR}/run_logs/${experiment_name}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "${PYTHON_BIN:-python}" "${python_args[@]}" 2>&1 \
  | tee "${VIFOS_RESULTS_DIR}/run_logs/${experiment_name}/test_only.log"
