#!/usr/bin/env bash
set -Eeuo pipefail
LEARNING_RATES=(5e-5) # 1e-3 5e-4 2e-4 1e-4 5e-5 2e-5 1e-5 5e-6 2e-6 1e-6
DROPOUT_RATES=(0.26)
BATCH_SIZES=(32)
NUMBLOCK=(2)
GSMAP_TIME_STEPS=(7)
ECMWF_TIME_STEPS=(7)
PATCH=(3)
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then read -r -a SEEDS <<<"$SEEDS_OVERRIDE"; else SEEDS=(52 62 72 82 92); fi
for gsmap in "${GSMAP_TIME_STEPS[@]}"; do
  for ecmwf in "${ECMWF_TIME_STEPS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for dr in "${DROPOUT_RATES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for block in "${NUMBLOCK[@]}"; do
            for pat in "${PATCH[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "${PYTHON_BIN:-python}" main.py --cfg config/default.yaml \
                    --name strans-v4b \
                    --seed "$seed" \
                    --gsmap_time_step "$gsmap"\
                    --ecmwf_time_step "$ecmwf"\
                    --in_channel 13 \
                    --adding_type 0 \
                    --dropout "$dr" \
                    --height 25 \
                    --width 25 \
                    --data_idx_dir "${DATA_IDX_ROOT:-/mnt/disk3/tunm/Subseasonal_Forecasting/data3}/data6789_reg_1_seed${seed}_new_all" \
                    --gauge_data_path "${GAUGE_DATA_PATH:-/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv}" \
                    --npyarr_dir "${NPYARR_DIR:-/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h}" \
                    --processed_ecmwf_dir "${PROCESSED_ECMWF_DIR:-/mnt/disk3/longnd/env_data/grid_base/data3_reg_1_new_all}" \
                    --esp_data_path "${ESP_DATA_PATH:-/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_1}" \
                    --lat_start 23.25 \
                    --lon_start 102.25 \
                    --height_esp 30 \
                    --width_esp 30 \
                    --lat_esp_start 23.25 \
                    --lon_esp_start 102.25 \
                    --use_layer_norm \
                    --loss_func weightedmse \
                    --lr "$lr" \
                    --use_lrscheduler \
                    --scheduler_type ReduceLROnPlateau \
                    --plateau_patience 3 \
                    --plateau_min_lr 1e-9 \
                    --plateau_factor 0.5 --plateau_verbose \
                    --num_vit_blocks "$block" \
                    --group_name data3-r1-test-vit-tiny-all-weekly \
                    --batch_size "$bs" \
                    --num_epochs 1000\
                    --patch_size "$pat" \
                    --output_norm \

                done
            done
          done
        done
      done
    done
  done
done
        
        
        
        
        

