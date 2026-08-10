LEARNING_RATES=(5e-4 2e-4 1e-4 5e-5 2e-5) # 1e-3 5e-4 2e-4 1e-4 5e-5 2e-5
DROPOUT_RATES=(0.25)
BATCH_SIZES=(64)
NUMBLOCK=(2)
GSMAP_TIME_STEPS=(1 2 3 4 5 6 7)
ECMWF_TIME_STEPS=(1)
PATCH=(3)
for bs in "${BATCH_SIZES[@]}"; do
  for dr in "${DROPOUT_RATES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for block in "${NUMBLOCK[@]}"; do
        for pat in "${PATCH[@]}"; do
          for gsmap in "${GSMAP_TIME_STEPS[@]}"; do
            for ecmwf in "${ECMWF_TIME_STEPS[@]}"; do
              CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/default.yaml \
              --name strans-v5 \
              --gsmap_time_step 3\
              --ecmwf_time_step 1\
              --in_channel 13 \
              --adding_type 0 \
              --dropout "$dr" \
              --height 19 \
              --width 35 \
              --data_idx_dir /mnt/disk3/tunm/Subseasonal_Forecasting/data3/data6789_reg_2_seed52_new_all \
              --gauge_data_path /mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_2.csv \
              --npyarr_dir /mnt/disk3/longnd/env_data/grid_base/nparr_reg2/Step24h \
              --processed_ecmwf_dir /mnt/disk3/longnd/env_data/grid_base/data3_reg_2_new_all \
              --esp_data_path /mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_2 \
              --lat_start 23 \
              --lon_start 103.75 \
              --height_esp 34 \
              --width_esp 54 \
              --lat_esp_start 23.55 \
              --lon_esp_start 103.25 \
              --use_layer_norm \
              --loss_func weightedmse \
              --lr "$lr" \
              --use_lrscheduler \
              --scheduler_type ReduceLROnPlateau \
              --plateau_patience 4 \
              --plateau_min_lr 1e-9 \
              --plateau_factor 0.5 --plateau_verbose \
              --num_vit_blocks "$block" \
              --group_name data3-r2-test-vit-tiny-all \
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
        
        
        
        
        
        