LEARNING_RATES=(5e-4)
DROPOUT_RATES=(0.3)
BATCH_SIZES=(64)
for bs in "${BATCH_SIZES[@]}"; do
  for dr in "${DROPOUT_RATES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/default.yaml \
        --name strans-v5 \
        --in_channel 13 \
        --adding_type 0 \
        --dropout "$dr" \
        --height 17 \
        --width 17 \
        --data_idx_dir /mnt/disk3/tunm/Subseasonal_Forecasting/data2/data6789_reg_1_seed52_new_long \
        --gauge_data_path /mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv \
        --npyarr_dir /mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h \
        --processed_ecmwf_dir /mnt/disk3/longnd/env_data/grid_base/data2_reg_1_new_long \
        --lat_start 22.75 \
        --lon_start 102.75 \
        --use_layer_norm \
        --loss_func mae \
        --lr "$lr" \
        --use_lrscheduler \
        --scheduler_type ReduceLROnPlateau \
        --plateau_patience 3 \
        --plateau_min_lr 2e-7 \
        --plateau_factor 0.5 --plateau_verbose \
        --group_name data2-r1-test-vit-tiny-long \
        --batch_size "$bs" \
        
    done
  done
done
        
        
        
        
        
        