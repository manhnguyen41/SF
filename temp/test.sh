CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/default.yaml  \
        --name strans-v4 \
        --in_channel 13 \
        --adding_type 0 \
        --dropout 0.41 \
        --height 17\
        --width 17 \
        --data_idx_dir /mnt/disk3/tunm/Subseasonal_Forecasting/data3/data6789_reg_1_seed52 \
        --gauge_data_path /mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_Region_1.csv \
        --npyarr_dir /mnt/disk3/longnd/env_data/S2S_0.125_old/S2S_0.125/nparr_reg_1/Step24h  \
        --processed_ecmwf_dir /mnt/disk3/longnd/env_data/S2S_0.125_small_crop/data3_reg_1  \
        --lat_start 22.75 \
        --lon_start 102.75 \
        --use_layer_norm \
        --loss_func mae \
        --lr 2e-5 \
        --use_lrscheduler \
        --scheduler_type ReduceLROnPlateau \
        --plateau_patience 3 \
        --plateau_min_lr 1e-8 \
        --plateau_factor 0.5 --plateau_verbose \
        --group_name data3-r1-test-vit-base \
        --batch_size 64 \
        
        
        
        