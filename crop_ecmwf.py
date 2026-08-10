import numpy as np
import os
import os
import numpy as np
from tqdm import tqdm
def crop_region_1():
    input_folder = '/mnt/disk3/longnd/env_data/S2S_0.125_small_crop/nparr/Step24h'
    output_folder = '/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h'
    os.makedirs(output_folder, exist_ok=True)
    # 25 to 8, 100 to 115
    lat_start = 25
    lon_start = 100
    step = 0.125
    lat_crop = 23.25
    lon_crop = 102.25
    h = 25
    w = 25

    # Ép kiểu int để tránh lỗi slicing
    y_begin = int(np.round((lat_start - lat_crop) / step))
    x_begin = int(np.round((lon_crop - lon_start) / step))
    
    for file_name in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file_name)
        data = np.load(file_path)
        # Cắt mảng
        data = data[:, :, :, y_begin:y_begin+h, x_begin:x_begin+w]
        padded = np.pad(data, ((0,0), (0,0), (0,0), (1,1), (1,1)), mode='reflect')
        shifts = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1),  (0, 0),  ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]
        neigh_sum = np.zeros_like(data, dtype=np.float32)
        for dy, dx in shifts:
            neigh_sum += padded[:, :, :, 1+dy : 1+dy + h,
                                        1+dx : 1+dx + w]

        # Trung bình 8 ô xung quanh
        data_avg8 = neigh_sum / 9.0
        np.save(os.path.join(output_folder, file_name), data_avg8)


def crop_region_2():
    input_folder = '/mnt/disk3/longnd/env_data/S2S_0.125_small_crop/nparr/Step24h'
    output_folder = '/mnt/disk3/longnd/env_data/grid_base/nparr_reg2/Step24h'
    os.makedirs(output_folder, exist_ok=True)

    lat_start = 25
    lon_start = 100
    step = 0.125
    lat_crop = 23
    lon_crop = 103.75
    h = 19
    w = 35

    # Ép kiểu int để tránh lỗi slicing
    y_begin = int(np.round((lat_start - lat_crop) / step))
    x_begin = int(np.round((lon_crop - lon_start) / step))

    for file_name in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file_name)
        data = np.load(file_path)
        # Cắt mảng
        data = data[:, :, :, y_begin:y_begin+h, x_begin:x_begin+w]
        np.save(os.path.join(output_folder, file_name), data)
def check():
    
    output_folder = '/mnt/disk3/longnd/env_data/grid_base/nparr_reg2/Step24h'
    #os.makedirs(output_folder, exist_ok=True)

    lat_start = 25
    lon_start = 100
    step = 0.125
    lat_crop = 22.75
    lon_crop = 102.75
    h = 19
    w = 35

    # Ép kiểu int để tránh lỗi slicing
    y_begin = int(np.round((lat_start - lat_crop) / step))
    x_begin = int(np.round((lon_crop - lon_start) / step))

    for file_name in tqdm(os.listdir(output_folder)):
        file_path = os.path.join(output_folder, file_name)
        data = np.load(file_path)
        # Cắt mảng
        
        if(data.shape[3]!=h or data.shape[4]!=w):
            print(f"File bị lỗi shape {file_name}")
            
crop_region_1()