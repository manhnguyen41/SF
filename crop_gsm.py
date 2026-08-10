import datetime
import numpy as np
import os
from tqdm import tqdm
import copy

def preprocess_region_1():
    fold_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/raw_region_1'
    output_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_1'
    os.makedirs(output_dir, exist_ok=True)

    # Tạo danh sách ngày từ 2024-01-01 đến 2024-12-31
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)

    dates = []
    current_date = start_date
    day_steps = [3, 4]  # bước nhảy 3-4 ngày xen kẽ trước 11/11
    step_index = 0

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        if current_date >= datetime.date(2024, 11, 11):
            current_date += datetime.timedelta(days=2)
        else:
            current_date += datetime.timedelta(days=day_steps[step_index])
            step_index = 1 - step_index

    h = 30
    w = 30

    for date in tqdm(dates, desc="Processing dates"):
        timestep = 15
        data = np.zeros((20, timestep, h, w))
        s = copy.copy(date)[4:]  # MMDD
        
        for year in range(2004, 2024):
            for lt in range(-timestep, 0):
                
                temp_date = datetime.datetime.strptime(date, "%Y%m%d")
                year_diff = 2024 - year

                try:
                    temp_date = temp_date.replace(year=temp_date.year - year_diff)
                except ValueError:
                    # Nếu gặp 29/2 mà năm mới không nhuận → dùng 28/2
                    temp_date = temp_date.replace(year=temp_date.year - year_diff, day=28)
                cur_date = temp_date + datetime.timedelta(days=lt)
                
                
                file_dir = f'{fold_dir}/{cur_date.strftime("%Y%m%d")}.npy'

                if os.path.exists(file_dir):
                    # Load file với mmap để tránh chiếm RAM quá nhiều
                    data_temp = np.load(file_dir, mmap_mode='r')
                    # Lấy vùng h x w bằng vectorized indexing
                    data_cur = data_temp.copy()
                    
                    data_cur[(np.isnan(data_cur)) | (data_cur <= 0)] = 0
                    data[year - 2004, (lt+timestep)] = data_cur
                else:
                    print(f'Warning: File {file_dir} not found')

        # Lưu file kết quả
        np.save(os.path.join(output_dir, s[:2] + '-' + s[2:] + '.npy'), data)

def cut_region_1():
    fold_dir = '/mnt/disk3/longnd/env_data/GSMaP/npy'
    output_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/raw_region_1'
    os.makedirs(output_dir, exist_ok=True)
    # Thông số lưới
    LAT_MIN, LAT_MAX = 20.25, 23.25  # max -> min
    LON_MIN, LON_MAX = 102.25, 105.25  # min -> max
    
    step_esp = 0.1
    h = 30
    w = 30
    start_date = datetime.date(2004, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    dates = []
    current_date = start_date

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        
        current_date += datetime.timedelta(days=1)
        

    lat_gsm_begin, lat_gsm_end = 24, 8
    lon_gsm_begin, lon_gsm_end = 102, 117
    # Tạo mảng lat/lon cho grid (vectorized)
    lat_idx = np.floor((lat_gsm_begin - (LAT_MAX - step_esp * np.arange(h)[:, None])) / step_esp).astype(int)
    lon_idx = np.floor((LON_MIN + step_esp * np.arange(w)[None, :] - lon_gsm_begin) / step_esp).astype(int)
    for date in tqdm(dates):
        
        data = np.zeros((h, w))
        for hour in range(24):
            file_dir = f'{fold_dir}/{date}_{hour:02d}00.npy'

            if os.path.exists(file_dir):
                # Load file với mmap để tránh chiếm RAM quá nhiều
                data_temp = np.load(file_dir, mmap_mode='r')
                # Lấy vùng h x w bằng vectorized indexing
                data_cur = data_temp[lat_idx, lon_idx].copy()
                padded = np.pad(data_cur, ((1,1), (1,1)), mode='reflect')
                shifts = [(-1,-1), (-1,0), (-1,1),
                        ( 0,-1), (0, 0), ( 0,1),
                        ( 1,-1), ( 1,0), ( 1,1)]
                neigh_sum = np.zeros_like(data_cur, dtype=np.float32)
                for dy, dx in shifts:
                    window = padded[1+dy : 1+dy + h,
                                                1+dx : 1+dx + w]
                    window[(np.isnan(window)) | (window <= 0)] = 0
                    neigh_sum += window
                neigh_sum /= 9.0
                
                data+= neigh_sum
            else:
                print(f'Warning: File {file_dir} not found')

        # Lưu file kết quả
        np.save(os.path.join(output_dir, date + f'.npy'), data)
        
        
def preprocess_region_2():
    fold_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/raw_region_2'
    output_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_2'
    os.makedirs(output_dir, exist_ok=True)

    # Tạo danh sách ngày từ 2024-01-01 đến 2024-12-31
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)

    dates = []
    current_date = start_date
    day_steps = [3, 4]  # bước nhảy 3-4 ngày xen kẽ trước 11/11
    step_index = 0

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        if current_date >= datetime.date(2024, 11, 11):
            current_date += datetime.timedelta(days=2)
        else:
            current_date += datetime.timedelta(days=day_steps[step_index])
            step_index = 1 - step_index

    h = 34
    w = 54

    for date in tqdm(dates, desc="Processing dates"):
        timestep = 15
        data = np.zeros((20, timestep, h, w))
        s = copy.copy(date)[4:]  # MMDD
        
        for year in range(2004, 2024):
            for lt in range(-timestep, 0):
                
                temp_date = datetime.datetime.strptime(date, "%Y%m%d")
                year_diff = 2024 - year

                try:
                    temp_date = temp_date.replace(year=temp_date.year - year_diff)
                except ValueError:
                    # Nếu gặp 29/2 mà năm mới không nhuận → dùng 28/2
                    temp_date = temp_date.replace(year=temp_date.year - year_diff, day=28)
                cur_date = temp_date + datetime.timedelta(days=lt)
                
                
                file_dir = f'{fold_dir}/{cur_date.strftime("%Y%m%d")}.npy'

                if os.path.exists(file_dir):
                    # Load file với mmap để tránh chiếm RAM quá nhiều
                    data_temp = np.load(file_dir, mmap_mode='r')
                    # Lấy vùng h x w bằng vectorized indexing
                    data_cur = data_temp.copy()
                    
                    data_cur[(np.isnan(data_cur)) | (data_cur <= 0)] = 0
                    data[year - 2004, (lt+timestep)] = data_cur
                else:
                    print(f'Warning: File {file_dir} not found')

        # Lưu file kết quả
        np.save(os.path.join(output_dir, s[:2] + '-' + s[2:] + '.npy'), data)

def cut_region_2():
    fold_dir = '/mnt/disk3/longnd/env_data/GSMaP/npy'
    output_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/raw_region_2'
    os.makedirs(output_dir, exist_ok=True)
    # Thông số lưới
    LAT_MIN, LAT_MAX = 20.25, 23.55  # max -> min
    LON_MIN, LON_MAX = 103.25, 108.55  # min -> max
    
    step_esp = 0.1
    h = 34
    w = 54
    start_date = datetime.date(2004, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    dates = []
    current_date = start_date

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        
        current_date += datetime.timedelta(days=1)
        

    lat_gsm_begin, lat_gsm_end = 24, 8
    lon_gsm_begin, lon_gsm_end = 102, 117
    # Tạo mảng lat/lon cho grid (vectorized)
    lat_idx = np.floor((lat_gsm_begin - (LAT_MAX - step_esp * np.arange(h)[:, None])) / step_esp).astype(int)
    lon_idx = np.floor((LON_MIN + step_esp * np.arange(w)[None, :] - lon_gsm_begin) / step_esp).astype(int)
    for date in tqdm(dates):
        
        data = np.zeros((h, w))
        for hour in range(24):
            file_dir = f'{fold_dir}/{date}_{hour:02d}00.npy'

            if os.path.exists(file_dir):
                # Load file với mmap để tránh chiếm RAM quá nhiều
                data_temp = np.load(file_dir, mmap_mode='r')
                # Lấy vùng h x w bằng vectorized indexing
                data_cur = data_temp[lat_idx, lon_idx].copy()
                padded = np.pad(data_cur, ((1,1), (1,1)), mode='reflect')
                shifts = [(-1,-1), (-1,0), (-1,1),
                        ( 0,-1), (0, 0), ( 0,1),
                        ( 1,-1), ( 1,0), ( 1,1)]
                neigh_sum = np.zeros_like(data_cur, dtype=np.float32)
                for dy, dx in shifts:
                    window = padded[1+dy : 1+dy + h,
                                                1+dx : 1+dx + w]
                    window[(np.isnan(window)) | (window <= 0)] = 0
                    neigh_sum += window
                neigh_sum /= 9.0
                
                data+= neigh_sum
            else:
                print(f'Warning: File {file_dir} not found')

        # Lưu file kết quả
        np.save(os.path.join(output_dir, date + f'.npy'), data)
if __name__ == "__main__":
    # cut_region_2()
    preprocess_region_2()
