import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import os
from tqdm import tqdm
import xarray as xr
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

def get_station_from_grid(y_pred, y):
    # y_pred: (h, w) 
    # y: (num_station, 3) 
    num_station = y.shape[0]
    lat_start = 25
    lon_start = 100
    step = 0.125  

    stations_lon = y[:, 1]  # (num_station)
    stations_lat = y[:, 2]  # (num_station)

    lat_idx = torch.floor((lat_start - stations_lat) / step).long()  # (num_station)
    lon_idx = torch.floor((stations_lon - lon_start) / step).long()  # (num_station)

    num_lat, num_lon = 17, 17
    lat_idx = torch.clamp(lat_idx, 0, num_lat - 1)  # (num_station)
    lon_idx = torch.clamp(lon_idx, 0, num_lon - 1)  # (num_station)
    
    pred_values = y_pred[lat_idx, lon_idx]  # (num_station)
    pred_values = pred_values.unsqueeze(-1)  # (num_station, 1)
    
    return pred_values

from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)

def convert_ecmwf_to_csv_station():
    idx_path = '/mnt/disk3/tunm/Subseasonal_Forecasting/data3/data6789_reg_1_seed52/test.csv'
    idx_df = pd.read_csv(idx_path).values
    processed_ecmwf_dir = '/mnt/disk3/longnd/env_data/S2S_0.125_old/data3_reg_1/processed_ecmwf/test'
    
    gauge_path = '/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_Region_1.csv'
    scaler_path = '/mnt/disk3/tunm/Subseasonal_Forecasting/data3/data6789_reg_1_seed52/scalers.pkl'
    output_csv = f'/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/ecmwf_temp.csv'
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Đọc dữ liệu gauge
    gauge_data = pd.read_csv(gauge_path)
    gauge_data['Day'] = pd.to_datetime(gauge_data['Day'])
    
    # Lấy thông tin trạm
    station_data = gauge_data[['Station', 'Lon', 'Lat']].drop_duplicates('Station')
    stations = station_data['Station'].values
    station_coords = station_data[['Lon', 'Lat']].to_numpy()
    
    # Load scaler từ file .pkl
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            ecmwf_scaler = pickle.load(f)
        print(f"Loaded ecmwf_scaler from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    def get_ground_truth(year, month, day, lead_time):
        start_date = datetime(year, month, day) + timedelta(days=lead_time - 6)
        end_date = datetime(year, month, day) + timedelta(days=lead_time)
        mask = (gauge_data['Day'] >= start_date) & (gauge_data['Day'] <= end_date)
        period_data = gauge_data.loc[mask]
        num_station = len(stations)
        total_rain = np.zeros((num_station, 1))
        for i, station in enumerate(stations):
            station_data = period_data[period_data['Station'] == station]
            total_rain[i, 0] = station_data['R'].sum()
        result = np.hstack((total_rain, station_coords))
        return result
    
    def inverse_transform_ecmwf_feature_12(ecmwf_data, scaler):
        """
        Reverse transform chỉ cho feature 12 (đặc trưng mưa) trong dữ liệu ECMWF.
        ecmwf_data: numpy array có shape (7, 13, 137, 121)
        scaler: scaler cho feature 12
        """
        ecmwf_data_inv = ecmwf_data.copy()
        rain_feature = ecmwf_data[:, 12, :, :].reshape(-1, 1)  # (7 * 137 * 121, 1)
        inv_rain = scaler.inverse_transform(rain_feature)  # Đảo ngược chuẩn hóa
        ecmwf_data_inv[:, 12, :, :] = inv_rain.reshape(7, 17, 17)
        return ecmwf_data_inv
    
    data = []
    
    for idx in tqdm(range(len(idx_df)), desc="Processing ecmwf_data"):
        ecmwf_path, lead_time, year, month, day = idx_df[idx]
        processed_ecmwf_path = f'{processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
        ecmwf_data = np.load(processed_ecmwf_path)  # (7, 13, 137, 121)
        
        # Reverse transform chỉ cho feature 12
        ecmwf_data = inverse_transform_ecmwf_feature_12(ecmwf_data, ecmwf_scaler[12])
        
        rain_data = ecmwf_data[:, 12, :, :]  # (7, 137, 121) - lấy đặc trưng mưa
        rain_data = np.sum(rain_data, axis=0)  # (137, 121) - tổng mưa theo thời gian
        
        y_grt = get_ground_truth(year, month, day, lead_time)  # (num_station, 3)
        
        if isinstance(rain_data, np.ndarray):
            rain_data = torch.from_numpy(rain_data)
        if isinstance(y_grt, np.ndarray):
            y_grt = torch.from_numpy(y_grt)
        
        y_prd = get_station_from_grid(rain_data, y_grt)  # (num_station, 1)

        # Chuyển y_grt sang numpy nếu nó là tensor và lấy cột đầu tiên
        if isinstance(y_grt, torch.Tensor):
            y_grt = y_grt.numpy()
        elif isinstance(y_grt, np.ndarray):
            y_grt = y_grt[:, 0]  # Lấy cột đầu tiên (total_rain)
        
        # Đảm bảo y_prd và y_grt là numpy arrays và loại bỏ chiều dư thừa
        if isinstance(y_prd, torch.Tensor):
            y_prd = y_prd.numpy()
        if isinstance(y_grt, torch.Tensor):
            y_grt = y_grt.numpy()
        
        y_prd = np.squeeze(y_prd)  # (num_station,)
        y_grt = np.squeeze(y_grt)  # (num_station,)
        
        # Thêm dữ liệu vào list theo định dạng CSV
        row = {
            'Prediction': y_prd,
            'Groundtruth': y_grt[0],
            'station': 355.0,
            'lead_time': lead_time,
            'year': year,
            'month': month,
            'day': day
        }
        data.append(row)
            
    
    # Tạo DataFrame và lưu vào CSV
    df_output = pd.DataFrame(data)
    df_output.to_csv(output_csv, index=False)
    
    print(f"Đã tạo file {output_csv} thành công!")    


convert_ecmwf_to_csv_station()