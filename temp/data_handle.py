import os
import glob
import xarray as xr
from tqdm import tqdm
import numpy as np
def process_mx2t6_data():
    param = 'mx2t6'
    input_folder = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step6h/121"
    output_folder = "/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/121"

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc="Processing files for mx2t6"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            # continue

        ds = xr.open_dataset(file_path)
        time_group_size = 1
        all_years_data = []
        
        for year in range(20):
            ds_year = ds.isel(time=slice(year*184, year*184+184))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            # 25 to 8, 100 to 115
            ds_reshaped = ds_year.coarsen(time=time_group_size, boundary="trim").max()
            new_time = ds_year.time[time_group_size-1::time_group_size]
            ds_final = ds_reshaped.assign_coords(time=new_time)

            # Chuyển từ độ K thành độ C
            ds_final_corrected = ds_final.copy()  
            ds_final_corrected[param] = ds_final_corrected[param] - 273.15 
            # print(ds_final_corrected.shape) 
            all_years_data.append(ds_final_corrected) 

        ds_all_years = xr.concat(all_years_data, dim='time')
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
            # print(ds_all_years[var].shape)
        # ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_mn2t6_data():
    param = 'mn2t6'
    input_folder = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step6h/122"
    output_folder = "/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/122"

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc="Processing files for mn2t6"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            continue

        ds = xr.open_dataset(file_path)
        time_group_size = 1
        years = ds['time'].dt.year.values
        all_years_data = []

        for year in range(20):
            ds_year = ds.isel(time=slice(year*184, year*184+184))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            ds_reshaped = ds_year.coarsen(time=time_group_size, boundary="trim").min()
            new_time = ds_year.time[time_group_size-1::time_group_size]
            ds_final = ds_reshaped.assign_coords(time=new_time)

            # Chuyển từ độ K thành độ C
            ds_final_corrected = ds_final.copy()  
            ds_final_corrected[param] = ds_final_corrected[param] - 273.15  
            all_years_data.append(ds_final_corrected) 

        ds_all_years = xr.concat(all_years_data, dim='time')
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
        ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_10u_data():
    input_folder = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step6h/165"
    output_folder = "/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/165"

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc="Processing files for 10u"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            continue

        ds = xr.open_dataset(file_path)
        time_group_size = 1
        years = ds['time'].dt.year.values
        all_years_data = []

        for year in range(20):
            ds_year = ds.isel(time=slice(year*185, year*185+185))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            ds_no_time0 = ds_year.isel(time=slice(1, None))
            ds_reshaped = ds_no_time0.coarsen(time=time_group_size, boundary="trim").mean()
            new_time = ds_no_time0.time[time_group_size-1::time_group_size]
            ds_final = ds_reshaped.assign_coords(time=new_time)
            all_years_data.append(ds_final)

        ds_all_years = xr.concat(all_years_data, dim='time')
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
        ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_10v_data():
    input_folder = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step6h/166"
    output_folder = "/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/166"

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc="Processing files for 10v"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            continue

        ds = xr.open_dataset(file_path)
        time_group_size = 1
        years = ds['time'].dt.year.values
        all_years_data = []

        for year in range(20):
            ds_year = ds.isel(time=slice(year*185, year*185+185))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            ds_no_time0 = ds_year.isel(time=slice(1, None))
            ds_reshaped = ds_no_time0.coarsen(time=time_group_size, boundary="trim").mean()
            new_time = ds_no_time0.time[time_group_size-1::time_group_size]
            ds_final = ds_reshaped.assign_coords(time=new_time)
            all_years_data.append(ds_final)

        ds_all_years = xr.concat(all_years_data, dim='time')
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
        ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_tp_data():
    param = 'tp'
    input_folder = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step6h/228"
    output_folder = "/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/228"

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc="Processing files for tp"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            continue

        ds = xr.open_dataset(file_path)
        time_group_size = 1
        years = ds['time'].dt.year.values
        all_years_data = []

        for year in range(20):
            ds_year = ds.isel(time=slice(year*185, year*185+185))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            ds_no_time0 = ds_year.isel(time=slice(1, None))
            ds_reshaped = ds_no_time0.coarsen(time=time_group_size, boundary="trim").max()
            new_time = ds_no_time0.time[time_group_size-1::time_group_size]
            ds_final = ds_reshaped.assign_coords(time=new_time)
            ds_final['tp'].values[1:, :, :] = ds_final['tp'].values[1:, :, :] - ds_final['tp'].values[:-1, :, :]
            
            all_years_data.append(ds_final)

        ds_all_years = xr.concat(all_years_data, dim='time')
        ds_all_years = ds_all_years.where(ds_all_years['tp'] > 0, 0)
        ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= 0, 0)
        threshold = 250 / 2
        ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= threshold, threshold)
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
        ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_cp():
    number = 143
    param = 'cp'
    input_folder = f"/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step24h/{number}"
    output_folder = f"/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/{number}"

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc=f"Processing files for {param} ({number})"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            continue

        ds = xr.open_dataset(file_path).load()
        years = ds['time'].dt.year.values
        all_years_data = []

        for year in range(20):
            ds_year = ds.isel(time=slice(year*47, year*47+47))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            ds_year[param].values[1:, :, :] = (ds_year[param].values[1:, :, :] - ds_year[param].values[:-1, :, :]) / 4
            ds_year = ds_year.isel(time=slice(1, None))
            ds_year = ds_year.isel(time=np.repeat(np.arange(ds_year.sizes['time']), 4))
            
            all_years_data.append(ds_year)

        ds_all_years = xr.concat(all_years_data, dim='time')
        ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= 0, 0)
        threshold = 56 / 2
        ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= threshold, threshold)
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
        ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_sro():
    number = 174008
    param = 'sro'
    input_folder = f"/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step24h/{number}"
    output_folder = f"/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/{number}"
    encoding = {
        param: {
            "dtype": "float64",
            "zlib": True,
            "complevel": 4
        }
    }

    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

    for file_path in tqdm(file_paths, desc=f"Processing files for {param} ({number})"):
        output_filename = os.path.join(output_folder, os.path.basename(file_path))
        if os.path.exists(output_filename):
            print(f"File {output_filename} đã tồn tại, bỏ qua.")
            continue

        ds = xr.open_dataset(file_path).load()
        years = ds['time'].dt.year.values
        all_years_data = []

        for year in range(20):
            ds_year = ds.isel(time=slice(year*47, year*47+47))
            ds_year = ds_year.where(
                (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                drop=True
            )
            ds_year[param].values[1:, :, :] = (ds_year[param].values[1:, :, :] - ds_year[param].values[:-1, :, :]) / 4
            ds_year = ds_year.isel(time=slice(1, None))
            ds_year = ds_year.isel(time=np.repeat(np.arange(ds_year.sizes['time']), 4))
            all_years_data.append(ds_year)

        ds_all_years = xr.concat(all_years_data, dim='time')
        ds_all_years.fillna(0)
        ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= 0, 0)
        threshold = 24 / 2
        ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= threshold, threshold)
        for var in ds_all_years.data_vars:
            ds_all_years[var] = ds_all_years[var].astype('float32')
        ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
        print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_all_radiation_data():
    params = ['sshf', 'slhf', 'strd', 'ssr', 'str', 'ttr']
    numbers = [146, 147, 175, 176, 177, 179]

    for param, number in zip(params, numbers):
        input_folder = f"/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step24h/{number}"
        output_folder = f"/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/{number}"
        encoding = {
            param: {
                "dtype": "float64",
                "zlib": True,
                "complevel": 4
            }
        }

        os.makedirs(output_folder, exist_ok=True)
        file_paths = glob.glob(os.path.join(input_folder, "*.nc"))

        for file_path in tqdm(file_paths, desc=f"Processing files for {param} ({number})"):
            output_filename = os.path.join(output_folder, os.path.basename(file_path))
            if os.path.exists(output_filename):
                print(f"File {output_filename} đã tồn tại, bỏ qua.")
                continue

            ds = xr.open_dataset(file_path).load()
            years = ds['time'].dt.year.values
            all_years_data = []

            for year in range(20):
                ds_year = ds.isel(time=slice(year*47, year*47+47))
                ds_year = ds_year.where(
                    (ds_year.latitude >= 8) & (ds_year.latitude <= 25) &
                    (ds_year.longitude >= 100) & (ds_year.longitude <= 115),
                    drop=True
                )
                ds_year[param] = -ds_year[param] / 84600
                ds_year[param].values[1:, :, :] = (ds_year[param].values[1:, :, :] - ds_year[param].values[:-1, :, :]) / 4
                
                ds_year = ds_year.isel(time=slice(1, None))
                ds_year = ds_year.isel(time=np.repeat(np.arange(ds_year.sizes['time']), 4))
                all_years_data.append(ds_year)

            ds_all_years = xr.concat(all_years_data, dim='time')
            if number == 146:
                low_threshold = -31 / 2
                high_threshold = 81 / 2
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= low_threshold, low_threshold)
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= high_threshold, high_threshold)
            elif number == 147:
                low_threshold = -1 / 2
                high_threshold = 430 / 2
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= low_threshold, low_threshold)
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= high_threshold, high_threshold)
            elif number == 175:
                high_threshold = -200 / 2
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= high_threshold, high_threshold)
            elif number == 176:
                high_threshold = 0
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] <= high_threshold, high_threshold)
            elif number == 177:
                low_threshold = 0
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= low_threshold, low_threshold)
            elif number == 179:
                low_threshold = 100 / 2
                ds_all_years[param] = ds_all_years[param].where(ds_all_years[param] >= low_threshold, low_threshold)

            for var in ds_all_years.data_vars:
                ds_all_years[var] = ds_all_years[var].astype('float32')
            ds_all_years.to_netcdf(output_filename, encoding={var: {'dtype': 'float32'} for var in ds_all_years.data_vars})
            print(f"Đã lưu kết quả gộp lại trong file {output_filename}")

def process_all_24h_data():
    process_cp()
    process_sro()
    process_all_radiation_data()

def process_all_6h_data():
    process_mx2t6_data()
    process_mn2t6_data()
    process_10u_data()
    process_10v_data()
    process_tp_data()

def process_all_data():
    process_all_6h_data()
    process_all_24h_data()
    

# Gọi hàm để xử lý tất cả dữ liệu
process_mx2t6_data()