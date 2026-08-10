from netCDF4 import Dataset, num2date
import os
import numpy as np
from tqdm import tqdm

params_dict = {143: "cp",
               146:"sshf",
               147:"slhf",
               175:"strd",
               176:"ssr",
               177:"str",
               179:"ttr",
               174008:"sro",
               121: "mx2t6",
               122: "mn2t6",
               165: "u10",
               166: "v10",
               228: "tp"}

different_dim_param = [121, 122, 165, 166, 228]

def read_nc():
    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"

    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk3/longnd/env_data/S2S_0.125_small_crop/nparr/Step24h"

    for day in tqdm(list_day):
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            core_data = variables[params_dict[param]][:]
            data_shape = core_data.shape
            reshaped_data = core_data.reshape(20,47,137,121)
            
            list_data.append(reshaped_data)
        new_arr = np.stack(list_data, 1)
        median_value = np.ma.median(new_arr)
        new_arr = np.ma.filled(new_arr, median_value) 
        #Save data as npy
        os.makedirs(f"{saved_path}/Step24h", exist_ok=True)
        np.save(f"{saved_path}/Step24h/{day[:-3]}.npy", new_arr)
    
def read_nc_region_1():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20.75, 22.75
    LON_MIN, LON_MAX = 102.75, 104.75

    folder_path = f"/mnt/disk3/longnd/env_data/grid_base/preprocessed_data"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk3/longnd/env_data/S2S_0.125_small_crop/nparr_reg_1"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            try:
                data = Dataset(f"{folder_path}/{param}/{day}")
                variables = data.variables
                
                # Lấy tọa độ latitude và longitude từ file .nc
                latitudes = variables['latitude'][:]
                longitudes = variables['longitude'][:]
                
                # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
                lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
                lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
                
                lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
                lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
                
                # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
                core_data = variables[params_dict[param]][:]
                # if param in different_dim_param:
                #     filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                #                             lon_indices[0]:lon_indices[-1]+1, :]
                #     filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
                # else:
                #     filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                #                             lon_indices[0]:lon_indices[-1]+1]
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                             lon_indices[0]:lon_indices[-1]+1]
                
                # Reshape dữ liệu đã lọc
                n_lat = len(lat_indices)
                n_lon = len(lon_indices)
                reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 37, 29)
                median_for_this_param = np.ma.median(reshaped_data)
                if np.ma.is_masked(median_for_this_param):
                    # Nếu không có dữ liệu hợp lệ nào để tính median, ta dùng giá trị mặc định là 0
                    print(f"    Cảnh báo: Toàn bộ dữ liệu cho {day}/{param} trong vùng đã chọn bị thiếu. Điền bằng 0.")
                    median_for_this_param = 0
                reshaped_data = np.ma.filled(reshaped_data, median_for_this_param)
                list_data.append(reshaped_data)
            
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc {day}: {e}")
                # return  # hoặc continue
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_1_6h():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20.75, 22.75
    LON_MIN, LON_MAX = 102.75, 104.75

    folder_path = f"/mnt/disk3/longnd/env_data/grid_base/preprocessed_data"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk3/longnd/env_data/grid_base/nparr_reg1"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step6h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step6h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            try:
                data = Dataset(f"{folder_path}/{param}/{day}")
                variables = data.variables
                
                # Lấy tọa độ latitude và longitude từ file .nc
                latitudes = variables['latitude'][:]
                longitudes = variables['longitude'][:]
                
                # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
                lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
                lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
                
                lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
                lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
                
                # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
                core_data = variables[params_dict[param]][:]
                
                # if param in different_dim_param:
                #     filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                #                             lon_indices[0]:lon_indices[-1]+1, :]
                #     filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
                # else:
                #     filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                #                             lon_indices[0]:lon_indices[-1]+1]
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                             lon_indices[0]:lon_indices[-1]+1]
                # Reshape dữ liệu đã lọc
                n_lat = len(lat_indices)
                n_lon = len(lon_indices)
                reshaped_data = filtered_data.reshape(20, 184, n_lat, n_lon) # (20, 47, 37, 29)
                median_for_this_param = np.ma.median(reshaped_data)
                if np.ma.is_masked(median_for_this_param):
                    # Nếu không có dữ liệu hợp lệ nào để tính median, ta dùng giá trị mặc định là 0
                    print(f"    Cảnh báo: Toàn bộ dữ liệu cho {day}/{param} trong vùng đã chọn bị thiếu. Điền bằng 0.")
                    median_for_this_param = 0
                reshaped_data = np.ma.filled(reshaped_data, median_for_this_param)
                list_data.append(reshaped_data)
            
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc {day}: {e}")
                # return  # hoặc continue
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")    
def read_nc_region_2():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20.75, 23
    LON_MIN, LON_MAX = 103.75, 108

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_2"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            try:
                data = Dataset(f"{folder_path}/{param}/{day}")
                variables = data.variables
                
                # Lấy tọa độ latitude và longitude từ file .nc
                latitudes = variables['latitude'][:]
                longitudes = variables['longitude'][:]
                
                # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
                lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
                lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
                
                lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
                lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
                
                # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
                core_data = variables[params_dict[param]][:]
                if param in different_dim_param:
                    filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                            lon_indices[0]:lon_indices[-1]+1, :]
                    filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
                else:
                    filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                            lon_indices[0]:lon_indices[-1]+1]
                
                # Reshape dữ liệu đã lọc
                n_lat = len(lat_indices)
                n_lon = len(lon_indices)
                reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 37, 29)
                
                list_data.append(reshaped_data)
            
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc {day}: {e}")
                # return  # hoặc continue
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_3():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20, 21.5
    LON_MIN, LON_MAX = 105, 107.75

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_3"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 13, 23)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_4():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 16, 20.5
    LON_MIN, LON_MAX = 104.25, 107.75

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_4"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            try:
                data = Dataset(f"{folder_path}/{param}/{day}")
                variables = data.variables
                
                # Lấy tọa độ latitude và longitude từ file .nc
                latitudes = variables['latitude'][:]
                longitudes = variables['longitude'][:]
                
                # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
                lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
                lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
                
                lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
                lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
                
                # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
                core_data = variables[params_dict[param]][:]
                if param in different_dim_param:
                    filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                            lon_indices[0]:lon_indices[-1]+1, :]
                    filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
                else:
                    filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                            lon_indices[0]:lon_indices[-1]+1]
                
                # Reshape dữ liệu đã lọc
                n_lat = len(lat_indices)
                n_lon = len(lon_indices)
                reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 37, 29)
                
                list_data.append(reshaped_data)
            
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc {day}: {e}")
                # return  # hoặc continue
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_5():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 10.5, 16.25
    LON_MIN, LON_MAX = 108, 109.5

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_5"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 47, 13)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_6():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 11.5, 14.75
    LON_MIN, LON_MAX = 106.75, 109

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_6"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 27, 19)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_7():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 8.5, 11.5
    LON_MIN, LON_MAX = 103.75, 112

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_7"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 25, 67)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")


def read_nc_month_6_7_8_9():
    input_folder = "/mnt/disk3/longnd/env_data/S2S_0.125_small_crop/nparr/Step24h"
    output_folder = "/mnt/disk3/longnd/env_data/S2S_0.125_small_crop/nparr_6789/Step24h"

    # Get all .npy files from input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    # Filter files for months 6, 7, 8, 9
    month_files = []
    for file_name in npy_files:
        # Extract month from filename (e.g., "06-01.npy" -> "06")
        month = file_name[:2]
        if month in ['04', '05']:
            month_files.append(file_name)
    
    # Sort files by date
    month_files = sorted(month_files)
    
    for file_name in tqdm(month_files):
        # Load the .npy file
        data = np.load(os.path.join(input_folder, file_name))  # Shape: (20, 13, 46, 137, 121)
        
        # Extract date from filename (e.g., "06-01.npy" -> "06-01")
        date_str = file_name[:-4]
        
        # Process each lead time
        lead_times = list(range(7, 47))
        for lead_time in tqdm(lead_times):
            # Process each year
            for year in range(2004, 2024):
                # Create output directory if it doesn't exist
                output_dir = os.path.join(output_folder, str(lead_time), str(year))
                os.makedirs(output_dir, exist_ok=True)
                
                # Create output file path
                output_file = os.path.join(output_dir, f"{date_str}.npy")
                
                # Skip if file already exists
                if os.path.exists(output_file):
                    continue
                
                # Extract data for this lead time and year
                # data shape: (20, 13, 46, 137, 121)
                # We want to get data for specific year and lead time window
                year_idx = year - 2004
                lead_time_data = data[year_idx, :, lead_time-7:lead_time, :, :]  # Shape: (13, 7, 137, 121)
                
                # Save the data
                np.save(output_file, lead_time_data)

read_nc_region_1_6h()