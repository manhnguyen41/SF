from netCDF4 import Dataset
import numpy as np

file_path = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step24h/143/01-01.nc"

try:
    with Dataset(file_path, mode='r') as nc_file:
        
        print("--- Các biến dữ liệu có trong file ---")
        print(list(nc_file.variables.keys()))
        print("\n" + "="*50 + "\n")

        # THAY ĐỔI Ở ĐÂY: Sử dụng tên biến 'cp' thay vì 't2m'
        variable_name = 'cp' 

        if variable_name in nc_file.variables:
            target_variable = nc_file.variables[variable_name]
            
            print(f"--- Thông tin về biến '{variable_name}' ---")
            print(target_variable)
            print("\n" + "="*50 + "\n")

            # Đọc dữ liệu của biến 'cp' vào mảng NumPy
            data = target_variable[:]
            
            print(f"Đã đọc thành công dữ liệu của biến '{variable_name}'.")
            print("Shape của mảng dữ liệu:", data.shape)
            print("Kiểu dữ liệu:", data.dtype)
            
            # In ra một vài giá trị để kiểm tra
            # Giả sử shape là (time, latitude, longitude)
            print("Giá trị tại điểm [0, 0, 0]:", data[0, 0, 0])

        else:
            # Dòng này sẽ không chạy nữa vì 'cp' chắc chắn có trong file
            print(f"Lỗi: Không tìm thấy biến '{variable_name}' trong file.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")