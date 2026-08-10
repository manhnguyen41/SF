import pickle
import pandas as pd
import numpy as np
import joblib
# Đường dẫn đến file scaler của bạn
scaler_file_path = '/mnt/disk3/tunm/Subseasonal_Forecasting/data2/data6789_reg_1_seed52_new_short/output_scaler.pkl'

# Mở và tải scaler
try:
    with open(scaler_file_path, 'rb') as file:
        loaded_scaler = joblib.load(scaler_file_path)
    print("Scaler đã được tải thành công!")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn: {scaler_file_path}")
    exit()
except Exception as e:
    print(f"Lỗi khi tải scaler: {e}")
    exit()
    # Tạo dữ liệu mẫu (ví dụ: một DataFrame hoặc mảng NumPy)
# Đảm bảo số lượng cột và thứ tự cột tương ứng với dữ liệu mà scaler đã được huấn luyện
# Ví dụ, nếu scaler được huấn luyện trên 5 đặc trưng, dữ liệu mới của bạn cũng phải có 5 đặc trưng.
new_data = [[0], [0], [0], [0]]
new_data = np.array(new_data)
print("\nDữ liệu gốc cần biến đổi:")
print(new_data)

# Biến đổi dữ liệu mới
transformed_data = loaded_scaler.inverse_transform(new_data)
# transformed_data = loaded_scaler.transform(new_data)
# Nếu bạn muốn chuyển lại về DataFrame để dễ nhìn
transformed_df = pd.DataFrame(transformed_data)

print("\nDữ liệu sau khi biến đổi:")
print(transformed_df)