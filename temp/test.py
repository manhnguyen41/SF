import xarray as xr
import numpy as np
file = '/mnt/disk3/longnd/env_data/grid_base/preprocessed_data/143/01-01.nc'
data = np.load('/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h/06-06.npy')
print(data.shape)
# Mở lười (lazy), chưa load vào RAM
ds = xr.open_dataset(file)  # có thể thêm engine="netcdf4" nếu cần
print(ds)                   # tóm tắt: dims, coords, data_vars

# Xem kích thước từng chiều, danh sách biến
print("dims:", ds.dims)
print("sizes:", ds.sizes)
print("data_vars:", list(ds.data_vars))

# Ví dụ: lấy một biến (đổi 'tp' thành biến thực tế trong file)
var_name = list(ds.data_vars)[0]   # lấy biến đầu tiên để thử
da = ds[var_name]                  # DataArray
print(var_name, da.shape, da.dtype)

# Lấy về numpy (cẩn thận RAM nếu dữ liệu lớn)
arr = da.values  # hoặc da.to_numpy()
print("numpy shape:", arr.shape)
