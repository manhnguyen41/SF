import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

fold_dir = '/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_1'
files = sorted([os.path.join(fold_dir, f) for f in os.listdir(fold_dir) if f.endswith('.npy') and f[1] in ['6', '7', '8', '9']])

print(f"Tổng số file: {len(files)}")

shapes = []
global_min, global_max = float("inf"), float("-inf")
global_sum, global_sqsum, global_count = 0.0, 0.0, 0
all_values_sampled = []   # <-- thêm list để lấy mẫu giá trị cho histogram

for f in tqdm(files, desc="Đang đọc dữ liệu"):
    arr = np.load(f)
    print(arr.shape)
    shapes.append(arr.shape)

    # Cập nhật thống kê
    global_min = min(global_min, arr.min())
    global_max = max(global_max, arr.max())
    global_sum += arr.sum()
    global_sqsum += (arr**2).sum()
    global_count += arr.size

    # Lấy mẫu ngẫu nhiên ~0.5% giá trị mỗi file để vẽ histogram (tránh quá nặng)
    n_samp = max(1, int(arr.size * 0.005))
    all_values_sampled.append(
        np.random.choice(arr.flatten(), size=n_samp, replace=False)
    )

# Gom toàn bộ mẫu lại
sampled_values = np.concatenate(all_values_sampled)

# Thống kê shapes
unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)
shape_stats = {tuple(s): int(c) for s, c in zip(unique_shapes, counts)}

# Mean & Std
global_mean = global_sum / global_count
global_std = np.sqrt(global_sqsum / global_count - global_mean**2)

print("\n📊 Thống kê dữ liệu:")
print(f"- Số file: {len(files)}")
print(f"- Các shape khác nhau và số lượng:")
for shape, c in shape_stats.items():
    print(f"  + Shape {shape}: {c} files")
print(f"- Giá trị min: {global_min}")
print(f"- Giá trị max: {global_max}")
print(f"- Giá trị mean: {global_mean:.4f}")
print(f"- Giá trị std: {global_std:.4f}")

# ====== Vẽ histogram & lưu ảnh ======
plt.figure(figsize=(8,5))
plt.hist(sampled_values, bins=100, color='steelblue', alpha=0.8)
plt.axvline(global_mean, color='red', linestyle='--', label=f'Mean {global_mean:.2f}')
plt.title("Histogram phân bố giá trị (mẫu)")
plt.xlabel("Giá trị")
plt.ylabel("Tần suất")
plt.legend()
plt.tight_layout()

out_path = "data_explore/gsm_distribution_hist.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"✅ Đã lưu histogram tại: {out_path}")