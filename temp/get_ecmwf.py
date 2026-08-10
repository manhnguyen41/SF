import numpy as np
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_feature_distribution_by_split(
    folder_path,
    num_features=13,
    sample_fraction=0.01,
    splits=( (0,13,'Train'), (13,16,'Val'), (16,19,'Test') )
):
    """
    Phân tích và trực quan hóa phân phối feature cho từng tập Train/Val/Test
    trong chiều 0 của mỗi file .npy.

    Parameters
    ----------
    folder_path : str
        Đường dẫn chứa các file .npy
    num_features : int
        Số feature (chiều 1)
    sample_fraction : float
        Tỷ lệ lấy mẫu ngẫu nhiên cho histogram
    splits : list(tuple)
        Danh sách các tuple (start_idx, end_idx, split_name)
    """
    if not os.path.isdir(folder_path):
        print(f"Không tồn tại thư mục: {folder_path}")
        return

    file_paths = glob.glob(os.path.join(folder_path, "*.npy"))
    if not file_paths:
        print("Không tìm thấy file .npy nào.")
        return

    print(f"Tìm thấy {len(file_paths)} file. Bắt đầu phân tích...")

    # Tạo dict: split_name -> list các list(feature)
    sampled_data = {
        sname: [ [] for _ in range(num_features) ]
        for _, _, sname in splits
    }

    for file_path in tqdm(file_paths, desc="Đang xử lý các file"):
        try:
            data_array = np.load(file_path)  # shape: (T, F, H, W)
            if data_array.ndim != 5 or data_array.shape[1] != num_features:
                print(f"Bỏ qua {file_path}, shape không hợp lệ {data_array.shape}")
                continue

            for start, end, sname in splits:
                if end > data_array.shape[0]:
                    # nếu file ngắn hơn dự kiến, bỏ qua phần dư
                    end = data_array.shape[0]
                # cắt theo chiều 0
                split_data = data_array[start:end]  # (len_split, F, H, W)

                for i in range(num_features):
                    feature_vals = split_data[:, i, :, :].flatten()
                    n_samp = int(len(feature_vals) * sample_fraction)
                    if n_samp > 0:
                        samples = np.random.choice(feature_vals, size=n_samp, replace=False)
                        sampled_data[sname][i].extend(samples)

        except Exception as e:
            print(f"Lỗi với file {os.path.basename(file_path)}: {e}")

    # ---- Tính toán thống kê ----
    all_stats = []
    for sname in sampled_data:
        for i in range(num_features):
            arr = np.array(sampled_data[sname][i])
            if len(arr) == 0:
                stat = {'split': sname, 'feature': i, 'count': 0}
            else:
                stat = {
                    'split': sname,
                    'feature': i,
                    'count': len(arr),
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    '25%': np.percentile(arr, 25),
                    '50% (median)': np.percentile(arr, 50),
                    '75%': np.percentile(arr, 75),
                    'max': np.max(arr),
                    'skewness': pd.Series(arr).skew()
                }
            all_stats.append(stat)

    stats_df = pd.DataFrame(all_stats)
    print("\n===== Bảng thống kê từng tập =====")
    print(stats_df.to_string(index=False))

    # ---- Vẽ histogram ----
    for sname in sampled_data:
        print(f"\n--- Vẽ biểu đồ phân phối: {sname} ---")
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols,
                                 figsize=(num_cols * 5, num_rows * 4))
        axes = axes.flatten()
        for i in range(num_features):
            ax = axes[i]
            vals = pd.Series(sampled_data[sname][i], name=f"Feature {i}")
            if not vals.empty:
                sns.histplot(vals, kde=True, ax=ax)
                skew = stats_df[
                    (stats_df['split']==sname) & (stats_df['feature']==i)
                ]['skewness'].values[0]
                ax.set_title(f"Feature {i}\nSkew: {skew:.2f}")
                ax.set_xlabel("Giá trị")
                ax.set_ylabel("Tần suất")
            else:
                ax.set_title(f"Feature {i}\n(Không dữ liệu)")
        for j in range(num_features, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"data_explore/hist_{sname}.png", dpi=150)
        plt.close()

# --- Cách sử dụng ---
folder_to_analyze = "/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h"
analyze_feature_distribution_by_split(folder_to_analyze)
