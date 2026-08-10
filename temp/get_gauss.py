import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

gauge_path = '/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv'
df = pd.read_csv(gauge_path)

# ====== TẠO CỘT NGÀY (nếu có Year/Month/Day) ======
if {'Year','Month','Day'}.issubset(df.columns):
    df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=df['Day']), errors='coerce')
elif 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
else:
    raise ValueError("Thiếu cột Day/Date để tạo DatetimeIndex.")

# ====== LỌC DỮ LIỆU ======
df = df[(df['Year'] >= 2004) & (df['Year'] <= 2020)].copy()
# df['R'] = df['R'].clip(lower=0, upper=40)
test_years  = [2018, 2019, 2020]
val_years = [2015, 2016, 2017]
print(df.shape)
train_df = df[~df['Year'].isin(val_years + test_years)].copy()
val_df   = df[df['Year'].isin(val_years)].copy()
test_df  = df[df['Year'].isin(test_years)].copy()

# ====== ROLL 7 NGÀY: TÍNH TỔNG MƯA 7D (R_7d) ======
# Nếu có nhiều trạm, rolling theo từng trạm
group_cols = [c for c in ['Station'] if c in df.columns]

def add_7d_total(d: pd.DataFrame) -> pd.DataFrame:
    """
    Yêu cầu cột: 'Station', 'Day', 'R'
    - Sort theo Day trong từng Station
    - Tính tổng lượng mưa 7 ngày (rolling window=7) cho từng Station
    - Scale min–max R_7d về [1, 5] trên toàn bộ dữ liệu
    - Sau đó áp dụng log1p
    Trả về:
      - R_7d_raw: tổng 7 ngày trước khi biến đổi
      - R_7d_scaled: sau khi scale về [1,5]
      - R_7d: log1p(R_7d_scaled) (kết quả cuối)
    """
    d = d.copy()

    # Đảm bảo Day là datetime
    if not np.issubdtype(d['Day'].dtype, np.datetime64):
        d['Day'] = pd.to_datetime(d['Day'], errors='coerce')

    # Nếu có giá trị R âm, kẹp về 0 cho an toàn
    
    #d['R'] = d['R'].clip(lower=0, upper=100)

    # Sort để rolling ổn định
    d = d.sort_values(['Station', 'Day'], kind='mergesort')

    # Tổng trượt 7 ngày theo từng Station
    d['R_7d'] = (
        d.groupby('Station', group_keys=False)['R']
         .transform(lambda x: x.rolling(window=1, min_periods=1).sum())
    )
    
    c = 10.0 
    
    return d
    # Cuối cùng mới log1p
    
    # d['R_7d'] = np.log10(d['R_7d'] + 1)
    # Min–max scale R_7d_raw về [1,5] (toàn bộ dữ liệu, bỏ qua NaN)
    rmin = d['R_7d'].min(skipna=True)
    rmax = d['R_7d'].max(skipna=True)
    if pd.isna(rmin) or pd.isna(rmax):
        # Không có đủ dữ liệu để tính rolling 7 ngày
        d['R_7d_scaled'] = np.nan
    elif rmax == rmin:
        # Trường hợp hằng: đặt giữa khoảng [1,5] = 3
        d['R_7d_scaled'] = np.where(d['R_7d'].notna(), 3.0, np.nan)
    else:
        d['R_7d_scaled'] = 1.0 + 4.0 * (d['R_7d'] - rmin) / (rmax - rmin)

    # log1p sau khi scale về [1,5]  → nằm khoảng [log(2), log(6)]
    R_7d_log1p =  d['R_7d_scaled'] # np.log1p(d['R_7d_scaled'])

    # Cuối cùng scale output về [-1, 1] trên toàn bộ dữ liệu
    y_min = R_7d_log1p.min(skipna=True)
    y_max = R_7d_log1p.max(skipna=True)
    if pd.isna(y_min) or pd.isna(y_max):
        d['R_7d'] = np.nan
    elif y_max == y_min:
        d['R_7d'] = np.where(R_7d_log1p.notna(), 0.0, np.nan)
    else:
        d['R_7d'] = -1.0 + 2.0 * (R_7d_log1p - y_min) / (y_max - y_min)

    return d
    return d

train_df = add_7d_total(train_df)
val_df   = add_7d_total(val_df)
test_df  = add_7d_total(test_df)

# ====== THỐNG KÊ PERCENTILE (trên R gốc, có thể đổi sang R_7d nếu cần) ======
q25  = train_df['R_7d'].quantile(0.25)
q50  = train_df['R_7d'].quantile(0.50)
q75  = train_df['R_7d'].quantile(0.75)
q99  = train_df['R_7d'].quantile(0.99)
q995 = train_df['R_7d'].quantile(0.99995)

print("-" * 40)
print("NGƯỠNG THỐNG KÊ DỰA TRÊN TRAIN (R)")
print("-" * 40)
print(f"P25   (25th):   {q25:.4f} mm")
print(f"P50   (50th):   {q50:.4f} mm")
print(f"P75   (75th):   {q75:.4f} mm")
print(f"P99   (99th):   {q99:.4f} mm")
print(f"P99.5 (99.5th): {q995:.4f} mm")
print("-" * 40)

# (tuỳ chọn) In trung bình tổng mưa 7 ngày
print("TRUNG BÌNH TỔNG MƯA 7 NGÀY (mm):")
print(f"Train: {train_df['R_7d'].mean(skipna=True):.4f}")
print(f"Valid: {val_df['R_7d'].mean(skipna=True):.4f}")
print(f"Test : {test_df['R_7d'].mean(skipna=True):.4f}")
print("-" * 40)

# ====== VẼ HISTOGRAM + BOXPLOT THEO R_7d ======
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

datasets = [
    (f"Train ({train_df['Year'].min()}–{train_df['Year'].max()})", train_df['R_7d'], "steelblue"),
    (f"Validation ({min(val_years)}–{max(val_years)})",            val_df['R_7d'],   "orange"),
    (f"Test ({min(test_years)}–{max(test_years)})",                test_df['R_7d'],  "green"),
]

# Hàng trên: histogram (R_7d)
for i, (title, data, color) in enumerate(datasets):
    ax = axes[0, i]
    ax.hist(data.dropna(), bins=100, color=color, alpha=0.7, density=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_title(f"Histogram – {title}")
    ax.set_ylabel("Tần suất (%)")
    ax.set_xlabel("Tổng mưa 7 ngày (mm)")

# Hàng dưới: boxplot (R_7d)
for i, (title, data, color) in enumerate(datasets):
    ax = axes[1, i]
    ax.boxplot(
        data.dropna(),
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor='black', linewidth=2, alpha=0.8),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none')
    )
    # ax.set_yscale('symlog', linthresh=0.01)
    # Giới hạn trục y phù hợp với tổng 7 ngày (có thể cao hơn R đơn ngày)
    ymax = np.nanpercentile(data, 99.5) * 1.2 if data.notna().any() else 1000
    ax.set_ylim(-1, max(1, ymax))
    ax.set_title(f"Boxplot (Symlog) – {title}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tổng mưa 7 ngày (mm) - Symlog", fontsize=10)
    ax.set_xlabel("Phân chia Dữ liệu")

plt.tight_layout()
plt.savefig("data_explore/rain_hist_box_each_split_7d.png", dpi=150)
plt.show()
