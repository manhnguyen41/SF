import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# =====================================
# Config
# =====================================
csv_path = "/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_2002_2024_Region_1.csv"

output_dir = "gauss/distribution"
os.makedirs(output_dir, exist_ok=True)

levels = np.linspace(0, 1, 100)

lat_start = 23.25
lon_start = 102.25
step = 0.125

HEIGHT = 25
WIDTH = 25

# =====================================
# Read csv
# =====================================
df = pd.read_csv(csv_path)
df["Day"] = pd.to_datetime(df["Day"])

# pixel -> list of weekly rainfall
pixel_data = defaultdict(list)

# =====================================
# Process every station
# =====================================
for station, group in tqdm(df.groupby("Station")):

    group = group.sort_values("Day")

    rain = group["R"].fillna(0).to_numpy(np.float32)

    if len(rain) < 7:
        continue

    weekly = np.convolve(
        rain,
        np.ones(7, dtype=np.float32),
        mode="valid"
    )

    # station coordinate (constant)
    lat = group.iloc[0]["Lat"]
    lon = group.iloc[0]["Lon"]

    lat_idx = int(np.floor((lat_start - lat) / step))
    lon_idx = int(np.floor((lon - lon_start) / step))

    lat_idx = np.clip(lat_idx, 0, HEIGHT - 1)
    lon_idx = np.clip(lon_idx, 0, WIDTH - 1)

    pixel_data[(lat_idx, lon_idx)].append(weekly)

# =====================================
# Save
# =====================================
for (i, j), values in pixel_data.items():

    values = np.concatenate(values)

    quantiles = np.quantile(values, levels)

    np.savez_compressed(
        os.path.join(output_dir, f"{i:02d}_{j:02d}.npz"),
        quantiles=quantiles.astype(np.float32),
        levels=levels.astype(np.float32),
    )

print(f"Saved {len(pixel_data)} pixel distributions.")