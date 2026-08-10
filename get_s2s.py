import os
import numpy as np
from tqdm import tqdm

# =====================================
# Config
# =====================================
input_dir = "/mnt/disk3/longnd/env_data/grid_base/nparr_reg1/Step24h"
output_dir = "s2s/distribution"

os.makedirs(output_dir, exist_ok=True)

files = sorted(
    f for f in os.listdir(input_dir)
    if f.endswith(".npy")
)

H = 25
W = 25
window = 7

# 100 quantiles
quantile_levels = np.linspace(0, 1, 100)

# lưu dữ liệu cho từng pixel
pixel_values = [[[] for _ in range(W)] for _ in range(H)]

# =====================================
# Read all files
# =====================================
for fname in tqdm(files, desc="Reading"):

    arr = np.load(os.path.join(input_dir, fname))

    # (20,46,25,25)
    rain = arr[:, -1]

    # rolling 7-day sum
    # (20,40,25,25)
    weekly = np.stack(
        [
            rain[:, i:i+window].sum(axis=1)
            for i in range(rain.shape[1] - window + 1)
        ],
        axis=1
    )

    for i in range(H):
        for j in range(W):
            pixel_values[i][j].append(
                weekly[:, :, i, j].reshape(-1)
            )

# =====================================
# Compute quantiles
# =====================================
for i in tqdm(range(H), desc="Saving quantiles"):

    for j in range(W):

        values = np.concatenate(pixel_values[i][j]).astype(np.float32)

        q = np.quantile(values, quantile_levels)

        # print(f"\n===== Pixel ({i:02d}, {j:02d}) =====")
        # for p, v in zip(quantile_levels, q):
        #     print(f"{p*100:6.2f}% : {v:.3f}")

        np.save(
            os.path.join(output_dir, f"{i:02d}_{j:02d}.npy"),
            q.astype(np.float32)
        )

print("Done!")