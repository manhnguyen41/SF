import numpy as np 
from src.utils.utils import load_model
from src.model.models import Model_Ver1, SwinTransformer_Ver3, SwinTransformer_Ver4
from src.utils.get_option import get_option
from src.utils.visualization import create_heatmap
import argparse

from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import wandb
import os 

from src.utils.get_option import get_option
from src.model.models import Model_Ver1
from src.utils.dataloader import CustomDataset
from src.utils import utils, get_scaler, train_func, test_func
from src.model import models 
from src.utils.loss import ExpMagnitudeWeightedMAELoss, WeightedMSELoss
checkpoint_path = f"saved_checkpoints/data3-r1-test/checkpoint/data3-r1-test_Strans-V4_PS-4_Lr-0.0001_LF-mae_DR-0.3_LN-True-ST_0_3-ON_False_Seed-52_LRS-True_ReduceLROnPlateau-min-0.5-3.pt"

args, config = get_option()
input_scaler, output_scaler = get_scaler.get_scaler(config)

# === 2. Load data ===
test_dataset = CustomDataset(mode='test', config=config, ecmwf_scaler=input_scaler, output_scaler=output_scaler)
sample = test_dataset[0]
input_data = torch.tensor(sample['x']).unsqueeze(0)
lead_time = torch.tensor(sample['lead_time']).unsqueeze(0)
target = torch.tensor(sample['y']).unsqueeze(0)

# === 3. Load model ===
model = SwinTransformer_Ver4(config)
load_model(model, checkpoint_path)
model.eval()

# === 4. Forward & extract embeddings ===
with torch.no_grad():
    output = model([input_data, lead_time])
        # Bạn cần đảm bảo đã lưu self.spatial_emb trong forward()
    res = model.res  # Tương tự
    prompt = model.delta_t                   # Vector prompt

# === 5. Vẽ heatmap ===
print(res.shape)
create_heatmap(res[0,:,:,0].cpu().numpy(), "res.png", cmap="Blues")

#create_heatmap(prompt.cpu().detach().numpy(), "prompt.png", "Prompt value distribution", "Order", "", cmap="hot")



# load_model()