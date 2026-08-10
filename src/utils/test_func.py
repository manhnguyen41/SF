import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)

import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from src.utils.loss import get_station_from_grid, get_station_from_gsmap
from src.utils import utils
import os


def to_float(x, device):
    if isinstance(x,list):
        list_x = []
        for x_i in x:
            x_i = x_i.to(device).float()
            list_x.append(x_i)
        x = list_x
    else:
        x = x.to(device).float()
        
    return x   

def cal_acc(y_prd, y_grt):
    """
    Hàm tính toán các chỉ số đánh giá, tương thích với các phiên bản sklearn cũ.
    """
    mae = mean_absolute_error(y_grt, y_prd)
    
    # Sửa ở đây:
    # 1. Tính MSE một cách bình thường, không có tham số 'squared'.
    #    Hàm này mặc định trả về MSE.
    mse = mean_squared_error(y_grt, y_prd)
    
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    
    # 2. Tính RMSE bằng cách lấy căn bậc hai của MSE.
    rmse = np.sqrt(mse)
    
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    
    return mae, mse, mape, rmse, r2, corr


def test_func(model, test_dataset, criterion, config, input_scaler, output_scaler, device):
    model.eval() 
    list_prd = []
    list_grt = []
    list_ecmwf = []
    
    epoch_loss = 0
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    print("********** Starting testing process **********")
    
    with torch.no_grad():
        
        for data in tqdm(test_dataloader):
            input_data, lead_time, y_grt, ecmwf = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['ecmwf'].to(device)
            
            #ecmwf [B, 13, 7, H, W]
            ecmwf = ecmwf[:,12,-config.MODEL.ECMWF_TIME_STEP:,:,:]# (B,1, H, W)

            ecmwf = torch.sum(ecmwf, dim=1) # (B, H, W)
            
            ecmwf = torch.unsqueeze(ecmwf, dim=-1)
            
            h = data.get("h", None)
            if h is not None:
                h = h.to(device)
                
                y_prd = model([input_data, lead_time, h])
            else:
                y_prd = model([input_data, lead_time])
            #esp [B, 7, H, W]
            
            
            y_prd = get_station_from_grid(y_prd, y_grt, config) # (batch_size, num_station, 1)
            ecmwf = get_station_from_grid(ecmwf, y_grt, config)
            
            y_prd = y_prd[:,:,0] # (batch_size, num_station)
            y_grt = y_grt[:,:,0] # (batch_size, num_station)
            ecmwf = ecmwf[:,:,0] # (batch_size, num_station)
            
            batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
            y_prd = y_prd.cpu().detach().numpy()
            y_grt = y_grt.cpu().detach().numpy()
            ecmwf = ecmwf.cpu().detach().numpy()
            
            if config.TRAIN.OUTPUT_NORM:
                
                y_prd = output_scaler.inverse_transform(y_prd)
                y_grt = output_scaler.inverse_transform(y_grt)
                
                
                
            y_prd = np.clip(y_prd, 0, config.DATA.RAIN_THRESHOLD)
            y_prd = np.squeeze(y_prd)
            y_grt = np.clip(y_grt, 0, config.DATA.RAIN_THRESHOLD)
            y_grt = np.squeeze(y_grt)
            ecmwf = np.squeeze(ecmwf)
            ecmwf = np.clip(ecmwf, 0, config.DATA.RAIN_THRESHOLD)
            list_prd.append(y_prd)
            list_grt.append(y_grt)
            list_ecmwf.append(ecmwf)
            
            epoch_loss += batch_loss.item()
            # breakpoint()
    list_prd = np.concatenate(list_prd, 0)
    list_grt = np.concatenate(list_grt,0)
    list_ecmwf = np.concatenate(list_ecmwf, 0)
    
    # breakpoint()
    mae, mse, mape, rmse, r2, corr_ = cal_acc(list_prd, list_grt)
    mae_ecm, mse_ecm, mape_ecm, rmse_ecm, r2_ecm, corr_ecm = cal_acc(list_ecmwf, list_grt)
    

    plot_idx = [i for i in range(10)]
    if config.WANDB.STATUS:
        wandb.log({"mae":mae, "mse":mse, "mape":mape, "rmse":rmse, "r2":r2, "corr":corr_})
        wandb.log({"mae_ecm":mae_ecm, "mse_ecm":mse_ecm, "mape_ecm":mape_ecm, "rmse_ecm":rmse_ecm, "r2_ecm":r2_ecm, "corr_ecm":corr_ecm})
        
        for i in plot_idx:
            plt.figure(figsize=(20, 5))
            plt.plot(list_prd[i], label='Predictions', marker='o')
            plt.plot(list_grt[i], label='Ground Truths', marker='x')
            plt.plot(list_ecmwf[i], label="ECMWF", marker = 's')
           
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Predictions vs Ground Truths')
            plt.legend()
            plt.grid(True)

            # Log the plot to W&B
            wandb.log({f"Output/Image{i}": wandb.Image(plt)})
            plt.close()
        

        # Flatten both arrays to 1D
        flattened1 = list_prd.flatten()  # Shape: (64 * 169,)
        flattened2 = list_grt.flatten()  # Shape: (64 * 169,)
        flattened3 = list_ecmwf.flatten() # Shape: (64 * 169,)
        
        data1 = np.stack([flattened1, flattened2], 0)
        table1 = wandb.Table(data=data1.T, columns=["Prediction", "Groundtruth"] )
        wandb.log({"Output/Table1": table1})
        data2 = np.stack([flattened3, flattened2], 0)
        table2 = wandb.Table(data=data2.T, columns=["Prediction", "Groundtruth"] )
        wandb.log({"Output/Table2": table2})
        
        wandb.finish()


    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")  
    print(f"MSE_ecm: {mse_ecm} MAE_ecm:{mae_ecm} MAPE_ecm:{mape_ecm} RMSE_ecm:{rmse_ecm} R2_ecm:{r2_ecm} Corr_ecm:{corr_ecm}")   
            
    return 

import glob
import os
import numpy as np


def load_quantile_distribution(grid_dir, gauge_dir):

    grid_distribution = {}

    for f in glob.glob(os.path.join(grid_dir, "*.npy")):
        key = os.path.basename(f).replace(".npy", "")
        grid_distribution[key] = np.load(f)

    gauge_distribution = {}

    for f in glob.glob(os.path.join(gauge_dir, "*.npz")):
        key = os.path.basename(f).replace(".npz", "")

        data = np.load(f)

        gauge_distribution[key] = (
            data["quantiles"],
            data["levels"]
        )

    return grid_distribution, gauge_distribution


def quantile_mapping(x,
                     grid_quantiles,
                     gauge_quantiles,
                     levels):

    x = np.clip(
        x,
        grid_quantiles[0],
        grid_quantiles[-1]
    )

    p = np.interp(
        x,
        grid_quantiles,
        levels
    )

    return np.interp(
        p,
        levels,
        gauge_quantiles
    )


def get_grid_index(lat,
                   lon,
                   config,
                   step=0.125):

    lat_idx = int(
        np.floor(
            (config.DATA.LAT_START-lat)/step
        )
    )

    lon_idx = int(
        np.floor(
            (lon-config.DATA.LON_START)/step
        )
    )

    lat_idx = np.clip(
        lat_idx,
        0,
        config.DATA.HEIGHT-1
    )

    lon_idx = np.clip(
        lon_idx,
        0,
        config.DATA.WIDTH-1
    )

    return lat_idx, lon_idx


def test_func_quantile(test_dataset, criterion, config, input_scaler, output_scaler, device):
    list_prd = []
    list_grt = []
    list_ecmwf = []
    grid_distribution, gauge_distribution = load_quantile_distribution(
        "s2s/distribution",
        "gauss/distribution"
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    print("********** Starting testing process **********")
    
    with torch.no_grad():
        
        for data in tqdm(test_dataloader):
            input_data, lead_time, y_grt, ecmwf = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['ecmwf'].to(device)
            
                        # ECMWF weekly rainfall
            ecmwf = ecmwf[:, 12, -config.MODEL.ECMWF_TIME_STEP:, :, :]
            ecmwf = torch.sum(ecmwf, dim=1)
            ecmwf = torch.unsqueeze(ecmwf, dim=-1)

            # lấy ECMWF tại station
            ecmwf_station = get_station_from_grid(ecmwf, y_grt, config)
            ecmwf_station = ecmwf_station[:, :, 0].cpu().numpy()
           
            coord = y_grt.cpu().numpy()

            B, N = ecmwf_station.shape

            y_prd = np.zeros((B, N), dtype=np.float32)

            for b in range(B):

                for s in range(N):

                    lon = coord[b, s, 1]
                    lat = coord[b, s, 2]

                    i, j = get_grid_index(
                        lat,
                        lon,
                        config
                    )

                    key = f"{i:02d}_{j:02d}"

                    x = ecmwf_station[b, s]

                    # nếu không có gauge hoặc grid distribution
                    if key not in grid_distribution or key not in gauge_distribution:
                        y_prd[b, s] = x
                        continue

                    grid_q = grid_distribution[key]

                    gauge_q, levels = gauge_distribution[key]

                    y_prd[b, s] = quantile_mapping(
                        x,
                        grid_q,
                        gauge_q,
                        levels
                    )
                    

            # ECMWF đã là numpy
            ecmwf = ecmwf_station

            # Ground truth rainfall
            y_grt = y_grt[:, :, 0].cpu().numpy()
            if config.TRAIN.OUTPUT_NORM:    
                
                y_grt = output_scaler.inverse_transform(y_grt)
                
                
                
            y_prd = np.clip(y_prd, 0, config.DATA.RAIN_THRESHOLD)
            y_prd = np.squeeze(y_prd)
            y_grt = np.clip(y_grt, 0, config.DATA.RAIN_THRESHOLD)
            y_grt = np.squeeze(y_grt)
            ecmwf = np.squeeze(ecmwf)
            ecmwf = np.clip(ecmwf, 0, config.DATA.RAIN_THRESHOLD)
            list_prd.append(y_prd)
            list_grt.append(y_grt)
            list_ecmwf.append(ecmwf)
            
            
            # breakpoint()
    list_prd = np.concatenate(list_prd, 0)
    list_grt = np.concatenate(list_grt,0)
    list_ecmwf = np.concatenate(list_ecmwf, 0)
    
    # breakpoint()
    mae, mse, mape, rmse, r2, corr_ = cal_acc(list_prd, list_grt)
    mae_ecm, mse_ecm, mape_ecm, rmse_ecm, r2_ecm, corr_ecm = cal_acc(list_ecmwf, list_grt)
    

    plot_idx = [i for i in range(10)]
    if config.WANDB.STATUS:
        wandb.log({"mae":mae, "mse":mse, "mape":mape, "rmse":rmse, "r2":r2, "corr":corr_})
        wandb.log({"mae_ecm":mae_ecm, "mse_ecm":mse_ecm, "mape_ecm":mape_ecm, "rmse_ecm":rmse_ecm, "r2_ecm":r2_ecm, "corr_ecm":corr_ecm})
        
        for i in plot_idx:
            plt.figure(figsize=(20, 5))
            plt.plot(list_prd[i], label='Predictions', marker='o')
            plt.plot(list_grt[i], label='Ground Truths', marker='x')
            plt.plot(list_ecmwf[i], label="ECMWF", marker = 's')
           
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Predictions vs Ground Truths')
            plt.legend()
            plt.grid(True)

            # Log the plot to W&B
            wandb.log({f"Output/Image{i}": wandb.Image(plt)})
            plt.close()
        

        # Flatten both arrays to 1D
        flattened1 = list_prd.flatten()  # Shape: (64 * 169,)
        flattened2 = list_grt.flatten()  # Shape: (64 * 169,)
        flattened3 = list_ecmwf.flatten() # Shape: (64 * 169,)
        
        data1 = np.stack([flattened1, flattened2], 0)
        table1 = wandb.Table(data=data1.T, columns=["Prediction", "Groundtruth"] )
        wandb.log({"Output/Table1": table1})
        data2 = np.stack([flattened3, flattened2], 0)
        table2 = wandb.Table(data=data2.T, columns=["Prediction", "Groundtruth"] )
        wandb.log({"Output/Table2": table2})
        
        wandb.finish()


    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")  
    print(f"MSE_ecm: {mse_ecm} MAE_ecm:{mae_ecm} MAPE_ecm:{mape_ecm} RMSE_ecm:{rmse_ecm} R2_ecm:{r2_ecm} Corr_ecm:{corr_ecm}")   
            
    return
