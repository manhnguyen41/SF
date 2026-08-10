import argparse
import time  # Thêm import time để đo thời gian
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import os
from tqdm import tqdm

from src.utils.dataloader import CustomDataset
from src.utils.get_scaler import get_scaler
from src.utils.get_option import get_option

if __name__ == "__main__":
    args = get_option()

    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg)) 
    
    # Preprocess data
    print("*************** Get scaler ***************")
    input_scalers, output_scaler = get_scaler(args)
    
    print("*************** Init dataset ***************")
    valid_dataset = CustomDataset(mode='valid', args=args, ecmwf_scaler=input_scalers, output_scaler=output_scaler)
    train_dataset = CustomDataset(mode='train', args=args, ecmwf_scaler=input_scalers, output_scaler=output_scaler)
    test_dataset = CustomDataset(mode='test', args=args, ecmwf_scaler=input_scalers, output_scaler=output_scaler)

    print(train_dataset[1])
    # Tạo DataLoader từ dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Đo thời gian load toàn bộ dữ liệu từ train_loader
    print("*************** Measuring time to load all data ***************")
    start_time = time.time()  # Bắt đầu đo thời gian
    
    # # Duyệt qua toàn bộ train_loader để load hết dữ liệu
    for data in tqdm(train_loader):
        pass  # Không cần xử lý gì, chỉ cần load hết dữ liệu
    
    end_time = time.time()  # Kết thúc đo thời gian
    elapsed_time = end_time - start_time  # Tính thời gian đã trôi qua
    
    print(f"Time to load all data from train_loader: {elapsed_time:.2f} seconds")

    # Lấy một batch từ train_loader để kiểm tra (nếu cần)
    first_batch = next(iter(train_loader))
    print("*************** First Batch ***************")
    for key, value in first_batch.items():
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")