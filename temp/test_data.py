import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from datetime import datetime, timedelta
import joblib

def get_scaler():
    # Path to data folder
    data_dir="/mnt/disk1/env_data/S2S_0.125/nparr_reg_1/Step24h"
    csv_path="/mnt/disk1/env_data/Gauge_thay_Tan/Final_Data_Region_1.csv"
    
    # If scaler files do not exist, create new scalers
    print("Creating scalers...")
    
    # Create a list of dates from 2022-01-03
    start_date = datetime(2022, 1, 3)
    end_date = datetime(2022, 12, 31)
    day_steps = [3, 4]
    step_index = 0
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=day_steps[step_index])
        step_index = 1 - step_index
    
    # Filter dates to only include July, August, and September
    dates = [date for date in dates if int(date.split('-')[1]) in [6, 7, 8, 9]]
    
    # Initialize scalers for 13 features
    scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(13)]
    
    # Calculate scalers for each feature
    first_pass = True
    for date in dates:
        file_path = os.path.join(data_dir, f"{date}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)  # Shape: (20, 13, 47, 137, 121)
            print(data.shape)
            for i in range(13):
                feature_data = data[:, i, :, :, :].reshape(-1, 1)
                if first_pass:
                    scalers[i].fit(feature_data)
                else:
                    scalers[i].partial_fit(feature_data)
            first_pass = False
    
    # Convert scalers to numpy array
    scalers_array = np.array(scalers)
    
    # Save scalers to file
    # joblib.dump(scalers_array, scaler_file)
    # print(f"Saved scalers into file: {scaler_file}")
    
    # Create output_scaler
    print("Creating output_scaler...")
    csv_data = pd.read_csv(csv_path)
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Reshape and fit output_scaler
    r_data = csv_data['R'].values.reshape(-1, 1)
    output_scaler.fit(r_data)
    
    # Save output_scaler
    # joblib.dump(output_scaler, output_scaler_file)
    # print(f"Saved output_scaler into file: {output_scaler_file}")
    
get_scaler()