import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
def get_scaler(config):
    data_dir=config.DATA.NPYARR_DIR
    esp_dir = config.DATA.ESP_DATA_PATH
    csv_path=config.DATA.GAUGE_DATA_PATH
    LOG_FEATURES_IDX = [0, 7, 12]
    POWER_FEATURES_IDX = [3, 5, 6, 8]
    STANDARD_FEATURES_IDX = [1, 2, 4, 9, 10, 11]
    NUM_FEATURES = 13
    scaler_file= f"{config.DATA.DATA_IDX_DIR}/scalers1.pkl"
    output_scaler_file=f"{config.DATA.DATA_IDX_DIR}/output_scaler1.pkl"
    esp_scaler_file = f"{config.DATA.DATA_IDX_DIR}/esp_scalers1.pkl"
    if os.path.exists(scaler_file) and os.path.exists(output_scaler_file):
        print(f"Load scalers from file: {scaler_file}")
        print(f"Load output_scaler from file: {output_scaler_file}")
        scalers = joblib.load(scaler_file)
        esp_scaler = joblib.load(esp_scaler_file)
        output_scaler = joblib.load(output_scaler_file)
        return scalers, esp_scaler, output_scaler
    
    
    print("Creating scalers...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    day_steps = [3, 4]
    step_index = 0
    dates = []
    current_date = start_date
    thresh_hold = config.DATA.RAIN_THRESHOLD 
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        # Từ 11-11 trở đi dùng bước 2 ngày
        if current_date >= datetime(2024, 11, 11):
            current_date += timedelta(days=2)
        else:
            current_date += timedelta(days=day_steps[step_index])
            step_index = 1 - step_index
    dates = [date for date in dates if int(date.split('-')[1]) in [6, 7, 8, 9]]
    file_paths = [os.path.join(data_dir, f"{date[-5:]}.npy") for date in dates]
    file_paths = [fp for fp in file_paths if os.path.exists(fp)]
    all_training_data = []
    print(f"Loading data from {len(file_paths)} training files...")
    for file_path in tqdm(file_paths, desc="Loading files"):
        data = np.load(file_path)
        data = data[:16,:,:,:,:]
        all_training_data.append(data)
        
    full_training_array = np.concatenate(all_training_data, axis=0)
    # full_training_array = np.clip(full_training_array, 0, thresh_hold)
    print(f"Full training data shape: {full_training_array.shape}")
    data_reshaped = full_training_array.transpose(0, 2, 3, 4, 1).reshape(-1, NUM_FEATURES)
    data_reshaped[:, 0] = np.clip(data_reshaped[:, 0], 0, thresh_hold )
    data_reshaped[:, 12] = np.clip(data_reshaped[:, 12], 0, thresh_hold )
    data_reshaped = np.nan_to_num(data_reshaped, nan=0.0)
    data_reshaped[:, LOG_FEATURES_IDX] = np.maximum(data_reshaped[:, LOG_FEATURES_IDX], 0)
    print(f"Reshaped data for fitting: {data_reshaped.shape}")
    del all_training_data
    del full_training_array
    print("Building and fitting the preprocessor pipeline...")
    log_pipeline_template = Pipeline(steps=[
        # ('scaler1', MinMaxScaler(feature_range=(1, 5))),
        ('log', FunctionTransformer(np.log1p, np.expm1, validate=False, check_inverse=False)),
        ('scaler', MinMaxScaler(feature_range=(-1, 1)))
    ])

    power_pipeline_template = Pipeline(steps=[
        ('scaler1', MinMaxScaler(feature_range=(1, 5))),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', MinMaxScaler(feature_range=(-1, 1)))
    ])

    standard_pipeline_template = Pipeline(steps=[
        ('scaler', MinMaxScaler(feature_range=(-1, 1)))
    ])
    scalers = [None] * NUM_FEATURES

    print("Initializing 13 individual scaler pipelines...")
    for i in range(NUM_FEATURES):
        if i in LOG_FEATURES_IDX:
            from sklearn import clone 
            scalers[i] = clone(log_pipeline_template)
            print(f"  - Index {i}: Assigned LOG pipeline.")
        elif i in POWER_FEATURES_IDX:
            from sklearn import clone
            scalers[i] = clone(power_pipeline_template)
            print(f"  - Index {i}: Assigned POWER pipeline.")
        elif i in STANDARD_FEATURES_IDX:
            from sklearn import clone
            scalers[i] = clone(standard_pipeline_template)
            print(f"  - Index {i}: Assigned STANDARD pipeline.")
    first_pass = True
    for date in dates:
        file_path = os.path.join(data_dir, f"{date[-5:]}.npy")
        if os.path.exists(file_path):
            for i in range(13):
                feature_data = data[:16, i, :, :, :].reshape(-1, 1)
                if i == 0 or i == 12:
                    feature_data = np.clip(feature_data, 0, thresh_hold)
                if i == 7:
                    feature_data = np.clip(feature_data, 0, 15)
                if first_pass:
                    scalers[i].fit(feature_data)
                else:
                    scalers[i].fit(feature_data)
            first_pass = False
    joblib.dump(scalers, scaler_file)
    print(f"Saved preprocessor into file: {scaler_file}")
    del data_reshaped
    esp_data = []
    print("Create ESP scaler file ....")
    for file_path in tqdm(os.listdir(esp_dir), desc="Loading files"):
        data = np.load(os.path.join(esp_dir, file_path))
        data = data[:16,:,:,:]
        data = np.clip(data, 0, thresh_hold)
        esp_data.append(data)
    esp_data = np.array(esp_data)  # shape = (num_files, 12, H, W)
    esp_data = np.clip(esp_data, 0, thresh_hold)
    # esp_data shape = (20, 46, H, W)
    B, T, L, H, W = esp_data.shape

    # Flatten batch + lead time
    esp_reshaped = esp_data.reshape(-1, 1)  # shape = (20*46, H*W)

    # Tạo pipeline
    esp_pipeline = Pipeline([
        # ('scaler1', MinMaxScaler(feature_range=(1, 5))),
        ('log', FunctionTransformer(np.log1p, np.expm1, validate=False, check_inverse=False)),
        ('scaler', MinMaxScaler(feature_range=(-1, 1)))
    ])

    # Fit
    esp_pipeline.fit(esp_reshaped)
    joblib.dump(esp_pipeline, esp_scaler_file)
    print(f"Saved preprocessor into file: {esp_scaler_file}")
    
    
    del esp_data
    # Create output_scaler
    print("Creating output_scaler...")
    csv_data = pd.read_csv(csv_path)
    output_transformer = Pipeline(steps=[
        #('scaler1', MinMaxScaler(feature_range=(1, 5))),
        # ('log', FunctionTransformer(np.log1p, np.expm1, validate=False, check_inverse=False)),
        ('log1', FunctionTransformer(np.log1p, np.expm1, validate=False, check_inverse=False)),
        ('scaler', MinMaxScaler(feature_range=(-1, 1)))
    ])
    
    # Reshape and fit output_scaler
    csv_data = csv_data[(csv_data['Month'] >= 1) & (csv_data['Month'] <= 12) & (csv_data['Year'] >= 2004) & (csv_data['Year'] <= 2020)]
    csv_data['R'] = csv_data['R'].clip(lower=0, upper=thresh_hold)
    val_years  = [2020, 2021]
    test_years = [2022, 2023]

    csv_data = csv_data[~csv_data['Year'].isin(val_years + test_years)]
    
    
    group_cols = [c for c in ['Station'] if c in csv_data.columns]
    def add_7d_total(d: pd.DataFrame) -> pd.DataFrame:
        
        d = d.copy()
        d['R'] = d['R'].clip(lower=0, upper=thresh_hold)
        if not np.issubdtype(d['Day'].dtype, np.datetime64):
            d['Day'] = pd.to_datetime(d['Day'], errors='coerce')
        d = d.sort_values(['Station', 'Day'], kind='mergesort')

        d['R'] = (
            d.groupby('Station', group_keys=False)['R']
            .transform(lambda x: x.rolling(window=7, min_periods=1).sum())
        )
        # d['R'] = np.log1p(d['R'])
        return d
    csv_data = add_7d_total(csv_data)
    
    
    r_data_train = csv_data['R'].values.reshape(-1, 1)
    
    output_transformer.fit(r_data_train)
    
    # Save output_scaler
    joblib.dump(output_transformer, output_scaler_file)
    print(f"Saved output_scaler into file: {output_scaler_file}")
    
    return scalers, esp_pipeline, output_transformer


def get_scaler1(config):
    # Path to data folder
    thresh_hold = config.DATA.RAIN_THRESHOLD
    data_dir=config.DATA.NPYARR_DIR
    csv_path=config.DATA.GAUGE_DATA_PATH
    esp_dir = config.DATA.ESP_DATA_PATH
    # Path to save scalers
    scaler_file= f"{config.DATA.DATA_IDX_DIR}/scalers.pkl"
    output_scaler_file=f"{config.DATA.DATA_IDX_DIR}/output_scaler.pkl"
    esp_scaler_file = f"{config.DATA.DATA_IDX_DIR}/esp_scalers.pkl"
    # Check if scaler files exist
    if os.path.exists(scaler_file) and os.path.exists(output_scaler_file):
        print(f"Load scalers from file: {scaler_file}")
        print(f"Load output_scaler from file: {output_scaler_file}")
        scalers = joblib.load(scaler_file)
        esp_scaler = joblib.load(esp_scaler_file)
        output_scaler = joblib.load(output_scaler_file)
        return scalers, esp_scaler, output_scaler
    
    # If scaler files do not exist, create new scalers
    print("Creating scalers...")
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    day_steps = [3, 4]
    step_index = 0
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        # Từ 11-11 trở đi dùng bước 2 ngày
        if current_date >= datetime(2024, 11, 11):
            current_date += timedelta(days=2)
        else:
            current_date += timedelta(days=day_steps[step_index])
            step_index = 1 - step_index
    dates = [date for date in dates if int(date.split('-')[1]) in [6, 7, 8, 9]]
    scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(13)]
    first_pass = True
    for date in dates:
        file_path = os.path.join(data_dir, f"{date[-5:]}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)  # Shape: (20, 13, 47, 137, 121)
            time_idx = np.r_[1:8, 9:16]
            for i in range(13):
                feature_data = data[:16, i, :, :, :].reshape(-1, 1)
                if i == 0 or i == 12:
                    feature_data = np.clip(feature_data, 0, thresh_hold)
                if i == 7:
                    feature_data = np.clip(feature_data, 0, 15)
                if first_pass:
                    scalers[i].fit(feature_data)
                else:
                    scalers[i].partial_fit(feature_data)
            first_pass = False
    scalers_array = np.array(scalers)
    esp_scaler = MinMaxScaler(feature_range=(-1, 1))
    joblib.dump(scalers_array, scaler_file)
    print(f"Saved scalers into file: {scaler_file}")
    first_pass = True
    for date in dates:
        file_path = os.path.join(esp_dir, f"{date[-5:]}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)  # Shape: (20, 13, 47, 137, 121)
            time_idx = np.r_[:16]
            feature_data = data[time_idx, :, :, :].reshape(-1, 1)
            feature_data = np.clip(feature_data, 0, thresh_hold)
            if first_pass:
                esp_scaler.fit(feature_data)
            else:
                esp_scaler.partial_fit(feature_data)
            first_pass = False
    joblib.dump(esp_scaler, esp_scaler_file)
    print("Saved esp scaler to ")
    
    # Create output_scaler
    print("Creating output_scaler...")
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data[(csv_data['Month'] >= 1) & (csv_data['Month'] <= 12) & (csv_data['Year'] >= 2004) & (csv_data['Year'] <= 2020)]
    csv_data['R'] = csv_data['R'].clip(lower=0, upper=thresh_hold)
    val_years  = [2020, 2021]
    test_years = [2022, 2023]

    csv_data = csv_data[~csv_data['Year'].isin(val_years + test_years)]
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    
    group_cols = [c for c in ['Station'] if c in csv_data.columns]
    def add_7d_total(d: pd.DataFrame) -> pd.DataFrame:
        
        d = d.copy()
        d['R'] = d['R'].clip(lower=0, upper=thresh_hold)
        if not np.issubdtype(d['Day'].dtype, np.datetime64):
            d['Day'] = pd.to_datetime(d['Day'], errors='coerce')
        d = d.sort_values(['Station', 'Day'], kind='mergesort')

        d['R'] = (
            d.groupby('Station', group_keys=False)['R']
            .transform(lambda x: x.rolling(window=7, min_periods=1).sum())
        )
        # d['R'] = np.log1p(d['R'])
        return d
    
    csv_data = add_7d_total(csv_data)
    
    
    # Reshape and fit output_scaler
    r_data_train = csv_data['R'].values.reshape(-1, 1)
    
    output_scaler.fit(r_data_train)
    
    # Save output_scaler
    joblib.dump(output_scaler, output_scaler_file)
    print(f"Saved output_scaler into file: {output_scaler_file}")
    
    return scalers, esp_scaler, output_scaler