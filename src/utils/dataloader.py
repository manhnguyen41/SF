import pandas as pd 
import numpy as np
import os 
import pickle
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import time
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, mode='train', ecmwf_scaler=None, output_scaler=None, config=None, shuffle = False):
        self.config = config
        self.mode = mode
        self.shuffle = shuffle
        self.output_norm = config.TRAIN.OUTPUT_NORM
        # Load gauge data
        self.gauge_data = pd.read_csv(config.DATA.GAUGE_DATA_PATH)
        self.gauge_data['Day'] = pd.to_datetime(self.gauge_data['Day'])
        
        # Load station data
        station_data = self.gauge_data[['Station', 'Lon', 'Lat']].drop_duplicates('Station')
        self.stations = station_data['Station'].values
        
        # Load station coords
        self.station_coords = station_data[['Lon', 'Lat']].to_numpy()
        
        # Load idx
        self.idx_df = pd.read_csv(f'{config.DATA.DATA_IDX_DIR}/{mode}.csv').values
        if self.shuffle:
            np.random.shuffle(self.idx_df)
        # Load scaler
        self.ecmwf_scaler = ecmwf_scaler
        self.output_scaler = output_scaler
        
        # Path to store ecmwf data
        self.processed_ecmwf_dir = f'{config.DATA.PROCESSED_ECMWF_DIR}/processed_ecmwf/{mode}'
        os.makedirs(self.processed_ecmwf_dir, exist_ok=True)
        
        # Process and save ecmwf data
        self.process_and_save_ecmwf_data()
        
    def calculate_leadtime_date(self, year, month, day, lead_time):
        """
        Calculate the date after adding `lead_time` days to the given (year, month, day).
        """
        start_date = datetime(year, month, day)
        leadtime_date = start_date + timedelta(days=lead_time)
        return leadtime_date

        
    def process_and_save_ecmwf_data(self):
        for idx in tqdm(range(len(self.idx_df)), desc=f"Processing ecmwf_data for {self.mode}"):
            # Create path to store processed ecmwf data
            processed_ecmwf_path = f'{self.processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
            
            # If processed ecmwf data not exist, process and save it
            if not os.path.exists(processed_ecmwf_path):
                ecmwf_path, lead_time, year, month, day = self.idx_df[idx]
                ecmwf_data = self.get_ecmwf(ecmwf_path, year, lead_time)  # (13, 7, 137, 121)
                
                ecmwf_data_scaled = self.transform_ecmwf(ecmwf_data)  # (13, 7, 137, 121)
                ecmwf_data_scaled = ecmwf_data_scaled.transpose(1, 0, 2, 3)  # (7, 13, 137, 121)
                np.save(processed_ecmwf_path, ecmwf_data_scaled)
    
    def get_ecmwf(self, ecmwf_path, year, lead_time):
        
        ecmwf_data = np.load(ecmwf_path) # (20, 13, 47, 137, 121)
        
        ecmwf_data = ecmwf_data[year-2002,:,lead_time-6:lead_time+1,:,:] # (13, 7, 137, 121) ecmwf_data[year-2002,:,:21,:,:] # 13, 21, 137, 121
        
        return ecmwf_data
    
    def get_ground_truth(self, year, month, day, lead_time):
        # Get start and end date
        start_date = datetime(year, month, day) + timedelta(days=lead_time - 6)
        end_date = datetime(year, month, day) + timedelta(days=lead_time)
        
        # Filter data by date
        mask = (self.gauge_data['Day'] >= start_date) & (self.gauge_data['Day'] <= end_date)
        period_data = self.gauge_data.loc[mask]
        
        # Init array to store ground truth
        num_station = len(self.stations)
        total_rain = np.zeros((num_station, 1))
        
        # Calculate total rain for each station
        for i, station in enumerate(self.stations):
            station_data = period_data[period_data['Station'] == station]
            total_rain[i, 0] = station_data['R'].sum()
           
        # Stack total rain and station coords 
        result = np.hstack((total_rain, self.station_coords))
        
        return result
    
    def transform_ecmwf(self, ecmwf_data):
        # Transform ecmwf_data
        # 1. Ghi nhớ shape ban đầu
        original_shape = ecmwf_data.shape  # (13, 7, H, W)
        num_features = original_shape[0]
        data_2d = ecmwf_data.reshape(num_features, -1).T
        data_processed_2d = self.ecmwf_scaler.transform(data_2d)
        
        data_processed_4d = data_processed_2d.T.reshape(original_shape)
        return data_processed_4d.astype(np.float32)
        # ecmwf_data_scaled = np.zeros_like(ecmwf_data, dtype=np.float32)
        # ecmwf_data_scaled = self.transform_ecmwf(ecmwf_data)
        # # print(ecmwf_data.shape)
        # for i in range(13):
        #     feature_data = ecmwf_data[i].reshape(-1, 1)
        #     scaled_feature = self.ecmwf_scaler[i].transform(feature_data)
        #     ecmwf_data_scaled[i] = scaled_feature.reshape(7, self.config.DATA.HEIGHT, self.config.DATA.WIDTH)
        # return ecmwf_data_scaled
    
    def transform_ground_truth(self, ground_truth):
        # Transform ground_truth
        ground_truth_scaled = ground_truth.copy()
        rain_data = ground_truth[:, 0].reshape(-1, 1)
        if self.output_norm:
            #scaled_rain = self.output_scaler.transform(rain_data)
            scaled_rain = self.output_scaler.transform(rain_data)
            scaled_rain = rain_data
            #scaled_rain = np.log2(3 + rain_data)
        else:
            scaled_rain = rain_data
            
        ground_truth_scaled[:, 0] = scaled_rain.flatten()
        return ground_truth_scaled
    
    def __getitem__(self, idx):
        ecmwf_path, lead_time, year, month, day = self.idx_df[idx]
        processed_ecmwf_path = f'{self.processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
        ecmwf_data = np.load(processed_ecmwf_path)  # (7, 13, 137, 121)  # (7, 13, 137, 121)
        
        ground_truth = self.get_ground_truth(year, month, day, lead_time)  # (num_station, 3)
        ground_truth = self.transform_ground_truth(ground_truth)
        
        x_leadtime = self.calculate_leadtime_date(year, month, day, lead_time)
        
        #return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth}
        
        return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth, 'x_leadtime': x_leadtime, 'ecmwf': self.get_ecmwf(ecmwf_path=ecmwf_path, year=year, lead_time=lead_time).transpose(1, 0, 2, 3)}
        # return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth, 'x_leadtime': x_leadtime, 'ecmwf': ecmwf_data}
    
    def __len__(self):
        return len(self.idx_df)
    
class CustomDataset2(Dataset):
    def __init__(self, mode='train', ecmwf_scaler=None, output_scaler=None, config=None, shuffle = False):
        self.config = config
        self.mode = mode
        self.shuffle = shuffle
        self.output_norm = config.TRAIN.OUTPUT_NORM
        # Load gauge data
        self.gauge_data = pd.read_csv(config.DATA.GAUGE_DATA_PATH)
        self.gauge_data['Day'] = pd.to_datetime(self.gauge_data['Day'])
        
        # Load station data
        station_data = self.gauge_data[['Station', 'Lon', 'Lat']].drop_duplicates('Station')
        self.stations = station_data['Station'].values
        
        # Load station coords
        self.station_coords = station_data[['Lon', 'Lat']].to_numpy()
        
        # Load idx
        self.idx_df = pd.read_csv(f'{config.DATA.DATA_IDX_DIR}/{mode}.csv').values
        if self.shuffle:
            np.random.shuffle(self.idx_df)
        # Load scaler
        self.ecmwf_scaler = ecmwf_scaler
        self.output_scaler = output_scaler
        
        # Path to store ecmwf data
        self.processed_ecmwf_dir = f'{config.DATA.PROCESSED_ECMWF_DIR}/processed_ecmwf/{mode}'
        os.makedirs(self.processed_ecmwf_dir, exist_ok=True)
        
        # Process and save ecmwf data
        self.process_and_save_ecmwf_data()
        
    def calculate_leadtime_date(self, year, month, day, lead_time):
        """
        Calculate the date after adding `lead_time` days to the given (year, month, day).
        """
        start_date = datetime(year, month, day)
        leadtime_date = start_date + timedelta(days=lead_time)
        return leadtime_date

        
    def process_and_save_ecmwf_data(self):
        for idx in tqdm(range(len(self.idx_df)), desc=f"Processing ecmwf_data for {self.mode}"):
            # Create path to store processed ecmwf data
            processed_ecmwf_path = f'{self.processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
            
            # If processed ecmwf data not exist, process and save it
            if not os.path.exists(processed_ecmwf_path):
                ecmwf_path, lead_time, year, month, day = self.idx_df[idx]
                ecmwf_data = self.get_ecmwf(ecmwf_path, year, lead_time)  # (13, 7, 137, 121)
                
                ecmwf_data_scaled = self.transform_ecmwf(ecmwf_data)  # (13, 7, 137, 121)
                ecmwf_data_scaled = ecmwf_data_scaled.transpose(1, 0, 2, 3)  # (7, 13, 137, 121)
                np.save(processed_ecmwf_path, ecmwf_data_scaled)
    
    def get_ecmwf(self, ecmwf_path, year, lead_time):
        
        ecmwf_data = np.load(ecmwf_path) # (20, 13, 47, 137, 121)
        
        ecmwf_data = ecmwf_data[year-2004,:,lead_time-7:lead_time,:,:] # (13, 7, 137, 121) ecmwf_data[year-2002,:,:21,:,:] # 13, 21, 137, 121
        
        return ecmwf_data
    
    def get_ground_truth(self, year, month, day, lead_time):
        # Get start and end date
        start_date = datetime(year, month, day) + timedelta(days=lead_time)
        end_date = datetime(year, month, day) + timedelta(days=lead_time)
        
        # Filter data by date
        mask = (self.gauge_data['Day'] >= start_date) & (self.gauge_data['Day'] <= end_date)
        period_data = self.gauge_data.loc[mask]
        
        # Init array to store ground truth
        num_station = len(self.stations)
        total_rain = np.zeros((num_station, 1))
        
        # Calculate total rain for each station
        for i, station in enumerate(self.stations):
            station_data = period_data[period_data['Station'] == station]
            
            total_rain[i, 0] = station_data['R'].sum()
           
        # Stack total rain and station coords 
        result = np.hstack((total_rain, self.station_coords))
        # print(result)
        return result
    
    def transform_ecmwf(self, ecmwf_data):
        
        
        original_shape = ecmwf_data.shape  # (13, 7, H, W)
        num_features = original_shape[0]
        data_2d = ecmwf_data.reshape(num_features, -1).T
        data_processed_2d = self.ecmwf_scaler.transform(data_2d)
        
        data_processed_4d = data_processed_2d.T.reshape(original_shape)
        return data_processed_4d.astype(np.float32)
    
        # Transform ecmwf_data
        # ecmwf_data_scaled = np.zeros_like(ecmwf_data, dtype=np.float32)
        
        # for i in range(13):
        #     feature_data = ecmwf_data[i].reshape(-1, 1)
        #     scaled_feature = self.ecmwf_scaler[i].transform(feature_data)
        #     ecmwf_data_scaled[i] = scaled_feature.reshape(7, self.config.DATA.HEIGHT, self.config.DATA.WIDTH)
        # return ecmwf_data_scaled
    
    def transform_ground_truth(self, ground_truth):
        
        
        # Transform ground_truth
        ground_truth_scaled = ground_truth.copy()
        rain_data = ground_truth[:, 0].reshape(-1, 1)
        rain_data[rain_data == -1] = 0
        
        if self.output_norm:
            scaled_rain = self.output_scaler.transform(rain_data)
            # scaled_rain = np.log2(3 + rain_data)
            # scaled_rain = self.output_scaler.transform(rain_data)
        else:
            scaled_rain = rain_data
            
        ground_truth_scaled[:, 0] = scaled_rain.flatten()
        return ground_truth_scaled
    
    def __getitem__(self, idx):
        ecmwf_path, lead_time, year, month, day = self.idx_df[idx]
        processed_ecmwf_path = f'{self.processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
        ecmwf_data = np.load(processed_ecmwf_path)  # (7, 13, 137, 121)  # (7, 13, 137, 121)
        
        ground_truth = self.get_ground_truth(year, month, day, lead_time)  # (num_station, 3)
        ground_truth = self.transform_ground_truth(ground_truth)
        
        x_leadtime = self.calculate_leadtime_date(year, month, day, lead_time)
        
        #return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth}
        
        return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth, 'x_leadtime': x_leadtime, 
                'ecmwf': self.get_ecmwf(ecmwf_path=ecmwf_path, year=year, lead_time=lead_time)}
        # return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth, 'x_leadtime': x_leadtime, 'ecmwf': ecmwf_data}
    
    def __len__(self):
        return len(self.idx_df)

class CustomDataset3(Dataset):
    def __init__(self, mode='train', ecmwf_scaler=None, esp_scaler=None,
                 output_scaler=None, config=None, shuffle=False):
        self.config = config
        self.mode = mode
        self.shuffle = shuffle
        self.output_norm = config.TRAIN.OUTPUT_NORM



        # Load dữ liệu gauge
        self.gauge_data = pd.read_csv(config.DATA.GAUGE_DATA_PATH)
        self.gauge_data['Day'] = pd.to_datetime(self.gauge_data['Day'])

        # Station
        station_data = self.gauge_data[['Station', 'Lon', 'Lat']].drop_duplicates('Station')
        self.stations = station_data['Station'].values
        self.station_coords = station_data[['Lon', 'Lat']].to_numpy()

        # Index
        self.idx_df = pd.read_csv(f'{config.DATA.DATA_IDX_DIR}/{mode}.csv').values
        if self.shuffle:
            np.random.shuffle(self.idx_df)

        self.ecmwf_scaler = ecmwf_scaler
        self.output_scaler = output_scaler
        self.esp_scaler = esp_scaler

        self.processed_ecmwf_dir = f'{config.DATA.PROCESSED_ECMWF_DIR}/processed_ecmwf/{mode}'
        os.makedirs(self.processed_ecmwf_dir, exist_ok=True)
        self.process_and_save_ecmwf_data()

    def calculate_leadtime_date(self, y, m, d, lead_time):
        try:
            start = datetime(y, m, d) + timedelta(days=lead_time)
        except ValueError:
            start = datetime(y, 2, 28) + timedelta(days=lead_time)
        return start

    def process_and_save_ecmwf_data(self):
        os.makedirs(f'{self.processed_ecmwf_dir}seed52_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}', exist_ok=True)
        for idx in tqdm(range(len(self.idx_df)), desc=f"Processing ecmwf_data for {self.mode}"):
            p_ecmwf = f'{self.processed_ecmwf_dir}seed52_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/ecmwf_data_{idx}.npy'
            p_esp   = f'{self.processed_ecmwf_dir}seed52_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/esp_data_{idx}.npy'

            if not os.path.exists(p_ecmwf):
                ecmwf_path, esp_path, lead_time, year, month, day = self.idx_df[idx]
                ecmwf_data = self.get_ecmwf(ecmwf_path, year, lead_time)
                ecmwf_data_scaled = self.transform_ecmwf(ecmwf_data)
                ecmwf_data_scaled = ecmwf_data_scaled.transpose(1, 0, 2, 3)  # (time, feature, H, W)
                np.save(p_ecmwf, ecmwf_data_scaled)

            if not os.path.exists(p_esp):
                ecmwf_path, esp_path, lead_time, year, month, day = self.idx_df[idx]
                esp_data = self.get_esp_data(esp_path, year, lead_time)
                esp_scaled = self.transform_esp_data(esp_data)
                np.save(p_esp, esp_scaled)

    def get_ecmwf(self, ecmwf_path, year, lead_time):
        data = np.load(ecmwf_path)  # (20,13,47,H,W)
        # print(data.shape)
        data = data[year-2004]
        data[0]  = np.clip(data[0], 0, self.config.DATA.RAIN_THRESHOLD)
        data[12] = np.clip(data[12], 0, self.config.DATA.RAIN_THRESHOLD)
        data[7] = np.clip(data[7], 0, 15)
        
        # data_pad = np.pad(data, pad_width=((0,0),(self.config.MODEL.ECMWF_TIME_STEP,0),(0,0),(0,0)), mode='edge')
        
        
        return data[:, lead_time-self.config.MODEL.ECMWF_TIME_STEP:lead_time, :, :]
    
    def get_ground_truth(self, year, month, day, lead_time):
        try:
            start = datetime(year, month, day) + timedelta(days=lead_time-self.config.MODEL.ECMWF_TIME_STEP+1)
        except ValueError:
            start = datetime(year, 2, 28) + timedelta(days=lead_time-self.config.MODEL.ECMWF_TIME_STEP)
            
        try:
            end = datetime(year, month, day) + timedelta(days=lead_time)
        except ValueError:
            end = datetime(year, 2, 28) + timedelta(days=lead_time)
        mask = (self.gauge_data['Day'] >= start) & (self.gauge_data['Day'] <= end)
        period = self.gauge_data.loc[mask]

        num_station = len(self.stations)
        total_rain = np.zeros((num_station, 1))
        for i, st in enumerate(self.stations):
            sdata = period[period['Station'] == st]
            sdata = sdata.copy() 
            sdata['R'] = sdata['R'].clip(lower=0, upper=self.config.DATA.RAIN_THRESHOLD)
            total_rain[i, 0] = sdata['R'].sum()
        return np.hstack((total_rain, self.station_coords))

    def get_esp_data(self, esp_path, year, lead_time):
        arr = np.load(esp_path)  # (20,47,H,W)
        yr_idx = year - 2004
        # time_idx = np.r_[:self.config.MODEL.TIME_STEP]
        arr = arr[yr_idx, -self.config.MODEL.TIME_STEP:, :, :]
        return np.clip(arr, 0, self.config.DATA.RAIN_THRESHOLD)

    def transform_ecmwf(self, ecmwf_data):
        # Transform ecmwf_data
        ecmwf_data_scaled = np.zeros_like(ecmwf_data, dtype=np.float32)
        for i in range(13):
            feature_data = ecmwf_data[i].reshape(-1, 1)
            scaled_feature = self.ecmwf_scaler[i].transform(feature_data)
            ecmwf_data_scaled[i] = scaled_feature.reshape(self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH)
        return ecmwf_data_scaled


    def transform_ground_truth(self, ground_truth):
        # Transform ground_truth
        ground_truth_scaled = ground_truth.copy()
        rain_data = ground_truth[:, 0].reshape(-1, 1)
        if self.output_norm:
            
            scaled_rain = self.output_scaler.transform(rain_data)
            
        else:
            scaled_rain = rain_data
            
        ground_truth_scaled[:, 0] = scaled_rain.flatten()
        return ground_truth_scaled

    def transform_esp_data(self, arr):
        shp = arr.shape
        flat = arr.reshape(-1, 1)
        scaled = self.esp_scaler.transform(flat)
        return scaled.reshape(shp)
        

    

    def __getitem__(self, idx):
        ecmwf_path, esp_path, lead_time, year, month, day = self.idx_df[idx]
        p_ecmwf = f'{self.processed_ecmwf_dir}seed52_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/ecmwf_data_{idx}.npy'
        p_esp   = f'{self.processed_ecmwf_dir}seed52_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/esp_data_{idx}.npy'

        ecmwf_data = np.load(p_ecmwf)    # (time, feature, H, W)
        esp_data   = np.load(p_esp)      # (8, H, W)
        ground_truth =  self.get_ground_truth(year, month, day, lead_time)
        
        ground_truth = self.transform_ground_truth(ground_truth)
        
        x_leadtime = self.calculate_leadtime_date(year, month, day, lead_time)
        return {
            'x': ecmwf_data,
            'lead_time': lead_time,
            'y': ground_truth,
            'x_leadtime': x_leadtime,
            'h': esp_data,
            'ecmwf': self.get_ecmwf(ecmwf_path, year, lead_time)
        }

    def __len__(self):
        return len(self.idx_df)



class CustomDataset3_2Head(Dataset):
    def __init__(self, mode='train', ecmwf_scaler=None, esp_scaler=None,
                 output_scaler=None, config=None, shuffle=False):
        self.config = config
        self.mode = mode
        self.shuffle = shuffle
        self.output_norm = config.TRAIN.OUTPUT_NORM



        # Load dữ liệu gauge
        self.gauge_data = pd.read_csv(config.DATA.GAUGE_DATA_PATH)
        self.gauge_data['Day'] = pd.to_datetime(self.gauge_data['Day'])

        # Station
        station_data = self.gauge_data[['Station', 'Lon', 'Lat']].drop_duplicates('Station')
        self.stations = station_data['Station'].values
        self.station_coords = station_data[['Lon', 'Lat']].to_numpy()

        # Index
        self.idx_df = pd.read_csv(f'{config.DATA.DATA_IDX_DIR}/{mode}.csv').values
        if self.shuffle:
            np.random.shuffle(self.idx_df)

        self.ecmwf_scaler = ecmwf_scaler
        self.output_scaler = output_scaler
        self.esp_scaler = esp_scaler

        self.processed_ecmwf_dir = f'{config.DATA.PROCESSED_ECMWF_DIR}/processed_ecmwf/{mode}'
        os.makedirs(self.processed_ecmwf_dir, exist_ok=True)
        self.process_and_save_ecmwf_data()

    def calculate_leadtime_date(self, y, m, d, lead_time):
        try:
            start = datetime(y, m, d) + timedelta(days=lead_time)
        except ValueError:
            start = datetime(y, 2, 28) + timedelta(days=lead_time)
        return start

    def process_and_save_ecmwf_data(self):
        os.makedirs(f'{self.processed_ecmwf_dir}seed{self.config.MODEL.SEED}_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}', exist_ok=True)
        for idx in tqdm(range(len(self.idx_df)), desc=f"Processing ecmwf_data for {self.mode}"):
            p_ecmwf = f'{self.processed_ecmwf_dir}seed{self.config.MODEL.SEED}_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/ecmwf_data_{idx}.npy'
            p_esp   = f'{self.processed_ecmwf_dir}seed{self.config.MODEL.SEED}_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/esp_data_{idx}.npy'

            if not os.path.exists(p_ecmwf):
                ecmwf_path, esp_path, lead_time, year, month, day = self.idx_df[idx]
                ecmwf_data = self.get_ecmwf(ecmwf_path, year, lead_time)
                ecmwf_data_scaled = self.transform_ecmwf(ecmwf_data)
                ecmwf_data_scaled = ecmwf_data_scaled.transpose(1, 0, 2, 3)  # (time, feature, H, W)
                np.save(p_ecmwf, ecmwf_data_scaled)

            if not os.path.exists(p_esp):
                ecmwf_path, esp_path, lead_time, year, month, day = self.idx_df[idx]
                esp_data = self.get_esp_data(esp_path, year, lead_time)
                esp_scaled = self.transform_esp_data(esp_data)
                np.save(p_esp, esp_scaled)

    def get_ecmwf(self, ecmwf_path, year, lead_time):
        data = np.load(ecmwf_path)  # (20,13,47,H,W)
        data = data[year-2004]
        data[0]  = np.clip(data[0], 0, self.config.DATA.RAIN_THRESHOLD*3)
        data[12] = np.clip(data[12], 0, self.config.DATA.RAIN_THRESHOLD*3)
        data[7] = np.clip(data[7], 0, 15)
        data_pad = np.pad(data, pad_width=((0,0),(1,1),(0,0),(0,0)), mode='edge')
        t0 = data_pad[:, 2:   , ...]              # x[t]
        t1 = data_pad[:, 1:-1, ...]               # x[t-1]
        t2 = data_pad[:,  :-2, ...]               # x[t-2]
        trend = (t0 + t1 + t2) / 3.0           # (V,T,H,W)

        # --- Season: s_t = x_t - 2*x_{t-1} + x_{t-2} (dùng cùng padding) ---
        season = t0 - 2.0*t1 + t2              # (V,T,H,W)
        data = data[:, lead_time-self.config.MODEL.ECMWF_TIME_STEP:lead_time, :, :]  # (13,1,H,W)
        
        return trend[:, lead_time-self.config.MODEL.ECMWF_TIME_STEP:lead_time, :, :]
    def get_ecmwf_only(self, ecmwf_path, year, lead_time):
        data = np.load(ecmwf_path)  # (20,13,47,H,W)
        data = data[year-2004]
        # data_pad = np.pad(data, pad_width=((0,0),(1,1),(0,0),(0,0)), mode='edge')
        data = data[:, lead_time-self.config.MODEL.ECMWF_TIME_STEP:lead_time, :, :]  # (13,1,H,W)
        data[0]  = np.clip(data[0], 0, self.config.DATA.RAIN_THRESHOLD)
        data[12] = np.clip(data[12], 0, self.config.DATA.RAIN_THRESHOLD)
        data[7] = np.clip(data[7], 0, 15)
        return data

    def get_ground_truth(self, year, month, day, lead_time):
        try:
            start = datetime(year, month, day) + timedelta(days=lead_time)
        except ValueError:
            start = datetime(year, 2, 28) + timedelta(days=lead_time)
            
        try:
            end = datetime(year, month, day) + timedelta(days=lead_time)
        except ValueError:
            end = datetime(year, 2, 28) + timedelta(days=lead_time)
        
        mask = (self.gauge_data['Day'] >= start) & (self.gauge_data['Day'] <= end)
        period = self.gauge_data.loc[mask]
        threshold = self.config.DATA.RAIN_PROB
        num_station = len(self.stations)
        total_rain = np.zeros((num_station, 1))
        rain_prob = np.zeros((num_station, 1))
        for i, st in enumerate(self.stations):
            sdata = period[period['Station'] == st]
            sdata = sdata.copy() 
            sdata['R'] = sdata['R'].clip(lower=0, upper=self.config.DATA.RAIN_THRESHOLD)
            total_rain[i, 0] = sdata['R'].sum()
            if(total_rain[i, 0] >= threshold): rain_prob[i, 0] = 1
            else: rain_prob[i, 0] = 0
        return np.hstack((total_rain, self.station_coords)), np.hstack((rain_prob, self.station_coords))

    def get_esp_data(self, esp_path, year, lead_time):
        arr = np.load(esp_path)  # (20,47,H,W)
        yr_idx = year - 2004
        # time_idx = np.r_[:self.config.MODEL.TIME_STEP]
        arr = arr[yr_idx, -self.config.MODEL.TIME_STEP:, :, :]
        return np.clip(arr, 0, self.config.DATA.RAIN_THRESHOLD*3)

    def transform_ecmwf(self, ecmwf_data):
        # Transform ecmwf_data
        ecmwf_data_scaled = np.zeros_like(ecmwf_data, dtype=np.float32)
        for i in range(13):
            feature_data = ecmwf_data[i].reshape(-1, 1)
            scaled_feature = self.ecmwf_scaler[i].transform(feature_data)
            ecmwf_data_scaled[i] = scaled_feature.reshape(self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH)
        return ecmwf_data_scaled


    def transform_ground_truth(self, ground_truth):
        # Transform ground_truth
        ground_truth_scaled = ground_truth.copy()
        rain_data = ground_truth[:, 0].reshape(-1, 1)
        if self.output_norm:
            
            scaled_rain = self.output_scaler.transform(rain_data)
            
        else:
            scaled_rain = rain_data
            
        ground_truth_scaled[:, 0] = scaled_rain.flatten()
        return ground_truth_scaled

    def transform_esp_data(self, arr):
        shp = arr.shape
        flat = arr.reshape(-1, 1)
        scaled = self.esp_scaler.transform(flat)
        return scaled.reshape(shp)
        

    

    def __getitem__(self, idx):
        ecmwf_path, esp_path, lead_time, year, month, day = self.idx_df[idx]
        p_ecmwf = f'{self.processed_ecmwf_dir}seed{self.config.MODEL.SEED}_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/ecmwf_data_{idx}.npy'
        p_esp   = f'{self.processed_ecmwf_dir}seed{self.config.MODEL.SEED}_ecmwf_{self.config.MODEL.ECMWF_TIME_STEP}_esp_{self.config.MODEL.TIME_STEP}/esp_data_{idx}.npy'

        ecmwf_data = np.load(p_ecmwf)    # (time, feature, H, W)
        esp_data   = np.load(p_esp)      # (8, H, W)
        ground_truth, rain_prob =  self.get_ground_truth(year, month, day, lead_time)
        
        ground_truth = self.transform_ground_truth(ground_truth)
        
        x_leadtime = self.calculate_leadtime_date(year, month, day, lead_time)
        return {
            'x': ecmwf_data,
            'lead_time': lead_time,
            'y': ground_truth,
            'prob': rain_prob,
            'x_leadtime': x_leadtime,
            'h': esp_data,
            'ecmwf': self.get_ecmwf_only(ecmwf_path, year, lead_time)
        }

    def __len__(self):
        return len(self.idx_df)