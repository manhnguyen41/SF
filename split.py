import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import os

def split():
    seed = 52
    folder_dir = f"/mnt/disk1/nxmanh/Hydrometeology/Subseasonal_Prediction/data789_seed{seed}"
    os.makedirs(folder_dir, exist_ok=True)

    # Create a list of dates from 2022-01-03 to 2022-12-31
    start_date = datetime.date(2022, 1, 3)
    end_date = datetime.date(2022, 12, 31)
    dates = []
    current_date = start_date
    day_steps = [3, 4]
    step_index = 0

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += datetime.timedelta(days=day_steps[step_index])
        step_index = 1 - step_index
        
    # Create a dataframe from the list of dates
    data = pd.DataFrame({"date": dates})

    # Add month and day columns
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day

    # Filter data for summer months (7, 8, 9)
    data = data[data['month'].isin([6, 7, 8, 9])]

    # Create a list of years from 2002 to 2021
    years = list(range(2002, 2022))
    lead_times = list(range(7, 47))  # 7, 8, ..., 46

    # Create index for each date, year, and lead time
    expanded_data = []
    for index, row in data.iterrows():
        date = row['date']
        month = row['month']
        day = row['day']
        for year in years:
            for lead_time in lead_times:
                path_ecmwf = f"/mnt/disk1/env_data/S2S_0.125/nparr/Step24h/{date}.npy"
                expanded_data.append({
                    "pathECMWF": path_ecmwf,
                    "leadTime": lead_time,
                    "year": year,
                    "month": month,
                    "day": day
                })

    data = pd.DataFrame(expanded_data)

    # Split data into train/valid/test
    # Take data from 2018 to 2021 as test data
    test_data = data[data["year"].isin([2018, 2019, 2020, 2021])]

    # Take data from 2002 to 2017 as remaining data
    remaining_data = data[~data["year"].isin([2018, 2019, 2020, 2021])]

    # Split remaining data into train and valid data with ratio 3:1
    train_data, valid_data = train_test_split(
        remaining_data, 
        test_size=0.25,  # Tỷ lệ 3:1 (train:valid)
        random_state=seed,
        stratify=remaining_data['leadTime']  
    )

    # Save data to csv files
    train_data.to_csv(f"{folder_dir}/train.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])
    valid_data.to_csv(f"{folder_dir}/valid.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])
    test_data.to_csv(f"{folder_dir}/test.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])

    # Print data shape
    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
def split_reg_4():
    seed = 52
    folder_dir = f"/mnt/disk1/nxmanh/Hydrometeology/Subseasonal_Prediction/data789_reg_4_seed{seed}"
    os.makedirs(folder_dir, exist_ok=True)

    # Create a list of dates from 2022-01-03 to 2022-12-31
    start_date = datetime.date(2022, 1, 3)
    end_date = datetime.date(2022, 12, 31)
    dates = []
    current_date = start_date
    day_steps = [3, 4]
    step_index = 0

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += datetime.timedelta(days=day_steps[step_index])
        step_index = 1 - step_index
        
    # Create a dataframe from the list of dates
    data = pd.DataFrame({"date": dates})

    # Add month and day columns
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day

    # Filter data for summer months (7, 8, 9)
    data = data[data['month'].isin([7, 8, 9])]

    # Create a list of years from 2002 to 2021
    years = list(range(2002, 2022))
    lead_times = list(range(7, 47))  # 7, 8, ..., 46

    # Create index for each date, year, and lead time
    expanded_data = []
    for index, row in data.iterrows():
        date = row['date']
        month = row['month']
        day = row['day']
        for year in years:
            for lead_time in lead_times:
                path_ecmwf = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_4/Step24h/{date}.npy"
                expanded_data.append({
                    "pathECMWF": path_ecmwf,
                    "leadTime": lead_time,
                    "year": year,
                    "month": month,
                    "day": day
                })

    data = pd.DataFrame(expanded_data)

    # Split data into train/valid/test
    # Take data from 2018 to 2021 as test data
    test_data = data[data["year"].isin([2018, 2019, 2020, 2021])]

    # Take data from 2002 to 2017 as remaining data
    remaining_data = data[~data["year"].isin([2018, 2019, 2020, 2021])]

    # Split remaining data into train and valid data with ratio 3:1
    train_data, valid_data = train_test_split(
        remaining_data, 
        test_size=0.25,  # Tỷ lệ 3:1 (train:valid)
        random_state=seed,
        stratify=remaining_data['leadTime']  
    )

    # Save data to csv files
    train_data.to_csv(f"{folder_dir}/train.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])
    valid_data.to_csv(f"{folder_dir}/valid.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])
    test_data.to_csv(f"{folder_dir}/test.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])

    # Print data shape
    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
def split_reg_lead_1(region):
    seed = 52
    folder_dir = f"/mnt/disk1/nxmanh/Hydrometeology/Subseasonal_Prediction/data789_reg_{region}__lead_1seed{seed}"
    os.makedirs(folder_dir, exist_ok=True)

    # Create a list of dates from 2022-01-03 to 2022-12-31
    start_date = datetime.date(2022, 1, 3)
    end_date = datetime.date(2022, 12, 31)
    dates = []
    current_date = start_date
    day_steps = [3, 4]
    step_index = 0

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += datetime.timedelta(days=day_steps[step_index])
        step_index = 1 - step_index
        
    # Create a dataframe from the list of dates
    data = pd.DataFrame({"date": dates})

    # Add month and day columns
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day

    # Filter data for summer months (7, 8, 9)
    data = data[data['month'].isin([7, 8, 9])]

    # Create a list of years from 2002 to 2021
    years = list(range(2002, 2022))
    lead_times = list(range(7, 27))  # 7, 8, ..., 46

    # Create index for each date, year, and lead time
    expanded_data = []
    for index, row in data.iterrows():
        date = row['date']
        month = row['month']
        day = row['day']
        for year in years:
            for lead_time in lead_times:
                path_ecmwf = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_{region}/Step24h/{date}.npy"
                expanded_data.append({
                    "pathECMWF": path_ecmwf,
                    "leadTime": lead_time,
                    "year": year,
                    "month": month,
                    "day": day
                })

    data = pd.DataFrame(expanded_data)

    # Split data into train/valid/test
    # Take data from 2018 to 2021 as test data
    test_data = data[data["year"].isin([2018, 2019, 2020, 2021])]

    # Take data from 2002 to 2017 as remaining data
    remaining_data = data[~data["year"].isin([2018, 2019, 2020, 2021])]

    # Split remaining data into train and valid data with ratio 3:1
    train_data, valid_data = train_test_split(
        remaining_data, 
        test_size=0.25,  # Tỷ lệ 3:1 (train:valid)
        random_state=seed,
        stratify=remaining_data['leadTime']  
    )

    # Save data to csv files
    train_data.to_csv(f"{folder_dir}/train.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])
    valid_data.to_csv(f"{folder_dir}/valid.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])
    test_data.to_csv(f"{folder_dir}/test.csv", 
                    index=False, 
                    columns=["pathECMWF", "leadTime", "year", "month", "day"])

    # Print data shape
    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
def split_reg(region):
    seed = 52
    range_lead_list = {"short": [7, 20], "medium": [21, 33], "long":[34, 46]}
    for range_lead, nums in range_lead_list.items():
        folder_dir = f"/mnt/disk3/tunm/Subseasonal_Forecasting/data1/data6789_reg_{region}_seed{seed}_new_{range_lead}"
    
        os.makedirs(folder_dir, exist_ok=True)

        # Create a list of dates from 2022-01-03 to 2022-12-31
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 12, 31)

        dates = []
        current_date = start_date
        day_steps = [3, 4]  # Trước 11-11
        step_index = 0

        # Tạo danh sách dates
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            # Từ 11-11 trở đi dùng bước 2 ngày
            if current_date >= datetime.date(2024, 11, 11):
                current_date += datetime.timedelta(days=2)
            else:
                current_date += datetime.timedelta(days=day_steps[step_index])
                step_index = 1 - step_index
            
        # Create a dataframe from the list of dates
        data = pd.DataFrame({"date": dates})

        # Add month and day columns
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day'] = pd.to_datetime(data['date']).dt.day

        # Filter data for summer months (7, 8, 9)
        data = data[data['month'].isin([6, 7, 8, 9])]
        # data = data[data['month'].isin([9, 10, 11])]

        # Create a list of years from 2004 to 2023
        years = list(range(2004, 2024))
        lead_times = list(range(nums[0], nums[1] + 1))  # 7, 8, ..., 46
        
        # Create index for each date, year, and lead time
        expanded_data = []
        for index, row in data.iterrows():
            date = row['date']
            month = row['month']
            day = row['day']
            
            for year in years:
                for lead_time in lead_times:
                    
                    path_ecmwf = f"/mnt/disk3/longnd/env_data/grid_base/nparr_reg{region}/Step24h/{date[-5:]}.npy"
                    path_esp = f"/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_{region}/{date[-5:]}.npy"
                    expanded_data.append({
                        "pathECMWF": path_ecmwf,
                        "pathESP": path_esp,
                        "leadTime": lead_time,
                        "year": year,
                        "month": month,
                        "day": day
                    })

        data = pd.DataFrame(expanded_data)

        # Split data into train/valid/test
        # Take data from 2018 to 2021 as test data
        test_data = data[data["year"].isin([2022, 2023])]

        # Take data from 2002 to 2017 as remaining data
        remaining_data = data[~data["year"].isin([2022, 2023])]
        remaining_data = remaining_data.copy()
        remaining_data['stratify_key'] = remaining_data['leadTime'].astype(str) + "_" + remaining_data['month'].astype(str)
        # Split remaining data into train and valid data with ratio 3:1
        train_data, valid_data = train_test_split(
            remaining_data, 
            test_size=2/9,  # Tỷ lệ 3:1 (train:valid)
            random_state=seed,
            stratify=remaining_data['stratify_key']
        )

        # Save data to csv files
        train_data.to_csv(f"{folder_dir}/train.csv", 
                        index=False, 
                        columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])
        valid_data.to_csv(f"{folder_dir}/valid.csv", 
                        index=False, 
                        columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])
        test_data.to_csv(f"{folder_dir}/test.csv", 
                        index=False, 
                        columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])

    # Print data shape
    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
def split_train_valid_reg(region):
    seed = 92
    # range_lead_list = {"short": [1, 7], "medium": [6, 10], "long":[34, 46], "all":[7, 46] }
    range_lead_list = {"all":[7, 46] }
    for range_lead, nums in range_lead_list.items():
        folder_dir = f"/mnt/disk3/tunm/Subseasonal_Forecasting/data3/data6789_reg_{region}_seed{seed}_new_{range_lead}"
    
        os.makedirs(folder_dir, exist_ok=True)

        # Create a list of dates from 2022-01-03 to 2022-12-31
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 12, 31)

        dates = []
        current_date = start_date
        day_steps = [3, 4]  # Trước 11-11
        step_index = 0

        # Tạo danh sách dates
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            # Từ 11-11 trở đi dùng bước 2 ngày
            if current_date >= datetime.date(2024, 11, 11):
                current_date += datetime.timedelta(days=2)
            else:
                current_date += datetime.timedelta(days=day_steps[step_index])
                step_index = 1 - step_index
            
        # Create a dataframe from the list of dates
        data = pd.DataFrame({"date": dates})

        # Add month and day columns
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day'] = pd.to_datetime(data['date']).dt.day

        # Filter data for summer months (7, 8, 9)
        # data = data[data['month'].isin([6, 7, 8, 9])]
        # data = data[data['month'].isin([9, 10, 11])]

        # Create a list of years from 2004 to 2023
        
        years = list(range(2004, 2024))
        lead_times = list(range(nums[0], nums[1] + 1))  # 7, 8, ..., 46

        # Create index for each date, year, and lead time
        expanded_data = []
        for index, row in data.iterrows():
            date = row['date']
            month = row['month']
            day = row['day']
            for year in years:
                for lead_time in lead_times:
                    path_ecmwf = f"/mnt/disk3/longnd/env_data/grid_base/nparr_reg{region}/Step24h/{date[-5:]}.npy"
                    path_esp = f"/mnt/disk3/longnd/env_data/grid_base/GMSaP/preprocess_region_{region}/{date[-5:]}.npy"
                    expanded_data.append({
                        "pathECMWF": path_ecmwf,
                        "pathESP": path_esp,
                        "leadTime": lead_time,
                        "year": year,
                        "month": month,
                        "day": day
                    })

        data = pd.DataFrame(expanded_data)

        # Split data into train/valid/test
        # Take data from 2018 to 2021 as test data
        val_years  = [2020, 2021]
        test_years = [2022, 2023]
        test_data = data[data["year"].isin(test_years)]

        valid_data = data[data["year"].isin(val_years)]

        # Take data from 2002 to 2017 as remaining data
        train_data = data[~data["year"].isin(val_years + test_years)]
        # remain_data = data[~data["year"].isin([2021, 2022, 2023])]
        # train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save data to csv files
        train_data.to_csv(f"{folder_dir}/train.csv", 
                        index=False, 
                        columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])
        valid_data.to_csv(f"{folder_dir}/valid.csv", 
                        index=False, 
                        columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])
        test_data.to_csv(f"{folder_dir}/test.csv", 
                        index=False, 
                        columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])
        # remain_data.to_csv(f"{folder_dir}/remain.csv",
        #                 index=False, 
        #                 columns=["pathECMWF", "pathESP", "leadTime", "year", "month", "day"])
        # Print data shape
        print(f"Train data shape {range_lead}: {train_data.shape}")
        print(f"Valid data shape {range_lead}: {valid_data.shape}")
        print(f"Test data shape {range_lead}: {test_data.shape}")

def splitv2_train_valid_reg(region):
    seed = 52
    range_lead_list = {"short": [7, 20], "medium": [21, 33], "long":[34, 46]}
    for range_lead, nums in range_lead_list:
        folder_dir = f"/mnt/disk1/tunn/Subseasonal_Prediction/data4/data6789_reg_{region}_seed{seed}_{range_lead}"
        os.makedirs(folder_dir, exist_ok=True)

        # Create a list of dates from 2022-01-03 to 2022-12-31
        start_date = datetime.date(2022, 1, 3)
        end_date = datetime.date(2022, 12, 31)
        dates = []
        current_date = start_date
        day_steps = [3, 4]
        step_index = 0

        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += datetime.timedelta(days=day_steps[step_index])
            step_index = 1 - step_index
            
        # Create a dataframe from the list of dates
        data = pd.DataFrame({"date": dates})

        # Add month and day columns
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day'] = pd.to_datetime(data['date']).dt.day

        # Filter data for summer months (7, 8, 9)
        data = data[data['month'].isin([6, 7, 8, 9])]
        # data = data[data['month'].isin([9, 10, 11])]

        # Create a list of years from 2002 to 2021
        years = list(range(2002, 2022))
        lead_times = list(range(nums[0], nums[1] + 1))  # 7, 8, ..., 46

        # Create index for each date, year, and lead time
        expanded_data = []
        for index, row in data.iterrows():
            date = row['date']
            month = row['month']
            day = row['day']
            for year in years:
                for lead_time in lead_times:
                    path_ecmwf = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg{region}/Step24h/{date}.npy"
                    expanded_data.append({
                        "pathECMWF": path_ecmwf,
                        "leadTime": lead_time,
                        "year": year,
                        "month": month,
                        "day": day
                    })

        data = pd.DataFrame(expanded_data)

        # Split data into train/valid/test
        # Take data from 2018 to 2021 as test data
        test_data = data[data["year"].isin([2018, 2019, 2020, 2021])]

        valid_data = data[data["year"].isin([2002, 2003, 2004, 2005])]

        # Take data from 2002 to 2017 as remaining data
        train_data = data[~data["year"].isin([2002, 2003, 2004, 2005, 2018, 2019, 2020, 2021])]
        
        #train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save data to csv files
        train_data.to_csv(f"{folder_dir}/train.csv", 
                        index=False, 
                        columns=["pathECMWF", "leadTime", "year", "month", "day"])
        valid_data.to_csv(f"{folder_dir}/valid.csv", 
                        index=False, 
                        columns=["pathECMWF", "leadTime", "year", "month", "day"])
        test_data.to_csv(f"{folder_dir}/test.csv", 
                        index=False, 
                        columns=["pathECMWF", "leadTime", "year", "month", "day"])

        # Print data shape
        print(f"Train data shape: {train_data.shape}")
        print(f"Valid data shape: {valid_data.shape}")
        print(f"Test data shape: {test_data.shape}")
    

# split_reg(1)
split_train_valid_reg(1)