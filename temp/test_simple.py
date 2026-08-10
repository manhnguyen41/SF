import numpy as np
import joblib

# Test pickling and unpickling a NumPy array
arr = np.array([1, 2, 3, 4, 5])
# joblib.dump(arr, '/mnt/disk1/tunn/Subseasonal_Prediction/data6789_reg_1_seed52/scalers.pkl')
loaded_arr = joblib.load('/mnt/disk1/tunn/Subseasonal_Prediction/data6789_reg_2_seed52/scalers.pkl')
print(loaded_arr)
