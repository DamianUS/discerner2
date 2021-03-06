import pandas as pd
import numpy as np
import joblib

import config
import utils
from config import *
from scalers.TanhScaler import TanhScaler

# DIVISION OF DATA BETWEEN TEST AND TRAINING

dataframe = utils.prepare_dataframe()
print(dataframe['queue_time_till_fully_scheduled'])

training_size = int(config.TEST_SPLIT * dataframe.shape[0])
data_test = dataframe[:training_size]
dataframe_training = dataframe[training_size:dataframe.shape[0]]

# SCALER GENERATION

print(dataframe_training[SELECTED_COLUMNS])
scaler = TanhScaler()
X, Y = utils.prepare_data_for_model(dataframe_training, SELECTED_COLUMNS, scaler)

#SCALER TESTS
transformed = scaler.transform(dataframe[SELECTED_COLUMNS])
print(transformed)

print(scaler.max)
print(scaler.min)

#SAVE SCALER
scaler_filename = "./tanh_scaler.save"
joblib.dump(scaler, scaler_filename)
