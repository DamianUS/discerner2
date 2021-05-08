import pandas as pd
import numpy as np
import joblib

import config
import utils
from config import *
from scalers.NTanhScaler import NTanhScaler

# DIVISION OF DATA BETWEEN TEST AND TRAINING

dataframe = utils.prepare_dataframe()
print(dataframe[config.PREDICTED_COLUMN_NAME])

training_size = int(config.TEST_SPLIT * dataframe.shape[0])
data_test = dataframe[:training_size]
dataframe_training = dataframe[training_size:dataframe.shape[0]]

# SCALER GENERATION

print(dataframe_training[SELECTED_COLUMNS])
scaler = NTanhScaler(config.PREDICTED_COLUMN_NAME, config.SPLIT_PERCENTILES)
X, Y = utils.prepare_data_for_model(dataframe_training, SELECTED_COLUMNS, scaler)

#SCALER TESTS
transformed = scaler.transform(dataframe[SELECTED_COLUMNS])
print(transformed)

#SAVE SCALER
scaler_filename = "./ntanh_scaler.save"
joblib.dump(scaler, scaler_filename)
