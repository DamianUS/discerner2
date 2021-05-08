import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import feature_column
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.models import model_from_json
import joblib
import config
import utils
from scipy import stats


def load_model(filename):
    json_file = open(filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + ".h5")
    return loaded_model


scaler = joblib.load(config.SCALER)
dataframe = utils.prepare_mesos_dataframe()[config.SELECTED_COLUMNS]
test_dataset_size = int(round(config.TEST_SPLIT * dataframe.shape[0]))
data_test = dataframe[:test_dataset_size]
dataframe_training = dataframe[test_dataset_size:dataframe.shape[0]]

# transformed = scaler.transform(data_test[config.SELECTED_COLUMNS])

y_test = []
x_test = []

for i in range(config.BATCH_SIZE, data_test.shape[0]):
    x_test.append(data_test[i - config.BATCH_SIZE:i])
    y_test.append(data_test[config.PREDICTED_COLUMN_NAME][i])

x_test_transformed = []
# for e in enumerate(x_test):
#     batch = e[1]
#     batch_transformed = scaler.transform(batch)
#     x_test_transformed.append(batch_transformed)

# Very slow
# x_test_transformed = [scaler.transform(batch) for batch in x_test]


models = []
for index in range(len(scaler.limits) + 1):
    model = load_model(config.MESOS_MODEL+"_"+str(index))
    model.compile(optimizer='adam', loss='mean_squared_error')
    models.append(model)

scalers_used = []
y_predicted_transformed = []
for batch in x_test:
    model_number = scaler.get_range_count(batch)
    #slow
    #batch_transformed = scaler.transform(batch)
    batch_transformed = scaler.scalers[model_number].transform(batch)
    model = models[model_number]
    scalers_used.append(model_number)
    x_test_transformed.append(batch_transformed)
    res = model.predict(batch_transformed.reshape(1, config.BATCH_SIZE, len(config.SELECTED_COLUMNS)))
    y_predicted_transformed.append(res[0][0])

x_test_transformed, y_test = np.array(x_test_transformed), np.array(y_test)
y_predicted_transformed = np.array(y_predicted_transformed)

#y_predicted_transformed = model.predict(x_test_transformed)
# CPU y MEM per batch
# y_predicted_transformed = np.c_[
#     np.ones(y_predicted_transformed.shape[0]), np.ones(y_predicted_transformed.shape[0]), np.ones(
#         y_predicted_transformed.shape[0]), np.ones(y_predicted_transformed.shape[0]), np.ones(
#         y_predicted_transformed.shape[0]), y_predicted_transformed]
# Num_tasks
y_predicted_transformed = np.c_[
    np.ones(y_predicted_transformed.shape[0]), np.ones(y_predicted_transformed.shape[0]), np.ones(
        y_predicted_transformed.shape[0]), np.ones(y_predicted_transformed.shape[0]), y_predicted_transformed]

y_predicted = []
# TanhScaler
if 'tanh' in config.SCALER:
    for index in range(y_predicted_transformed.shape[0]):
        row = y_predicted_transformed[index]
        scaler_index = scalers_used[index]
        prediction = row.reshape(1,len(config.SELECTED_COLUMNS))
        y_predicted.append(scaler.inverse_transform(prediction, config.SELECTED_COLUMNS, scaler_index)[0])
    y_predicted = np.array(y_predicted)
# QuantileScaler
elif 'quantile' in config.SCALER:
    y_predicted = scaler.inverse_transform(y_predicted_transformed)
y_predicted = y_predicted[:, config.PREDICTED_COLUMN_INDEX]
y_test, y_predicted = y_test.reshape(-1, 1), y_predicted.reshape(-1, 1)
y_error = y_test - y_predicted
y_error = np.absolute(y_error)

mode = stats.mode(y_error)
mean = y_error.mean()
median = np.percentile(y_error, 50)
p75 = np.percentile(y_error, 75)
p90 = np.percentile(y_error, 90)

plt.figure(figsize=(14, 5))
plt.plot(y_error, color='r', label="Error in prediction")

plt.title("Prediction error Mesos")
plt.xlabel("# Job")
plt.ylabel("queue_time_till_fully_scheduled")
plt.legend()
plt.show()

print("mode error: ", mode, "mean error: ", mean, "median error: ", median, "p75 error: ", p75, "p90 error: ", p90)

# plt.figure(figsize=(14, 5))
# plt.plot(y_test, color='r', label="Real value")
# plt.plot(y_predicted, color='b', label="Predicted value")
#
# plt.title("Real and predicted values for Mesos")
# plt.xlabel("Submission time")
# plt.ylabel("queue_time_till_fully_scheduled")
# plt.legend()
# plt.show()
