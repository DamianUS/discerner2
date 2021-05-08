import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
import csv
from tensorflow import feature_column
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
import joblib
import config
import utils
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='cpu')


def save_model(name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5")
    print("Saved model to disk")


def plot_metric(history, metric, title):
    train_metrics = history.history[metric]
    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo', label="training " + metric)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

scaler = joblib.load(config.SCALER)
dataframe = utils.prepare_mesos_dataframe()
test_dataset_size = int(round(config.TEST_SPLIT * dataframe.shape[0]))

data_test = dataframe[:test_dataset_size]
dataframe_training = dataframe[test_dataset_size:dataframe.shape[0]]

data_training = dataframe_training[config.SELECTED_COLUMNS]
data_training = scaler.transform(data_training)
# CPU y MEM per batch
# transformed_dataset = pd.DataFrame(
#     {'estimated_task_duration': data_training[:, 0], 'data_center_utilization_at_submission': data_training[:, 1],
#      'cpus_per_batch': data_training[:, 2], 'mem_per_batch': data_training[:, 3],
#      'submission_time_transformed': data_training[:, 4], 'queue_time_till_fully_scheduled': data_training[:, 5]})
# Num_tasks
transformed_dataset = pd.DataFrame(
    {'estimated_task_duration': data_training[:, 0], 'data_center_utilization_at_submission': data_training[:, 1],
     'num_tasks': data_training[:, 2],
     'submission_time_transformed': data_training[:, 3], 'queue_time_till_fully_scheduled': data_training[:, 4]})
transformed_dataset.hist()
plt.title("Histogram for transformed values")
plt.show()

x_train = []
y_train = []

for i in range(config.BATCH_SIZE, data_training.shape[0]):
    x_train.append(data_training[i - config.BATCH_SIZE:i])
    # index of queue_time_till_fully_scheduled column is 1
    y_train.append(data_training[i, config.PREDICTED_COLUMN_INDEX])

x_train, y_train = np.array(x_train), np.array(y_train)

transformed_dataset = pd.DataFrame(y_train)
transformed_dataset.hist()
plt.title("Histogram of prediction column")
plt.show()

model = Sequential()
model.add(LSTM(units=10, activation='tanh', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=20, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=30, activation='tanh'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
result = model.fit(x_train, y_train, epochs=config.EPOCHS, batch_size=config.ANN_BATCH_SIZE, validation_split=config.VALIDATION_SPLIT)

result.history["loss"]
numpy_loss_history = np.array(result.history["loss"])
np.savetxt("./loss_history_mesos.csv", result.history["loss"], delimiter=",")

pd.DataFrame(result.history).to_csv("history.csv")
plot_metric(result, 'loss', 'Training loss')

save_model(config.MESOS_MODEL)
