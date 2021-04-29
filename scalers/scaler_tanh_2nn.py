import pandas as pd
import numpy as np
import joblib

from BiTanhScaler import BiTanhScaler

from config import *

def prepare_dataframe(path, columns):
    dataframe = pd.read_csv(path)
    dataframe = dataframe.filter(columns)
    dataframe = dataframe.astype(float)
    return dataframe


def create_column_associated_with_num_task(dataframe, columns):
    for x, y in columns.items():
        dataframe[x] = dataframe.apply(lambda row: row['num_tasks'] * row[y], axis=1)
    to_delete = [*columns.values()]
    to_delete.append('num_tasks')
    dataframe = dataframe.drop(columns=to_delete)
    return dataframe


def transform_submission_time(dataframe):
    dataframe['submission_time_transformed'] = dataframe.apply(
        lambda row: (float(row['submission_time']) % 86400) / 60.0, axis=1)
    dataframe = dataframe.drop(['submission_time'], axis=1)
    return dataframe


def prepare_data_for_model(dataframe, columns, scaler):
    data_training = dataframe[columns]
    data_training = scaler.fit_transform(data_training, PREDICTED_COLUMN_NAME)
    X_train = []
    Y_train = []
    for i in range(60, data_training.shape[0]):
        X_train.append(data_training[i - 60:i])
        # index of guessed column (e.g. queue_time_till_fully_scheduled, makespan -1 if column is provided at the end)
        Y_train.append(data_training[i])
    return np.array(X_train), np.array(Y_train)


# DIVISION OF DATA BETWEEN TEST AND TRAINING

dataframe = prepare_dataframe(DATA, COLUMNS)
print(dataframe['queue_time_till_fully_scheduled'])

dataframe = create_column_associated_with_num_task(dataframe, ADDITIONAL_COLUMNS)
print(dataframe)

dataframe = transform_submission_time(dataframe)
data_test = dataframe[:1000]
dataframe_training = dataframe[1000:dataframe.shape[0]]

# SCALER GENERATION

print(dataframe_training[SELECTED_COLUMNS])
scaler = BiTanhScaler(0.75)
X, Y = prepare_data_for_model(dataframe_training, SELECTED_COLUMNS, scaler)

#SCALER TESTS
transformed = scaler.transform(dataframe[SELECTED_COLUMNS])
print(transformed)

print(scaler.max)
print(scaler.min)

#SAVE SCALER
scaler_filename = "./scaler_tanh.save"
joblib.dump(scaler, scaler_filename)
