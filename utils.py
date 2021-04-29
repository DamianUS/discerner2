import pandas as pd
import numpy as np
import joblib

import config
from config import *


def read_dataframe(path, columns):
    dataframe = pd.read_csv(path)
    dataframe = dataframe.filter(columns)
    dataframe = dataframe.astype(float)
    return dataframe


def create_column_associated_with_num_task(dataframe, columns):
    for x, y in columns.items():
        dataframe[x] = dataframe.apply(lambda row: row['num_tasks'] * row[y], axis=1)
    # to_delete = [*columns.values()]
    # to_delete.append('num_tasks')
    # dataframe = dataframe.drop(columns=to_delete)
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


def prepare_dataframe():
    dataframe = read_dataframe(DATA, COLUMNS)
    print(dataframe['queue_time_till_fully_scheduled'])
    dataframe = create_column_associated_with_num_task(dataframe, ADDITIONAL_COLUMNS)
    print(dataframe)
    dataframe = transform_submission_time(dataframe)
    if config.UNTAIL:
        dataframe = untail_column(dataframe, config.PREDICTED_COLUMN_NAME, config.UNTAIL_MIN_PERCENTILE, config.UNTAIL_MAX_PERCENTILE)
    return dataframe


def prepare_mesos_dataframe():
    dataframe = read_dataframe(MESOS_DATA, COLUMNS)
    print(dataframe['queue_time_till_fully_scheduled'])
    dataframe = create_column_associated_with_num_task(dataframe, ADDITIONAL_COLUMNS)
    print(dataframe)
    dataframe = transform_submission_time(dataframe)
    if config.UNTAIL:
        dataframe = untail_column(dataframe, config.PREDICTED_COLUMN_NAME, config.UNTAIL_MIN_PERCENTILE, config.UNTAIL_MAX_PERCENTILE)
    return dataframe


def prepare_omega_dataframe():
    dataframe = read_dataframe(OMEGA_DATA, COLUMNS)
    print(dataframe['queue_time_till_fully_scheduled'])
    dataframe = create_column_associated_with_num_task(dataframe, ADDITIONAL_COLUMNS)
    print(dataframe)
    dataframe = transform_submission_time(dataframe)
    if config.UNTAIL:
        dataframe = untail_column(dataframe, config.PREDICTED_COLUMN_NAME, config.UNTAIL_MIN_PERCENTILE, config.UNTAIL_MAX_PERCENTILE)
    return dataframe


def untail_column(dataframe, column, min_percentile=0.0, max_percentile=1.0):
    min_limit = dataframe[column].quantile(min_percentile)
    max_limit = dataframe[column].quantile(max_percentile)

    if config.UNTAIL_MODE == "collapse":
        dataframe[column] = dataframe.apply(lambda row: max_limit if float(row[column]) > max_limit else float(row[column]), axis=1)
        dataframe[column] = dataframe.apply(lambda row: min_limit if float(row[column]) < min_limit else float(row[column]),axis=1)
    elif config.UNTAIL_MODE == "remove":
        dataframe = dataframe.loc[(dataframe[config.PREDICTED_COLUMN_NAME] < max_limit)]
        dataframe = dataframe.loc[(dataframe[config.PREDICTED_COLUMN_NAME] > min_limit)]
        dataframe_numpy = dataframe.to_numpy()
        dataframe = pd.DataFrame({dataframe.columns[0]: dataframe_numpy[:, 0], dataframe.columns[1]: dataframe_numpy[:, 1], dataframe.columns[2]: dataframe_numpy[:, 2], dataframe.columns[3]: dataframe_numpy[:, 3], dataframe.columns[4]: dataframe_numpy[:, 4], dataframe.columns[5]: dataframe_numpy[:, 5], dataframe.columns[6]: dataframe_numpy[:, 6], dataframe.columns[7]: dataframe_numpy[:, 7], dataframe.columns[8]: dataframe_numpy[:, 8], dataframe.columns[9]: dataframe_numpy[:, 9]})
    return dataframe
