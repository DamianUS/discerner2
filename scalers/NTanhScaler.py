import pandas as pd
import numpy as np

class NTanhScaler:

    def __init__(self, splitting_column, percentiles=[0.5]):
        for threshold in percentiles:
            if threshold <= 0.0 or threshold >= 1.0:
                raise ValueError('Incorrect threshold. Values must be in the range (0-1])')
        self.percentiles = percentiles
        self.splitting_column = splitting_column

    def fit_transform(self, data_training, column_name):
        self.min_threshold = data_training.quantile(0.0)[self.splitting_column]
        self.max_threshold = data_training.quantile(1.0)[self.splitting_column]
        self.limits = [data_training.quantile(threshold) for threshold in self.percentiles]
        self.scalers = []
        pandas = []
        for index in range(len(self.limits) + 1):
            min_threshold = data_training.quantile(0.0)[self.splitting_column]
            max_threshold = data_training.quantile(1.0)[self.splitting_column]
            if index > 0:
                min_threshold = self.limits[index - 1][self.splitting_column]
            if index < len(self.limits):
                max_threshold = self.limits[index][self.splitting_column]
            if min_threshold == max_threshold:
                raise ValueError("The minimum and maximum threshold are equal. Increase the threshold: " + str(
                    self.percentiles[index]) + " to something larger")
            if index == len(self.limits):
                mask = (data_training[self.splitting_column] >= min_threshold) & (
                            data_training[self.splitting_column] <= max_threshold)
            else:
                mask = (data_training[self.splitting_column] >= min_threshold) & (
                            data_training[self.splitting_column] < max_threshold)
            data = data_training[mask]
            scaler = TanhScaler()
            panda = scaler.fit_transform(data)
            pandas.append(panda)
            self.scalers.append(scaler)
        return np.array(pandas)

    def transform(self, data):
        if len(self.scalers) > 1:
            # TODO: More strategies to be added, at the moment I use the scaler that matches the higher number of rows.
            return self.scalers[self.get_range_count(data)].transform(data)
        else:
            return self.scalers[0].transform(data)

    def get_range_count(self, data):
        counts = []
        for index in range(len(self.limits) + 1):
            min_threshold = self.min_threshold
            max_threshold = self.max_threshold
            if index > 0:
                min_threshold = self.limits[index - 1][self.splitting_column]
            if index < len(self.limits):
                max_threshold = self.limits[index][self.splitting_column]
            if min_threshold == max_threshold:
                raise ValueError("The minimum and maximum threshold are equal. Increase the threshold: " + str(
                    self.percentiles[index]) + " to something larger")
            if index == len(self.limits):
                mask = (data[self.splitting_column] >= min_threshold) & (
                        data[self.splitting_column] <= max_threshold)
            else:
                mask = (data[self.splitting_column] >= min_threshold) & (
                        data[self.splitting_column] < max_threshold)
            data_filtered = data[mask]
            counts.append(len(data_filtered))
        return counts.index(max(counts))

    def inverse_transform(self, nparray, columnas, scaler_index):
        return self.scalers[scaler_index].inverse_transform(nparray, columnas)

    def split_dataframe(self, data):
        datas = []
        for index in range(len(self.limits) + 1):
            min_threshold = data.quantile(0.0)[self.splitting_column]
            max_threshold = data.quantile(1.0)[self.splitting_column]
            if index > 0:
                min_threshold = self.limits[index - 1][self.splitting_column]
            if index < len(self.limits):
                max_threshold = self.limits[index][self.splitting_column]
            if min_threshold == max_threshold:
                raise ValueError("The minimum and maximum threshold are equal. Increase the threshold: " + str(
                    self.percentiles[index]) + " to something larger")
            if index == len(self.limits):
                mask = (data[self.splitting_column] >= min_threshold) & (
                        data[self.splitting_column] <= max_threshold)
            else:
                mask = (data[self.splitting_column] >= min_threshold) & (
                        data[self.splitting_column] < max_threshold)
            data_filtered = data[mask]
            data_filtered_numpy = data_filtered.to_numpy()
            data_filtered = pd.DataFrame(
                {data_filtered.columns[0]: data_filtered_numpy[:, 0], data_filtered.columns[1]: data_filtered_numpy[:, 1],
                 data_filtered.columns[2]: data_filtered_numpy[:, 2], data_filtered.columns[3]: data_filtered_numpy[:, 3],
                 data_filtered.columns[4]: data_filtered_numpy[:, 4]})
            datas.append(data_filtered)
        return datas

class TanhScaler:
    m = 0.0
    std = 0.0
    max = None
    min = None

    def fit_transform(self, data_training):
        self.m = np.mean(data_training, axis=0)
        self.std = np.std(data_training, axis=0)
        self.max = np.max(data_training, axis=0)
        self.min = np.min(data_training, axis=0)

        panda = 0.5 * (np.tanh(0.01 * ((data_training - self.m) / self.std)) + 1)
        return panda.to_numpy()

    def transform(self, data):
        panda = 0.5 * (np.tanh(0.01 * ((data - self.m) / self.std)) + 1)
        return panda.to_numpy()

    def inverse_transform(self, nparray, columnas):
        panda = pd.DataFrame(nparray, columns=columnas)
        panda = self.std[columnas] * 100 * np.arctanh(2 * panda[columnas] - 1) + self.m[columnas]
        return panda.to_numpy()