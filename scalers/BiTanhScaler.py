import pandas as pd
import numpy as np


class BiTanhScaler:
    m1 = 0.0
    std1 = 0.0
    max1 = 0.0
    min1 = 0.0

    m2 = 0.0
    std2 = 0.0
    max2 = 0.0
    min2 = 0.0

    def __init__(self, percentile=0.5):
        self.percentile = percentile

    def fit_transform(self, data_training, splitting_column):
        self.limits = data_training.quantile(self.percentile)
        mask = data_training[splitting_column] > self.limits[splitting_column]
        data_training_1 = data_training[mask]
        data_training_2 = data_training[~mask]

        self.m1 = np.mean(data_training_1, axis=0)
        self.std1 = np.std(data_training_1, axis=0)
        self.max1 = np.max(data_training_1, axis=0)
        self.min1 = np.min(data_training_1, axis=0)

        self.m2 = np.mean(data_training_2, axis=0)
        self.std2 = np.std(data_training_2, axis=0)
        self.max2 = np.max(data_training_2, axis=0)
        self.min2 = np.min(data_training_2, axis=0)

        panda1 = 0.5 * (np.tanh(0.01 * ((data_training_1 - self.m1) / self.std1)) + 1)
        panda2 = 0.5 * (np.tanh(0.01 * ((data_training_2 - self.m2) / self.std2)) + 1)
        panda = pd.concat([panda1, panda2])
        return panda.to_numpy()

    def transform(self, data):
        panda = 0.5 * (np.tanh(0.01 * ((data - self.m) / self.std)) + 1)
        return panda.to_numpy()

    def inverse_transform(self, nparray, columnas):
        panda = pd.DataFrame(nparray, columns=columnas)
        panda = self.std[columnas] * 100 * np.arctanh(2 * panda[columnas] - 1) + self.m[columnas]
        return panda.to_numpy()
