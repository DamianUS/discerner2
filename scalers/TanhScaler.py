import pandas as pd
import numpy as np


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
