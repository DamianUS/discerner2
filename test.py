import tensorflow as tf
print(tf.__version__)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='cpu')

import numpy as np
import pandas as pd
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Dropout
import time

train = pd.read_csv("./data/mnist_train.csv", header=None)
test = pd.read_csv("./data/mnist_test.csv", header=None)

trainX = train[train.columns[1:]].to_numpy().reshape((-1, 28, 28))
trainy = train[train.columns[0]].to_numpy()
testX = test[test.columns[1:]].to_numpy().reshape((-1, 28, 28))
testy = test[test.columns[0]].to_numpy()

print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], trainX.shape[2], 1))
testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2], 1))

in_shape = trainX.shape[1:]
in_shape

n_classes = len(unique(trainy))
n_classes

# Normalize
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# Create CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
model.add(MaxPool2D((2, 2), strides=(2,2)))
model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
model.add(MaxPool2D((2, 2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
model.fit(trainX, trainy, epochs=10, batch_size=128, verbose=1)
elapsed_time = time.time() - start_time

print(f"Time: {elapsed_time}")

# Evaluate
loss, acc = model.evaluate(testX, testy, verbose=0)
print('Accuracy: %.3f' % acc)