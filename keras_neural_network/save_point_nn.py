from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
import keras.callbacks
import numpy as np
import os

BATCH_SIZE = 128
NUM_EPOCHS = 20
MODEL_DIR = "/tmp"

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = xtrain.reshape(60000, 784).astype("float32") / 255
xtest = xtest.reshape(10000, 784).astype("float32") / 255
ytrain = np_utils.to_categorical(ytrain, 10)
ytest = np_utils.to_categorical(ytest, 10)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# сохранить модель

#checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))

checkpoint = keras.callbacks.TensorBoard(log_dir="/tmp/logs", histogram_freq=0, write_graph=True, write_images=False)
model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=0.1, callbacks=[checkpoint])
#from quiver_engine import server
#server.launch(model)