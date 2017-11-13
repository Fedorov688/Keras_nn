from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671) ## для воспроизводимости результатов

# сеть и ее обучение
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # количество результатов = количество цифр
OPTIMIZER = SGD() # СГС-оптимизатор
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # какая часть обучающего набора зарезервирована для контроля

# данные: случайно перетасованы и разбиты на обучающий и тестовый набор

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train содержит 60000 изобаажений размера 28х28 --> преобразуем в массив 60000х784
RESHAPED = 784
x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# нормировать

x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# преобразовать векторы классов в бинарные матрицы классов
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

#M_HIDDEN скрытых слоев
# 10 выходов
# на последнем этапе softmax
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# компиляция модели в Keras
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# обучение модели в Keras
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# result
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])