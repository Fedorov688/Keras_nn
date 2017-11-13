from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

NUM_TO_AUGMENT = 5

# набор 10 содержит 60К изображений 32х32 с 3 каналами
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# константы
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# загрузить набор данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train.shape:', x_train.shape)
print(x_train[0], 'train samples')
print(x_test[0], 'test samples')

# пополнение
print("Augmenting training set images...")
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
xtas, ytas = [], []
for i in range(x_train.shape[0]):
    num_aug = 0
    x = x_train[i] # (3, 32, 32)
    x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
    for x_aug in datagen.flow(x, batch_size=1,
                              save_to_dir='preview',
                              save_prefix='cifar',
                              save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1

# инициализировать генератор
datagen.fit(x_train)

# преобразовать к категориальному виду
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# преобразовать к формату с плавающей точкой и нормировать
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# сеть
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# обучение v1
#model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
#model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
#score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
#print("Test score:", score[0])
#print("Test accuracy", score[1])

# обучение v2
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), samples_per_epoch = x_train.shape[0],
#                              epochs=NB_EPOCH, verbose=VERBOSE)
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy", score[1])
version_time = time.clock()
# сохранить модель
model_json = model.to_json()
open('./arch/cifar_architecture' + str(time.strftime("%a_%d_%b_%Y_%H:%M:%S")) + '.json', 'w').write(model_json)
# и веса, вычисленные в результате обучения сети
model.save_weights('./weights/cifar10_weights' + str(time.strftime("%a_%d_%b_%Y_%H:%M:%S")) + '.h5', overwrite=True)