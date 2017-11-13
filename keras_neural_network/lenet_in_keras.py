from keras import backend as k
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

# define the ConvNet

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        # CONV => RELU => POOL

        model.add(Conv2D(50, kernel_size=5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #слои Flatten => RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        #softmax - классификатор
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

# сеть и ее обучение
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28 # размеры входного изображения
NB_CLASSES = 10 # число выходов = число цифр
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

# данные: перетасованы и разбиты на обучающий и тестовый набор
(x_train, y_train), (x_test, y_test) = mnist.load_data()
k.set_image_dim_ordering("th")

# рассматрваем как числа с плавающей точкой и нормируем
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# нам нужна форма 60К х [1 х 28 х 28], подаваемая на вход сверточной сети
x_train = x_train[:, np.newaxis, :, :]
x_test = x_test[:, np.newaxis, :, :]
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# преобразуем векторы классов в бинарные матрицы классов
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# инициализировать оптимизатор и модель
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
#print("Test score:", score[0])
#print("Test accuracy:", score[1])

# перечислить все данные в истории
print(history.history.keys())

# построить график изменения верности
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# построить график изменения потери
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()