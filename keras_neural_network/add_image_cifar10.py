from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
NUM_TO_AUGMENT = 5

# загрузить набор данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

# обучить
#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=))