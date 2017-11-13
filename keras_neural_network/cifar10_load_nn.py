import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD
#import keras.callbacks
from PIL import Image, ImageDraw, ImageFont
import subprocess

PWD = '/home/ubskvm/test/pycharmproject/PycharmProjects/untitled/Keras_neural_network/'
BATCH_SIZE = 128
NUM_EPOCHS = 20
FONT_NAMES = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

# загрузить модель
model_architecture = './arch/cifar_architectureFri_10_Nov_2017_15:31:07.json'
model_weights = './weights/cifar10_weightsFri_10_Nov_2017_15:31:07.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# загрузить изображения
img_names = ['cat.jpg', 'dog.jpg', 'J0317vtW42g.jpg', 'uauuigsdM58.jpg', '1477494809_samolet_na_zakate.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

# обучить
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# список классов
list_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# предсказать
predictions = model.predict_classes(imgs)
print(predictions)
i = 0
for img_name in img_names:
    with Image.open(str(PWD) + img_names[i]) as img:
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype(font=FONT_NAMES, size=30)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), str(list_classes[predictions[i]]), (155, 155, 155), font=font)
        img.show()
    print(predictions[i])
    i += 1
#checkpoint = keras.callbacks.TensorBoard(log_dir="/tmp/logs", histogram_freq=0, write_graph=True, write_images=True)
#model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=0.1, callbacks=[checkpoint])
#keras.callbacks.BaseLogger()