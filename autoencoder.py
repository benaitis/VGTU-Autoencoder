import glob
import random

import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
import cv2
import numpy

lr = 0.0000001
img_dim = 124

labels = {
    'l': 0,
    'b': 0.5,
    'i': 1
}

# load data from files ((original_image, label), expected_image)
data_x1 = []
data_x2 = []
data_y = []
for label in labels:
    imgs = glob.glob("C:\Test\labai\*_*_*_*_" + label + ".png")

    for img in imgs:
        data_x1.append(cv2.resize(cv2.imread(img[:-8] + ".png", 0), (img_dim, img_dim)))
        data_x2.append(labels[label])
        data_y.append(cv2.resize(cv2.imread(img, 0), (img_dim, img_dim)))

#shuffle
data = list(zip(data_x1, data_x2, data_y))
random.seed(42)
random.shuffle(data)

data_x1, data_x2, data_y = zip(*data)

data_x1 = numpy.array(list(data_x1))
data_x2 = numpy.array(list(data_x2))
data_y = numpy.array(list(data_y))

x1_max = float(data_x1.max())
y_max = float(data_y.max())

data_x1 = data_x1.astype('float32') / x1_max
data_y = data_y.astype('float32') / y_max

data_x1 = data_x1.reshape((len(data_x1), img_dim, img_dim, 1))
data_y = data_y.reshape((len(data_y), img_dim, img_dim, 1))


label_as_dimension = []
c = 0
for d in data_x1:
    to_add = numpy.full((img_dim, img_dim), data_x2[c])
    label_as_dimension.append(numpy.dstack((d, to_add)))
    c += 1

data_x1 = numpy.array(label_as_dimension)

#80% train 20% test
train_size = 0.8

train_data_x1 = data_x1[:(int)(len(data_x1) * train_size)]
train_data_y = data_y[:(int)(len(data_y) * train_size)]

test_data_x1 = data_x1[len(train_data_x1):]
test_data_y = data_y[len(train_data_y):]

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(124,124,2)))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualization
autoencoder.add(Flatten())

autoencoder.summary()

autoencoder.add(Reshape((16, 16, 8)))

# Decoder Layers
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(train_data_x1, train_data_y,
                epochs=10,
                batch_size=128,
                validation_data=(test_data_x1, test_data_y))

print("Saving model to JSON")
model_json = autoencoder.to_json()
with open('./model.json', "w") as json_file:
    json_file.write(model_json)
print("Saving weights to HDF5")
autoencoder.save_weights('./weights.h5')
print("Saved model to disk")