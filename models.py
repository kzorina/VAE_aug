from __future__ import print_function
import os
from collections import defaultdict
from PIL import Image
from six.moves import range
import keras.backend as K
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
from keras.utils.generic_utils import Progbar

import matplotlib.pyplot as plt
from keras.layers.noise import GaussianNoise

try:
    import cPickle as pickle
except ImportError:
    import pickle

from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model

LATENT_DIM = 110


def generator_model():

    cnn = Sequential()
    cnn.add(Dense(384 * 4 * 4, input_dim=LATENT_DIM, activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(Reshape((384, 4, 4)))

    cnn.add(Conv2DTranspose(192, kernel_size=5, strides=2, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2DTranspose(96, kernel_size=5, strides=2, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    #cnn.summary()

    latent = Input(shape=(LATENT_DIM,))
    image_class = Input(shape=(1,), dtype='int32')
    cls = Flatten()(Embedding(10, LATENT_DIM,
                              embeddings_initializer='glorot_normal')(image_class))

    h = layers.multiply([latent, cls])
    fake_image = cnn(h)
    return Model([latent, image_class], fake_image)

generator = generator_model()
generator.load_weights("C:\\Users\kzorina\Studing\ML\project\\from_colab\params_generator_epoch_015.hdf5")