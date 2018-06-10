from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.datasets import mnist, cifar10


DIM = 32*32*3
HIDDEN_DIM = 256
LATENT_DIM = 2

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def build_decoder(decoder_h, decoder_mean):
    model = Sequential()
    model.add(decoder_h)
    model.add(decoder_mean)
    return model


def neg_log_lh(true, pred):
    return K.sum(K.binary_crossentropy(true, pred), axis=-1)


# MNIST
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, DIM) / 255.
# x_test = x_test.reshape(-1, DIM) / 255.

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(-1, DIM) / 255.
x_test = x_test.reshape(-1, DIM) / 255.

decoder = Sequential([
    Dense(HIDDEN_DIM, input_dim=LATENT_DIM, activation='relu'),
    Dense(DIM, activation='sigmoid')
])

x = Input(shape=(DIM,))
h = Dense(HIDDEN_DIM, activation='relu')(x)

z_mu = Dense(LATENT_DIM)(h)
z_log_var = Dense(LATENT_DIM)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=1.0,
                                   shape=(K.shape(x)[0], LATENT_DIM)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=neg_log_lh)



vae.fit(x_train,
        x_train,
        shuffle=True,
        epochs=60,
        batch_size=100,
        validation_data=(x_test, x_test))

encoder = Model(x, z_mu)

# display a 2D plot of the digit classes in the latent space
# z_test = encoder.predict(x_test, batch_size=100)
# plt.figure(figsize=(6, 6))
# plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
#             alpha=.4, s=3**2, cmap='viridis')
# plt.colorbar()
# plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 32

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(n*n, 2))
x_decoded = x_decoded.reshape(n, n, digit_size, digit_size, 3)

plt.figure(figsize=(10, 10))
plt.imshow(np.block(list(map(list, x_decoded))))
plt.show()


element_size = 32*32
def show_near():
    figure = np.zeros((element_size * 3, element_size * 3))
    xi = 0.05
    yi = 0.05
    for i in range(3):
        for j in range(3):
            z_sample = np.array([[xi + i, yi + j]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(element_size, element_size, 3)
            figure[i * element_size: (i + 1) * element_size,
            j * element_size: (j + 1) * element_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()

show_near()

