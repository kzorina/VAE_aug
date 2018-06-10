import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist, cifar10

import pickle



#NUM_CLASSES = 101 #caltech
NUM_CLASSES = 10
DATA_AUGMENTATION = False
#SIZE = [100, 150, 3]
#original_dim = 150*100*3
#intermediate_dim = 3000
# latent_dim = 20
original_dim = 32*32*3
intermediate_dim = 128
latent_dim = 32
batch_size = 64
epochs = 50
epsilon_std = 1.0
load_model = False
#weights_path = "weights_90_DIM_5_epochs.hdf5"


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """ KL divergence for loss. """

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


decoder_h = Dense(intermediate_dim, input_dim=latent_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

decoder = Sequential([
    decoder_h,
    decoder_mean
])

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)
encoder = Model([x, eps], z)
vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)



with open('caltech_dataset.pickle', 'rb') as data:
    x_train, x_test, y_train, y_test = pickle.load(data)
x_train = x_train.reshape(-1, original_dim) / 255.
x_test = x_test.reshape(-1, original_dim) / 255.

#x_train = x_train[:1000]
#y_train = x_train[:1000]

from collections import Counter
c=Counter()
for d in y_train:
    c[str(d)] += 1
print(c)
def visualize_closest(latent_point, axis1, axis2, step):
    n = 3
    #img_size = 32
    figure = np.zeros((SIZE[0] * 3, SIZE[1] * 3, 3))

    for i in range(-1, 2):
        for j in range(-1, 2):
            temp = latent_point
            temp[axis1] += i*step
            temp[axis2] += j * step
            x_decoded = generator.predict(np.array([temp]))
            img = x_decoded[0].reshape(SIZE)
            figure[(i+1) * SIZE[0]: (i + 2) * SIZE[0], (j+1) * SIZE[1]: (j + 2) * SIZE[1]] = img

    plt.figure(figsize=(20, 20))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()



print(y_train[0])
print(encoder.predict(x_train[0:2]))
decoder_input = Input(shape=(latent_dim,))

_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


if load_model:
    vae.load_weights(weights_path)
else:
    vae.fit(x_train,
            x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))
    vae.save_weights("weights_{}_DIM_{}_epochs.hdf5".format(latent_dim, epochs))

lat1, lat2 = encoder.predict(x_train[0:2])
plt.imshow(x_train[0].reshape(SIZE))
plt.show()
visualize_closest(lat1, 1, 5, 0.1)
plt.imshow(x_train[1].reshape(SIZE))
plt.show()
visualize_closest(lat2, 1, 5, 0.1)
print(x_train.shape)


x = x_train[0]
print(x.shape)
eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))


n = 10  # figure with 15x15 digits
digit_size = 32



grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

n = 15
img_size = 32
figure = np.zeros((img_size * n, img_size * n, 3))


for i in range(n):
    for j in range(n):
        z_sample = np.array([np.random.uniform(-1,1 ,size=latent_dim)])
        x_decoded = generator.predict(z_sample)
        img = x_decoded[0].reshape(img_size, img_size, 3)
        figure[i * img_size: (i + 1) * img_size,j * img_size: (j + 1) * img_size] = img



plt.figure(figsize=(20, 20))
plt.imshow(figure)
plt.show()
