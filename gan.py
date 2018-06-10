import keras
from keras.datasets import cifar10

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, ZeroPadding2D, BatchNormalization, Input, UpSampling2D, Dropout, Flatten
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
EPOCHS = 15
BATCH_SIZE = 60
NUM_CLASSES = 10
LATENT_DIM = 100
SAVE_INTERVAL = 5

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    print(gen_imgs.shape)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    # fig.suptitle("DCGAN: Generated digits", fontsize=12)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("cifar_%d.png" % epoch)
    plt.close()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Generator
model_gen = Sequential()
model_gen.add(Dense(2 * 2 * 512, activation="relu", input_shape=(LATENT_DIM, )))
model_gen.add(Reshape((2, 2, 512)))
#model_gen.add(BatchNormalization(momentum=0.8))
#model_gen.add(UpSampling2D())
model_gen.add(Conv2DTranspose(256, kernel_size=5, strides=2, activation="relu" ,padding="same"))

model_gen.add(BatchNormalization(momentum=0.8))
model_gen.add(Conv2DTranspose(128, kernel_size=5, strides=2, activation="relu" ,padding="same"))
model_gen.add(BatchNormalization(momentum=0.8))
model_gen.add(Conv2DTranspose(64, kernel_size=5, strides=2, activation="relu" ,padding="same"))
#model_gen.add(BatchNormalization(momentum=0.8))
model_gen.add(Conv2DTranspose(3, kernel_size=3, strides=2,activation="tanh", padding="same"))
print(model_gen.summary())
noise = Input(shape=(LATENT_DIM,))
noisy_img = model_gen(noise)
generator = Model(noise, noisy_img)


generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())


#Discriminator

model_dis = Sequential()

model_dis.add(Conv2D(64, kernel_size=5, strides=2, input_shape=x_train.shape[1:], padding="same"))
model_dis.add(LeakyReLU(alpha=0.2))
#model_dis.add(Dropout(0.25))
model_dis.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
#model_dis.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
model_dis.add(LeakyReLU(alpha=0.2))
model_dis.add(Dropout(0.25))
model_dis.add(BatchNormalization(momentum=0.8))
model_dis.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
model_dis.add(LeakyReLU(alpha=0.2))
#model_dis.add(Dropout(0.25))
model_dis.add(BatchNormalization(momentum=0.8))
model_dis.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
model_dis.add(LeakyReLU(alpha=0.2))
model_dis.add(Dropout(0.25))
model_dis.add(Flatten())
model_dis.add(Dense(1, activation='sigmoid'))
print(model_dis.summary())
random_ing = Input(shape=x_train.shape[1:])
validity = model_dis(random_ing)
discriminator = Model(random_ing, validity)
discriminator.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
#Combined generator and discriminator
z = Input(shape=(LATENT_DIM,))
combined = Model(z, discriminator(generator(z)))
combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
for epoch in range(EPOCHS):


    idx = np.random.randint(0, x_train.shape[0], int(BATCH_SIZE / 2))
    imgs = x_train[idx]

    noise = np.random.normal(0, 1, (int(BATCH_SIZE / 2), 100))
    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(imgs, np.ones((int(BATCH_SIZE / 2), 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((int(BATCH_SIZE / 2), 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))

    g_loss = combined.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    if epoch % SAVE_INTERVAL == 0:
        save_imgs(epoch)


