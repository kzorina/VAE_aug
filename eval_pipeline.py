from ganVAE import vanila_cnn
from keras.datasets import cifar10, mnist
import numpy as np
#datasets = ['mnist', 'cifar10', 'caltech101']
dataset = 'mnist'
num_images = 50000
baseline = vanila_cnn.build_model(data_augmentation = False, dataset = dataset)
typical_aug = vanila_cnn.build_model(data_augmentation = True, dataset = dataset)
#vae_add_near =
#vae_add_random =
from ganVAE import gan_working as gan
from keras.callbacks import ModelCheckpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
scores = []
losses = []

models_to_compare = [gan]#, vae_add_near, vae_add_random, gan]
fake_ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]
for model in models_to_compare:
    for ratio in fake_ratio:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[:num_images*(1 - fake_ratio)]
        fake, sampled_labels = model.generated_images(num_images*fake_ratio)



        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        augmented = np.concatenate((x_train, fake))
        print(augmented.shape)
        augmented_y = np.concatenate((y_train, sampled_labels))
        print(augmented_y.shape)
        model.fit(augmented, augmented_y,
                  batch_size=64,
                  epochs=50,
                  verbose=2,
                  validation_data=(x_test, y_test),
                  callbacks=callbacks_list)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        scores.append(score[0])
        print('Test accuracy:', score[1])
        losses.append(score[1])