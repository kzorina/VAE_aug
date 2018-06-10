import numpy as np
import matplotlib.pyplot as plt
from ganVAE.models import generator_model

GEN_WEIGHTS_PATH = 'params_generator_epoch_015.hdf5'
DIS_WEIGHTS_PATH = 'params_discriminator_epoch_{0:03d}.hdf5'

latent_size = 110
num_classes = 10
generator = generator_model()
generator.summary()

# load weights
generator.load_weights("C:\\Users\kzorina\Studing\ML\project\\from_colab\params_generator_epoch_015.hdf5")

print("HERE")

r = 6
c = 8
noise = np.random.normal(0, 1, (r * c, 100))
gen_imgs = generator.predict(noise)
print(gen_imgs.shape)
# Rescale to 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(r, c)
# fig.suptitle("DCGAN: Generated digits", fontsize=12)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt, :, :, :])
        axs[i, j].axis('off')
        cnt += 1
plt.show()
plt.close()