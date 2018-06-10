import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
import pickle
IMAGE_SIZE = [100, 150, 3]


dataset_path = "C:\\Users\\kzorina\\Downloads\\new\\101_ObjectCategories.tar\\101_ObjectCategories"
all_categories = sorted(os.listdir(dataset_path))
custom_catgories = ['accordion', 'barrel','brain','elephant','headphone','pyramid','scissors','umbrella','metronome']
exclude_categories = ['BACKGROUND_Google']

categories = custom_catgories#[x for x in all_categories if x not in exclude_categories]

print(categories)

x = []
y = []
for i, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    for file in os.listdir(category_path):
        path = os.path.join(category_path, file)
        img = Image.open(path).convert('RGB')
        new_img = imresize(img, IMAGE_SIZE)
        if (new_img.shape == (100, 150)):
            print(path)
        else:
            x.append(new_img)
            y.append(i)

x = np.stack(x, axis=0)
y = np.stack(y, axis=0)


print(x.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)

# with open('caltech_dataset.pickle', 'wb') as output:
#     pickle.dump((X_train, X_test, y_train, y_test), output)
with open('small_caltech_dataset.pickle', 'wb') as output:
    pickle.dump((X_train, X_test, y_train, y_test), output)
