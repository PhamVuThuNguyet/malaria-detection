from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from imutils import paths
import os

from PIL import Image
import pandas as pd
import numpy as np
from imutils import paths
import os
import random

import joblib
from tqdm import tqdm

# Utilities for ImageNet data preprocessing & prediction decoding
from keras.applications import imagenet_utils
import tensorflow as tf

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
from keras.utils import img_to_array, load_img
import keras

efficientnet_model = keras.models.load_model('./output/models/EfficientNet/finetune')

classes = ['gametocyte',  'leukocyte' ,'ring', 'schizont', 'trophozoite']

imagePaths = list(paths.list_images("./output/cell_images_test"))
random.shuffle(imagePaths)
print(imagePaths)

# it will extract the labels from the path of each image
labels = [p.split(os.path.sep)[-1].split('_')[1].split('.')[0]
          for p in imagePaths]
print(labels)

final_targets = [classes.index(i) for i in labels]

X = []
for i in tqdm(imagePaths):
    image = load_img(i, target_size=(256, 256))  # loading image by there paths
    image = img_to_array(image)  # converting images into arrays
    X.append(image)
X = np.array(X)

y_pred = efficientnet_model.predict(x=X, batch_size=8, verbose=2)
predicted_categories = np.argmax(y_pred, axis = 1)

print(classification_report(final_targets, predicted_categories, target_names=classes))
