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
# importing Keras implementation of the pre-trained VGG16 network
from keras.applications import EfficientNetV2L
# Utilities for ImageNet data preprocessing & prediction decoding
from keras.applications import imagenet_utils

import numpy as np
from PIL import Image


imagePaths = list(paths.list_images("output/cell_images_test"))
svc_model = joblib.load('output/models/EfficientNet-SVM/model_SVC_4.pkl')

efficientnet_model = EfficientNetV2L(weights="imagenet", include_top=False)

classes = ['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']
labels = []
targets = []

for i in imagePaths:
    targets.append(i.split(os.path.sep)[1].split('_')[1].split('.')[0])

    im = Image.open(i)
    cr_img = im.resize((256, 256))

    data = np.array(cr_img)
    data = np.expand_dims(data, axis=0)

    data = imagenet_utils.preprocess_input(data)

    data = efficientnet_model.predict(data)

    data = np.array(data)
    data = data.reshape(1, 1280 * 8 * 8)

    pred = svc_model.predict(data)

    label = pred[0]
    print(targets[len(targets) - 1], label)
    labels.append(label)

final_targets = [classes.index(i) for i in targets]
# final_labels = [classes.index(i) for i in labels]

print(classification_report(final_targets, labels, target_names=classes))
