import numpy as np
import os
from keras.applications import EfficientNetV2L

from keras.applications import imagenet_utils

from keras.utils import img_to_array, load_img

import random
from tqdm import tqdm

from imutils import paths

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
import joblib

# it contains path for each image in our folder
imagePaths = list(paths.list_images("output/cell_images"))
random.shuffle(imagePaths)

# it will extract the labels from the path of each image
labels = [p.split(os.path.sep)[1].split('_')[1].split('.')[0]
          for p in imagePaths]
classNames = [str(x) for x in np.unique(labels)]

# loading the EfficientNetV2L pre-trained on imagenet network
model = EfficientNetV2L(weights="imagenet", include_top=False)


# list which will have 81,920 featurs for each image
data = []
for i in tqdm(imagePaths):
    image = load_img(i, target_size=(256, 256))  # loading image by there paths
    image = img_to_array(image)  # converting images into arrays
    # inserting a new dimension because keras need extra dimensions
    image = np.expand_dims(image, axis=0)
    # preprocessing image according to imagenet data
    image = imagenet_utils.preprocess_input(image)
    features = model.predict(image)  # extracting those features from the model
    data.append(features)  # appending features to the list

data = np.array(data)  # converting list into array
data = data.reshape(data.shape[0], 1280*8*8)

print(model.summary())


class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(labels),
                                                  y=labels)
weights = {}
for i in range(5):
    weights[i] = class_weights[i]

# convert the labels from integers to vectors
le = LabelEncoder()
labels = le.fit_transform(labels)

print(le.classes_)

# hyper-parameter tuning parameters for logistic regression
params = {"C": [10.0],
          "gamma": [0.05, 0.1, 0.5]}

model = GridSearchCV(estimator=SVC(class_weight=weights,
                     verbose=1), param_grid=params, cv=5, verbose=2)
model.fit(data, labels)

print(model.cv_results_)
print("The best classifier is: ", model.best_estimator_)

if not os.path.exists('output/models'):
    os.makedirs('output/models')
# Save the model as a pickle in a file
joblib.dump(model.best_estimator_, 'output/models/model_SVC.pkl')



