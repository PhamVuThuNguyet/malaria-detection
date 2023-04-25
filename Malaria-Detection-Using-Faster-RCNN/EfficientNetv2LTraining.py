import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras.applications import EfficientNetV2L
from keras.utils import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import random
from tqdm import tqdm

from imutils import paths

from sklearn.utils import class_weight

import joblib

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


log_dir = "./output/models/EfficientNet"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)


def plot_hist_acc(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show(block=True)


def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show(block=True)


img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


# it contains path for each image in our folder
imagePaths = list(paths.list_images("./output/cell_images"))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-1].split('_')[1].split('.')[0]
          for p in imagePaths]
classNames = [str(x) for x in np.unique(labels)]

# convert the labels from integers to vectors
le = LabelEncoder()
labels = le.fit_transform(labels)

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(labels),
                                                  y=labels)


onehot_encoder = OneHotEncoder(sparse=False)
labels = labels.reshape(len(labels), 1)
labels = onehot_encoder.fit_transform(labels)

X = []
for i in tqdm(imagePaths):
    image = load_img(i, target_size=(512, 512))  # loading image by there paths
    image = img_to_array(image)  # converting images into arrays
    X.append(image)
X = np.array(X)


# loading the EfficientNetV2L pre-trained on imagenet network
backbone = EfficientNetV2L(
    include_top=False, weights="imagenet", input_shape=(512, 512, 3), pooling="max")
backbone.trainable = False

print(backbone.summary())
print(le.classes_)

inputs = layers.Input(shape=(512, 512, 3))
# TODO get class from input -> modify more augmentation  
x = img_augmentation(inputs)

# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = backbone(x, training=False)

# Rebuild top
# x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.BatchNormalization()(x)
# x = layers.Dense(64)(x)
top_dropout_rate = 0.4 # decrease to 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
x = layers.Dense(128)(x)
x = layers.Dense(64)(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs, name="EfficientNetv2L-transfer")

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())

weights = {}
for i in range(5):
    weights[i] = class_weights[i]

hist = model.fit(x=X, y=labels, batch_size=8, epochs=200,
                 validation_split=0.3, class_weight=weights, verbose=2, callbacks=[tensorboard_callback])

# plot_hist_acc(hist)
# plot_hist_loss(hist)

if not os.path.exists('./output/models/EfficientNet/transfer'):
    os.makedirs('./output/models/EfficientNet/transfer')

model.save('./output/models/EfficientNet/transfer')

# Unfreeze the base model
backbone.trainable = True

# Keep freezing BatchNormalization
for layer in model.layers[:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# Train end-to-end. Be careful to stop before you overfit!
hist_2 = model.fit(x=X, y=labels, batch_size=8, epochs=15,
                   validation_split=0.3, class_weight=weights, verbose=2, callbacks=[tensorboard_callback])
# plot_hist_acc(hist_2)
# plot_hist_loss(hist_2)

if not os.path.exists('./output/models/EfficientNet/finetune'):
    os.makedirs('./output/models/EfficientNet/finetune')

model.save('./output/models/EfficientNet/finetune')

joblib.dump(hist.history, 'output/models/EfficientNet/model_transfer_hist.pkl')
joblib.dump(hist_2.history,
            'output/models/EfficientNet/model_transfer_finetune_hist.pkl')
