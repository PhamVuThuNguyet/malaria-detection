import pandas as pd
import numpy as np
from collections import defaultdict


from tqdm import tqdm
import joblib
# importing Keras implementation of the pre-trained VGG16 network
from keras.applications import EfficientNetV2L
# Utilities for ImageNet data preprocessing & prediction decoding
from keras.applications import imagenet_utils

from collections import defaultdict
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np

from PIL import Image


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

# Inference on all training data


train = pd.read_json('../data/malaria_bbbc_80k/malaria/training.json')

data = []
for i in tqdm(range(train.shape[0])):
    for j in range(len(train.iloc[i, 1])):
        img_name = train.iloc[i, 0]['pathname'].split('/')[2]
        label = train.iloc[i, 1][j]['category']
        x_min = train.iloc[i, 1][j]['bounding_box']['minimum']['c']
        x_max = train.iloc[i, 1][j]['bounding_box']['maximum']['c']
        y_min = train.iloc[i, 1][j]['bounding_box']['minimum']['r']
        y_max = train.iloc[i, 1][j]['bounding_box']['maximum']['r']

        data.append([img_name, label, x_min, y_min, x_max, y_max])

df_train = pd.DataFrame(
    data, columns=['img_name', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])

non_rbc = ['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']

# converting all cells other than rbc to non-rbc
for i in range(df_train.shape[0]):
    if df_train.iloc[i, 1] in non_rbc:
        df_train.iloc[i, 1] = 'non_rbc'

df_train.img_name = df_train.img_name.apply(
    lambda x: "annotated_data/training_images/"+str(x))


# dataframe with only two labels RBC and NON-RBC for FasterRCNN Detector
df_train_two = df_train[df_train['label'] != "difficult"]


data = []
for i in tqdm(range(train.shape[0])):
    for j in range(len(train.iloc[i, 1])):
        img_name = train.iloc[i, 0]['pathname'].split('/')[2]
        label = train.iloc[i, 1][j]['category']
        x_min = train.iloc[i, 1][j]['bounding_box']['minimum']['c']
        x_max = train.iloc[i, 1][j]['bounding_box']['maximum']['c']
        y_min = train.iloc[i, 1][j]['bounding_box']['minimum']['r']
        y_max = train.iloc[i, 1][j]['bounding_box']['maximum']['r']

        data.append([img_name, label, x_min, y_min, x_max, y_max])

df_train = pd.DataFrame(
    data, columns=['img_name', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])

df_train.img_name = df_train.img_name.apply(
    lambda x: "annotated_data/training_images/"+str(x))

# dataframe with all labels
df_train_all = df_train[df_train['label'] != "difficult"]

# loading trained model and classes for faster-rcnn ||already trained and saved on disk
# saved Faster-RCNN model graph
modelPath = "output/models/frcnn_3/saved_model"
labels_loc = "output/records/classes.pbtxt"  # saved classes files
min_confidence = 0.5


training_images = np.unique(df_train_all.img_name.values)

svc_model = joblib.load('output/models/EfficientNet-SVM/model_SVC_4.pkl')

detect_fn = tf.saved_model.load(modelPath)
categoryIdx = label_map_util.create_category_index_from_labelmap(labels_loc,
                                                                 use_display_name=True)

classes = ['gametocyte', 'leukocyte', 'ring', 'schizont', 'trophozoite']

train_prediction_rnn = {}
training_predicition_efficientnet = {}
predicted_boxes_stacked_train = defaultdict(dict)


efficientnet_model = EfficientNetV2L(
    weights="imagenet", include_top=False)

for img in tqdm(training_images):
    print('Running inference for {}... '.format(img), end='')

    image_np = load_image_into_numpy_array(img)

    (H, W) = image_np.shape[:2]

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    boxes = detections['detection_boxes']

    scores = detections['detection_scores']

    labels = detections['detection_classes']

    # boxes = np.squeeze(boxes)
    # scores = np.squeeze(scores)
    # labels = np.squeeze(labels)

    o = []
    boxes_nm = []

    for (box, score, label) in zip(boxes, scores, labels):
        if score < min_confidence:
            continue

        (startY, startX, endY, endX) = box

        startX = int(startX*W)
        startY = int(startY*H)
        endX = int(endX*W)
        endY = int(endY*H)

        if img in predicted_boxes_stacked_train:
            predicted_boxes_stacked_train[img]['boxes'].append(
                [startX, startY, endX, endY])
            predicted_boxes_stacked_train[img]['scores'].append(
                float(score))
        else:
            predicted_boxes_stacked_train[img]['boxes'] = [
                [startX, startY, endX, endY]]
            predicted_boxes_stacked_train[img]['scores'] = [float(score)]

        if categoryIdx[label]['name'] == "non_rbc":

            if categoryIdx[label]['name'] in train_prediction_rnn:
                train_prediction_rnn[categoryIdx[label]
                                     ['name']] += 1
            else:
                train_prediction_rnn[categoryIdx[label]
                                     ['name']] = 1

            b_box = [startX, startY, endX, endY]

            im = Image.open(img)
            cr_img = im.crop(b_box)
            cr_img = cr_img.resize((256, 256))

            data = np.array(cr_img)
            data = np.expand_dims(data, axis=0)
            data = imagenet_utils.preprocess_input(data)

            data = efficientnet_model.predict(data)

            data = np.array(data)
            data = data.reshape(1, 1280 * 8 * 8)

            pred = svc_model.predict(data)
            label = classes[pred[0]]

            if label in training_predicition_efficientnet:
                training_predicition_efficientnet[label] += 1
            else:
                training_predicition_efficientnet[label] = 1

        else:
            label = categoryIdx[label]
            idx = int(label["id"])-1
            label = label['name']

            if label in train_prediction_rnn:
                train_prediction_rnn[label] += 1
            else:
                train_prediction_rnn[label] = 1

            if label in training_predicition_efficientnet:
                training_predicition_efficientnet[label] += 1
            else:
                training_predicition_efficientnet[label] = 1


print("Ground Truth for F-RCNN::", dict(df_train_two.label.value_counts()))
print("Prediction for F-RCNN::", train_prediction_rnn)

print("Ground Truth for Finetuned EfficientNet::",
      dict(df_train_all.label.value_counts()))
print("Prediction for Finetuned EfficientNet::",
      training_predicition_efficientnet)

import json
with open('./output/records/predicted_boxes_stacked_train.json', 'w+') as fp:
    json.dump(predicted_boxes_stacked_train, fp, separators=(',', ':'), sort_keys=True, indent=4)
