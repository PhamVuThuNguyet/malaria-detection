import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import joblib
# importing Keras implementation of the pre-trained VGG16 network
from keras.applications import EfficientNetV2L
# Utilities for ImageNet data preprocessing & prediction decoding
from keras.applications import imagenet_utils

from collections import defaultdict
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np

matplotlib.use('TkAgg')

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

detect_fn = tf.saved_model.load("./output/models/frcnn_3/saved_model/")
category_index = label_map_util.create_category_index_from_labelmap("./output/records/full_classes.pbtxt",
                                                                    use_display_name=True)
categoryIdx = label_map_util.create_category_index_from_labelmap("./output/records/classes.pbtxt",
                                                                 use_display_name=True)

min_confidence = 0.5
svc_model = joblib.load('output/models/EfficientNet-SVM/model_SVC_4.pkl')
efficientnet_model = EfficientNetV2L(
    weights="imagenet", include_top=False)

IMAGE_PATHS = ["./annotated_data/training_images/0d2aba33-6920-4001-bd54-59fe0bf9f50e.png"]

for img in IMAGE_PATHS:

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
    
    final_labels = []

    for (box, score, label) in zip(boxes, scores, labels):
        if score < min_confidence:
            continue

        (startY, startX, endY, endX) = box

        startX = int(startX*W)
        startY = int(startY*H)
        endX = int(endX*W)
        endY = int(endY*H)

        if categoryIdx[label]['name'] == "non_rbc":

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
            label = pred[0] + 2
            final_labels.append(label)

        else:
            final_labels.append(1)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          final_labels,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    
plt.show(block=True)