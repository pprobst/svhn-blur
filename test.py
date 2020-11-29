import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_reader import retrieve_data, process_image

SIZE = (32, 32)

def find_bbox_blur(img, model, size=SIZE):
    original_img = np.array(img).copy()

    iw, ih = img.shape[0], img.shape[1]
    proc_img = process_image(img)
    proc_img = proc_img[np.newaxis, ...]

    sbox = model.predict(proc_img)[0] # Predicted sbox.

    # Rescales bboxes.
    sbox[0] = sbox[0]/float(size[0]/iw)
    sbox[1] = sbox[1]/float(size[1]/ih)
    sbox[2] = sbox[2]/float(size[1]/ih)
    sbox[3] = sbox[3]/float(size[0]/iw)

    y1 = np.clip(sbox[0], 1, iw).astype(np.int32)
    y2 = np.clip(sbox[0]+sbox[3], 1, iw).astype(np.int32)
    x1 = np.clip(sbox[1], 1, ih).astype(np.int32)
    x2 = np.clip(sbox[1]+sbox[2], 1, ih).astype(np.int32)

    # Draw rectangle (bbox).
    red = (255, 0, 0)
    input_img = cv2.rectangle(img, (x1, y1), (x2, y2), red, 1)

    top_left = (x1+1, y1+1)
    bottom_right = (x2, y2)
    x, y = top_left[0], top_left[1]
    w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

    # Blur region inside rectangle.
    roi = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(roi, (51, 51), 0)

    input_img[y:y+h, x:x+w] = blur

    # Show original image and censores image side by side.
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()

# Read processed SVHN data.
_, test, train_imgs, test_imgs, train_labels, test_labels = retrieve_data(4)

# Load CNN model.
model = tf.keras.models.load_model('cnn_model.h5')

for i in range(100):
    find_bbox_blur(test['img'][i], model)
