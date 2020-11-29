import cv2
import pandas as pd
import numpy as np

SIZE = (32, 32)
#SIZE = (64, 64)

def process_image(img, size=SIZE):
    '''
    Resizes, turns into grayscale and normalizes an image.
    About resizing: must be a 'square' size, because CNNs work on squares.
    size is set by default on 32x32 because higher sizes may consume a lot of
    memory (> 16 GiB) depending on how the neural network model is structured.
    '''
    img_gray = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY).astype(np.float64) # convert image to grayscale
    img_normalized = cv2.normalize(img_gray, 0, 1, cv2.NORM_MINMAX) # normalize image (0..255 to 0..1)
    return img_normalized[..., np.newaxis] # increase dimension by 1 (x, y, z)

def get_image_data(d, size=SIZE):
    '''
    Get processed image data.
    '''
    imgs = d['img']
    img_list = []
    for img in imgs:
       img_list.append(process_image(img))
    return np.array(img_list)

def scale_bbox(d, size=SIZE):
    '''
    Scales the bbox data to the size given as parameter.
    '''
    w, h = size[0], size[1]
    iw, ih = d['img_width'], d['img_height']
    d['top']     =  (d['top']     *  (w / ih)).astype(np.int32)
    d['left']    =  (d['left']    *  (h / iw)).astype(np.int32)
    d['bottom']  =  (d['bottom']  *  (w / ih)).astype(np.int32)
    d['right']   =  (d['right']   *  (h / iw)).astype(np.int32)
    d['width']   =  (d['right']   -  d['left']).astype(np.int32)
    d['height']  =  (d['bottom']  -  d['top']).astype(np.int32)
    return d

def retrieve_data(max_num_digits, size=SIZE):
    '''
    Read data and process it to 'fit' into a CNN.
    '''
    train_data = pd.read_hdf('train_data.h5', 'table')
    train_data = scale_bbox(train_data, size)

    test_data = pd.read_hdf('test_data.h5', 'table')
    test_data = scale_bbox(test_data, size)

    train_imgs = get_image_data(train_data)
    test_imgs = get_image_data(test_data)

    train_labels = np.array(train_data[['top', 'left', 'width', 'height']])
    test_labels = np.array(test_data[['top', 'left', 'width', 'height']])

    return train_data, test_data, train_imgs, test_imgs, train_labels, test_labels
