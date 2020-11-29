import keras
import numpy as np
import cv2
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_reader import retrieve_data

np.random.seed(123)

def cnn_model(in_shape):
    pool_size = (2, 2)
    kernel_size = (3, 3)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size, padding='same', activation='relu', input_shape=in_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4),
    ])

    return model

def cnn_model_2(in_shape):
    pool_size = (2, 2)
    kernel_size = (3, 3)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size, padding='same', activation='relu', input_shape=in_shape),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4),
    ])

    return model


train, test, train_imgs, test_imgs, train_labels, test_labels = retrieve_data(4)

print(train_imgs.shape)
print(train_labels.shape)

model = cnn_model((32, 32, 1))
#model = cnn_model_2((64, 64, 1))
model.compile(optimizer="adam", loss="mse")
h = model.fit(train_imgs, train_labels, batch_size=128, epochs=10, validation_split=0.2)

# 32x32 modelo 1
# loss: 5.0
# val_loss: 4.7
# demora 5 min!

# 32x32 modelo 2
# loss: 5.0
# val_loss: 8.0
# demora 5 min também, mas dá overfitting nos dados de validação.

# 64x64 modelo 1
# loss: 18.8338
# val_loss: 20.9874
# não é reliable, muitas flutuações em val_loss e desempenho ruim no geral
# demora 20 min!

# 64x64 modelo 2
# melhor otimizado para 64x64
# loss: 14.59
# val_loss: 16.1
# Teríamos que usar mais épocas para treinar, o que levaria muito tempo (~1h+).

model.save('cnn_model_1.h5')
