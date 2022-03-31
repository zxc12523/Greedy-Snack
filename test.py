import random
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D

model = tf.keras.Sequential([Conv2D(input_shape=(16, 16, 1), filters=10,
                                            kernel_size=3, strides=1, activation='relu', padding='same'),
                                     MaxPool2D(),
                                     Conv2D(filters=4, kernel_size=3, padding='same', activation='relu'),
                                     MaxPool2D(),
                                     Flatten(),
                                     Dropout(0.1),
                                     Dense(10, activation='linear')
                                     ])
# model.summary()
#  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()