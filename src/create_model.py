import kagglehub
import cv2
import tqdm
import os
from keras.api.utils import plot_model, img_to_array
from keras.api import models, layers, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Model

def down_block(x, filters, kernel_size, apply_batch_normalization=True):
    x = layers.Conv2D(filters, kernel_size, padding="same", strides=2)(x)
    if apply_batch_normalization:
        x = layers.BatchNormalization()(x)

    x = layers.LeakyReLU()(x)
    return x

def up_block(x, skip, filters, kernel_size, dropout=False):
    x = layers.Conv2DTranspose(filters, kernel_size, padding="same", strides=2)(x)
    if dropout:
        x = layers.Dropout(0.1)(x)

    x = layers.LeakyReLU()(x)
    x = layers.concatenate([x, skip])
    return x

def build_model():
    inputs = layers.Input(shape=[240, 320, 3])

    # Downsampling
    d1 = down_block(inputs, 128, (3, 3), apply_batch_normalization=False)
    d2 = down_block(d1, 128, (3, 3), apply_batch_normalization=False)
    d3 = down_block(d2, 256, (3, 3), apply_batch_normalization=True)
    d4 = down_block(d3, 512, (3, 3), apply_batch_normalization=True)
    #d5 = down_block(d4, 512, (3, 3), apply_batch_normalization=True)

    # Upsampling
    #u1 = up_block(d5, d4, 512, (3, 3), dropout=False)
    u2 = up_block(d4, d3, 256, (3, 3), dropout=False)
    u3 = up_block(u2, d2, 128, (3, 3), dropout=False)
    u4 = up_block(u3, d1, 128, (3, 3), dropout=False)

    # Final upsampling
    u5 = layers.Conv2DTranspose(3, (3, 3), padding='same', strides=2)(u4)
    u5 = layers.LeakyReLU()(u5)
    u5 = layers.concatenate([u5, inputs])

    u6 = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same", activation='sigmoid')(u5)
    u6 = layers.LeakyReLU()(u6)
    

    # Output layer
    #outputs = layers.Conv2D(3, (2, 2), padding='same', strides=1)(u6)
    return Model(inputs=inputs, outputs=u6)


def load_images(path, size):
    
    files = os.listdir(path)

    images = []

    for file in tqdm.tqdm(files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype("float32") / 255.
        img = img_to_array(img)
        images.append(img)

    images = np.array(images)
    return images

def load_res_images(path, width, height, count = float("inf")):
    files = os.listdir(path)

    high_res_images = []
    low_res_images = []

    for file in tqdm.tqdm(files):
        if len(high_res_images) >= count:
            break
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        h, w, channels = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype("float32") / 255.
        
        # Create low res image 
        low_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        img = img_to_array(img)
        low_img = img_to_array(low_img)

        low_res_images.append(low_img)
        high_res_images.append(img)
        #print(len(images))

    high_res_images = np.array(high_res_images)
    low_res_images = np.array(low_res_images)
    return low_res_images, high_res_images