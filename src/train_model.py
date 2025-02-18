import kagglehub
import cv2
import tqdm
import os
from keras.api.utils import Sequence
from keras.api import models, layers, optimizers, callbacks
from keras.api.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Model
import tensorflow as tf

from create_model import build_model, load_res_images

# Download latest version
#path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")

#print("Path to dataset files:", path)
sep = 'src'
stripped = __file__.split(sep, 1)[0]
print(stripped)
path = stripped + "/coco2017"

high_res_train_path = path + "/train2017"

high_res_test_path = path + "/test2017"

print("\n\nloading images\n\n")
train_low_images, train_high_images = load_res_images(high_res_train_path, 320, 240, 800)

test_low_images, test_high_images = load_res_images(high_res_test_path, 320, 240, 200)


# Simple test to check if images are being loaded properly and check if downscale isn't too sharp
'''fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

axes[0].imshow(train_low_images[10])
axes[0].set_title("Low-Resolution Image")
axes[0].axis("off")

axes[1].imshow(train_high_images[10])
axes[1].set_title("High-Resolution Image")
axes[1].axis("off")

plt.show()'''

# Build the model
model = build_model()

model.summary()

model.compile(
    optimizer = optimizers.Adam(learning_rate=0.001),
    loss = "mean_absolute_error",
    metrics = ["accuracy"]
)

# Callback for savinf models
checkpoint_path = "../models/coco.weights.h5"
cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Shennanigans to keep the dataset outside of vram
with tf.device('/CPU:0'):
    train_high_images = tf.constant(train_high_images)
    train_low_images = tf.constant(train_low_images)
    test_low_images = tf.constant(test_low_images)
    test_high_images = tf.constant(test_high_images)



history = model.fit(
    train_low_images,
    train_high_images,
    epochs = 10,
    batch_size = 16,
    validation_data=(test_low_images, test_high_images),
    verbose = 1,
    callbacks=[cp_callback]
)