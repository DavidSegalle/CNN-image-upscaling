import kagglehub
import cv2
import tqdm
import os
from keras.api.utils import plot_model, img_to_array
from keras.api import models, layers, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Model

from create_model import build_model, load_images

# Download latest version
path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")

print("Path to dataset files:", path)

low_res_train_path = path + "/dataset/train/low_res"
high_res_train_path = path + "/dataset/train/high_res"

low_res_val_path = path + "/dataset/val/low_res"
high_res_val_path = path + "/dataset/val/high_res"

SIZE = 256

train_low_images = load_images(low_res_train_path, size=SIZE)

train_high_images = load_images(high_res_train_path, size = SIZE)

val_low_images = load_images(low_res_val_path, size=SIZE)

val_high_images = load_images(high_res_val_path, size=SIZE)

'''fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

axes[0].imshow(train_low_images[10])
axes[0].set_title("Low-Resolution Image")
axes[0].axis("off")

axes[1].imshow(train_high_images[10])
axes[1].set_title("High-Resolution Image")
axes[1].axis("off")

plt.show()'''

model = build_model(size=SIZE)

model.summary()

model.compile(
    optimizer = optimizers.Adam(learning_rate=0.001),
    loss = "mean_absolute_error",
    metrics = ["accuracy"]
)

# Callback for savinf models
checkpoint_path = "../models/256Train.weights.h5"
cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(
    train_low_images,
    train_high_images,
    epochs = 100,
    batch_size = 16,
    validation_data=(val_low_images, val_high_images),
    verbose = 1,
    callbacks=[cp_callback]
)