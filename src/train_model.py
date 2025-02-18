import kagglehub
import cv2
import tqdm
import os
from keras.api.utils import plot_model, img_to_array
from keras.api import models, layers, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Model

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

print("loading images")
train_low_images, train_high_images = load_res_images(high_res_train_path, 320, 240, 1200)

test_low_images, test_high_images = load_res_images(high_res_test_path, 640, 480, 350)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

axes[0].imshow(train_low_images[10])
axes[0].set_title("Low-Resolution Image")
axes[0].axis("off")

axes[1].imshow(train_high_images[10])
axes[1].set_title("High-Resolution Image")
axes[1].axis("off")

plt.show()

'''model = build_model(size=SIZE)

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
    validation_data=(test_low_images, test_high_images),
    verbose = 1,
    callbacks=[cp_callback]
)'''