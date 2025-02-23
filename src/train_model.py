import kagglehub
import cv2
import tqdm
import os
from keras.api.utils import Sequence
from keras.api import models, layers, optimizers, callbacks
from keras.api.utils import image_dataset_from_directory
from keras.api.layers import Rescaling
from keras.api.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Model
import tensorflow as tf
import pandas as pd

import datetime

from create_model import build_model, load_res_images

# Download latest version
#path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")

#print("Path to dataset files:", path)
sep = 'src'
stripped = __file__.split(sep, 1)[0]
print(stripped)
high_path = stripped + "coco2017"
low_path = stripped + "low_coco"

'''high_res_train_path = path + "/train2017"

high_res_test_path = path + "/test2017"

print("\n\nloading images\n\n")
train_low_images, train_high_images = load_res_images(high_res_train_path, 320, 240, 800)

test_low_images, test_high_images = load_res_images(high_res_test_path, 320, 240, 200)'''

class ImagePairGenerator(Sequence):
    def __init__(self, low_res_path, high_res_path, batch_size, input_size, output_size):
        self.low_res_files = sorted(os.listdir(low_res_path))
        self.high_res_files = sorted(os.listdir(high_res_path))
        self.low_res_path = low_res_path
        self.high_res_path = high_res_path
        self.batch_size = batch_size
        self.input_size = input_size   # Low-resolution size
        self.output_size = output_size # High-resolution size

    def __len__(self):
        return int(np.floor(len(self.low_res_files) / self.batch_size))

    def __getitem__(self, index):
        batch_low = self.low_res_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_high = self.high_res_files[index * self.batch_size:(index + 1) * self.batch_size]

        # Load and resize images
        low_res_imgs = np.array([
            cv2.resize(cv2.imread(os.path.join(self.low_res_path, f)), self.input_size)
            for f in batch_low
        ]) / 255.0
        
        high_res_imgs = np.array([
            cv2.resize(cv2.imread(os.path.join(self.high_res_path, f)), self.output_size)
            for f in batch_high
        ]) / 255.0

        return low_res_imgs, high_res_imgs

input_size = (320, 240)   # Example for low-resolution images
output_size = (640, 480)
print(low_path)
print(high_path + "\n\n\n")
print(os.listdir(high_path))
train_generator = ImagePairGenerator(low_path + "/train2017", high_path + "/train2017", batch_size=16, input_size=input_size, output_size=output_size)
test_generator = ImagePairGenerator(low_path + "/test2017", high_path + "/test2017", batch_size=16, input_size=input_size, output_size=output_size)


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

now = datetime.datetime.now()
print(now.time())

history = model.fit(
    train_generator,
    epochs = 15,
    validation_data=test_generator,
    verbose = 1,
    callbacks=[cp_callback]
)

now = datetime.datetime.now()
print(now.time())

history_df = pd.DataFrame(history.history)
history_df.head()


plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend(["train", "valid"])
plt.show()

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend(["train", "valid"])
plt.show()

