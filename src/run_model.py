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

SIZE = 256

model = build_model(size=SIZE)

model.load_weights("../models/256Train.weights.h5")

def predict_images(test_low, test_high, count=5, size=224):
    for _ in range(count):
        random_idx = np.random.randint(len(test_low))
        predicted = model.predict(test_low[random_idx].reshape(1, size, size, 3), verbose=0)
        predicted = np.clip(predicted, 0.0, 1.0).reshape(size, size, 3)
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        
        axes[0].imshow(test_low[random_idx])
        axes[0].set_title("Low-Resolution Image")
        axes[0].axis("off")
        
        axes[1].imshow(test_high[random_idx])
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(predicted)
        axes[2].set_title("Enhanced Image")
        axes[2].axis("off")
        
        plt.show()

path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")

print("Path to dataset files:", path)

low_res_val_path = path + "/dataset/val/low_res"
high_res_val_path = path + "/dataset/val/high_res"

SIZE = 256

val_low_images = load_images(low_res_val_path, size=SIZE)

val_high_images = load_images(high_res_val_path, size=SIZE)

predict_images(val_low_images, val_high_images, count=5, size=SIZE)
