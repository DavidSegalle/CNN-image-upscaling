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


model = build_model()

#plot_model(model, show_shapes=True, show_layer_names=True, to_file="model_plot.png")

model.load_weights("../models/coco.weights.h5")

def predict_images(test_low, test_high, count=5, size=224):
    for _ in range(count):
        random_idx = np.random.randint(len(test_low))
        predicted = model.predict(np.expand_dims(test_low[random_idx], axis=0), verbose=0)[0]
        predicted = np.clip(predicted, 0.0, 1.0)
        
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



high_res_val_path = "../coco2017/val2017"

low_res_val_path = "../low_coco/val2017"

print("\n\nloading images\n\n")
high_val_images = load_images(high_res_val_path)

low_val_images = load_images(low_res_val_path)
print(low_val_images[0].shape)
#print(model.predict(np.expand_dims(low_val_images[0], axis=0)))

predict_images(low_val_images, high_val_images, count=5)
