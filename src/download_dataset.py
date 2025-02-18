import kagglehub
from create_model import load_res_images
import matplotlib.pyplot as plt
import os
from PIL import Image

# Download latest version
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

path += "/coco2017"

print("Path to dataset files:", path)

print(__file__)

sep = 'src'
stripped = __file__.split(sep, 1)[0]
print(stripped)

os.system("cp -r " + path + " " + stripped)
os.system("rm -rf " + stripped + "/coco2017/annotations")

def delete_non_640x480_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(file_path) as img:
                if img.size != (640, 480):
                    os.remove(file_path)
                    #print(f"Deleted: {filename} (Size: {img.size})")
                #else:
                #    print(f"Kept: {filename} (Size: {img.size})")
        except Exception as e:
            print(f"Skipping: {filename} (Error: {e})")

# Example usage
folder = "../coco2017/test2017"  # Change this to your image folder path
delete_non_640x480_images(folder)
folder = "../coco2017/train2017"  # Change this to your image folder path
delete_non_640x480_images(folder)
folder = "../coco2017/val2017"  # Change this to your image folder path
delete_non_640x480_images(folder)

'''low_res_train_path = path + "/coco2017/train2017"
high_res_train_path = path + "/coco2017/train2017"

low_res_test_path = path + "/coco2017/test2017"
high_res_test_path = path + "/coco2017/test2017"

WIDTH = 640
HEIGHT = 480

#train_low_images = load_res_images(low_res_train_path, WIDTH, HEIGHT, 1500)

train_high_images = load_res_images(high_res_train_path, WIDTH, HEIGHT, 1500)

#val_low_images = load_res_images(low_res_test_path, WIDTH, HEIGHT, 150)

test_high_images = load_res_images(high_res_test_path, WIDTH, HEIGHT, 150)'''

'''fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

axes[0].imshow(train_low_images[10])
axes[0].set_title("Low-Resolution Image")
axes[0].axis("off")

axes[1].imshow(train_high_images[10])
axes[1].set_title("High-Resolution Image")
axes[1].axis("off")

plt.show()'''