import kagglehub
from create_model import load_res_images
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Download latest version
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

path += "/coco2017"

sep = 'src'
stripped = __file__.split(sep, 1)[0]

print("Copying dataset to desired location")

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

print("Deleting wrong resolution test images")
folder = "../coco2017/test2017"
delete_non_640x480_images(folder)

print("Deleting wrong resolution train images")
folder = "../coco2017/train2017"
delete_non_640x480_images(folder)

print("Deleting wrong resolution val images")
folder = "../coco2017/val2017"
delete_non_640x480_images(folder)

def rename_images(folder_path, prefix="image"):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    images.sort()

    for index, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]  # Get file extension
        new_name = f"{prefix}_{index}{ext}"
        print(str(index) + new_name + "\n")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        #print(f"Renamed: {filename} -> {new_name}")
    
    #print("Renaming completed!")

# Renaming images back into ascending order without skips (important for indexing reasons)
print("Renaming train images")
folder_path = "../coco2017/train2017"
rename_images(folder_path)

print("Renaming val images")
folder_path = "../coco2017/val2017"
rename_images(folder_path)

print("Renaming test images")
folder_path = "../coco2017/test2017"
rename_images(folder_path)

# Creating a copy of the dataset in low resolution (in order to allow only loading parts of it)
def downscale_images(folder_path, save_path):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    images.sort()
    
    if not os.path.isdir("../low_coco"):
        os.mkdir("../low_coco")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for index, filename in enumerate(images, start=1):
        img = cv2.imread(folder_path + "/" + filename)
        resized_image = cv2.resize(img, (320, 240))
        cv2.imwrite(save_path + "/" + filename, resized_image)

print("Creating downscaled test images")
downscale_images("../coco2017/test2017", "../low_coco/test2017")

print("Creating downscaled train images")
downscale_images("../coco2017/train2017", "../low_coco/train2017")

print("Creating downscaled val images")
downscale_images("../coco2017/val2017", "../low_coco/val2017")
