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

def rename_images(folder_path, prefix="image"):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    images.sort()

    for index, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]  # Get file extension
        new_name = f"{prefix}_{index}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
    
    print("Renaming completed!")

# Renaming images back into ascending order without skips (important for indexing reasons)
folder_path = "../coco2017/train2017"
rename_images(folder_path)

folder_path = "../coco2017/val2017"
rename_images(folder_path)

folder_path = "../coco2017/test2017"
rename_images(folder_path)