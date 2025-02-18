import os
def rename_images(folder_path, prefix="image"):
    # Get a list of image files in the folder
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    # Sort files to ensure ascending order
    images.sort()
    
    # Rename files
    for index, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]  # Get file extension
        new_name = f"{prefix}_{index}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
    
    print("Renaming completed!")

# Example usage
folder_path = "../coco2017/train2017"  # Change this to your folder path
rename_images(folder_path)