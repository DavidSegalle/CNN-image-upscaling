import cv2

# Load the image
image = cv2.imread('image.jpg')

# Get the original dimensions
(h, w) = image.shape[:2]

# Desired width
new_width = 800

# Calculate the aspect ratio
aspect_ratio = h / w
new_height = int(new_width * aspect_ratio)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)