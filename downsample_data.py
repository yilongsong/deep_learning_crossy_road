
# prepare_data.py
# Apr 28, 2023

# Perform downsampling and smart cropping on all images

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Loop through the directory
for folder in os.listdir('dataset/raw'):
    if folder == '.DS_Store':
        continue
    print(folder)
    for file in os.listdir('dataset/raw/'+folder):
        if file.endswith(".png"):
            print(file)
            # Load image
            img_path = os.path.join('dataset/raw/', folder, file)
            print(img_path)
            img = cv2.imread(img_path)

            # Downsample
            resized_img = cv2.resize(img, (236,186))

            # Save the blurred image with the new filename
            new_img_path = os.path.join('dataset/downsampled/', folder, file)
            if not cv2.imwrite(new_img_path, resized_img):
                raise Exception("Could not write image")

            print(f"Downsampled {file} -> Saved {file}")

print("Downsampling applied to all images in the directory.")
