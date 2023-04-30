
# prepare_data.py
# Apr 28, 2023

import os
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
            # Load the original image
            img_path = os.path.join('dataset/raw/', folder, file)
            print(img_path)
            img = cv2.imread(img_path)

            # Downsample
            pil_img = Image.fromarray(img)

            resize_transform = transforms.Resize((186, 236))
            resized_img = resize_transform(pil_img)
            resized_img = np.array(resized_img)

            # Define the new filename for the blurred image
            new_filename = "downsampled_" + file
            new_foldername = "downsampled_" + folder

            # Save the blurred image with the new filename
            new_img_path = os.path.join('dataset', new_foldername, new_filename)
            if not cv2.imwrite(new_img_path, resized_img):
                raise Exception("Could not write image")

            print(f"Processed {file} -> Saved {new_filename}")

print("Downsampling applied to all images in the directory.")