
# prepare_data.py
# Apr 28, 2023

import os
import cv2


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
            downsampled_img = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)

            # Define the new filename for the blurred image
            new_filename = "downsampled_" + file
            new_foldername = "downsampled_" + folder

            # Save the blurred image with the new filename
            new_img_path = os.path.join('dataset', new_foldername, new_filename)
            if not cv2.imwrite(new_img_path, downsampled_img):
                raise Exception("Could not write image")

            print(f"Processed {file} -> Saved {new_filename}")

print("Downsampling applied to all images in the directory.")