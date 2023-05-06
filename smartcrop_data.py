

# tracking.py

# May 3, 2023

import cv2
import numpy as np
import os
from functools import reduce

def load_to_clear(l): # Load pixels to not consider in tracking into list l
    for x in range(224):
        for y in range(167):
            if y < 70 or y < (112/224)*x:
                l.append((x,y))

def clear_region(a, l): # Set pixels in l to 0 in tracking map
    a_copy = a.copy()
    for x, y in l:
        a_copy[y,x] = 0
    return a_copy

def load_templates(templates):
    for file in os.listdir('dataset/templates'):
        if file == '.DS_Store':
            continue
        template = cv2.imread('dataset/templates/'+file)
        templates.append(template)

def smart_crop(img, templates, to_clear): # Track and crop
    res = []
    for t in templates:
        res.append(cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED))


    result = reduce(np.maximum, res)
    result = clear_region(result, to_clear)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw rectangle:
    h, w = templates[0].shape[:-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop
    start_x, start_y = 0, max_loc[1] - 66
    h, w = 88, 236
    img = img[start_y:start_y+h, start_x:start_x+w]

    return img



def main():
    to_clear = [] # Pixels that we exclude from tracking
    templates = []
    load_to_clear(to_clear)
    load_templates(templates)

    for folder in os.listdir('dataset/downsampled'):
        if folder == '.DS_Store':
            continue
        print(folder)
        for file in os.listdir('dataset/downsampled/'+folder):
            if file.endswith(".png"):
                print(file)
                img_path = os.path.join('dataset/downsampled/', folder, file)
                img = cv2.imread(img_path)
                img = smart_crop(img, templates, to_clear)
                new_img_path = os.path.join('dataset/smart_cropped/', folder, file)
                if not cv2.imwrite(new_img_path, img):
                    raise Exception("Could not write image")
                print(f"Downsampled {file} -> Saved {new_img_path}")

if __name__ == "__main__":
    main()
