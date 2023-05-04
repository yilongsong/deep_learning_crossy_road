# play.py

# For running trained models
# Apr 28, 2023

from model import ConvNet
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

import pyautogui # For taking screenshots
import time # For taking screenshots at an interval (IS THIS NECESSARY?)
from pynput import keyboard # For taking screenshots upon keypresses
from pynput.keyboard import Key

import numpy as np

import matplotlib.pyplot as plt

import cv2

import mss
import mss.tools

from smartcrop_data import load_to_clear, load_templates, smart_crop

import random


number_screenshots_per_second = 10

def screenshot(to_clear, templates):
    # Takes screenshot, downsamples, and smart crops
    with mss.mss() as sct:
        img = sct.grab({'top': 25, 'left': 0, 'width': 788, 'height': 619})

    img_np = np.array(img)
    img_np = img_np[:,:,:3]


    # Downsample
    resized_np = cv2.resize(img_np, (236,186))

    # smart crop
    resized_np = smart_crop(resized_np, templates, to_clear)

    resized_np = np.transpose(resized_np, (2,1,0))

    # convert to tensor and normalize
    resized_tensor = torch.from_numpy(resized_np)
    resized_tensor = resized_tensor/255


    return resized_tensor


def move(n):
    if n==0:
        pyautogui.press('up') 
        print('up')
    elif n==1:
        pyautogui.press('down') 
        print('down')
    elif n==2:
        pyautogui.press('left') 
        print('left')
    elif n==3:
        pyautogui.press('right') 
        print('right')
    else:
        print('noop')
    

def play(model, to_clear, templates):
    softmax = nn.Softmax(dim=1)

    while True:
        x = screenshot(to_clear, templates)
        pred = softmax(model.forward(x.unsqueeze(0)))
        max_index = torch.argmax(pred[0])

        move(max_index)
        time.sleep(1/number_screenshots_per_second)
        



def main():
    model = torch.load('convnet_trained.pth', map_location=torch.device('cpu'))
    model.eval()

    to_clear = [] # Pixels that we exclude from tracking
    templates = []
    load_to_clear(to_clear)
    load_templates(templates)

    play(model, to_clear, templates)



if __name__ == '__main__':
    main()