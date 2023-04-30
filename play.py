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


number_screenshots_per_second = 50

def screenshot():
    # Takes screenshot and downsamples
    with mss.mss() as sct:
        img = sct.grab({'top': 25, 'left': 0, 'width': 788, 'height': 619}) # This is the correct region to capture on my mac
        # When moving the BlueStacks simulator window to the top left corner of the screen.
    
    img_np = np.array(img)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    image_pil = transforms.ToPILImage()(img_rgb)

    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(image_pil)

    resized_img = transforms.Resize((186,236), interpolation=transforms.InterpolationMode.BILINEAR)(img_tensor)
    resized_img = resized_img/255
    

    return resized_img


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
    

def play(model):
    softmax = nn.Softmax(dim=1)

    while True:
        x = screenshot()
        pred = softmax(model.forward(x.unsqueeze(0)))
        max_index = torch.argmax(pred[0])

        move(max_index)
        time.sleep(0.2)
        



def main():
    model = torch.load('convnet_trained.pth')
    model.eval()

    play(model)



if __name__ == '__main__':
    main()