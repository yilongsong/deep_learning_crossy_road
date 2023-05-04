
    
# collect_data.py

# yilong song
# Apr 28, 2023

# Run this script with python3 collect_data.py before each session of data collection
# Press escape key to exit
# Need to clean up images collected in the beginning and near the end

from pynput import keyboard
from pynput.keyboard import Key
import threading
import time
import random

import mss
import mss.tools

number_screenshots_per_second = 5
lock = False

def screenshot(directory):
    with mss.mss() as sct:
        # Screenshot
        img = sct.grab({'top': 25, 'left': 0, 'width': 788, 'height': 619}) # Adjusted for juliet playing crossy road

        # Save file
        mss.tools.to_png(img.rgb, img.size, output='./dataset/raw/'+directory+'/'+str(random.randint(100000, 1000000))+'.png')
    

def on_press(key):
    if key == Key.up:
        global lock
        lock = True
        screenshot('up')
        print('up')
        time.sleep(1/number_screenshots_per_second)
        lock = False
    elif key == Key.esc: # To exit
        print("EXIT")
        exit()

def start_key_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        global lock
        while True:
            time.sleep(1/number_screenshots_per_second)
            if lock == False:
                screenshot('noop')
                print('noop')

def main():
    key_listener_thread = threading.Thread(target=start_key_listener)
    key_listener_thread.start()


if __name__ == '__main__':
    main()

