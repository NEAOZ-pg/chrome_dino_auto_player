import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy
import time
import mss
from utils.Dino_Preprocess import Dino_Preprocess

top = 150
left = 1000
width = 1700
height = 500

data_name = "dino_CHZ_12_1__11"

if len(sys.argv) == 3:
    if sys.argv[1] == "-o":
        if "dino_" in sys.argv[2]:
            data_name = sys.argv[2]
            print(f"output file{data_name}")
        else:
            raise ValueError("dataset name should begin with `dino_`")
            exit
    else:
        raise ValueError("only support `-o`")
        exit
elif len(sys.argv) == 1:
    print(f"default output file{data_name}")
else:
    raise ValueError("input style `-o dino_*`")
    exit

root_path = os.path.join("../dataset", data_name)

from pynput import mouse, keyboard
import threading
from pynput.mouse import Listener

mouse_x_pos, mouse_y_pos = None, None

# Function to handle mouse events
def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.left:
        global mouse_x_pos, mouse_y_pos
        mouse_x_pos, mouse_y_pos = x, y
        return True

# Run the listener in a separate thread
def run_mouse_listener():
    with Listener(on_click=on_click) as listener:
        listener.join()

mouse_listener_thread = threading.Thread(target=run_mouse_listener)
mouse_listener_thread.daemon = True
mouse_listener_thread.start()

keyboard_key = None

def on_press(key):
    global keyboard_key
    print(key)
    try:
        # If it's a character key (like 'a', '1', etc.)
        keyboard_key = key.char     
        print(f"Character key pressed: {keyboard_key}")
    except AttributeError:
        keyboard_key = str(key)
        print(f"Special key pressed: {keyboard_key}")
    return True

def on_release(key):
    global keyboard_key
    keyboard_key = None
    return True

def run_keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

keyboard_listener_thread = threading.Thread(target=run_keyboard_listener)
keyboard_listener_thread.daemon = True
keyboard_listener_thread.start()

keyboard_control = keyboard.Controller()


FPS = 30
time_pre_frame = 1 / FPS

with mss.mss() as sct:
    # Part of the screen to capture
    os.makedirs(root_path, exist_ok=True)
    
    jump_path = os.path.join(root_path, data_name + r"_jump")
    down_path = os.path.join(root_path, data_name + r"_down")
    stay_path = os.path.join(root_path, data_name + r"_stay")

    os.makedirs(jump_path, exist_ok=True)
    os.makedirs(down_path, exist_ok=True)
    os.makedirs(stay_path, exist_ok=True)

    # mouse_controller = mouse.Controller()
    # print("click on the left up side first")
    # while mouse_x_pos == None and mouse_y_pos == None:
    #     True
    # left, top = mouse_x_pos, mouse_y_pos
    # mouse_x_pos, mouse_y_pos = None, None

    # print("then click on the right up side.")    # listener_thread.start()
    # while mouse_x_pos == None and mouse_y_pos == None:
    #     True
    # x, y = mouse_x_pos, mouse_y_pos
    # mouse_x_pos, mouse_y_pos = None, None
    # width = x - left

    # print("finally, click on the left lower side")
    # # listener_thread.start()
    # while mouse_x_pos == None and mouse_y_pos == None:
    #     True
    # x, y = mouse_x_pos, mouse_y_pos
    # mouse_x_pos, mouse_y_pos = None, None
    # height = y - top
    # # listener_thread.stop()

    top, left, width, height = 111, 36, 663, 211
    # top, left, width, height = 184, 76, 644, 187
    monitor = {"top": top, "left": left, "width": width, "height": height}
    print("size of windows for capture is ", monitor)

    print("press `left shift` for recording!")
    while keyboard_key != "Key.shift":
        True
    keyboard_key = None
    print("start recording")

    frame_num = 0
    last_time = time.time()
    while "Screen capturing":
        while time.time() - last_time <= time_pre_frame:
            True
        print("FPS = ", 1 / (time.time() - last_time))
        last_time = time.time()  
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # Press "Key.shift" to quit
        if keyboard_key == "Key.shift" or frame_num > 2800:
            print("stop!!!!")
            sys.exit()
        key_input = 0
        if keyboard_key == str(keyboard.Key.space):
            key_input = 1
        elif keyboard_key == str(keyboard.Key.down):
            key_input = 2
        # Display the picture
        # cv2.imshow("OpenCV/Numpy normal", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = Dino_Preprocess.flip_image(img)
        if img is None:
            continue
        crop_image = Dino_Preprocess.crop_blue_frame(img)
        if crop_image is None:
            continue
        
        gray_img = Dino_Preprocess.resize_and_gray_frame(crop_image)
         
        # Display the picture in grayscale
        # cv2.imshow("OpenCV/Gray image", gray_img)

        if key_input == 1:
            cv2.imwrite(os.path.join(jump_path, data_name + f"_frame_{frame_num}_jump_{key_input}.png"), gray_img)
        elif key_input == 2:
            cv2.imwrite(os.path.join(down_path, data_name + f"_frame_{frame_num}_down_{key_input}.png"), gray_img)
        else:
            cv2.imwrite(os.path.join(stay_path, data_name + f"_frame_{frame_num}_stay_{key_input}.png"), gray_img)
        frame_num += 1


