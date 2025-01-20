import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import numpy
import cv2
import time
import mss
from pynput import mouse, keyboard
from utils.Dino_Preprocess import Dino_Preprocess

import mindspore
from models.mindspore.Dino_CNN_LSTM_FC import Dino_CNN_LSTM_FC
mindspore.context.set_context(device_target="CPU")

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

def run_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

keyboard_listener_thread = threading.Thread(target=run_keyboard_listener)
keyboard_listener_thread.start()

keyboard_control = keyboard.Controller()

FPS = 30
time_pre_frame = 1 / FPS

def load_model(model_name, label_num, hidden_size, num_layers, model_path):
    model = eval(f"{model_name}({label_num}, {hidden_size}, {num_layers})")
    mindspore.load_checkpoint(model_path, model)
    return model 

if __name__ == "__main__":
    model_name = "Dino_CNN_LSTM_FC"
    label_num = 2
    hidden_size = 512
    num_layers = 1 

    # replace the model path below
    model_path0 = "./ckpt/mindspore/LY_epoch_1867_loss_0.002432126324856654.pt.ckpt"
    
    model0 = load_model(model_name, label_num, hidden_size, num_layers, model_path0)
    print(f"model: {model_name}0 load successfully")
    
    # confirm the postion of the dino
    # must be larger than the blue frame of the windows
    # annotate stopping here
    mouse_controller = mouse.Controller()
    print("click on the left up side first")
    while mouse_x_pos == None and mouse_y_pos == None:
        True
    left, top = mouse_x_pos, mouse_y_pos
    mouse_x_pos, mouse_y_pos = None, None

    print("then click on the right up side.")    # listener_thread.start()
    while mouse_x_pos == None and mouse_y_pos == None:
        True 
    x, y = mouse_x_pos, mouse_y_pos
    mouse_x_pos, mouse_y_pos = None, None
    width = x - left

    print("finally, click on the left lower side")
    # listener_thread.start()
    while mouse_x_pos == None and mouse_y_pos == None:
        True
    x, y = mouse_x_pos, mouse_y_pos
    mouse_x_pos, mouse_y_pos = None, None
    height = y - top
    # listener_thread.stop()

    # if you just play it on your pc and the windows in the same postion
    # you can fill the parameter below on your first try and just use the annotate the code above
    # top, left, width, height = 111, 36, 663, 211
    monitor = {"top": int(top), "left": int(left), "width": int(width), "height": int(height)}

    print("the windows sizes has been collected")
    print(monitor)

    with mss.mss() as sct:
        print("press `left shift` for auto play!\n And `left shift` again to stop")
        while keyboard_key != "Key.shift":
            True
        keyboard_key = None
        print("here AI come")

        last_time = time.time()

        h_0 = mindspore.Tensor(numpy.zeros((num_layers, 1, hidden_size)), dtype=mindspore.float32)
        h_0 = mindspore.ops.StopGradient()(h_0)
        c_0 = mindspore.Tensor(numpy.zeros((num_layers, 1, hidden_size)), dtype=mindspore.float32)
        c_0 = mindspore.ops.StopGradient()(c_0)

        frame = 0
        while "Screen capturing":
            while time.time() - last_time <= time_pre_frame:
                True
            print("FPS = ", 1 / (time.time() - last_time))
            last_time = time.time()
            frame += 1
            print("frame = ", frame)
            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = Dino_Preprocess.flip_image(img)
            if img is None:
                continue
            crop_image = Dino_Preprocess.crop_blue_frame(img)
            if crop_image is None:
                print("No Blue Frame has been detected!")
            else:
                gray_img = Dino_Preprocess.resize_and_gray_frame(crop_image)
                binary_image = Dino_Preprocess.convert_Gray2binary(gray_img)
                dino_image = Dino_Preprocess.crop_binaryframe_dino(binary_image)

                # nn inference
                dino_image = mindspore.tensor(dino_image, dtype=mindspore.float32).expand_dims(0).expand_dims(0)

                nn_out0, h_0, c_0 = model0(dino_image, h_0, c_0)
                inference_result = int(mindspore.Tensor.argmax(nn_out0))
                # nn result

                # mouse.move(500, 300, absolute=True)

                if  inference_result == 1:
                    keyboard_control.press(keyboard.Key.space)
                    print("`space` pressed")
                elif inference_result == 2:
                    keyboard_control.press(keyboard.Key.down)
                    print("`down` pressed")
                else:
                    # Release all pressed keys
                    keyboard_control.release(keyboard.Key.space)
                    keyboard_control.release(keyboard.Key.down)

            # Press the left shift to quit
            if keyboard_key == "Key.shift":
                print("stop!!!!")
                break
