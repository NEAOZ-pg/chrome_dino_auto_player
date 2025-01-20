import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import time
import pandas as pd
from utils.Dino_Preprocess import Dino_Preprocess

DATASET_NAME = "dino_flip_2"
FPS = 30

DATASET_PATH = os.path.join("../", "dataset") 
ABS_PATH = os.path.join("/data/dataset")
print(f"DATASET_PATH: {DATASET_PATH}")

time_pre_frame = 1 / FPS

if __name__ == "__main__":
    dataset_dir = os.path.join(DATASET_PATH, DATASET_NAME)
    csv_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(dataset_dir, f))]
    print(csv_files)
    
    if len(csv_files) != 1:
        raise ValueError(f"there are {len(csv_files)} in {dataset_dir}")
        exit()

    csv_file = pd.read_csv(filepath_or_buffer=os.path.join(DATASET_PATH, csv_files[0]), index_col="index") 
    
    last_time = time.time()
    for image_path, text in zip(csv_file["Image_Path"], csv_file["Label"]):
        while time.time() - last_time <= time_pre_frame - 0.001:
            True
        print("FPS = ", 1 / (time.time() - last_time))
        last_time = time.time()
        image = cv2.imread(os.path.join("../../", image_path), cv2.IMREAD_GRAYSCALE)
        binary_image = Dino_Preprocess.convert_Gray2binary(image, lower_threshold=127) 
        cv2.putText(binary_image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128), 1, cv2.LINE_AA)
        cv2.imshow(DATASET_NAME, binary_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()