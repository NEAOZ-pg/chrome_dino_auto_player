import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import cv2
from utils.Dino_Preprocess import Dino_Preprocess

from torch.utils.data import DataLoader

class Dino_DataLoader(torch.utils.data.Dataset):

    def __init__(self, csv_path, workspace_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        self.csv_path = csv_path
        self.workspace_path = workspace_path
        self.df = pd.read_csv(self.csv_path) 

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.workspace_path, row['Image_Path'].replace("\\", "/"))
        label = int(row['Value'])
    
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        width, height = image.shape

        if height != 512 or width != 128:
            raise ValueError(f"the expected shape of image is 512x128, but the data in set is {width}x{height}")
            exit
            
        binary_image = Dino_Preprocess.convert_Gray2binary(image) 

        #inspect whether it is a proper threshold
        # cv2.imshow("binary_image", binary_image)
        dino_image, number_image = Dino_Preprocess.crop_binaryframe_dino_number(binary_image)

        return dino_image, number_image, label

    def __len__(self):
        return len(self.df)
    


# Example about how to use it
if __name__ == "__main__":
    dino_dataset = Dino_DataLoader(r"C:\Users\25836\Desktop\huawei_ICT\chrome_dino\data\dataset\dino_combine\combined_dino_images.csv")    
    dino_dataloader = DataLoader(dataset=dino_dataset, batch_size=1, shuffle=True)

    for dino_image, number_image, label in dino_dataloader:
        # convert type from tensor to numpy
        print(dino_image)
        cv2.imshow("dino_image", dino_image.numpy()[0])
        print(number_image)
        cv2.imshow("number_image", number_image.numpy()[0])
        print("label = ", label)
        break
    while cv2.waitKey(0) & 0xFF != ord('q'):
        continue
    cv2.destroyAllWindows()
