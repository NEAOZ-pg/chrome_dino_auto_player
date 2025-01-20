import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import cv2
import numpy as np
from utils.Dino_Preprocess import Dino_Preprocess

from torch.utils.data import DataLoader

class Dino_Dataset_Series(torch.utils.data.Dataset):

    def __init__(self, csv_path, workspace_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        self.csv_path = csv_path
        self.workspace_path = workspace_path
        self.series_files = pd.read_csv(filepath_or_buffer=self.csv_path, index_col="index") 
        self.img_csv_files_len = []

        self.csv_dino_image = []
        self.csv_dino_label = []
        for file_name in self.series_files["csv_path"]:
            csv_file = pd.read_csv(filepath_or_buffer=os.path.join(self.workspace_path, file_name), index_col="index")
            self.img_csv_files_len.append(len(csv_file))
            csv_img_series = []
            csv_label_series = []
            for row in csv_file.iloc:
                image_path = os.path.join(self.workspace_path, row['Image_Path'])
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                width, height = image.shape

                if height != 512 or width != 128:
                    raise ValueError(f"the expected shape of image is 512x128, but the data in set is {width}x{height}")
                    exit()
            
                binary_image = Dino_Preprocess.convert_Gray2binary(image, lower_threshold=180) 

                #inspect whether it is a proper threshold
                # cv2.imshow("binary_image", binary_image)
                csv_img_series.append(Dino_Preprocess.crop_binaryframe_dino(binary_image, dino_size=320))
                csv_label_series.append(row['Value'])
            
            self.csv_dino_image.append(torch.tensor(np.array(csv_img_series)).to(torch.float32))
            self.csv_dino_label.append(torch.tensor(np.array(csv_label_series)).to(torch.long))

    def __getitem__(self, index):

        return self.csv_dino_image[index], self.csv_dino_label[index]

    def __len__(self):
        return len(self.series_files)
    


# Example about how to use it
if __name__ == "__main__":
    # NEAOZ PC
    # dino_dataset = Dino_Dataset_Series(r"C:\\Users\\25836\\Desktop\\huawei_ICT\\chrome_dino\\data\\dataset\\csv_path.csv")
    # NEAOZ MAC
    dino_dataset = Dino_Dataset_Series(r"/Users/neaoz/Desktop/chrome_dino/data/dataset/csv_path.csv")    

    dino_dataloader = DataLoader(dataset=dino_dataset, batch_size=1, shuffle=True)

    print("data length = ", len(dino_dataset))
    for dino_image, label in dino_dataloader:
        # convert type from tensor to numpy
        print(dino_image.shape)
        cv2.imshow("dino_image", dino_image.squeeze(0)[20].numpy())
        print("label = ", label.squeeze(0)[0])
        break
    while cv2.waitKey(0) & 0xFF != ord('q'):
        continue
    cv2.destroyAllWindows()
