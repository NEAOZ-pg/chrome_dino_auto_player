import os
import sys
import re
import pandas as pd

DATASET_PATH = os.path.join("../", "dataset") 
print(f"DATASET_PATH: {DATASET_PATH}")


def list_folders_one_level(root_path):
    dirlist = []
    dirpath = []
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path):
            dirlist.append(item)
            dirpath.append(full_path)
            # print(f"Folder: {full_path}")
    return dirlist, dirpath


def list_csv_files_one_level(root_path):
    csvlist = []
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        if os.path.isfile(full_path) and item.endswith('.csv'):
            return full_path
            # print(f"csv file: {full_path}")


def trim_path(png_file):
    split_lst = re.split(r'[\\/]', png_file)
    path = os.path.join(*split_lst[1:])
    path = os.path.join(r"data", path)
    return path.replace("\\", "/")


if __name__ == "__main__":
    datasets_name, datasets_dir = list_folders_one_level(DATASET_PATH)
    print(datasets_dir)
    
    csv_path = []

    for dataset_name, dataset_dir in zip(datasets_name, datasets_dir):
        csv_file = list_csv_files_one_level(dataset_dir)
        csv_path.append(trim_path(csv_file))

    data = {
        r"csv_path" : csv_path,
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATASET_PATH, r"csv_path.csv"), index_label="index")

    print(f"generate csv path file successfully")
