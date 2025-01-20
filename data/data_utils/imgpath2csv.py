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


def list_png_files_one_level(root_path):
    pnglist = []
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        if os.path.isfile(full_path) and item.endswith('.png'):
            pnglist.append(full_path)
            # print(f"PNG file: {full_path}")
    return pnglist


def extract_frame_number(file_name):
    matches = re.search(r'frame_(\d+)_', file_name)
    if matches is not None:
        return int(matches.group(1)) 
    raise ValueError(f"the png file name `{file_name}` is wrong")
    sys.exit()


def extract_label_number(file_name):
    matches = re.search(r'_(\d+)\.png$', file_name)
    if matches is not None:
        return int(matches.group(1)) 
    raise ValueError(f"the png file name `{file_name}` is wrong")
    sys.exit()


def extract_label_name(file_name):
    match = re.search(r'_frame_\d+_(\w+)_\d+\.png$', file_name)
    if match is not None:
        return str(match.group(1)) 
    raise ValueError(f"the png file name `{file_name}` is wrong")
    sys.exit()
    return 


def trim_path(png_file):
    split_lst = re.split(r'[\\/]', png_file)
    path = os.path.join(*split_lst[1:])
    path = os.path.join(r"data", path)
    return path.replace("\\", "/")


if __name__ == "__main__":
    datasets_name, datasets_dir = list_folders_one_level(DATASET_PATH)
    print(datasets_dir)
    
    for dataset_name, dataset_dir in zip(datasets_name, datasets_dir):

        # prevent for re generate csv
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(dataset_dir, f))]
        # if csv_files:
        #     print(f"csv file under dir `{dataset_name}` has been gnerated before")
        #     continue

        label_name, label_dir = list_folders_one_level(dataset_dir)
        
        png_list = []
        for label in label_dir:
            png_list += list_png_files_one_level(label)

        png_sorted = sorted(png_list, key=extract_frame_number)

        value_list = []
        label_list = []
        img_list = []
        for png_file in png_sorted:
            value_list.append(extract_label_number(png_file))
            label_list.append(extract_label_name(png_file))
            img_list.append(trim_path(png_file))

        data = {
            r"Image_Path" : img_list,
            r"Label" : label_list,
            r"Value" : value_list
        }
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(dataset_dir, dataset_name + r".csv"), index_label="index")

        print(f"generate csv file under dir `{dataset_name}` successfully")

