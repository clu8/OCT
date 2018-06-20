"""
Preprocess raw data into numpy cubes. 
"""

import glob
import os

import numpy as np

import constants


def case_to_split(case):
    h = hash(case) % 10
    if h == 0:
        return constants.SPLITS[0] # train
    elif h == 1:
        return constants.SPLITS[1] # val
    else:
        return constants.SPLITS[2] # test

def img_to_np(path):
    dtype = np.uint8
    shape = (200, 1024, 200)
    
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype)
        
    img = data.reshape(shape)
    return img

def process_all_data():
    cubes_path = os.path.join(constants.PROCESSED_DATA_PATH, constants.CUBES_FOLDER)
    
    # create processed dirs
    for split in constants.SPLITS:
        for c in constants.CLASSES:
            os.makedirs(os.path.join(cubes_path, split, c), exist_ok=True)
    
    for label, top_dirs in [('pos', constants.POS_DIRS), ('neg', constants.NEG_DIRS)]:
        for top_dir in top_dirs:
            top_dir_path = os.path.join(constants.RAW_DATA_PATH, top_dir)
            for case_dir in os.listdir(top_dir_path):
                case_dir_path = os.path.join(top_dir_path, case_dir)
                split = case_to_split(case_dir)
                print(f'{case_dir} | {split}')
                
                counter = 1
                for visit_dir in os.listdir(case_dir_path):
                    visit_dir_path = os.path.join(case_dir_path, visit_dir)
                    files = glob.glob(os.path.join(visit_dir_path, '*Optic Disc Cube 200x200*raw.img'))

                    for img_path in files:
                        print(img_path)
                        np_img = img_to_np(img_path)
                        output_path = os.path.join(cubes_path, split, label, f'Stanford_{case_dir}_{counter:02d}.npy')
                        np.save(output_path, np_img)
                        counter += 1

if __name__ == '__main__':
    process_all_data()
