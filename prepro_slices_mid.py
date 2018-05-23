import glob
import matplotlib.pyplot as plt
import numpy as np
import os


RAW_DATA_PATH = '../mount'

NEG_DIRS = ['Non referral1', 'Non Referral2', 'NonReferral3']
POS_DIRS = ['Referral1', 'Referral2', 'Referral3']

PROCESSED_DATA_PATH = '../oct-data/data-2'

def img_to_np(path):
    dtype = np.uint8
    shape = (200, 1024, 200)
    
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype)
        
    img = data.reshape(shape)
    return img

def np_mid_slice(img):
    return img[:, :, 100]

def case_to_split(case_dir):
    h = hash(case_dir) % 20
    if h == 0:
        return 'val'
    elif h == 1:
        return 'test'
    else:
        return 'train'

def process_all_data():
    for label, top_dirs in [('pos', POS_DIRS), ('neg', NEG_DIRS)]:
        counter = 0
        for top_dir in top_dirs:
            top_dir_path = os.path.join(RAW_DATA_PATH, top_dir)
            for case_dir in os.listdir(top_dir_path):
                split = case_to_split(case_dir)
                case_dir_path = os.path.join(top_dir_path, case_dir)
                for visit_dir in os.listdir(case_dir_path):
                    visit_dir_path = os.path.join(case_dir_path, visit_dir)
                    files = glob.glob(os.path.join(visit_dir_path, '*Optic Disc Cube 200x200*raw.img'))

                    for img_path in files:
                        print(img_path)
                        np_img = np_mid_slice(img_to_np(img_path))
                        output_path = os.path.join(PROCESSED_DATA_PATH, split, label, f'{counter:04d}.npy')
                        np.save(output_path, np_img)
                        counter += 1

if __name__ == '__main__':
    process_all_data()
