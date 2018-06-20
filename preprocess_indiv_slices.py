"""
Turn preprocessed cubes into individual 2d numpy slices. 
"""

import os

import numpy as np

import constants


def to_slices(cubes_path, out_path, slice_start, slice_end, triple_channels=False):
    if triple_channels:
        raise NotImplementedError()
    
    print(f'Cubes path: {cubes_path}\nOut path: {out_path}\nSlice range: {slice_start} - {slice_end}\nTriple channels: {triple_channels}')

    # create processed dirs
    for split in constants.SPLITS:
        for c in constants.CLASSES:
            os.makedirs(os.path.join(out_path, split, c), exist_ok=True)
    
    for split in os.listdir(cubes_path):
        split_dir = os.path.join(cubes_path, split)
        for c in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, c)
            for cube_file in os.listdir(class_dir):
                cube_path = os.path.join(class_dir, cube_file)
                cube = np.load(cube_path)
                for slice_idx in range(slice_start, slice_end):
                    slice_ = cube[:, :, slice_idx]
                    output_path = os.path.join(out_path, split, c, f"{cube_file.split('.')[0]}_{slice_idx:03d}.npy")
                    print(output_path)
                    np.save(output_path, slice_)


if __name__ == '__main__':
    cubes_path = os.path.join(constants.PROCESSED_DATA_PATH, constants.CUBES_FOLDER)
    slice_start = 80
    slice_end = 120
    out_path_1 = os.path.join(constants.PROCESSED_DATA_PATH, f'slices_{slice_start}_{slice_end}')
    # out_path_3 = os.path.join(constants.PROCESSED_DATA_PATH, f'slices_{slice_start}_{slice_end}_3c')
    to_slices(cubes_path, out_path_1, slice_start, slice_end, False)
    # to_slices(cubes_path, out_path_3, slice_start, slice_end, True)
