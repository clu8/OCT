import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class OctRandomSliceDataset(Dataset):
    """
    Dataset which loads cubes and takes a random slice. 
    """
    def __init__(self, data_dir, slice_min=80, slice_max=120):
        assert 0 <= slice_min <= slice_max < 200
        self.slice_min = slice_min
        self.slice_max = slice_max
        
        pos_dir = os.path.join(data_dir, 'pos')
        pos_paths = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
        
        neg_dir = os.path.join(data_dir, 'neg')
        neg_paths = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
        
        self.cube_paths = pos_paths + neg_paths
        self.labels = [1.] * len(pos_paths) + [0.] * len(neg_paths)
        
        self.transforms = T.Compose([
            T.Resize([200, 200]),
            T.ToTensor()
        ])
        
        assert len(self.labels) == len(self.cube_paths)
        print(f'Number of cubes: {len(self.labels)}')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        cube = np.load(self.cube_paths[i])
        label = self.labels[i]
        
        slice_idx = random.randint(self.slice_min, self.slice_max)
        
        slice_ = cube[:, :, slice_idx]
        img = Image.fromarray(slice_)
        return torch.tensor(self.transforms(img)), torch.tensor(label)