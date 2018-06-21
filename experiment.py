import os
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import constants
from dataset import OctSliceDataset
import model2d
from training import evaluate, train_epoch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

slice_start = 80
slice_end = 120

slices_dir = f'slices_{slice_start}_{slice_end}'

train_dir = os.path.join(constants.PROCESSED_DATA_PATH, slices_dir, 'train')
val_dir = os.path.join(constants.PROCESSED_DATA_PATH, slices_dir, 'val')
test_dir = os.path.join(constants.PROCESSED_DATA_PATH, slices_dir, 'test')

def train(num_epochs=10, eval_every=3, verbose=True):
    print('==============')
    print(f'Starting training')
    
    train_dataset = OctSliceDataset(train_dir, triple_channels=True, transforms=[
#         T.RandomApply([
#             T.RandomCrop(size=(175, 896))
#         ], 0.5),
#         T.RandomVerticalFlip(p=0.5),
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomRotation(degrees=10, resample=False, expand=False, center=None),
    ])
    val_dataset = OctSliceDataset(val_dir, triple_channels=True)
    test_dataset = OctSliceDataset(test_dir, triple_channels=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    net = model2d.SliceDensenet201(finetune=True).to(device)

    if verbose: print('------ Evaluating ------')
    evaluate(net, val_loader, device, verbose)
    for epoch in range(1, num_epochs + 1):
        if verbose: print(f'====== Epoch {epoch} ======')
        train_loss = train_epoch(net, train_loader, device, verbose)

        if epoch % eval_every == 0 or epoch == num_epochs:
            if verbose: print('------ Evaluating ------')
            val_loss, auprc, auroc = evaluate(net, val_loader, device, verbose)
        
    return train_loss, val_loss, auprc, auroc

if __name__ == '__main__':
    train(num_epochs=30)

