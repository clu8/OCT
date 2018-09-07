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

# pretrained_name = 'densenet201_noaug_0'
pretrained_name = None
experiment_name = 'densenet201_data6_noaug_0'

def train(num_epochs=10, eval_every=1, verbose=True):
    print('==============')
    print(f'Starting training')
    
    train_dataset = OctSliceDataset(train_dir, triple_channels=True, transforms=[
#         T.RandomVerticalFlip(p=0.5),
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomApply([
#             T.RandomCrop(size=(175, 896))
#         ], 0.25),
#         T.ColorJitter(0.5, 0.5, 0.5, 0.1)
    ])
    val_dataset = OctSliceDataset(val_dir, triple_channels=True)
    test_dataset = OctSliceDataset(test_dir, triple_channels=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    net = model2d.SliceDensenet201(finetune=True).to(device)
    if pretrained_name:
        pretrained_path = os.path.join(constants.MODELS_PATH, f'{pretrained_name}.pyt')
        net.load_state_dict(torch.load(pretrained_path))
        print(f'Loaded model params at {pretrained_path}')

    if verbose: print('------ Evaluating ------')
    val_loss, auprc, auroc = evaluate(net, val_loader, device, verbose)
    best_auroc = auroc
    for epoch in range(1, num_epochs + 1):
        if verbose: print(f'====== Epoch {epoch} ======')
        train_loss = train_epoch(net, train_loader, device, verbose)

        if epoch % eval_every == 0 or epoch == num_epochs:
            if verbose: print('------ Evaluating ------')
            val_loss, auprc, auroc = evaluate(net, val_loader, device, verbose)
            if auroc > best_auroc:
                print(f'NEW BEST AUROC!!!')
                best_auroc = auroc
                torch.save(net.state_dict(), os.path.join(constants.MODELS_PATH, f'{experiment_name}.pyt'))
        
    return train_loss, val_loss, auprc, auroc

if __name__ == '__main__':
    train(num_epochs=50)

