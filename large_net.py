import os
import random
import statistics
import time

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

import constants


class OctSliceDataset(Dataset):
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
            T.RandomResizedCrop(size=200, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            T.RandomVerticalFlip(p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, resample=False, expand=False, center=None),
            T.Resize([200, 200]),
            T.ToTensor(),
            T.Lambda(lambda s: s.repeat(3, 1, 1))
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
    
class OctSliceResnet(nn.Module):
    def __init__(self, finetune=False):
        super(OctSliceResnet, self).__init__()
        
        self.net = torchvision.models.resnet18(pretrained=True)
        if finetune:
            for param in self.net.parameters():
                param.requires_grad = False
        
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, slice_):
        return self.net(slice_).squeeze(dim=1)
    
    def train_step(self, slice_, targets):
        logits = self(slice_)
        loss = self.loss_fn(logits, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
def evaluate(net, loader, verbose=True):
    start = time.time()
    all_logits = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            logits = net(X.to(device))
            loss = net.loss_fn(logits, y.to(device))
            all_logits.extend(list(logits.cpu().numpy()))
            all_labels.extend(list(y))
            all_losses.append(loss.item())

    val_loss = statistics.mean(all_losses)
    auprc = average_precision_score(all_labels, all_logits)
    auroc = roc_auc_score(all_labels, all_logits)
    
    if verbose:
        print(f'Average precision score: {auprc}')
        print(f'AUROC: {auroc}')
        print(f'Validation loss (approximate): {val_loss}')
        print(f'Elapsed: {time.time() - start}')
    return val_loss, auprc, auroc

def train(num_epochs=10, eval_every=3, verbose=True):
    print('==============')
    print(f'Starting training')
    
    train_dataset = OctSliceDataset(train_dir, slice_min, slice_max)
    val_dataset = OctSliceDataset(val_dir, slice_min, slice_max)
    test_dataset = OctSliceDataset(test_dir, slice_min, slice_max)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    net = OctSliceResnet().to(device)

    if verbose: print('------ Evaluating ------')
    evaluate(net, val_loader, verbose)
    for epoch in range(1, num_epochs + 1):
        if verbose: print(f'====== Epoch {epoch} ======')
        losses = []
        for X, y in train_loader:
            loss = net.train_step(X.to(device), y.to(device))
            loss = loss.item()
            losses.append(loss)
        train_loss = statistics.mean(losses)
        if verbose: print(f'Train loss (approximate): {train_loss}')

        if epoch % eval_every == 0 or epoch == num_epochs:
            if verbose: print('------ Evaluating ------')
            val_loss, auprc, auroc = evaluate(net, val_loader, verbose)
        
    return train_loss, val_loss, auprc, auroc

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dir = os.path.join(constants.PROCESSED_DATA_PATH, 'train')
    val_dir = os.path.join(constants.PROCESSED_DATA_PATH, 'val')
    test_dir = os.path.join(constants.PROCESSED_DATA_PATH, 'test')

    slice_min = 80
    slice_max = 120
    
    train(num_epochs=30, eval_every=10)

