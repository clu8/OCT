import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

class SimpleSliceNet(nn.Module):
    def __init__(self):
        super(SimpleSliceNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            Flatten(),
            nn.Linear(8464, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, slice_):
        return self.cnn(slice_).squeeze(dim=1)
    
    def train_step(self, slice_, targets):
        logits = self(slice_)
        loss = self.loss_fn(logits, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
