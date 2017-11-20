import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten3d(nn.Module):
    def forward(self, x):
        N, C, D, H, W = x.size()
        return x.view(N, -1)

class OctNet(nn.Module):
    def __init__(self):
        super(OctNet, self).__init__()
    
    def forward(self, cubes):
        return self.cnn(cubes).squeeze()
    
    def train_step(self, cubes, targets):
        logits = self(cubes)
        loss = self.loss_fn(logits, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

class Oct200Net(OctNet):
    def __init__(self):
        super(Oct200Net, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=8),
            nn.BatchNorm3d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=4),
            nn.Conv3d(16, 16, kernel_size=4),
            nn.BatchNorm3d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=4),
            Flatten3d(),
            nn.Linear(224, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.BCEWithLogitsLoss()
