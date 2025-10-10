import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=28*28, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)
        self.layers = [self.fc1, self.fc2, self.fc3, self.out]
        self.activations = ["relu", "relu", "relu", "none"]
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
    def forward_cache(self, x):
        h_list = []
        z_list = []
        x = x.view(x.size(0), -1)
        h_list.append(x)
        z1 = self.fc1(x)
        z_list.append(z1)
        h1 = F.relu(z1)
        h_list.append(h1)
        z2 = self.fc2(h1)
        z_list.append(z2)
        h2 = F.relu(z2)
        h_list.append(h2)
        z3 = self.fc3(h2)
        z_list.append(z3)
        h3 = F.relu(z3)
        h_list.append(h3)
        return h_list, z_list
