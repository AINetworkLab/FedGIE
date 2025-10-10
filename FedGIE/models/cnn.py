import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, num_classes, 1, padding=0, bias=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.layers = [self.conv1, self.conv2, self.conv3]
        self.activations = ["relu", "relu", "none"]
    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        s = self.conv3(h2)
        g = self.gap(s).view(s.size(0), s.size(1))
        return g
    def forward_cache(self, x):
        h_list = []
        z_list = []
        h_list.append(x)
        z1 = self.conv1(x)
        z_list.append(z1)
        h1 = F.relu(z1)
        h_list.append(h1)
        z2 = self.conv2(h1)
        z_list.append(z2)
        h2 = F.relu(z2)
        h_list.append(h2)
        z3 = self.conv3(h2)
        z_list.append(z3)
        h3 = z3
        h_list.append(h3)
        return h_list, z_list
