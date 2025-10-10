import random
import torch
import torch.nn.functional as F

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def one_hot(y, c, device):
    return F.one_hot(y, c).float().to(device)
