import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def dirichlet_split(dataset, num_clients, alpha, num_classes):
    label_indices = [[] for _ in range(num_classes)]
    for i in range(len(dataset)):
        _, y = dataset[i]
        label_indices[y].append(i)
    for c in range(num_classes):
        random.shuffle(label_indices[c])
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        n = len(label_indices[c])
        if n == 0:
            continue
        probs = torch.distributions.Dirichlet(torch.ones(num_clients) * alpha).sample().tolist()
        counts = [int(p * n) for p in probs]
        diff = n - sum(counts)
        for j in range(diff):
            counts[j % num_clients] += 1
        start = 0
        for cid in range(num_clients):
            end = start + counts[cid]
            client_indices[cid].extend(label_indices[c][start:end])
            start = end
    for cid in range(num_clients):
        random.shuffle(client_indices[cid])
    return client_indices

def get_dataset(name):
    if name == "mnist":
        tr = transforms.Compose([transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root="./data", train=True, download=True, transform=tr)
        test = datasets.MNIST(root="./data", train=False, download=True, transform=te)
        num_classes = 10
        in_dim = 28*28
        return train, test, num_classes, in_dim
    if name == "fashion_mnist":
        tr = transforms.Compose([transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tr)
        test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=te)
        num_classes = 10
        in_dim = 28*28
        return train, test, num_classes, in_dim
    raise ValueError("unsupported dataset")

def build_test_loader(test, batch_size):
    return DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
