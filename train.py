import argparse
import torch
from fedgie.server import Server
from fedgie.client import Client
from fedgie.utils import set_seed
from fedgie.models.mlp import MLP
from fedgie.data.partition import dirichlet_split, get_dataset, build_test_loader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))
    train, test, num_classes, in_dim = get_dataset(args.dataset)
    model_fn = lambda: MLP(in_dim=in_dim, num_classes=num_classes)
    server = Server(model_fn, device)
    parts = dirichlet_split(train, args.clients, args.alpha, num_classes)
    clients = []
    base = server.broadcast()
    for cid in range(args.clients):
        c = Client(cid, model_fn, train, parts[cid], device, num_classes)
        c.load_state_dict(base)
        clients.append(c)
    test_loader = build_test_loader(test, 256)
    for r in range(1, args.rounds+1):
        states = []
        base = server.broadcast()
        for c in clients:
            c.load_state_dict(base)
        for c in clients:
            sd = c.local_update(args.batch)
            states.append(sd)
        server.aggregate(states)
        acc = server.eval_accuracy(test_loader)
        print(f"round={r} acc={acc:.4f}")
if __name__ == "__main__":
    main()
