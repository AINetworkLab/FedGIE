import torch

class Server:
    def __init__(self, model_fn, device):
        self.global_model = model_fn().to(device)
        self.device = device
    def broadcast(self):
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
    def aggregate(self, client_states):
        keys = list(client_states[0].keys())
        avg_state = {}
        for k in keys:
            s = None
            for cs in client_states:
                v = cs[k].to(self.device)
                if s is None:
                    s = v.clone()
                else:
                    s += v
            s /= len(client_states)
            avg_state[k] = s.detach().cpu()
        self.global_model.load_state_dict(avg_state, strict=True)
    def eval_accuracy(self, loader):
        self.global_model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.global_model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / max(1, total)
