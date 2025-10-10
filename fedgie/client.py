import random
import torch
from .utils import one_hot

class Client:
    def __init__(self, cid, model_fn, dataset, indices, device, num_classes):
        self.cid = cid
        self.model = model_fn().to(device)
        self.dataset = dataset
        self.indices = indices
        self.device = device
        self.num_classes = num_classes
    def sample_batch(self, batch_size):
        if len(self.indices) == 0:
            raise RuntimeError("empty client partition")
        idx = random.choices(self.indices, k=batch_size)
        xs = []
        ys = []
        for i in idx:
            x, y = self.dataset[i]
            xs.append(x.unsqueeze(0))
            ys.append(y)
        x = torch.cat(xs, dim=0).to(self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y
    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    def load_state_dict(self, sd):
        self.model.load_state_dict(sd, strict=True)
    def local_update(self, batch_size):
        self.model.train()
        x, y = self.sample_batch(batch_size)
        with torch.no_grad():
            h_list, z_list = self.model.forward_cache(x)
        B = x.size(0)
        Fcur = one_hot(y, self.num_classes, self.device).T
        layers = self.model.layers
        acts = self.model.activations
        for i in reversed(range(len(layers))):
            layer = layers[i]
            h_prev = h_list[i]
            h_prev_t = h_prev.detach().T
            ones = torch.ones(1, B, device=self.device)
            htilde = torch.cat([ones, h_prev_t], dim=0)
            pinv_htilde = torch.linalg.pinv(htilde)
            Wtilde = Fcur @ pinv_htilde
            W = Wtilde[:, 1:]
            b = Wtilde[:, 0]
            with torch.no_grad():
                layer.weight.data.copy_(W)
                layer.bias.data.copy_(b)
            temp = Fcur - b.unsqueeze(1).expand(-1, B)
            pinvW = torch.linalg.pinv(layer.weight.data)
            h_hat = pinvW @ temp
            if i > 0:
                z_prev = z_list[i-1].detach().T
                if acts[i-1] == "relu":
                    der = (z_prev > 0).float()
                else:
                    der = torch.ones_like(z_prev)
                Fcur = h_hat * der
        return self.state_dict()
