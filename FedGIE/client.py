import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    layers = self.model.layers
    acts = self.model.activations
    is_conv = isinstance(layers[-1], nn.Conv2d)
    if is_conv:
        z_top = z_list[-1]
        N, Ctop, Htop, Wtop = z_top.shape
        Fcur_map = one_hot(y, Ctop, self.device).view(N, Ctop, 1, 1).expand(N, Ctop, Htop, Wtop) / float(Htop * Wtop)
        for i in reversed(range(len(layers))):
            layer = layers[i]
            if isinstance(layer, nn.Conv2d):
                h_prev = h_list[i]
                N, Cin, Hin, Win = h_prev.shape
                kH, kW = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                sH, sW = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
                pH, pW = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                dH, dW = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)
                X = F.unfold(h_prev, kernel_size=(kH, kW), dilation=(dH, dW), padding=(pH, pW), stride=(sH, sW))
                N, Ck, L = X.shape
                Xcol = X.permute(1,0,2).reshape(Ck, N*L)
                Fmap = Fcur_map
                N2, Cout, Hout, Wout = Fmap.shape
                Fcol = Fmap.permute(1,0,2,3).reshape(Cout, N2*Hout*Wout)
                ones = torch.ones(1, N*L, device=self.device)
                Htilde = torch.cat([ones, Xcol], dim=0)
                pinv_H = torch.linalg.pinv(Htilde)
                Wtilde = Fcol @ pinv_H
                Wcol = Wtilde[:, 1:]
                b = Wtilde[:, 0]
                Wnew = Wcol.view(layer.out_channels, Cin, kH, kW)
                with torch.no_grad():
                    layer.weight.data.copy_(Wnew)
                    layer.bias.data.copy_(b)
                temp = Fcol - b.unsqueeze(1).expand(-1, N*L)
                pinvW = torch.linalg.pinv(Wcol)
                Hhat_col = pinvW @ temp
                Hhat_col = Hhat_col.view(Ck, N, L).permute(1,0,2)
                Hhat = F.fold(Hhat_col, output_size=(Hin, Win), kernel_size=(kH, kW), dilation=(dH, dW), padding=(pH, pW), stride=(sH, sW))
                if i > 0:
                    z_prev = z_list[i-1]
                    if acts[i-1] == "relu":
                        der = (z_prev > 0).float()
                    else:
                        der = torch.ones_like(z_prev)
                    Fcur_map = Hhat * der
        return self.state_dict()
    else:
        Fcur = one_hot(y, self.num_classes, self.device).T
        for i in reversed(range(len(layers))):
            layer = layers[i]
            h_prev = h_list[i]
            if isinstance(layer, nn.Linear):
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
