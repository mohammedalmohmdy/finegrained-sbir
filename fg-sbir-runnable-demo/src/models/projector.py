import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden)]
        if use_bn:
            layers += [nn.BatchNorm1d(hidden)]
        layers += [nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
