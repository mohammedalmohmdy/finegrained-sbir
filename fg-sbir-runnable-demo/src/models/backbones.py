from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .projector import Projector


class Encoder(nn.Module):
    def __init__(self, backbone_name: str = 'resnet18', embed_dim: int = 256, proj_hidden: int = 256, use_bn: bool = True):
        super().__init__()
        if backbone_name != 'resnet18':
            raise ValueError('Only resnet18 is implemented in the demo. Extend as needed.')
        base = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        self.projector = Projector(512, embed_dim, hidden=proj_hidden, use_bn=use_bn)

    def forward(self, x):
        feats = self.feature_extractor(x).flatten(1)
        z = self.projector(feats)
        z = F.normalize(z, dim=-1)
        return z


class DualEncoder(nn.Module):
    def __init__(self, backbone_name='resnet18', embed_dim=256, proj_hidden=256, use_bn=True):
        super().__init__()
        self.sketch_enc = Encoder(backbone_name, embed_dim, proj_hidden, use_bn)
        self.image_enc = Encoder(backbone_name, embed_dim, proj_hidden, use_bn)

    def forward(self, x, branch: str):
        if branch == 'sketch':
            return self.sketch_enc(x)
        elif branch == 'image':
            return self.image_enc(x)
        else:
            raise ValueError('branch must be sketch or image')
