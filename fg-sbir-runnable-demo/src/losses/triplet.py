import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletMinerLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, z_s: torch.Tensor, z_i: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Simple within-batch miner: anchor=sketch, positive=paired image, negatives=other images with different labels
        B = z_s.size(0)
        pos = (labels.view(-1,1) == labels.view(1,-1))  # [B,B]
        mask_neg = ~pos
        dists = 1 - (z_s @ z_i.t())  # cosine distance
        losses = []
        for a in range(B):
            p = dists[a, a]  # paired
            neg_d = dists[a][mask_neg[a]]
            if neg_d.numel() == 0:
                continue
            n = neg_d.min()
            losses.append(F.relu(p - n + self.margin))
        if not losses:
            return torch.tensor(0.0, device=z_s.device, requires_grad=True)
        return torch.stack(losses).mean()
