import torch
import torch.nn as nn

class PrototypeAlignment(nn.Module):
    """Compute simple per-class prototypes within batch and penalize sketch vs image prototype mismatch.
    L2 distance between class means, averaged across present classes.
    """
    def __init__(self, weight: float = 0.3):
        super().__init__()
        self.weight = weight

    def forward(self, z_s: torch.Tensor, z_i: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if classes.numel() == 0:
            return torch.tensor(0.0, device=z_s.device, requires_grad=True)
        loss = 0.0
        cnt = 0
        for c in classes:
            idx = (labels == c)
            zs_c = z_s[idx]
            zi_c = z_i[idx]
            if zs_c.numel() == 0 or zi_c.numel() == 0:
                continue
            proto_s = zs_c.mean(dim=0)
            proto_i = zi_c.mean(dim=0)
            loss += (proto_s - proto_i).pow(2).sum().sqrt()
            cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=z_s.device, requires_grad=True)
        return self.weight * (loss / cnt)
