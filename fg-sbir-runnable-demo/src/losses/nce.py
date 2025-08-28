import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """Batch-wise NT-Xent between aligned pairs (sketch_i, image_i) with negatives from other items.
    Assumes inputs are L2-normalized.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z_s: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        logits = z_s @ z_i.t() / self.t          # [B,B]
        labels = torch.arange(z_s.size(0), device=z_s.device)
        loss_si = F.cross_entropy(logits, labels)
        loss_is = F.cross_entropy(logits.t(), labels)
        return (loss_si + loss_is) * 0.5
