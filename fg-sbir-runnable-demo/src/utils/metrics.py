from typing import Tuple
import torch
import torch.nn.functional as F

class RetrievalEvaluator:
    def __init__(self, topk=(1,5)):
        self.topk = topk
        self.zs, self.zi, self.labels = [], [], []
    def add_batch(self, zs, zi, labels):
        self.zs.append(zs.detach().cpu())
        self.zi.append(zi.detach().cpu())
        self.labels.append(labels.detach().cpu())
    def compute(self):
        zs = torch.cat(self.zs, dim=0)
        zi = torch.cat(self.zi, dim=0)
        y = torch.cat(self.labels, dim=0)
        zs = torch.nn.functional.normalize(zs, dim=-1)
        zi = torch.nn.functional.normalize(zi, dim=-1)
        sims = zs @ zi.t()  # [Q,G]
        ranks = sims.argsort(dim=1, descending=True)  # indices in gallery
        metrics = {}
        for k in self.topk:
            topk_idx = ranks[:, :k]
            topk_labels = y[topk_idx]
            correct = (topk_labels == y.view(-1,1)).any(dim=1).float().mean()
            metrics[f'P@{k}'] = correct
        # mAP (simplified)
        ap = []
        for i in range(sims.size(0)):
            order = ranks[i]
            rel = (y[order] == y[i]).float()
            if rel.sum() == 0:
                ap.append(torch.tensor(0.0))
                continue
            cumsum = rel.cumsum(0)
            precision = cumsum / torch.arange(1, len(rel)+1)
            ap.append((precision * rel).sum() / rel.sum())
        metrics['mAP'] = torch.stack(ap).mean()
        return metrics
