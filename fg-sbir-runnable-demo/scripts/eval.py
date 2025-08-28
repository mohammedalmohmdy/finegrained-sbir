import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
import yaml

from src.models.backbones import DualEncoder
from src.data.dataset import build_synthetic_pair_dataset
from src.utils.metrics import RetrievalEvaluator


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default=str(ROOT/'configs/sbir_resnet18.yaml'))
    ap.add_argument('--ckpt', type=str, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    val_set = build_synthetic_pair_dataset(train=False)
    val_loader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False,
                            num_workers=cfg['train']['num_workers'])

    # Model
    model = DualEncoder(backbone_name=cfg['model']['backbone'], embed_dim=cfg['model']['embed_dim'],
                        proj_hidden=cfg['model'].get('proj_hidden', 256), use_bn=cfg['model'].get('proj_bn', True)).to(device)

    # Load last/best ckpt if unspecified
    if args.ckpt is None:
        # pick most recent exp best.pt
        exps = sorted((Path(cfg['output_dir']).glob('exp_*/ckpts/best.pt')), key=lambda p: p.stat().st_mtime)
        if not exps:
            raise FileNotFoundError("No checkpoints found. Run training first or pass --ckpt")
        ckpt_path = exps[-1]
    else:
        ckpt_path = Path(args.ckpt)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    evaluator = RetrievalEvaluator(topk=(1,5))
    with torch.no_grad():
        for sketches, images, labels in val_loader:
            sketches, images, labels = sketches.to(device), images.to(device), labels.to(device)
            zs = model(sketches, branch='sketch')
            zi = model(images, branch='image')
            evaluator.add_batch(zs, zi, labels)
    metrics = evaluator.compute()

    print("Evaluation:", json.dumps({k: float(v) for k,v in metrics.items()}, indent=2))


if __name__ == '__main__':
    main()
