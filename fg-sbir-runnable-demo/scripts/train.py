import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.models.backbones import DualEncoder
from src.losses.nce import NTXentLoss
from src.losses.triplet import TripletMinerLoss
from src.losses.prototype import PrototypeAlignment
from src.data.dataset import build_synthetic_pair_dataset
from src.engines.train_engine import AverageMeter, set_seed
from src.utils.metrics import RetrievalEvaluator


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default=str(ROOT/'configs/sbir_resnet18.yaml'))
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.epochs is not None: cfg['train']['epochs'] = args.epochs
    if args.batch_size is not None: cfg['train']['batch_size'] = args.batch_size
    if args.output_dir is not None: cfg['output_dir'] = args.output_dir

    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_set = build_synthetic_pair_dataset(train=True)
    val_set = build_synthetic_pair_dataset(train=False)
    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False,
                            num_workers=cfg['train']['num_workers'])

    # Model
    model = DualEncoder(backbone_name=cfg['model']['backbone'], embed_dim=cfg['model']['embed_dim'],
                        proj_hidden=cfg['model'].get('proj_hidden', 256), use_bn=cfg['model'].get('proj_bn', True)).to(device)

    # Losses
    nce_loss = NTXentLoss(temperature=cfg['loss']['temperature']) if cfg['loss']['use_nce'] else None
    triplet_loss = TripletMinerLoss(margin=cfg['loss']['triplet_margin']) if cfg['loss']['use_triplet'] else None
    proto_loss = PrototypeAlignment(weight=cfg['loss']['proto_weight']) if cfg['loss']['use_proto'] else None

    # Optimizer
    if cfg['train']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])

    # Logging & ckpts
    exp_dir = Path(cfg['output_dir']) / f"exp_{int(time.time())}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir/"ckpts").mkdir(exist_ok=True)

    best_p1 = 0.0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            sketches, images, labels = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            zs = model(sketches, branch='sketch')
            zi = model(images, branch='image')

            loss = 0.0
            if nce_loss is not None:
                loss += nce_loss(zs, zi)
            if triplet_loss is not None:
                loss += triplet_loss(zs, zi, labels)
            if proto_loss is not None:
                loss += proto_loss(zs, zi, labels)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), n=sketches.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Validate
        model.eval()
        evaluator = RetrievalEvaluator(topk=(1,5))
        with torch.no_grad():
            for sketches, images, labels in val_loader:
                sketches, images, labels = sketches.to(device), images.to(device), labels.to(device)
                zs = model(sketches, branch='sketch')
                zi = model(images, branch='image')
                evaluator.add_batch(zs, zi, labels)
        metrics = evaluator.compute()
        p1 = float(metrics['P@1'])
        is_best = p1 > best_p1
        best_p1 = max(best_p1, p1)

        # Save checkpoint and log
        ckpt = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'config': cfg,
        }
        torch.save(ckpt, exp_dir/"ckpts"/f"epoch_{epoch+1}.pt")
        with open(exp_dir/"log.jsonl", 'a') as f:
            f.write(json.dumps({
                'epoch': epoch+1,
                'loss': loss_meter.avg,
                **{k: float(v) for k,v in metrics.items()}
            }) + "\n")

        if is_best:
            torch.save(ckpt, exp_dir/"ckpts"/"best.pt")

    print("Training complete. Best P@1:", best_p1)


if __name__ == '__main__':
    main()
