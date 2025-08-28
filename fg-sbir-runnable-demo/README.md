# Fine-Grained Instance-Level SBIR — Reference Implementation (Runnable Demo)

This repository provides a **fully runnable, minimal reference implementation** for a Sketch-Based Image Retrieval (SBIR) pipeline with:

- Multi-branch encoders (sketch & image) built on **ResNet-18**
- A light **projection head** (BN → ReLU → Linear)
- Losses: **NT-Xent (InfoNCE)**, **Triplet**, and a simple **Cross-Modal Prototype Alignment**
- End‑to‑end **training & evaluation** on a **synthetic demo dataset** derived from CIFAR‑10 (images) with edge‑filtered **sketch surrogates** (so the code runs immediately without external datasets)
- Top‑K retrieval metrics (P@1, P@5, mAP)

> ⚖️ **Integrity note**: This repo is production‑quality and **runs end‑to‑end** out of the box on the synthetic demo. To reproduce paper‑level metrics on research datasets (Sketchy, QMUL FG‑SBIR…), plug in their loaders (stubs included) and follow their licenses. **Do not claim results from the synthetic demo as real dataset benchmarks.**

## Quick Start
```bash
# 1) Create env
conda create -y -n sbir python=3.10
conda activate sbir
pip install -r requirements.txt

# 2) (Optional) Torch CUDA wheels: see https://pytorch.org for your system
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Train on synthetic demo
python scripts/train.py --cfg configs/sbir_resnet18.yaml --epochs 3 --batch_size 64

# 4) Evaluate (uses the last checkpoint by default if path not specified)
python scripts/eval.py --cfg configs/sbir_resnet18.yaml
```

Expected files are created under `results/exp_<timestamp>/...` containing logs, ckpts, and JSON metrics.

## Repo Layout
```
fg-sbir/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .gitattributes
├─ LICENSE
├─ CITATION.cff
├─ configs/
│  └─ sbir_resnet18.yaml
├─ scripts/
│  ├─ train.py
│  ├─ eval.py
│  └─ prepare_synth_data.py
├─ src/
│  ├─ models/
│  │  ├─ backbones.py
│  │  └─ projector.py
│  ├─ losses/
│  │  ├─ nce.py
│  │  ├─ triplet.py
│  │  └─ prototype.py
│  ├─ data/
│  │  └─ dataset.py
│  ├─ engines/
│  │  └─ train_engine.py
│  └─ utils/
│     ├─ common.py
│     └─ metrics.py
└─ tests/
   └─ test_smoke.py
```

## Configuration
See `configs/sbir_resnet18.yaml`.

## Reproducibility
- Seed: 42 (configurable)
- Deterministic flags set where possible
- Logs and metrics saved under `results/`

## License
Apache-2.0 (see `LICENSE`).

## Citation
See `CITATION.cff`.
