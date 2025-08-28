Fine-Grained Instance-Level SBIR — Reference Implementation (Runnable Demo)
This repository provides a fully runnable, minimal reference implementation for a Sketch-Based Image Retrieval (SBIR) pipeline with:

Multi-branch encoders (sketch & image) built on ResNet-18
A light projection head (BN → ReLU → Linear)
Losses: NT-Xent (InfoNCE), Triplet, and a simple Cross-Modal Prototype Alignment
End‑to‑end training & evaluation on a synthetic demo dataset derived from CIFAR‑10 (images) with edge‑filtered sketch surrogates (so the code runs immediately without external datasets)
Top‑K retrieval metrics (P@1, P@5, mAP)
⚖️ Integrity note: This repo is production‑quality and runs end‑to‑end out of the box on the synthetic demo. To reproduce paper‑level metrics on research datasets (Sketchy, QMUL FG‑SBIR…), plug in their loaders (stubs included) and follow their licenses. Do not claim results from the synthetic demo as real dataset benchmarks.


