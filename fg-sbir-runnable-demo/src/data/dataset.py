from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import ImageFilter


def _make_transforms(img_size: int = 224):
    # Image pipeline (RGB normalization for ResNet)
    image_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # Sketch surrogate: edge filtered + grayscale -> 3ch
    sketch_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda im: im.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return sketch_tf, image_tf


class CIFAR10Pairs(Dataset):
    def __init__(self, train: bool = True, img_size: int = 224):
        self.ds = datasets.CIFAR10(root='.cache', train=train, download=True)
        self.sketch_tf, self.image_tf = _make_transforms(img_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        sketch = self.sketch_tf(img)
        image = self.image_tf(img)
        return sketch, image, torch.tensor(label, dtype=torch.long)


def build_synthetic_pair_dataset(train: bool = True):
    return CIFAR10Pairs(train=train, img_size=224)
