"""Data utilities (Pl@ntNet-300K-v2, 1000 classes). Set PLANTNET_DIR to the dataset root."""

import json
import math
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2

DATA = Path(os.environ.get("PLANTNET_DIR", "data/plantnet300kv2"))
NUM_CLASSES = 1000
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_image(path, min_size=512):
    """Fast JPEG decode: let libjpeg downscale by 1/2, 1/4, 1/8 while keeping both sides >= min_size."""
    img = Image.open(path)
    img.draft("RGB", (min_size, min_size))
    return img.convert("RGB")


class PlantNet(datasets.ImageFolder):
    """images/{split}/{0000..0999}/*.jpg; class index == species_id (sorted folder names)."""

    def __init__(self, split, transform, min_size=512):
        super().__init__(DATA / "images" / split, transform=transform,
                         loader=lambda p: load_image(p, min_size))


def eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, size=224):
    """Geometry of the official Pl@ntNet-300K-v2 ValTransform (square bicubic resize, no aspect crop)."""
    return v2.Compose([v2.Resize((size, size), interpolation=v2.InterpolationMode.BICUBIC), v2.ToImage(),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


def train_class_counts():
    return torch.bincount(torch.tensor(PlantNet("train", None).targets), minlength=NUM_CLASSES)


def cosine(step, total, start, end):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(step / total, 1.0)))


class Logger:
    """Append-only JSONL log."""

    def __init__(self, path):
        self.f, self.t0 = open(path, "a"), time.time()

    def log(self, d):
        self.f.write(json.dumps({"time": time.time() - self.t0, **d}) + "\n")
        self.f.flush()
