# conda env: starch_vit
"""Dataset for per-granule species classification.

Preprocessing mirrors the earlier whole-photo StarchDataset's reasoning
(dataset.py): pad to square (not crop) so no part of the granule is
discarded, using a per-image estimated background color for the pad/rotation
fill so the fill doesn't read as an obvious artifact. Augmentation is full
0-360 rotation plus flips (granule orientation is arbitrary under the
microscope), no color jitter (same reasoning as before: unclear whether
tonal differences are real signal or incidental session variation).
"""

import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from paths import REPO_ROOT

LABELS = {"DA": 0, "DB": 1, "CE": 2, "IB": 3}
LABEL_NAMES = ["DA", "DB", "CE", "IB"]

CROPS_DIR = os.path.join(REPO_ROOT, "crops")
GRANULE_MANIFEST = os.path.join(REPO_ROOT, "granule_manifest.csv")


def estimate_border_color(img: Image.Image, border_px: int = 5) -> tuple:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    bp = min(border_px, h // 2, w // 2) or 1
    top, bottom = arr[:bp, :, :], arr[-bp:, :, :]
    left, right = arr[:, :bp, :], arr[:, -bp:, :]
    border = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)])
    mean = border.mean(axis=0)
    return tuple(int(v) for v in mean)


class PadThenRotate:
    def __init__(self, train: bool):
        self.train = train

    def __call__(self, img: Image.Image) -> Image.Image:
        fill = estimate_border_color(img)
        w, h = img.size
        size = max(w, h)
        canvas = Image.new(img.mode, (size, size), fill)
        canvas.paste(img, ((size - w) // 2, (size - h) // 2))
        if self.train:
            angle = transforms.RandomRotation.get_params([0, 360])
            canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)
        return canvas


def load_granule_manifest(path: str = None):
    path = path or GRANULE_MANIFEST
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class GranuleDataset(Dataset):
    def __init__(self, rows, image_size: int, image_mean, image_std, train: bool):
        self.rows = rows
        self.train = train

        base = [PadThenRotate(train=train)]
        if train:
            base.append(transforms.RandomHorizontalFlip())
            base.append(transforms.RandomVerticalFlip())
        base += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
        self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = os.path.join(CROPS_DIR, row["crop_filename"])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = LABELS[row["species"]]
        return x, y
