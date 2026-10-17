# conda env: starch_vit
"""Grad-CAM diagnostic: trains one fold of the granule classifier (reusing
the current photo-grouped split) and visualizes which pixels drove each
test prediction, to check whether the model is attending to the granule
itself or to background/lighting artifacts -- directly testing the leakage
concern raised by the suspiciously high accuracy from train_granule_classifier.py.

Target layer is layer4 (ResNet18's last conv block), the standard Grad-CAM
choice: late enough to carry class-discriminative semantic information,
early enough to still have spatial resolution to visualize.

Run: python grad_cam.py   (writes outputs/grad_cam/)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from granule_dataset import LABEL_NAMES, LABELS, GranuleDataset, load_granule_manifest
from paths import REPO_ROOT
from train_granule_classifier import train_granule_model

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "grad_cam")
IMAGE_SIZE = 224
BATCH_SIZE = 32
N_SAMPLES_PER_SPECIES = 3
SPLIT_SEED = 0


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, int(out.argmax(dim=1).item())


def overlay_heatmap(img: Image.Image, cam: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm

    heatmap = cm.jet(cam)[:, :, :3]  # (H, W, 3) in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
    base = np.array(img.convert("RGB"))
    blended = (0.5 * base + 0.5 * heatmap).astype(np.uint8)
    return Image.fromarray(blended)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows = load_granule_manifest()
    y_species = [r["species"] for r in all_rows]
    groups = [r["source_image_path"] for r in all_rows]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    train_idx, test_idx = next(skf.split(all_rows, y_species, groups))
    train_rows = [all_rows[i] for i in train_idx]
    test_rows = [all_rows[i] for i in test_idx]
    print(f"train: {len(train_rows)}, test: {len(test_rows)}")

    print("training one fold for Grad-CAM diagnostic...")
    model, mean, std = train_granule_model(train_rows, device)

    # First pass: find misclassified test rows (no grad needed).
    full_test_ds = GranuleDataset(test_rows, IMAGE_SIZE, mean, std, train=False)
    full_test_loader = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for x, y in full_test_loader:
            x = x.to(device)
            out = model(x)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())

    misclassified = [(row, pred) for row, pred in zip(test_rows, all_preds) if pred != LABELS[row["species"]]]
    print(f"\nmisclassified: {len(misclassified)} / {len(test_rows)}")
    for row, pred in misclassified:
        print(f"  true={row['species']} pred={LABEL_NAMES[pred]}  {row['crop_filename']}")

    cam = GradCAM(model, model.backbone.layer4)

    # Visualize all misclassified examples (or cap if there are many).
    samples = [row for row, _ in misclassified][:24]

    sample_preds = {id(row): pred for row, pred in misclassified}
    single_ds = GranuleDataset(samples, IMAGE_SIZE, mean, std, train=False)
    for i, row in enumerate(samples):
        x, y = single_ds[i]
        x = x.unsqueeze(0).to(device)
        x.requires_grad_(False)

        expected_pred = sample_preds[id(row)]
        heatmap, pred = cam(x, class_idx=expected_pred)  # CAM for the WRONG class, to see what drove the mistake
        true_name = LABEL_NAMES[y]
        pred_name = LABEL_NAMES[pred]

        # Reload the original crop (undo normalization) for a clean overlay base.
        orig = Image.open(os.path.join(REPO_ROOT, "crops", row["crop_filename"])).convert("RGB")
        orig_padded = single_ds.transform.transforms[0](orig)  # PadThenRotate (train=False, no rotation)
        orig_resized = orig_padded.resize((IMAGE_SIZE, IMAGE_SIZE))

        blended = overlay_heatmap(orig_resized, heatmap)
        correct = "CORRECT" if pred == y else "WRONG"
        fname = f"{i:02d}_true-{true_name}_pred-{pred_name}_{correct}.png"
        blended.save(os.path.join(OUT_DIR, fname))
        print(f"{fname}")

    print(f"\nwrote {len(samples)} Grad-CAM overlays to {OUT_DIR}")


if __name__ == "__main__":
    main()
