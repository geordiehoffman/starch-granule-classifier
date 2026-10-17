# conda env: starch_vit
"""Guided Grad-CAM: combines the coarse, class-discriminative Grad-CAM
heatmap (layer4) with pixel-level guided backpropagation saliency, giving a
sharp map of exactly which pixels on the granule (edge, hilum-like center
mark, a specific facet, surface shading) drove the prediction -- not just
"somewhere on the granule" like plain Grad-CAM shows.

Guided backprop works by registering a backward hook on every ReLU that
zeroes out negative gradients in addition to ReLU's normal (forward-pass)
zeroing of negative activations -- this suppresses noisy/inhibitory
gradient paths and produces a much cleaner, higher-resolution saliency map
than a raw input-gradient would.

Guided Grad-CAM = guided backprop map * upsampled Grad-CAM heatmap
(element-wise), per Selvaraju et al. 2017 -- same technique named in the
phytolith paper referenced in conversation.

Run: python guided_grad_cam.py   (writes outputs/guided_grad_cam/)
"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from grad_cam import GradCAM, overlay_heatmap
from granule_dataset import LABEL_NAMES, LABELS, GranuleDataset, load_granule_manifest
from paths import REPO_ROOT
from train_granule_classifier import train_granule_model

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "guided_grad_cam")
IMAGE_SIZE = 224
BATCH_SIZE = 32
SPLIT_SEED = 0
N_CORRECT_SAMPLES = 8
N_WRONG_SAMPLES = 8


class GuidedBackprop:
    """Registers hooks on every ReLU to zero negative gradients during
    backward (in addition to ReLU's own forward-pass zeroing of negative
    activations), producing a sharp pixel-level saliency map.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False  # in-place ReLU breaks the backward-hook view/modify check
                self.handles.append(module.register_full_backward_hook(self._relu_backward_hook))

    @staticmethod
    def _relu_backward_hook(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, x, class_idx):
        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, class_idx]
        score.backward()
        saliency = x.grad[0].detach().cpu().numpy()  # (3, H, W)
        saliency = np.transpose(saliency, (1, 2, 0))  # (H, W, 3)
        saliency = np.abs(saliency).sum(axis=2)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency

    def remove(self):
        for h in self.handles:
            h.remove()


def colorize_saliency(saliency: np.ndarray) -> Image.Image:
    # Grayscale intensity map -- brighter = more influential pixel.
    arr = (saliency * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def guided_gradcam_overlay(img: Image.Image, guided_saliency: np.ndarray, cam_heatmap: np.ndarray) -> Image.Image:
    combined = guided_saliency * cam_heatmap
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    arr = (combined * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


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

    print("training one fold...")
    model, mean, std = train_granule_model(train_rows, device)

    full_test_ds = GranuleDataset(test_rows, IMAGE_SIZE, mean, std, train=False)
    full_test_loader = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for x, y in full_test_loader:
            out = model(x.to(device))
            all_preds.extend(out.argmax(dim=1).cpu().tolist())

    # Sample evenly across species, not just the first N in file order.
    correct_by_species, wrong_by_species = {}, {}
    for r, p in zip(test_rows, all_preds):
        if p == LABELS[r["species"]]:
            correct_by_species.setdefault(r["species"], []).append(r)
        else:
            wrong_by_species.setdefault(r["species"], []).append((r, p))

    n_per_species_correct = max(1, N_CORRECT_SAMPLES // len(LABELS))
    n_per_species_wrong = max(1, N_WRONG_SAMPLES // len(LABELS))
    correct_rows = []
    wrong_rows_preds = []
    for species in LABELS:
        correct_rows.extend(correct_by_species.get(species, [])[:n_per_species_correct])
        wrong_rows_preds.extend(wrong_by_species.get(species, [])[:n_per_species_wrong])

    samples = correct_rows + [r for r, _ in wrong_rows_preds]
    pred_override = {id(r): p for r, p in wrong_rows_preds}

    cam = GradCAM(model, model.backbone.layer4)
    guided = GuidedBackprop(model)

    single_ds = GranuleDataset(samples, IMAGE_SIZE, mean, std, train=False)
    for i, row in enumerate(samples):
        x, y = single_ds[i]
        x = x.unsqueeze(0).to(device)
        class_idx = pred_override.get(id(row), y)

        cam_heatmap, pred = cam(x, class_idx=class_idx)
        guided_saliency = guided(x, class_idx=class_idx)

        orig = Image.open(os.path.join(REPO_ROOT, "crops", row["crop_filename"])).convert("RGB")
        orig_padded = single_ds.transform.transforms[0](orig)
        orig_resized = orig_padded.resize((IMAGE_SIZE, IMAGE_SIZE))

        true_name = LABEL_NAMES[y]
        pred_name = LABEL_NAMES[pred]
        status = "CORRECT" if pred == y else "WRONG"

        cam_img = overlay_heatmap(orig_resized, cam_heatmap)
        guided_img = colorize_saliency(guided_saliency)
        combined_img = guided_gradcam_overlay(orig_resized, guided_saliency, cam_heatmap)

        base_name = f"{i:02d}_true-{true_name}_pred-{pred_name}_{status}"
        orig_resized.save(os.path.join(OUT_DIR, f"{base_name}_0orig.png"))
        cam_img.save(os.path.join(OUT_DIR, f"{base_name}_1cam.png"))
        guided_img.save(os.path.join(OUT_DIR, f"{base_name}_2guided.png"))
        combined_img.save(os.path.join(OUT_DIR, f"{base_name}_3guidedgradcam.png"))
        print(base_name)

    guided.remove()
    print(f"\nwrote {len(samples)} guided Grad-CAM sets to {OUT_DIR}")


if __name__ == "__main__":
    main()
