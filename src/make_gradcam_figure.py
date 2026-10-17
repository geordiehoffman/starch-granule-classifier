# conda env: starch_vit
"""Builds the coauthor-facing Grad-CAM figure: for one correctly-classified
granule per species, shows the original crop, the coarse class-discriminative
Grad-CAM heatmap (highlights roughly where on the granule the decision came
from), and the sharper Guided Grad-CAM saliency (highlights which specific
pixels drove it) -- together making the case that the model attends to
granule morphology, not background.

Run: python make_gradcam_figure.py   (writes ../results/granule_classifier/figures/4_guided_gradcam.png)
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

from paths import REPO_ROOT

GRAD_CAM_DIR = os.path.join(REPO_ROOT, "outputs", "guided_grad_cam")
OUT_DIR = os.path.join(REPO_ROOT, "results", "granule_classifier", "figures")

SPECIES_ORDER = ["DA", "DB", "CE", "IB"]
SPECIES_FULL = {"DA": "D. alata", "DB": "D. bulbifera", "CE": "C. esculenta", "IB": "I. batatas"}
COLORS = {"DA": "#2a78d6", "DB": "#eb6834", "CE": "#1baf7a", "IB": "#eda100"}

# (original, coarse Grad-CAM overlay (already colorized by guided_grad_cam.py),
#  raw guided Grad-CAM saliency (grayscale, recolorized below))
GRAD_CAM_EXAMPLES = {
    "DA": ("00_true-DA_pred-DA_CORRECT_0orig.png", "00_true-DA_pred-DA_CORRECT_1cam.png", "00_true-DA_pred-DA_CORRECT_3guidedgradcam.png"),
    "DB": ("02_true-DB_pred-DB_CORRECT_0orig.png", "02_true-DB_pred-DB_CORRECT_1cam.png", "02_true-DB_pred-DB_CORRECT_3guidedgradcam.png"),
    "CE": ("04_true-CE_pred-CE_CORRECT_0orig.png", "04_true-CE_pred-CE_CORRECT_1cam.png", "04_true-CE_pred-CE_CORRECT_3guidedgradcam.png"),
    "IB": ("06_true-IB_pred-IB_CORRECT_0orig.png", "06_true-IB_pred-IB_CORRECT_1cam.png", "06_true-IB_pred-IB_CORRECT_3guidedgradcam.png"),
}

ROW_LABELS = ["Original", "Grad-CAM", "Guided Grad-CAM"]

TEXT_SECONDARY = "#52514e"
GRID = "#e5e4de"


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
        }
    )


def stretch_contrast(saliency: np.ndarray) -> np.ndarray:
    # Percentile stretch, not just min/max, so a few extreme pixels don't
    # wash out the hotspots that actually matter.
    lo, hi = np.percentile(saliency, 1), np.percentile(saliency, 99)
    return np.clip((saliency - lo) / (hi - lo + 1e-8), 0, 1)


def overlay_saliency(orig: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    saliency = stretch_contrast(saliency)
    heat = cm.inferno(saliency)[:, :, :3]
    alpha = (saliency**0.4)[:, :, None]  # stronger boost than a plain linear blend
    return orig * (1 - alpha) + heat * alpha


def load_normalized(path: str) -> np.ndarray:
    arr = imread(path)[:, :, :3].astype(float)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def main():
    style()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(len(ROW_LABELS), len(SPECIES_ORDER), figsize=(12, 10), dpi=200)
    for col, species in enumerate(SPECIES_ORDER):
        orig_fname, cam_fname, guided_fname = GRAD_CAM_EXAMPLES[species]
        orig = load_normalized(os.path.join(GRAD_CAM_DIR, orig_fname))
        cam_overlay = load_normalized(os.path.join(GRAD_CAM_DIR, cam_fname))
        guided_saliency = load_normalized(os.path.join(GRAD_CAM_DIR, guided_fname))[:, :, 0]

        axes[0, col].imshow(orig)
        axes[1, col].imshow(cam_overlay)
        axes[2, col].imshow(overlay_saliency(orig, guided_saliency))

        axes[0, col].set_title(SPECIES_FULL[species], fontsize=12.5, color=COLORS[species], fontweight="bold")
        for row in range(len(ROW_LABELS)):
            axes[row, col].axis("off")

    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].text(
            -0.08, 0.5, label, transform=axes[row, 0].transAxes, rotation=90,
            fontsize=11.5, color=TEXT_SECONDARY, ha="center", va="center",
        )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "4_guided_gradcam.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
