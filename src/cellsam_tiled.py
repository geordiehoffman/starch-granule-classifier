# conda env: starch_cellsam
"""Sliding-window tiling wrapper around CellSAM: splits each full image into
overlapping tiles, runs CellSAM per tile, translates detections back to
full-image coordinates, and deduplicates overlap-region detections by
centroid distance (simple NMS, not a rigorous IoU merge -- good enough for
this comparison test).

Run on a sample spanning all species from manifest.csv to check whether the
resolution fix (confirmed on one image in cellsam_resolution_test.py) holds
up generally. Run: python cellsam_tiled.py   (writes outputs/cellsam_tiled/)
"""

import csv
import os
import random
import time

import cv2
import numpy as np
from cellSAM import get_model, segment_cellular_image
from PIL import Image, ImageDraw

from paths import DATA_ROOT, REPO_ROOT

MANIFEST = os.path.join(REPO_ROOT, "manifest.csv")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "cellsam_tiled")

TILE_SIZE = 800
STRIDE = 600  # 200px overlap between adjacent tiles
DEDUP_DIST = 40  # px; detections whose centroids are closer than this are treated as duplicates
# (legacy centroid check, kept as a fast pre-filter; see IOU_MERGE_THRESHOLD below for the fix
# to the bug it missed: two overlapping PARTIAL detections of one granule, e.g.
# DB_LS5_Ton23_DB_LS5_P4_DIC_g00/g01 -- real IoU ~0.30, centroids 66px apart, so the old
# centroid-only check let both through as "different objects" even though they're the same
# granule split into two incomplete pieces by tile-boundary stitching.)
IOU_MERGE_THRESHOLD = 0.15  # below typical NMS (~0.5) on purpose: partial/fragmented duplicate
# detections of the same object can have modest overlap without being fully coincident, so a
# stricter threshold would miss exactly the case above

N_PER_SPECIES = 6
SEED = 0

COLORS = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]

# Post-filter thresholds for dropping blank-tile hallucinations and tile-seam
# artifacts, measured against the FULL original image (not the tile) so tile
# boundaries can't bias the measurement.
MIN_CONTENT_STD = 15.0  # grayscale std inside the detected bbox; real granules show
# visible internal shading/edges, hallucinated boxes on blank background don't.
# Empirically grounded (cellsam_filter_diagnostic.py): real detections scored
# 16.5-65 std, blank-tile hallucinations scored 1.5-3.2 -- clean gap, no tuning needed.
MAX_ASPECT_RATIO = 4.0  # reject long thin boxes typical of tile-seam line artifacts;
# real detections scored aspect <=2.07, seam artifacts scored 2.4-25.7


def sample_images():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_species = {}
    for r in rows:
        by_species.setdefault(r["species"], []).append(r["image_path"])
    random.seed(SEED)
    picked = []
    for species, paths in by_species.items():
        n = min(N_PER_SPECIES, len(paths))
        for p in random.sample(paths, n):
            picked.append((species, p))
    return picked


def tile_bounds(w, h, tile_size, stride):
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    bounds = []
    for y in ys:
        for x in xs:
            bounds.append((x, y, min(x + tile_size, w), min(y + tile_size, h)))
    return bounds


def bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_roundness(mask_local: np.ndarray) -> tuple:
    """Returns (area, perimeter, roundness) for a binary mask using its
    largest contour. Roundness = 4*pi*Area/Perimeter^2 (Guo et al.), 1.0 for
    a perfect circle, lower for elongated/jagged/fragment shapes.
    """
    area = float(mask_local.sum())
    contours, _ = cv2.findContours(mask_local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return area, 0.0, 0.0
    perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), closed=True)
    roundness = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    return area, float(perimeter), float(roundness)


def run_tiled(img: Image.Image, model, filter_detections: bool = True):
    w, h = img.size
    detections = []  # list of dicts: cx, cy, x0, y0, x1, y1, area, perimeter, roundness, touches_border
    gray = np.array(img.convert("L"), dtype=np.float32) if filter_detections else None

    for (x0, y0, x1, y1) in tile_bounds(w, h, TILE_SIZE, STRIDE):
        tile = np.array(img.crop((x0, y0, x1, y1)))
        try:
            mask, _, _ = segment_cellular_image(tile, model=model, device="cuda")
        except AttributeError:
            mask = None
        if mask is None:
            continue

        for label in range(1, int(mask.max()) + 1):
            local_mask = mask == label
            ys, xs = np.where(local_mask)
            if len(xs) == 0:
                continue
            gx0, gx1 = xs.min() + x0, xs.max() + x0
            gy0, gy1 = ys.min() + y0, ys.max() + y0
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2

            if filter_detections:
                region = gray[int(gy0) : int(gy1), int(gx0) : int(gx1)]
                std = float(region.std()) if region.size else 0.0
                bw, bh = max(1, gx1 - gx0), max(1, gy1 - gy0)
                aspect = max(bw / bh, bh / bw)
                if std < MIN_CONTENT_STD or aspect > MAX_ASPECT_RATIO:
                    continue  # blank-tile hallucination or tile-seam artifact

            new_box = (gx0, gy0, gx1, gy1)
            merged_into = None
            for d in detections:
                centroid_dup = ((d["cx"] - cx) ** 2 + (d["cy"] - cy) ** 2) ** 0.5 < DEDUP_DIST
                iou = bbox_iou(new_box, (d["x0"], d["y0"], d["x1"], d["y1"]))
                if centroid_dup or iou >= IOU_MERGE_THRESHOLD:
                    merged_into = d
                    break

            area, perimeter, roundness = mask_roundness(local_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])

            if merged_into is not None:
                # Same object, two partial detections (e.g. split across a tile
                # boundary) -- take the union bbox so neither partial crop survives
                # on its own, rather than arbitrarily discarding the second one.
                merged_into["x0"] = min(merged_into["x0"], gx0)
                merged_into["y0"] = min(merged_into["y0"], gy0)
                merged_into["x1"] = max(merged_into["x1"], gx1)
                merged_into["y1"] = max(merged_into["y1"], gy1)
                merged_into["cx"] = (merged_into["x0"] + merged_into["x1"]) / 2
                merged_into["cy"] = (merged_into["y0"] + merged_into["y1"]) / 2
                merged_into["area"] = max(merged_into["area"], area)  # best available estimate post-merge
                continue

            touches_border = gx0 <= 1 or gy0 <= 1 or gx1 >= w - 1 or gy1 >= h - 1

            detections.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "x0": gx0,
                    "y0": gy0,
                    "x1": gx1,
                    "y1": gy1,
                    "area": area,
                    "perimeter": perimeter,
                    "roundness": roundness,
                    "touches_border": touches_border,
                }
            )

    return detections


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = sample_images()
    print(f"sampling {len(samples)} images across species: {[s[0] for s in samples]}")

    model = get_model(model="cellsam_general")

    results = []
    t_start = time.time()
    for species, rel_path in samples:
        path = os.path.join(DATA_ROOT, rel_path)
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        detections = run_tiled(img, model)
        dt = time.time() - t0
        stem = f"{species}_{os.path.splitext(os.path.basename(rel_path))[0]}"
        print(f"{stem}: {len(detections)} granules ({dt:.1f}s)")
        results.append((species, stem, len(detections)))

        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, d in enumerate(detections):
            draw.rectangle([d["x0"], d["y0"], d["x1"], d["y1"]], outline=COLORS[i % len(COLORS)], width=4)
        draw_img.save(os.path.join(OUT_DIR, f"{stem}_boxes.png"))

    total_dt = time.time() - t_start
    print(f"\ntotal time: {total_dt:.1f}s for {len(samples)} images ({total_dt/len(samples):.1f}s/image avg)")

    print("\nper-species average granule count:")
    by_species = {}
    for species, _, n in results:
        by_species.setdefault(species, []).append(n)
    for species, counts in by_species.items():
        print(f"  {species}: {counts} (avg {sum(counts)/len(counts):.1f})")


if __name__ == "__main__":
    main()
