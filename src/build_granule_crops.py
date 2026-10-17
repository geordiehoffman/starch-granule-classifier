# conda env: starch_cellsam
"""Builds the per-granule crop dataset: runs tiled+filtered CellSAM
(cellsam_tiled.run_tiled) across every image in manifest.csv, crops a padded
bounding box around each surviving detection, and writes both the crops and
a tracking CSV (granule_manifest.csv) linking each crop back to its source
image.

Crops go to crops/<species>/<crop_filename>.jpg (repo-local, not the
read-only data repo). Padding is 25% of the bbox's larger dimension on each
side, clipped to image bounds -- deliberately looser than the mask itself,
since build_manifest.py found segmentation can under-cover a granule's
fainter edge (confirmed visually on the C_esculenta small-granule case).

Run: python build_granule_crops.py   (writes ../crops/ and ../granule_manifest.csv)
"""

import csv
import os
import time

from cellSAM import get_model
from PIL import Image

from cellsam_tiled import run_tiled
from paths import DATA_ROOT, REPO_ROOT

MANIFEST = os.path.join(REPO_ROOT, "manifest.csv")
OUT_CSV = os.path.join(REPO_ROOT, "granule_manifest.csv")
CROPS_DIR = os.path.join(REPO_ROOT, "crops")

PAD_FRAC = 0.25


def pad_bbox(x0, y0, x1, y1, w, h, frac):
    bw, bh = x1 - x0, y1 - y0
    pad = frac * max(bw, bh)
    nx0 = max(0, x0 - pad)
    ny0 = max(0, y0 - pad)
    nx1 = min(w, x1 + pad)
    ny1 = min(h, y1 + pad)
    return int(nx0), int(ny0), int(nx1), int(ny1)


def main():
    os.makedirs(CROPS_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    print(f"processing {len(rows)} source images")

    model = get_model(model="cellsam_general")

    out_rows = []
    t_start = time.time()
    for i, row in enumerate(rows):
        rel_path = row["image_path"]
        species = row["species"]
        specimen = row["specimen"]
        fold = row["fold"]

        species_dir = os.path.join(CROPS_DIR, species)
        os.makedirs(species_dir, exist_ok=True)

        path = os.path.join(DATA_ROOT, rel_path)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] ERROR opening {rel_path}: {e}")
            continue

        w, h = img.size
        detections = run_tiled(img, model, filter_detections=True)

        stem = os.path.splitext(os.path.basename(rel_path))[0]
        for g_idx, d in enumerate(detections):
            x0, y0, x1, y1 = d["x0"], d["y0"], d["x1"], d["y1"]
            px0, py0, px1, py1 = pad_bbox(x0, y0, x1, y1, w, h, PAD_FRAC)
            crop = img.crop((px0, py0, px1, py1))
            crop_filename = f"{species}_{specimen}_{stem}_g{g_idx:02d}.jpg"
            crop.save(os.path.join(species_dir, crop_filename), quality=95)

            out_rows.append(
                {
                    "source_image_path": rel_path,
                    "crop_filename": f"{species}/{crop_filename}",
                    "species": species,
                    "specimen": specimen,
                    "fold": fold,
                    "bbox_x0": int(x0),
                    "bbox_y0": int(y0),
                    "bbox_x1": int(x1),
                    "bbox_y1": int(y1),
                }
            )

        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        remaining = avg * (len(rows) - i - 1)
        print(
            f"[{i+1}/{len(rows)}] {rel_path}: {len(detections)} granules "
            f"({elapsed:.0f}s elapsed, ~{remaining/60:.1f}min remaining)"
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_image_path",
            "crop_filename",
            "species",
            "specimen",
            "fold",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nwrote {len(out_rows)} granule crops from {len(rows)} source images to {CROPS_DIR}")
    print(f"wrote tracking CSV to {OUT_CSV}")

    by_species = {}
    for r in out_rows:
        by_species.setdefault(r["species"], 0)
        by_species[r["species"]] += 1
    for species, n in by_species.items():
        print(f"  {species}: {n} crops")


if __name__ == "__main__":
    main()
