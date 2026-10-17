# conda env: starch_vit
"""Removes scale-bar-text false positives from granule_manifest.csv/crops/.
Confirmed on CE_LS15_Tong_ref_25_LS15--01_DIC_g00.jpg: a crop of the "20" in
the "20 um" scale bar label, which passed the std/aspect filter in
build_granule_crops.py since text has legitimate sharp edges.

Uses the same bottom-right CORNER_FRAC as check_overlay_contamination.py's
scale-bar exclusion zone. A detection is dropped if its bbox *center* falls
inside that corner.

Run: python clean_scalebar_crops.py   (rewrites ../granule_manifest.csv, deletes bad crop files)
"""

import csv
import os

from PIL import Image

from paths import DATA_ROOT, REPO_ROOT

MANIFEST = os.path.join(REPO_ROOT, "granule_manifest.csv")
CROPS_DIR = os.path.join(REPO_ROOT, "crops")
CORNER_FRAC = 0.22


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    print(f"checking {len(rows)} crops")

    size_cache = {}
    kept, dropped = [], []

    for row in rows:
        src = row["source_image_path"]
        if src not in size_cache:
            size_cache[src] = Image.open(os.path.join(DATA_ROOT, src)).size
        w, h = size_cache[src]

        x0, y0, x1, y1 = int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        in_corner = cx > w * (1 - CORNER_FRAC) and cy > h * (1 - CORNER_FRAC)
        if in_corner:
            dropped.append(row)
        else:
            kept.append(row)

    print(f"dropping {len(dropped)} scale-bar-corner detections, keeping {len(kept)}")

    for row in dropped:
        crop_path = os.path.join(CROPS_DIR, row["crop_filename"])
        if os.path.exists(crop_path):
            os.remove(crop_path)

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nwrote {len(kept)} rows to {MANIFEST}")
    by_species = {}
    for r in kept:
        by_species.setdefault(r["species"], 0)
        by_species[r["species"]] += 1
    for species, n in by_species.items():
        print(f"  {species}: {n} crops")


if __name__ == "__main__":
    main()
