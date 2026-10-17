# conda env: starch_vit
"""Two literature-grounded quality filters applied post-hoc to
granule_manifest.csv/crops/ (no re-detection needed, both are computable
from data already on hand):

1. Per-species bottom-5th-percentile area filter (Wilson et al. used an even
   more aggressive top-20%-only comparison subset). Deliberately
   *per-species*, not a global pixel threshold -- species differ hugely in
   real granule size (C. esculenta median bbox area 990px^2 vs D. alata
   17112px^2, a real taxonomic difference, not noise), so a flat cutoff
   would disproportionately gut small-granule species. Calibration against
   5 known good/bad IB crops showed area, not roundness, is what actually
   separates genuine granules from debris/fragments here.
2. Border-touching exclusion (Wilson et al.: discard granules touching the
   image edge, to avoid partial-object bias).

Run: python filter_fragments.py   (rewrites ../granule_manifest.csv, deletes bad crop files)
"""

import csv
import os

from PIL import Image

from paths import DATA_ROOT, REPO_ROOT

MANIFEST = os.path.join(REPO_ROOT, "granule_manifest.csv")
CROPS_DIR = os.path.join(REPO_ROOT, "crops")
BOTTOM_PERCENTILE = 5
BORDER_MARGIN = 1  # px


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    print(f"checking {len(rows)} crops")

    size_cache = {}
    for row in rows:
        src = row["source_image_path"]
        if src not in size_cache:
            size_cache[src] = Image.open(os.path.join(DATA_ROOT, src)).size
        row["_w"], row["_h"] = size_cache[src]
        row["_area"] = (int(row["bbox_x1"]) - int(row["bbox_x0"])) * (int(row["bbox_y1"]) - int(row["bbox_y0"]))

    # Per-species area threshold.
    by_species_areas = {}
    for row in rows:
        by_species_areas.setdefault(row["species"], []).append(row["_area"])
    thresholds = {}
    for species, areas in by_species_areas.items():
        areas.sort()
        idx = max(0, int(len(areas) * BOTTOM_PERCENTILE / 100) - 1)
        thresholds[species] = areas[idx]
    print("per-species area thresholds (bottom 5th percentile):", thresholds)

    kept, dropped_small, dropped_border = [], [], []
    for row in rows:
        x0, y0, x1, y1 = int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"])
        w, h = row["_w"], row["_h"]
        touches_border = x0 <= BORDER_MARGIN or y0 <= BORDER_MARGIN or x1 >= w - BORDER_MARGIN or y1 >= h - BORDER_MARGIN

        if touches_border:
            dropped_border.append(row)
        elif row["_area"] < thresholds[row["species"]]:
            dropped_small.append(row)
        else:
            kept.append(row)

    print(f"dropping {len(dropped_small)} small-fragment crops, {len(dropped_border)} border-touching crops")
    print(f"keeping {len(kept)}")

    for row in dropped_small + dropped_border:
        crop_path = os.path.join(CROPS_DIR, row["crop_filename"])
        if os.path.exists(crop_path):
            os.remove(crop_path)

    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\nwrote {len(kept)} rows to {MANIFEST}")
    by_species = {}
    for r in kept:
        by_species.setdefault(r["species"], 0)
        by_species[r["species"]] += 1
    for species, n in by_species.items():
        print(f"  {species}: {n} crops")


if __name__ == "__main__":
    main()
