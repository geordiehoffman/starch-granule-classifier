# conda env: starch_vit
"""Builds the final deduplicated training manifest from the manually-cleaned
data/ layout (data/<Species>/<Specimen>/JPG/{DIC,Other,POL}).

Usable images = JPG/DIC + JPG/Other (POL and the old top-level Other/
measurement-dump folders are excluded entirely -- POL is confirmed to hold
the measurement-overlay images regardless of filename, per project owner).

Two problems get resolved here, not just file listing:
1. Overlay leakage into JPG/DIC: even after manual sorting, some DIC files
   still carry burned-in measurement overlays (confirmed e.g. on
   D_bulbifera/LS2/JPG/DIC/LS2_17_DIC_40x.jpg) -- caught via the same
   near-white-pixel-outside-scale-bar-corner heuristic as check_overlay_contamination.py.
2. Near-duplicates (near_duplicates.csv, from find_near_duplicates.py):
   within each cluster of near-identical shots of the same field, only the
   single lowest-overlay-score (cleanest) image is kept; the rest are dropped
   as redundant/contaminated duplicates of the kept one.

Run: python build_manifest.py   (writes ../manifest.csv)
"""

import csv
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from paths import DATA_ROOT, REPO_ROOT

DATA_DIR = os.path.join(DATA_ROOT, "data")
NEAR_DUP_CSV = os.path.join(REPO_ROOT, "near_duplicates.csv")
OUT_CSV = os.path.join(REPO_ROOT, "manifest.csv")

SPECIES_DIRS = {
    "D_alata": "DA",
    "D_bulbifera": "DB",
    "C_esculenta": "CE",
    "I_batatas": "IB",
}

WHITE_THRESHOLD = 235
CORNER_FRAC = 0.22


def overlay_score(path: str) -> int:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    h, w, _ = arr.shape
    white = np.all(arr >= WHITE_THRESHOLD, axis=2)
    corner_h = int(h * CORNER_FRAC)
    corner_w = int(w * CORNER_FRAC)
    white[h - corner_h :, w - corner_w :] = False
    return int(white.sum())


def collect_usable():
    """Returns list of (abs_path, species_code, specimen)."""
    rows = []
    for species_dir, species_code in SPECIES_DIRS.items():
        species_path = os.path.join(DATA_DIR, species_dir)
        if not os.path.isdir(species_path):
            continue
        for specimen in os.listdir(species_path):
            specimen_path = os.path.join(species_path, specimen)
            if not os.path.isdir(specimen_path):
                continue
            for sub in ("DIC", "Other"):
                sub_path = os.path.join(specimen_path, "JPG", sub)
                if not os.path.isdir(sub_path):
                    continue
                for root, _, files in os.walk(sub_path):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg")):
                            rows.append((os.path.join(root, f), species_code, specimen))
    return rows


def load_clusters():
    """Returns {rel_path: cluster_id} for every image in a near-dup cluster."""
    if not os.path.exists(NEAR_DUP_CSV):
        print(f"WARNING: {NEAR_DUP_CSV} not found, skipping dedup step")
        return {}
    cluster_of = {}
    with open(NEAR_DUP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cluster_of[row["image_path"]] = row["cluster_id"]
    return cluster_of


def main():
    usable = collect_usable()
    print(f"usable candidate images: {len(usable)}")

    cluster_of = load_clusters()

    # Score every image once (needed both for overlay exclusion and for
    # picking the cleanest representative within each duplicate cluster).
    scored = []
    for path, species_code, specimen in usable:
        rel = os.path.relpath(path, DATA_ROOT).replace(os.sep, "/")
        try:
            score = overlay_score(path)
        except Exception as e:
            print(f"ERROR scoring {path}: {e}")
            continue
        cluster_id = cluster_of.get(rel)
        scored.append(
            {
                "path": path,
                "rel": rel,
                "species": species_code,
                "specimen": specimen,
                "score": score,
                "cluster_id": cluster_id,
            }
        )

    # Resolve clusters: keep only the lowest-score (cleanest) image per cluster.
    by_cluster = defaultdict(list)
    singles = []
    for row in scored:
        if row["cluster_id"] is not None:
            by_cluster[row["cluster_id"]].append(row)
        else:
            singles.append(row)

    kept = list(singles)
    dropped_as_duplicate = []
    for cluster_id, rows in by_cluster.items():
        rows.sort(key=lambda r: r["score"])
        kept.append(rows[0])
        dropped_as_duplicate.extend(rows[1:])

    print(f"clusters resolved: {len(by_cluster)}")
    print(f"dropped as redundant duplicates: {len(dropped_as_duplicate)}")

    # Fold assignment: leave-one-specimen-out per species, in encounter order.
    specimens_by_species = defaultdict(list)
    for row in kept:
        if row["specimen"] not in specimens_by_species[row["species"]]:
            specimens_by_species[row["species"]].append(row["specimen"])

    kept.sort(key=lambda r: r["rel"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "species", "specimen", "fold", "overlay_score"]
        )
        writer.writeheader()
        for row in kept:
            fold = specimens_by_species[row["species"]].index(row["specimen"])
            writer.writerow(
                {
                    "image_path": row["rel"],
                    "species": row["species"],
                    "specimen": row["specimen"],
                    "fold": fold,
                    "overlay_score": row["score"],
                }
            )

    print(f"\nwrote {len(kept)} rows to {OUT_CSV}")
    for species_code in SPECIES_DIRS.values():
        n = sum(1 for r in kept if r["species"] == species_code)
        print(f"  {species_code}: {n}")

    # Flag remaining high-overlay-score singles for manual review -- these are
    # NOT auto-excluded (score alone isn't reliable per earlier false-positive
    # findings on bright DIC highlights), just surfaced.
    high_score_singles = [r for r in kept if r["score"] > 5000]
    if high_score_singles:
        print(f"\n{len(high_score_singles)} kept images with overlay_score > 5000 -- worth a manual look:")
        for r in sorted(high_score_singles, key=lambda r: -r["score"])[:20]:
            print(f"  {r['score']:6d}  {r['rel']}")


if __name__ == "__main__":
    main()
