# conda env: starch_vit
"""Near-duplicate scan (perceptual dHash) across the cleaned usable image set
(JPG/DIC + JPG/Other per specimen, POL and top-level Other/old-measurements
excluded), to find bracketed/near-identical shots of the same field (e.g.
different focus or lighting) that exact byte hashing won't catch.

dHash implemented by hand (no new dependency): resize to 9x8 grayscale,
threshold adjacent-pixel differences -> 64-bit hash. Hamming distance <=
MAX_DIST is treated as a near-duplicate pair; pairs are then grouped via
union-find into clusters.

Run: python find_near_duplicates.py   (writes ../near_duplicates.csv)
"""

import csv
import os
from itertools import combinations

import numpy as np
from PIL import Image

from paths import DATA_ROOT, REPO_ROOT

DATA_DIR = os.path.join(DATA_ROOT, "data")
MAX_DIST = 3  # hamming distance threshold out of 64 bits -- tightened after a cross-species
# false-positive cluster at MAX_DIST=6 showed the hash was keying off generic
# background/scale-bar structure, not granule content
OUT_CSV = os.path.join(REPO_ROOT, "near_duplicates.csv")

SPECIES_DIRS = ["D_alata", "D_bulbifera", "C_esculenta", "I_batatas"]


def collect_usable():
    """Returns {(species, specimen): [paths]} -- comparisons are restricted to
    within the same specimen, since two different specimens (let alone two
    different species) can never share a genuine duplicate field of view.
    """
    groups = {}
    for species in SPECIES_DIRS:
        species_path = os.path.join(DATA_DIR, species)
        if not os.path.isdir(species_path):
            continue
        for specimen in os.listdir(species_path):
            specimen_path = os.path.join(species_path, specimen)
            if not os.path.isdir(specimen_path):
                continue
            paths = []
            for sub in ("DIC", "Other"):
                sub_path = os.path.join(specimen_path, "JPG", sub)
                if not os.path.isdir(sub_path):
                    continue
                for root, _, files in os.walk(sub_path):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg")):
                            paths.append(os.path.join(root, f))
            if paths:
                groups[(species, specimen)] = paths
    return groups


def dhash(path: str, hash_size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    specimen_groups = collect_usable()
    paths = [p for g in specimen_groups.values() for p in g]
    print(f"usable images (JPG/DIC + JPG/Other): {len(paths)}")
    print(f"specimens: {len(specimen_groups)}")

    pair_rows = []
    clusters = []
    for (species, specimen), group_paths in specimen_groups.items():
        hashes = []
        for p in group_paths:
            try:
                hashes.append(dhash(p))
            except Exception as e:
                print(f"ERROR hashing {p}: {e}")
                hashes.append(None)

        uf = UnionFind(len(group_paths))
        for i, j in combinations(range(len(group_paths)), 2):
            if hashes[i] is None or hashes[j] is None:
                continue
            d = hamming(hashes[i], hashes[j])
            if d <= MAX_DIST:
                uf.union(i, j)
                pair_rows.append((d, group_paths[i], group_paths[j]))

        local_groups = {}
        for i, p in enumerate(group_paths):
            root = uf.find(i)
            local_groups.setdefault(root, []).append(p)
        clusters.extend(g for g in local_groups.values() if len(g) > 1)

    clusters.sort(key=lambda g: -len(g))

    print(f"near-duplicate pairs (hamming <= {MAX_DIST}): {len(pair_rows)}")
    print(f"near-duplicate clusters (size > 1): {len(clusters)}")
    n_in_clusters = sum(len(g) for g in clusters)
    print(f"images involved in some cluster: {n_in_clusters} / {len(paths)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "image_path"])
        for cid, g in enumerate(clusters):
            for p in g:
                writer.writerow([cid, os.path.relpath(p, DATA_ROOT).replace(os.sep, "/")])
    print(f"wrote {OUT_CSV}")

    print("\nlargest 10 clusters:")
    for g in clusters[:10]:
        print(f"  size {len(g)}:")
        for p in g:
            print(f"    {os.path.relpath(p, DATA_ROOT)}")


if __name__ == "__main__":
    main()
