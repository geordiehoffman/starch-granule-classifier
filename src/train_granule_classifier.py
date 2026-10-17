# conda env: starch_vit
"""Stratified k-fold training for the per-granule species classifier.

Per project owner direction: specimen identity is not used for splitting.
First attempt (plain StratifiedKFold, no grouping at all) leaked badly --
95%+ balanced accuracy, 100% specimen accuracy, implausibly high given
Wilson et al.'s per-granule results -- because crops from the same source
photo could land in both train and test, letting the model memorize
photo-specific background/lighting instead of granule morphology.

This version groups by source photo (sklearn StratifiedGroupKFold, group =
source_image_path) instead of specimen: folds are still random and
species-balanced, but all crops from one photo stay together on one side of
the split, closing the most obvious leakage path without reintroducing
specimen-level grouping.

Reports both per-granule accuracy (the raw classifier output) and
per-specimen accuracy (summed softmax probabilities across all granules
from one specimen, argmax -- the aggregation step the literature review
found essential: Wilson et al. 2010 showed single-granule classification is
weak but pooling many granules per sample is markedly more reliable). Note
specimens can now have some of their granules in train and some in test
since specimen is no longer a split boundary.

Run: python train_granule_classifier.py   (writes ../results/granule_classifier/)
"""

import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from granule_dataset import LABEL_NAMES, LABELS, GranuleDataset, load_granule_manifest
from granule_model import GranuleResNetClassifier
from paths import REPO_ROOT

N_FOLDS = 5
SPLIT_SEED = 0

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
UNFREEZE_LAST_N_BLOCKS = 1
OUT_DIR = os.path.join(REPO_ROOT, "results", "granule_classifier")


def class_weights(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["species"]] += 1
    total = len(rows)
    n_classes = len(LABELS)
    weights = torch.zeros(n_classes)
    for species, idx in LABELS.items():
        weights[idx] = total / (n_classes * max(1, counts[species]))
    return weights


def train_granule_model(train_rows: list, device: str, log_prefix: str = ""):
    """Trains one GranuleResNetClassifier on train_rows and returns it in
    eval mode. Shared by train_granule_classifier.py, grad_cam.py, and
    guided_grad_cam.py, which each need a fold-trained model but differ in
    what they do with it afterward.
    """
    model = GranuleResNetClassifier(num_classes=len(LABELS), unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS).to(device)
    mean, std = model.image_mean, model.image_std

    train_ds = GranuleDataset(train_rows, IMAGE_SIZE, mean, std, train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    weights = class_weights(train_rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"  {log_prefix}epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_ds):.4f}")

    model.eval()
    return model, mean, std


def run_fold(fold: int, train_rows: list, test_rows: list, device: str):
    model, mean, std = train_granule_model(train_rows, device, log_prefix=f"fold {fold} ")
    test_ds = GranuleDataset(test_rows, IMAGE_SIZE, mean, std, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Evaluate: per-granule predictions + probabilities, kept alongside each
    # row's specimen so we can aggregate to specimen-level afterward.
    all_probs, all_labels, all_specimens, all_species = [], [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    for r in test_rows:
        all_specimens.append(r["specimen"])
        all_species.append(r["species"])

    return all_probs, all_labels, all_specimens, all_species, test_rows


def balanced_accuracy(preds, labels, n_classes):
    per_class_acc = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class_acc.append((preds[mask] == c).mean())
    return float(np.mean(per_class_acc))


def specimen_level_accuracy(probs, labels, specimens, species):
    # Sum softmax probabilities across all granules from the same specimen,
    # argmax -- per Wilson et al.'s aggregation finding.
    by_specimen_probs = defaultdict(lambda: np.zeros(probs.shape[1]))
    by_specimen_true = {}
    for p, y, spec, sp in zip(probs, labels, specimens, species):
        key = (sp, spec)
        by_specimen_probs[key] += p
        by_specimen_true[key] = y

    correct = 0
    total = 0
    per_specimen_results = {}
    for key, summed in by_specimen_probs.items():
        pred = int(np.argmax(summed))
        true = int(by_specimen_true[key])
        per_specimen_results[f"{key[0]}/{key[1]}"] = {"pred": LABEL_NAMES[pred], "true": LABEL_NAMES[true]}
        correct += int(pred == true)
        total += 1
    return correct / total if total else 0.0, per_specimen_results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    all_rows = load_granule_manifest()
    print(f"total crops: {len(all_rows)}")

    y_species = [r["species"] for r in all_rows]
    groups = [r["source_image_path"] for r in all_rows]
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)

    fold_results = []
    all_probs_list, all_labels_list, all_specimens_list, all_species_list = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_rows, y_species, groups)):
        print(f"\n=== fold {fold} ===")
        train_rows = [all_rows[i] for i in train_idx]
        test_rows = [all_rows[i] for i in test_idx]
        probs, labels, specimens, species, test_rows = run_fold(fold, train_rows, test_rows, device)
        preds = np.argmax(probs, axis=1)

        bal_acc = balanced_accuracy(preds, labels, len(LABELS))
        raw_acc = float((preds == labels).mean())
        spec_acc, _ = specimen_level_accuracy(probs, labels, specimens, species)

        print(f"fold {fold}: per-granule acc={raw_acc:.3f} balanced_acc={bal_acc:.3f} specimen_acc={spec_acc:.3f}")

        fold_results.append(
            {
                "fold": fold,
                "n_test_granules": len(test_rows),
                "per_granule_accuracy": raw_acc,
                "balanced_accuracy": bal_acc,
                "specimen_accuracy": spec_acc,
            }
        )
        all_probs_list.append(probs)
        all_labels_list.append(labels)
        all_specimens_list.extend(specimens)
        all_species_list.extend(species)

    mean_bal_acc = np.mean([f["balanced_accuracy"] for f in fold_results])
    std_bal_acc = np.std([f["balanced_accuracy"] for f in fold_results])
    mean_spec_acc = np.mean([f["specimen_accuracy"] for f in fold_results])

    print("\n=== summary ===")
    print(f"mean balanced accuracy (per-granule): {mean_bal_acc:.3f} +/- {std_bal_acc:.3f}")
    print(f"mean specimen-level accuracy: {mean_spec_acc:.3f}")

    out_path = os.path.join(OUT_DIR, "results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fold", "n_test_granules", "per_granule_accuracy", "balanced_accuracy", "specimen_accuracy"]
        )
        writer.writeheader()
        for r in fold_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
        writer.writerow(
            {
                "fold": "mean",
                "n_test_granules": sum(r["n_test_granules"] for r in fold_results),
                "per_granule_accuracy": np.mean([r["per_granule_accuracy"] for r in fold_results]),
                "balanced_accuracy": mean_bal_acc,
                "specimen_accuracy": mean_spec_acc,
            }
        )
        writer.writerow({"fold": "std", "balanced_accuracy": std_bal_acc})
    print(f"\nwrote results to {out_path}")


if __name__ == "__main__":
    main()
