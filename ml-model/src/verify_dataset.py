"""
Dataset Verification & Validation Script for AgriShield (SIH 26131).
Verifies:
1. Dataset directory structure and raw class folders
2. Stratified 70/15/15 train/val/test split generation and zero data leakage
3. Image loading, Albumentations transforms, and tensor shapes
4. Class distribution profiling, imbalance ratio, and class weight generation
5. DataLoader batch iteration
"""

import sys
from pathlib import Path

# Add ml-model to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from src import config
from src.dataset import (
    create_stratified_splits,
    analyze_class_distribution,
    compute_class_weights,
    get_dataloaders
)


def run_dataset_verification():
    print("=" * 75)
    print("AgriShield Part 3: Dataset Pipeline & Preprocessing Verification")
    print("=" * 75)

    # 1. Verify Raw Dataset Structure
    raw_dir = config.RAW_DATA_DIR
    class_folders = sorted([d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"\n[STEP 1] Raw Dataset Inspection:")
    print(f"  Raw Directory Path     : {raw_dir}")
    print(f"  Total Class Directories: {len(class_folders)} of {config.NUM_CLASSES}")
    assert len(class_folders) == config.NUM_CLASSES, f"Expected 38 classes, found {len(class_folders)}"

    # 2. Execute Stratified Splitting
    print(f"\n[STEP 2] Stratified 70/15/15 Splitting (Seed = {config.RANDOM_SEED}):")
    train_df, val_df, test_df = create_stratified_splits(
        raw_dir=config.RAW_DATA_DIR,
        processed_dir=config.PROCESSED_DATA_DIR,
        class_indices_path=config.CLASS_INDICES_PATH,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=config.RANDOM_SEED
    )

    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"  Total Valid Images Cataloged: {total_samples}")
    print(f"  Training Set   : {len(train_df):5d} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  Validation Set : {len(val_df):5d} samples ({len(val_df)/total_samples*100:.1f}%)")
    print(f"  Testing Set    : {len(test_df):5d} samples ({len(test_df)/total_samples*100:.1f}%)")

    # 3. Verify Zero Data Leakage
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert len(train_paths.intersection(val_paths)) == 0, "Leakage: Train <-> Val"
    assert len(train_paths.intersection(test_paths)) == 0, "Leakage: Train <-> Test"
    assert len(val_paths.intersection(test_paths)) == 0, "Leakage: Val <-> Test"
    print(f"  Data Leakage Check: PASSED (Strict disjoint image paths across all splits)")

    # 4. Class Distribution & Imbalance Analysis
    print(f"\n[STEP 3] Class Distribution & Imbalance Analysis:")
    summary = analyze_class_distribution(train_df)
    print(f"  Total Training Classes Detected : {summary['num_classes']}")
    print(f"  Min Samples per Class           : {summary['min_samples_per_class']}")
    print(f"  Max Samples per Class           : {summary['max_samples_per_class']}")
    print(f"  Median Samples per Class        : {summary['median_samples_per_class']:.1f}")
    print(f"  Imbalance Ratio (Max / Min)     : {summary['imbalance_ratio']}x")
    print(f"  Dataset Status                  : {'IMBALANCED (Class weighting recommended)' if summary['is_imbalanced'] else 'BALANCED'}")

    # Display Top-3 and Bottom-3 classes in training set
    dist_items = sorted(summary["per_class_distribution"].items(), key=lambda x: x[1]["count"], reverse=True)
    print("\n  Top-3 Represented Classes in Training Split:")
    for idx, d in dist_items[:3]:
        print(f"    Class {idx:2d} ({d['class_name']}): {d['count']} images ({d['percentage']}%)")
    print("  Bottom-3 Represented Classes in Training Split:")
    for idx, d in dist_items[-3:]:
        print(f"    Class {idx:2d} ({d['class_name']}): {d['count']} images ({d['percentage']}%)")

    # 5. Class Weights Generation
    print(f"\n[STEP 4] Class Imbalance Mechanism:")
    class_weights = compute_class_weights(train_df)
    print(f"  Computed Balanced Class Weights Tensor Shape: {class_weights.shape}")
    print(f"  Weight Range: Min = {class_weights.min().item():.3f}, Max = {class_weights.max().item():.3f}")
    print(f"  Ready for nn.CrossEntropyLoss(weight=class_weights) during training (Part 5)")

    # 6. DataLoader Verification
    print(f"\n[STEP 5] DataLoader Batch Inspection:")
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=config.PROCESSED_DATA_DIR,
        batch_size=16,
        num_workers=0
    )

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"  Train Batch Image Tensor Shape : {batch_imgs.shape} (Expected: [16, 3, 224, 224])")
    print(f"  Train Batch Tensor Dtype       : {batch_imgs.dtype}")
    print(f"  Train Batch Labels Shape       : {batch_labels.shape}")
    print(f"  Train Batch Label Range        : [{batch_labels.min().item()} .. {batch_labels.max().item()}]")
    print(f"  Pixel Value Range (Normalized) : [{batch_imgs.min().item():.2f} .. {batch_imgs.max().item():.2f}]")

    val_imgs, val_labels = next(iter(val_loader))
    print(f"  Val Batch Image Tensor Shape   : {val_imgs.shape} (Deterministic Preprocessing)")

    test_imgs, test_labels = next(iter(test_loader))
    print(f"  Test Batch Image Tensor Shape  : {test_imgs.shape} (Deterministic Preprocessing)")

    assert batch_imgs.shape == (16, 3, 224, 224), "Incorrect training tensor dimensions!"
    assert val_imgs.shape == (16, 3, 224, 224), "Incorrect validation tensor dimensions!"
    assert test_imgs.shape == (16, 3, 224, 224), "Incorrect test tensor dimensions!"

    print("\n" + "=" * 75)
    print("ALL PART 3 VERIFICATIONS PASSED SUCCESSFULLY.")
    print("NO MODEL WAS TRAINED. NO MODEL WEIGHTS CREATED.")
    print("=" * 75)


if __name__ == "__main__":
    run_dataset_verification()
