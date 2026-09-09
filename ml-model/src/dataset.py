"""
Dataset Setup, Preprocessing, and DataLoader Module for AgriShield (SIH 26131).
Implements PlantVillageDataset, data validation, stratified splitting (70/15/15),
safe data augmentation pipelines, class distribution profiling, and imbalance weighting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import config

# Setup module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Image Preprocessing & Augmentation Pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """
    Returns training data augmentation pipeline.
    Applies only safe, agricultural-image augmentations (geometric and photometric)
    without altering pathological lesion structures.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        # Natural rotational and reflection symmetries of plant leaves
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Handheld camera shift, mild scale, and rotation angle variations
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=30,
            interpolation=1,  # cv2.INTER_LINEAR
            border_mode=0,    # cv2.BORDER_CONSTANT
            p=0.5
        ),
        # Natural sunlight and shade brightness/contrast variations
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5
        ),
        # Channel-wise ImageNet standardization for EfficientNet-B0
        A.Normalize(
            mean=config.IMAGE_MEAN,
            std=config.IMAGE_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_val_test_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """
    Returns deterministic validation/test preprocessing pipeline.
    Strictly NO random augmentations to avoid data corruption or evaluation bias.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=config.IMAGE_MEAN,
            std=config.IMAGE_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset Implementation
# ---------------------------------------------------------------------------

class PlantVillageDataset(Dataset):
    """
    Custom PyTorch Dataset for PlantVillage Leaf Disease Classification.
    Safely loads images, converts to RGB, validates file integrity, and applies transforms.
    """

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, Path],
        transform: Optional[A.Compose] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            data_source: DataFrame or path to manifest CSV (train.csv, val.csv, test.csv).
            transform: Albumentations transformation pipeline.
            base_dir: Base directory to resolve relative paths if necessary.
        """
        if isinstance(data_source, (str, Path)):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy().reset_index(drop=True)
        else:
            raise ValueError("data_source must be a DataFrame or path to CSV manifest")

        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        raw_path = Path(row["image_path"])

        if not raw_path.is_absolute() and self.base_dir:
            img_path = self.base_dir / raw_path
        else:
            img_path = raw_path

        label = int(row["label_idx"])

        # Safe image loading and verification
        try:
            with Image.open(img_path) as pil_img:
                # Convert to RGB (handles RGBA, Grayscale, CMYK gracefully)
                rgb_img = pil_img.convert("RGB")
                img_array = np.array(rgb_img)
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            raise IOError(f"Unreadable or corrupted image: {img_path}") from e

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=img_array)
            img_tensor = augmented["image"]
        else:
            # Fallback: simple scaling and tensor conversion
            img_array = img_array.transpose((2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_array)

        return img_tensor, label


# ---------------------------------------------------------------------------
# 3. Stratified Splitting & Data Leakage Prevention
# ---------------------------------------------------------------------------

def create_stratified_splits(
    raw_dir: Path = config.RAW_DATA_DIR,
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    class_indices_path: Path = config.CLASS_INDICES_PATH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans raw dataset directory, performs stratified 70/15/15 splitting,
    verifies file readability, guarantees zero data leakage across splits,
    and writes reproducible train.csv, val.csv, and test.csv manifests.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load class mapping
    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    dir_to_meta = {}
    for idx_str, meta in class_indices.items():
        dir_to_meta[meta["raw_name"]] = {
            "label_idx": int(idx_str),
            "crop": meta["crop"],
            "disease": meta["disease"],
            "is_healthy": meta["is_healthy"],
        }

    # Scan dataset files
    records = []
    corrupted_files = []

    logger.info(f"Scanning raw images from: {raw_dir}")
    class_folders = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for folder in sorted(class_folders):
        folder_name = folder.name
        if folder_name not in dir_to_meta:
            logger.warning(f"Unrecognized class folder found: {folder_name}. Skipping.")
            continue

        meta = dir_to_meta[folder_name]
        image_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))

        for img_p in image_files:
            # Check basic file integrity
            try:
                if img_p.stat().st_size == 0:
                    corrupted_files.append((str(img_p), "Zero-byte file"))
                    continue
                records.append({
                    "image_path": str(img_p.resolve()),
                    "class_name": folder_name,
                    "label_idx": meta["label_idx"],
                    "crop": meta["crop"],
                    "disease": meta["disease"],
                    "is_healthy": meta["is_healthy"],
                })
            except Exception as e:
                corrupted_files.append((str(img_p), str(e)))

    if corrupted_files:
        logger.warning(f"Found {len(corrupted_files)} corrupted/unreadable files during scan:")
        for cf, reason in corrupted_files[:10]:
            logger.warning(f"  {cf} -> {reason}")

    all_df = pd.DataFrame(records)
    total_samples = len(all_df)
    logger.info(f"Total valid images cataloged: {total_samples} across {all_df['label_idx'].nunique()} classes.")

    # Stratified Split 1: Train vs. (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        all_df,
        test_size=temp_ratio,
        stratify=all_df["label_idx"],
        random_state=seed,
        shuffle=True
    )

    # Stratified Split 2: Val vs. Test (equal 50/50 split of the temp subset = 15% and 15% overall)
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_df["label_idx"],
        random_state=seed,
        shuffle=True
    )

    # Data Leakage Prevention Verification
    train_set = set(train_df["image_path"])
    val_set = set(val_df["image_path"])
    test_set = set(test_df["image_path"])

    assert len(train_set.intersection(val_set)) == 0, "DATA LEAKAGE DETECTED: Overlap between Train and Val sets!"
    assert len(train_set.intersection(test_set)) == 0, "DATA LEAKAGE DETECTED: Overlap between Train and Test sets!"
    assert len(val_set.intersection(test_set)) == 0, "DATA LEAKAGE DETECTED: Overlap between Val and Test sets!"
    logger.info("Data leakage verification passed: Exactly 0 overlapping image paths between all splits.")

    # Save processed manifest CSVs
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved Train manifest: {train_path} ({len(train_df)} samples, {len(train_df)/total_samples:.1%})")
    logger.info(f"Saved Val manifest:   {val_path} ({len(val_df)} samples, {len(val_df)/total_samples:.1%})")
    logger.info(f"Saved Test manifest:  {test_path} ({len(test_df)} samples, {len(test_df)/total_samples:.1%})")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 4. Class Distribution & Imbalance Analysis
# ---------------------------------------------------------------------------

def analyze_class_distribution(
    manifest_source: Union[pd.DataFrame, str, Path],
    class_indices_path: Path = config.CLASS_INDICES_PATH
) -> Dict:
    """
    Analyzes class distribution across the dataset to measure balance and imbalance ratios.
    """
    if isinstance(manifest_source, (str, Path)):
        df = pd.read_csv(manifest_source)
    else:
        df = manifest_source.copy()

    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    counts = df["label_idx"].value_counts().sort_index()
    total = len(df)

    distribution = {}
    for idx, count in counts.items():
        meta = class_indices[str(idx)]
        distribution[int(idx)] = {
            "class_name": meta["raw_name"],
            "crop": meta["crop"],
            "disease": meta["disease"],
            "is_healthy": meta["is_healthy"],
            "count": int(count),
            "percentage": round((count / total) * 100, 2)
        }

    counts_arr = np.array(list(counts.values))
    min_count = int(counts_arr.min())
    max_count = int(counts_arr.max())
    median_count = float(np.median(counts_arr))
    imbalance_ratio = round(max_count / min_count, 2)

    summary = {
        "total_samples": total,
        "num_classes": len(counts),
        "min_samples_per_class": min_count,
        "max_samples_per_class": max_count,
        "median_samples_per_class": median_count,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": (imbalance_ratio > 3.0),
        "per_class_distribution": distribution
    }

    return summary


# ---------------------------------------------------------------------------
# 5. Class Imbalance Mitigation Mechanisms
# ---------------------------------------------------------------------------

def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = config.NUM_CLASSES
) -> torch.FloatTensor:
    """
    Computes balanced class weights for use with nn.CrossEntropyLoss(weight=weights).
    Weight formula: total_samples / (num_classes * class_count).
    """
    classes = np.arange(num_classes)
    y_train = train_df["label_idx"].to_numpy()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    tensor_weights = torch.FloatTensor(weights)
    return tensor_weights


def create_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler for DataLoader to oversample minority classes
    and present a balanced distribution per batch during training.
    """
    class_counts = train_df["label_idx"].value_counts().to_dict()
    total_samples = len(train_df)

    # Class weight = total / class_count
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    sample_weights = [class_weights[label] for label in train_df["label_idx"]]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ---------------------------------------------------------------------------
# 6. DataLoader Builders
# ---------------------------------------------------------------------------

def get_dataloaders(
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampler: bool = False,
    pin_memory: bool = config.PIN_MEMORY
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds and returns (train_loader, val_loader, test_loader).
    """
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"Processed CSV manifests not found in {processed_dir}. "
            f"Run create_stratified_splits() first."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = PlantVillageDataset(train_df, transform=get_train_transforms())
    val_dataset = PlantVillageDataset(val_df, transform=get_val_test_transforms())
    test_dataset = PlantVillageDataset(test_df, transform=get_val_test_transforms())

    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
