"""
Training Pipeline Module for AgriShield Crop Disease Detection (SIH 26131).
Implements the training loop with class-weighted CrossEntropyLoss,
validation metrics tracking, early stopping, and best model checkpointing.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Ensure ml-model is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd

from src import config
from src.model import create_model, freeze_backbone, unfreeze_backbone, count_parameters
from src.dataset import get_dataloaders, compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, accuracy_percentage)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {current_loss:.4f} | Acc: {current_acc:.2f}% | Elapsed: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs evaluation on validation or test set and returns (mean_loss, accuracy_percentage)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    epochs: int = 2,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.PHASE1_LR,
    use_class_weights: bool = True,
    device: torch.device = config.DEVICE,
    save_path: Path = config.BEST_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Executes training on the PlantVillage dataset with checkpointing.
    Saves the best model state dict, metadata, and optimizer state to disk.
    """
    logger.info(f"Starting Training for {model_name.upper()} on device: {device}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Initializing DataLoaders from processed split manifests...")
    train_loader, val_loader, _ = get_dataloaders(
        processed_dir=config.PROCESSED_DATA_DIR,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
    )

    # 2. Setup Loss Criterion (with Class Imbalance Mitigation)
    if use_class_weights:
        train_df = pd.read_csv(config.TRAIN_MANIFEST_PATH)
        weights = compute_class_weights(train_df, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        logger.info("Class-weighted CrossEntropyLoss configured for class imbalance handling.")
    else:
        criterion = nn.CrossEntropyLoss()

    # 3. Instantiate Model & Freeze Backbone (Phase 1 Transfer Learning Warmup)
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=config.DROPOUT_RATE,
    )
    freeze_backbone(model)
    model.to(device)

    param_info = count_parameters(model)
    logger.info(
        f"Model Parameter Summary: Total={param_info['total_parameters']:,} | "
        f"Trainable={param_info['trainable_parameters']:,} | "
        f"Frozen={param_info['frozen_parameters']:,}"
    )

    # 4. Optimizer for Trainable Head
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_loss, val_acc = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"--> Epoch [{epoch}/{epochs}] Summary ({elapsed:.1f}s): "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Checkpointing
        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc

            checkpoint = {
                "model_name": model_name,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "epoch": epoch,
                "image_size": config.IMAGE_SIZE,
                "image_mean": config.IMAGE_MEAN,
                "image_std": config.IMAGE_STD,
                "class_indices_path": str(config.CLASS_INDICES_PATH),
            }

            torch.save(checkpoint, save_path)
            logger.info(f"[SAVE] New best model saved to: {save_path} (Val Acc: {best_val_acc:.2f}%)")

            # Also save periodic epoch checkpoint
            epoch_ckpt_path = config.CHECKPOINTS_DIR / f"{model_name}_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_ckpt_path)

    logger.info(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")
    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "model_path": str(save_path),
    }


if __name__ == "__main__":
    train_model(epochs=2)
