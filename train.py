"""
===========================================================================
  COMPLETE TRAINING PIPELINE
  Two-Phase Transfer Learning: Warmup → Full Fine-Tuning
===========================================================================

TRAINING LOOP OVERVIEW
-----------------------
A training "epoch" consists of:
  1. Training loop:   iterate over all training batches, update weights
  2. Validation loop: iterate over val set, NO weight updates, track accuracy
  3. LR scheduling:   adjust learning rate based on epoch/progress
  4. Checkpointing:   save model if val_acc improved

This script handles:
  Phase 1 — Warmup (frozen backbone, head only)
  Phase 2 — Fine-tuning (full model, low LR)
  Early stopping, progress bars, metrics logging
"""

import os
import sys
import time
import json
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau




from pathlib import Path
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))


from eurosat_dataset import get_dataloaders, EUROSAT_CLASSES
from classifier import SatelliteClassifier, build_model, save_model
from config import MODEL_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, CLASS_NAMES


# ─── Metrics Tracker ─────────────────────────────────────────────────────────
class MetricsTracker:
    """
    Stores and retrieves per-epoch training metrics.
    Saved to JSON for later plotting and analysis.
    """

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0

    def update(self, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.train_accs.append(float(train_acc))
        self.val_accs.append(float(val_acc))
        self.learning_rates.append(float(lr))
        self.epoch_times.append(float(epoch_time))

        if val_acc > self.best_val_acc:
            self.best_val_acc = float(val_acc)
            self.best_epoch = len(self.val_accs) - 1

    def to_dict(self) -> Dict:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
        }

    def save(self, path: str):
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Metrics saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        with open(path) as f:
            data = json.load(f)
        tracker = cls()
        for k, v in data.items():
            setattr(tracker, k, v)
        return tracker


# ─── Training loop (one epoch) ───────────────────────────────────────────────
def train_one_epoch(
    model: SatelliteClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_every: int = 20,
    scaler=None,  # GradScaler for AMP
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    At each step:
      1. Forward pass: model(images) → predicted logits
      2. Loss: CrossEntropy(logits, true_labels)
      3. Backward: compute gradient of loss w.r.t. every weight
      4. Optimizer step: update weights in direction of -gradient

    Returns:
        (epoch_loss, epoch_accuracy)
    """
    model.train()  # Enable dropout, batch norm in training mode

    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients from previous step
        # (gradients accumulate by default — we must reset each batch)
        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            logits = model(images)             # [B, 10]
            loss = criterion(logits, labels)   # scalar

            # Backward pass: compute gradients
            loss.backward()

            # Gradient clipping: prevent exploding gradients
            # Clips gradient norm to max_norm=1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights using gradients
            optimizer.step()

        # Track statistics
        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Progress output
        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == n_batches:
            running_acc = correct / total * 100
            running_loss = total_loss / total
            print(
                f"  Epoch {epoch+1} | Batch {batch_idx+1}/{n_batches} | "
                f"Loss: {running_loss:.4f} | Acc: {running_acc:.1f}%",
                end="\r"
            )

    print()  # newline after \r
    epoch_loss = total_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ─── Validation loop ─────────────────────────────────────────────────────────
@torch.no_grad()   # Disable gradient computation — saves memory and time
def validate(
    model: SatelliteClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one validation epoch (no weight updates).

    The @torch.no_grad() decorator disables the automatic gradient tracking
    that PyTorch normally does. This is faster and uses less memory.

    Returns:
        (val_loss, val_accuracy)
    """
    model.eval()  # Disable dropout, use running stats for batch norm

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ─── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when validation accuracy stops improving.

    WHY EARLY STOPPING?
    If we train for too many epochs, the model starts memorising the
    training data (overfitting) and generalisation degrades. Early stopping
    halts training when the validation metric plateaus.

    We keep a copy of the best model weights and restore them at the end.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_acc = 0.0
        self.best_weights = None
        self.stop = False

    def __call__(self, val_acc: float, model: SatelliteClassifier) -> bool:
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop

    def restore_best(self, model: SatelliteClassifier):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored best weights (val_acc = {self.best_val_acc:.4f})")


# ─── Optimizer factory ────────────────────────────────────────────────────────
def build_optimizer(
    model: SatelliteClassifier,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    Create optimizer with parameter groups.

    We use different learning rates for backbone vs head:
      - Backbone (pretrained): lower LR (lr / 10)
      - Head (new): full LR

    This is called "discriminative learning rates" and preserves
    pretrained features while allowing the head to adapt faster.
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    if optimizer_name == "adamw":
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr / 10, "weight_decay": weight_decay},
            {"params": head_params,     "lr": lr,       "weight_decay": weight_decay},
        ])
    elif optimizer_name == "adam":
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params,     "lr": lr},
        ])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD([
            {"params": backbone_params, "lr": lr / 10, "weight_decay": weight_decay, "momentum": 0.9},
            {"params": head_params,     "lr": lr,       "weight_decay": weight_decay, "momentum": 0.9},
        ], nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


# ─── Main training function ──────────────────────────────────────────────────
def train(
    model_config: Dict = None,
    dataset_config: Dict = None,
    training_config: Dict = None,
) -> Tuple[SatelliteClassifier, MetricsTracker]:
    """
    Full two-phase training pipeline.

    Phase 1 (warmup): Train head only for warmup_epochs.
    Phase 2 (finetune): Unfreeze backbone, train all layers.

    Returns:
        (trained_model, metrics_tracker)
    """
    # Merge configs with defaults
    mc = {**MODEL_CONFIG,    **(model_config    or {})}
    dc = {**DATASET_CONFIG,  **(dataset_config  or {})}
    tc = {**TRAINING_CONFIG, **(training_config or {})}

    # ── Device setup ────────────────────────────────────────────────────────
    if tc["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(tc["device"])
    print(f"\nTraining device: {device}")
    if device.type == "cpu":
        print("  (CPU training: ~2-4h for EfficientNet-B0 × 30 epochs)")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading EuroSAT dataset...")
    train_loader, val_loader, _ = get_dataloaders(
        root=dc["data_root"],
        batch_size=tc["batch_size"],
        num_workers=tc["num_workers"],
        val_split=dc["val_split"],
        test_split=dc["test_split"],
        use_weighted_sampler=dc["use_weighted_sampler"],
        seed=dc["seed"],
        img_size=dc["img_size"],
        download=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        model_name=mc["model_name"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
        pretrained=mc["pretrained"],
        device=str(device),
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss with label smoothing
    # Label smoothing: instead of target=[0,0,1,...], use [0.01, 0.01, 0.89, ...]
    # This prevents the model from becoming overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=tc.get("label_smoothing", 0.1))

    # ── AMP scaler (for GPU mixed precision, no-op on CPU) ────────────────────
    scaler = torch.cuda.amp.GradScaler() if (tc.get("use_amp") and device.type == "cuda") else None

    # ── Metrics and early stopping ─────────────────────────────────────────
    tracker = MetricsTracker()
    early_stopper = EarlyStopping(patience=tc["early_stopping_patience"])

    # ── Output paths ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = mc["model_name"]
    best_model_path = Path(tc["save_dir"]) / f"{model_name}_best_{timestamp}.pth"
    metrics_path    = Path(tc["log_dir"])  / f"metrics_{model_name}_{timestamp}.json"
    os.makedirs(tc["save_dir"], exist_ok=True)
    os.makedirs(tc["log_dir"],  exist_ok=True)

    total_epochs   = tc["num_epochs"]
    warmup_epochs  = tc["warmup_epochs"]
    best_val_acc   = 0.0

    # ════════════════════════════════════════════════════════════════════════
    #  PHASE 1: WARMUP — Train head only (backbone frozen)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Warmup Training (epochs 1–{warmup_epochs})")
    print(f"  Backbone FROZEN | Training head only")
    print(f"{'='*60}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc["warmup_lr"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=tc["min_lr"])

    for epoch in range(warmup_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, tc["print_every"], scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]

        tracker.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)

        print(
            f"  Epoch {epoch+1:3d}/{warmup_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(best_model_path), optimizer, epoch, val_acc,
                       extra={"phase": "warmup"})

    # ════════════════════════════════════════════════════════════════════════
    #  PHASE 2: FULL FINE-TUNING — Unfreeze backbone
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Full Fine-Tuning (epochs {warmup_epochs+1}–{total_epochs})")
    print(f"  Backbone UNFROZEN | LR = {tc['finetune_lr']:.1e}")
    print(f"{'='*60}")

    model.unfreeze_backbone()
    model.param_summary()

    optimizer = build_optimizer(
        model,
        optimizer_name=tc["optimizer"],
        lr=tc["finetune_lr"],
        weight_decay=tc["weight_decay"],
    )

    remaining_epochs = total_epochs - warmup_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=tc["min_lr"])

    for epoch_idx in range(remaining_epochs):
        epoch = warmup_epochs + epoch_idx
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, tc["print_every"], scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]

        tracker.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)

        print(
            f"  Epoch {epoch+1:3d}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.0f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(best_model_path), optimizer, epoch, val_acc,
                       extra={"phase": "finetune", "class_names": CLASS_NAMES})
            print(f"  ✓ New best model saved! (val_acc = {val_acc*100:.2f}%)")

        # Early stopping check
        if early_stopper(val_acc, model):
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            early_stopper.restore_best(model)
            break

    # ── Save final metrics ─────────────────────────────────────────────────
    tracker.save(str(metrics_path))

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}% (epoch {tracker.best_epoch+1})")
    print(f"  Model saved:       {best_model_path}")
    print(f"  Metrics saved:     {metrics_path}")
    print(f"  Total time:        {sum(tracker.epoch_times)/60:.1f} min")
    print(f"{'='*60}\n")

    return model, tracker


# ─── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EuroSAT Satellite Classifier")
    parser.add_argument("--model",   default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "mobilenet_v3", "vit_tiny"])
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--warmup",  type=int, default=5)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data",    default=str(_SCRIPT_DIR / "data"), help="Dataset root")
    parser.add_argument("--fast",    action="store_true", help="Fast mode (15 epochs)")
    args = parser.parse_args()

    if args.fast:
        print("Running in FAST mode (15 epochs, MobileNetV3)")
        from config import PRESET_FAST as tc
        tc["num_epochs"] = 15
    else:
        tc = {**TRAINING_CONFIG}

    tc.update({
        "num_epochs": args.epochs,
        "warmup_epochs": args.warmup,
        "finetune_lr": args.lr,
        "batch_size": args.batch,
        "num_workers": args.workers,
    })

    mc = {**MODEL_CONFIG, "model_name": args.model}
    dc = {**DATASET_CONFIG, "data_root": args.data}

    model, tracker = train(model_config=mc, dataset_config=dc, training_config=tc)
    print(f"\nFinal best validation accuracy: {tracker.best_val_acc*100:.2f}%")