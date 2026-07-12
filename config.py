"""
===========================================================================
  TRAINING CONFIGURATION
  All Hyperparameters with Explanations
===========================================================================

HYPERPARAMETER GUIDE FOR BEGINNERS
------------------------------------
Every number in this file is a "hyperparameter" — a setting we choose
BEFORE training that controls HOW the model learns.

Unlike model weights (which are learned), hyperparameters are set manually
and require experimentation to optimise. This is called "hyperparameter tuning".

Key hyperparameters and their effects:
  learning_rate:  How big each weight update step is. Too high → unstable.
                  Too low → very slow learning. 1e-3 is a standard starting point.
  batch_size:     How many images to process at once before updating weights.
                  Larger batches → more stable gradients, more RAM needed.
  num_epochs:     How many times to loop over the entire training set.
  weight_decay:   L2 regularization — penalises large weights to prevent overfitting.
  dropout:        Randomly zeros some neurons during training — prevents overfitting.
"""
from pathlib import Path
# ─── Project Root (folder where this file lives) ─────────────────────────────
_PROJECT_DIR = Path(__file__).parent.resolve()

# ─── Model Configuration ──────────────────────────────────────────────────────
MODEL_CONFIG = {
    # Architecture: "efficientnet_b0" | "resnet50" | "mobilenet_v3" | "vit_tiny"
    # Recommendation: Start with efficientnet_b0 (best accuracy/speed tradeoff)
    "model_name": "efficientnet_b0",

    # Number of output classes (10 for EuroSAT)
    "num_classes": 10,

    # Dropout probability in classification head
    # Range: 0.0 (no dropout) to 0.5 (aggressive regularization)
    # 0.3 is a good default for fine-tuning
    "dropout": 0.3,

    # Load pretrained ImageNet weights (ALWAYS True for transfer learning)
    "pretrained": True,
}

# ─── Dataset Configuration ────────────────────────────────────────────────────
DATASET_CONFIG = {
    # Path to store EuroSAT dataset (will auto-download ~90MB)
     "data_root": str(_PROJECT_DIR / "data"),

    # Image size for the model (224 = standard for ImageNet pretrained models)
    "img_size": 224,

    # Fraction of data for validation (15% = ~4,050 images)
    "val_split": 0.15,

    # Fraction of data for testing (15% = ~4,050 images)
    "test_split": 0.15,

    # Random seed for reproducible train/val/test split
    "seed": 42,

    # Use weighted sampling to balance class distribution
    # Set True if classes are very imbalanced (EuroSAT is roughly balanced)
    "use_weighted_sampler": False,
 # ── NEW: multispectral TIF support ────────────────────────────────────────
    # Set input_mode to "tif" to train on Sentinel-2 GeoTIFFs instead of JPEGs.
    # "rgb"  → use torchvision EuroSAT JPEG dataset (current behaviour)
    # "tif"  → use MultiSpectralTifDataset from ms_data_root
    "input_mode": "tif",
    "ms_data_root": r"C:\Users\Lkd\Desktop\proj\SATML\data",

}

# ─── Training Configuration ──────────────────────────────────────────────────
TRAINING_CONFIG = {
    # ── Phase 1: Train head only (backbone frozen) ──────────────────────────
    # Number of epochs to train only the classification head
    # WHY: First, let the new randomly-initialised head stabilise
    #      before unfreezing the pretrained backbone.
    "warmup_epochs": 5,

    # Learning rate for Phase 1 (head training)
    # Higher LR OK here because we're training a fresh layer
    "warmup_lr": 1e-4,

    # ── Phase 2: Full fine-tuning (backbone unfrozen) ────────────────────────
    # Total epochs (including warmup)
    # For EuroSAT + EfficientNet-B0: ~30 epochs → ~90% accuracy
    "num_epochs": 30,

    # Learning rate for Phase 2 (full model)
    # MUST be much lower than Phase 1 to avoid destroying pretrained features
    # Rule of thumb: 10× to 100× lower than Phase 1
    "finetune_lr": 1e-5,

    # ── Optimizer Configuration ──────────────────────────────────────────────
    # Optimizer: "adamw" | "adam" | "sgd"
    # AdamW is preferred for transfer learning:
    #   - Adaptive learning rates per parameter
    #   - Weight decay applied correctly (unlike Adam)
    #   - Converges faster than SGD
    "optimizer": "adamw",

    # Weight decay (L2 regularization)
    # Adds a penalty proportional to the square of each weight
    # Prevents overfitting by keeping weights small
    # Range: 1e-4 to 1e-2 (1e-4 is conservative and safe)
    "weight_decay": 1e-5,

    # ── Batch Size ───────────────────────────────────────────────────────────
    # Number of images per training step
    # 32 is ideal for 16GB RAM + CPU (EfficientNet-B0 + 224×224)
    # Reduce to 16 if you get OOM (Out of Memory) errors
    "batch_size": 32,

    # Number of DataLoader worker processes
    # 2 is safe for most systems (0 = no multiprocessing, safest)
    "num_workers": 2,

    # ── Learning Rate Scheduler ───────────────────────────────────────────────
    # Scheduler type: "cosine" | "step" | "plateau" | "none"
    # Cosine annealing slowly reduces LR following a cosine curve
    # This allows faster initial learning and careful final convergence
    "scheduler": "cosine",

    # Minimum LR for cosine scheduler (don't go below this)
    "min_lr": 1e-6,

    # For StepLR: reduce LR by gamma every step_size epochs
    "lr_step_size": 10,
    "lr_gamma": 0.5,

    # ── Regularisation ────────────────────────────────────────────────────────
    # Label smoothing: instead of hard 0/1 targets, use 0.1/0.9
    # Prevents overconfident predictions and improves calibration
    "label_smoothing": 0.1,

    # ── Early Stopping ────────────────────────────────────────────────────────
    # Stop training if validation accuracy doesn't improve for N epochs
    "early_stopping_patience": 7,

    # ── Checkpointing ────────────────────────────────────────────────────────
    "save_dir": str(_PROJECT_DIR / "results" / "models"),
    "save_best_only": True,         # Only save when val accuracy improves

    # ── Logging ──────────────────────────────────────────────────────────────
     "log_dir": str(_PROJECT_DIR / "results" / "metrics"),
    "print_every": 10,              # Print batch stats every N batches

    # ── Device ────────────────────────────────────────────────────────────────
    # "auto" → use GPU if available, else CPU
    "device": "auto",

    # ── Mixed Precision (FP16) ────────────────────────────────────────────────
    # Speeds up training if GPU is available (no effect on CPU-only)
    "use_amp": False,



    "mixup_alpha": 0.2, 
}

# ─── Hyperparameter Presets ───────────────────────────────────────────────────
# Ready-to-use configurations for different scenarios

PRESET_FAST = {
    # Quick experiment (few hours on CPU)
    **TRAINING_CONFIG,
    "warmup_epochs": 3,
    "num_epochs": 15,
    "batch_size": 64,
    "model_name": "mobilenet_v3",   # Fastest model
}

PRESET_BEST = {
    # Best accuracy (overnight CPU run)
    **TRAINING_CONFIG,
    "model_name": "efficientnet_b0",
    "warmup_epochs": 7,
    "num_epochs": 50,
    "warmup_lr": 5e-4,
    "finetune_lr": 5e-5,
    "weight_decay": 5e-5,
    "label_smoothing": 0.05,
    "early_stopping_patience": 12,
}

PRESET_COMPARISON = {
    # For comparing multiple architectures
    **TRAINING_CONFIG,
    "num_epochs": 20,
    "save_best_only": False,   # Save every epoch for comparison plots
}

# ─── Class Labels ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# ─── Expected Performance Benchmarks ─────────────────────────────────────────
BENCHMARKS = {
    # Architecture → (expected_val_acc, approx_cpu_train_time_hours)
    "mobilenet_v3":   (0.86, 1.5),
    "efficientnet_b0": (0.92, 2.5),
    "resnet50":        (0.90, 4.0),
    "vit_tiny":        (0.88, 5.0),
}
