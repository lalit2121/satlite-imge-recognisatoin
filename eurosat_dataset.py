"""
EuroSAT Dataset Loader & Utilities
Compatible with: classifier.py, train.py, evaluate.py, app.py, plot_results.py
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path

# ─── Class Metadata ──────────────────────────────────────────────────────────
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

EUROSAT_COLORS = [
    "#F4A460",  # AnnualCrop — sandy brown
    "#228B22",  # Forest — forest green
    "#90EE90",  # HerbaceousVegetation — light green
    "#696969",  # Highway — dim grey
    "#FF6347",  # Industrial — tomato red
    "#98FB98",  # Pasture — pale green
    "#8B4513",  # PermanentCrop — saddle brown
    "#FFD700",  # Residential — gold
    "#1E90FF",  # River — dodger blue
    "#00CED1",  # SeaLake — dark turquoise
]

# ImageNet normalization for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Transforms ──────────────────────────────────────────────────────────────
def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_inference_transforms(img_size=224):
    """Used by app.py for single-image classification."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def denormalize(tensor):
    """Reverse ImageNet normalization for display (used by app.py & plot_results.py)."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

# ─── GeoTIFF Support ─────────────────────────────────────────────────────────
def tif_to_rgb(path):
    """Convert Sentinel-2 GeoTIFF to RGB PIL Image (used by app.py)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            if src.count >= 3:
                # Sentinel-2 RGB = B04 (Red), B03 (Green), B02 (Blue)
                rgb = src.read([4, 3, 2])
            else:
                rgb = src.read([1, 1, 1])
            rgb = np.transpose(rgb, (1, 2, 0))
            # Robust percentile normalization
            low, high = np.percentile(rgb, [2, 98])
            rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1) * 255
            return Image.fromarray(rgb.astype(np.uint8))
    except ImportError:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# ─── DataLoaders ─────────────────────────────────────────────────────────────
def get_dataloaders(
    root="./data",
    batch_size=32,
    num_workers=2,
    val_split=0.15,
    test_split=0.15,
    use_weighted_sampler=False,
    seed=42,
    img_size=224,
    download=True,
):
    """
    Create EuroSAT train/val/test DataLoaders.
    Called by train.py and evaluate.py.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Download / load dataset with training transforms
    try:
        full_dataset = datasets.EuroSAT(
            root=str(root),
            download=download,
            transform=get_train_transforms(img_size),
        )
    except Exception as e:
        print(f"[WARNING] Could not load EuroSAT: {e}")
        return None, None, None

    n_total = len(full_dataset)
    n_test  = int(n_total * test_split)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Val/Test need different transforms — random_split returns Subsets that share
    # the same parent dataset, so we wrap them to apply transforms independently.
    class TransformSubset(Subset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.indices = subset.indices
            self.transform = transform
            self.dataset = subset.dataset  # keep reference for compatibility

        def __getitem__(self, idx):
            x, y = self.subset.dataset[self.subset.indices[idx]]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset.indices)

    val_ds   = TransformSubset(val_ds,   get_val_transforms(img_size))
    test_ds  = TransformSubset(test_ds,  get_val_transforms(img_size))

    # Optional weighted sampler for class imbalance
    sampler = None
    if use_weighted_sampler:
        targets = np.array([full_dataset[i][1] for i in train_ds.indices])
        class_counts = np.bincount(targets, minlength=10)
        weights = 1.0 / class_counts[targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
