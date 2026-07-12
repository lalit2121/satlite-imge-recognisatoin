"""
===========================================================================
  EVALUATION MODULE
  Accuracy · Confusion Matrix · Precision / Recall / F1 · Per-Class Stats
===========================================================================

WHY ACCURACY ALONE IS NOT ENOUGH
----------------------------------
Overall accuracy = correct / total

But if 90% of images are "Forest", a model that always predicts
"Forest" achieves 90% accuracy without learning anything!

Better metrics:
  Precision: of all times I predicted class X, how often was I right?
             Precision = TP / (TP + FP)

  Recall:    of all actual class X images, how many did I find?
             Recall = TP / (TP + FN)

  F1 Score:  harmonic mean of precision and recall
             F1 = 2 × (Precision × Recall) / (Precision + Recall)

  Confusion matrix: N×N grid showing what the model predicted vs truth.
             Cell (i,j) = how many class-i images were predicted as class-j.
             Perfect model → diagonal matrix.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ─── Anchor to this script's folder (flat structure) ─────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from eurosat_dataset import EUROSAT_CLASSES
from classifier import SatelliteClassifier, load_model


# ─── Full evaluation pipeline ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(
    model: SatelliteClassifier,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Complete model evaluation on a dataset split.

    Computes:
      - Overall accuracy
      - Per-class precision, recall, F1
      - Confusion matrix
      - Top-5 accuracy (useful for verification)
      - Predictions and true labels (for visualisation)

    Args:
        model:       Trained SatelliteClassifier
        loader:      DataLoader (val or test split)
        device:      torch.device
        class_names: List of class name strings
        verbose:     Print results table

    Returns:
        dict with all metrics and raw predictions
    """
    if class_names is None:
        class_names = EUROSAT_CLASSES

    model.eval()
    num_classes = len(class_names)

    all_preds   = []
    all_labels  = []
    all_probs   = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    # ── Overall accuracy ─────────────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean()

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support   = cm[c, :].sum()

        per_class[class_names[c]] = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   int(support),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }

    # ── Macro & weighted averages ─────────────────────────────────────────────
    macro_precision = np.mean([per_class[c]["precision"] for c in class_names])
    macro_recall    = np.mean([per_class[c]["recall"]    for c in class_names])
    macro_f1        = np.mean([per_class[c]["f1"]        for c in class_names])

    supports = np.array([per_class[c]["support"] for c in class_names])
    total_support = supports.sum()
    weights = supports / total_support

    weighted_f1        = np.sum([per_class[c]["f1"]        * w for c, w in zip(class_names, weights)])
    weighted_precision = np.sum([per_class[c]["precision"] * w for c, w in zip(class_names, weights)])
    weighted_recall    = np.sum([per_class[c]["recall"]    * w for c, w in zip(class_names, weights)])

    # ── Print results table ───────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 75)
        print(f"  EVALUATION RESULTS")
        print("=" * 75)
        print(f"  Overall Accuracy:  {overall_acc*100:.2f}%")
        print(f"  Macro F1:          {macro_f1*100:.2f}%")
        print(f"  Weighted F1:       {weighted_f1*100:.2f}%")
        print("-" * 75)
        print(f"  {'Class':<<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 75)
        for cls in class_names:
            m = per_class[cls]
            print(
                f"  {cls:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['support']:>10}"
            )
        print("-" * 75)
        print(
            f"  {'Macro avg':<<25} {macro_precision:>10.4f} {macro_recall:>10.4f} "
            f"{macro_f1:>10.4f} {total_support:>10}"
        )
        print(
            f"  {'Weighted avg':<<25} {weighted_precision:>10.4f} {weighted_recall:>10.4f} "
            f"{weighted_f1:>10.4f} {total_support:>10}"
        )
        print("=" * 75)

    results = {
        "overall_accuracy":    float(overall_acc),
        "macro_precision":     float(macro_precision),
        "macro_recall":        float(macro_recall),
        "macro_f1":            float(macro_f1),
        "weighted_f1":         float(weighted_f1),
        "weighted_precision":  float(weighted_precision),
        "weighted_recall":     float(weighted_recall),
        "per_class_metrics":   per_class,
        "confusion_matrix":    cm.tolist(),
        "class_names":         class_names,
        "all_predictions":     all_preds.tolist(),
        "all_true_labels":     all_labels.tolist(),
        "all_probabilities":   all_probs.tolist(),
        "num_samples":         int(total_support),
    }

    return results


def save_evaluation_results(results: Dict, save_path: str):
    """Save evaluation results to JSON (excluding large arrays for readability)."""
    # Only save the summary metrics, not the full prediction arrays
    summary = {k: v for k, v in results.items()
                if k not in ("all_predictions", "all_true_labels", "all_probabilities")}
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation results saved to: {save_path}")


def get_misclassified_examples(
    model: SatelliteClassifier,
    loader: DataLoader,
    device: torch.device,
    n_examples: int = 16,
) -> List[Dict]:
    """
    Find examples where the model made mistakes.

    Useful for error analysis — understanding WHAT the model gets wrong
    often reveals how to improve it.

    Returns:
        List of dicts: {image, true_label, predicted_label, confidence}
    """
    model.eval()
    mistakes = []

    with torch.no_grad():
        for images, labels in loader:
            if len(mistakes) >= n_examples:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            wrong = preds != labels
            for i in wrong.nonzero(as_tuple=True)[0]:
                mistakes.append({
                    "image":      images[i].cpu(),
                    "true_label": labels[i].item(),
                    "pred_label": preds[i].item(),
                    "confidence": probs[i, preds[i]].item(),
                    "true_prob":  probs[i, labels[i]].item(),
                })
                if len(mistakes) >= n_examples:
                    break

    return mistakes


def _find_latest_checkpoint(models_dir: Path) -> Optional[str]:
    """
    Scan the models directory for .pth files and return the most recent one.
    Falls back to None if no checkpoints are found.
    """
    if not models_dir.exists():
        return None
    checkpoints = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoints[0]) if checkpoints else None


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from eurosat_dataset import get_dataloaders

    _RESULTS_DIR = _SCRIPT_DIR / "results"
    _MODELS_DIR  = _RESULTS_DIR / "models"

    # Auto-detect latest checkpoint if best_model.pth doesn't exist
    default_model = _MODELS_DIR / "best_model.pth"
    if not default_model.exists():
        detected = _find_latest_checkpoint(_MODELS_DIR)
        if detected:
            default_model = Path(detected)
        else:
            default_model = _MODELS_DIR / "best_model.pth"  # keep as fallback

    parser = argparse.ArgumentParser(description="Evaluate trained EuroSAT model")
    parser.add_argument(
        "--model_path",
        default=str(default_model),
        help="Path to .pth checkpoint (auto-detects latest if not specified)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--save",
        default=str(_RESULTS_DIR / "metrics" / "eval_results.json"),
        help="Path to save evaluation JSON"
    )
    args = parser.parse_args()

    # Verify the checkpoint exists
    if not Path(args.model_path).exists():
        print(f"ERROR: Checkpoint not found: {args.model_path}")
        print(f"Checked in: {_MODELS_DIR}")
        print("Run training first: python train.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {args.model_path}")
    model, _ = load_model(args.model_path, device=str(device))

    _, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch, num_workers=2, download=False
    )
    loader = test_loader if args.split == "test" else val_loader

    results = evaluate_model(model, loader, device, verbose=True)
    save_evaluation_results(results, args.save)