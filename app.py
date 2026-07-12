"""
Streamlit Dashboard — EuroSAT Satellite Image Classifier
Run: streamlit run app.py

Integrated with plot_results.py functions for automatic visualization
Auto-loads metrics from: results/metrics/*.json
Auto-loads models from:  results/models/*.pth
"""
import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image

st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️", layout="wide")

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from eurosat_dataset import (
    EUROSAT_CLASSES, EUROSAT_COLORS, get_inference_transforms, denormalize, tif_to_rgb
)
from classifier import load_model, build_model, SUPPORTED_MODELS
from config import BENCHMARKS, CLASS_NAMES

# ═════════════════════════════════════════════════════════════
# MATPLOTLIB / PLOTTING SETUP
# ═════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "navy": "#0D1B2A", "electric": "#1565C0", "sky": "#1E88E5",
    "teal": "#00ACC1", "gold": "#FFB300", "success": "#2E7D32",
    "error": "#C62828", "grey": "#546E7A",
}
CLASS_COLORS = [
    "#F4A460", "#228B22", "#90EE90", "#696969", "#FF6347",
    "#98FB98", "#8B4513", "#FFD700", "#1E90FF", "#00CED1",
]

def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
        "axes.grid": True, "grid.alpha": 0.4, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
    })

# ═════════════════════════════════════════════════════════════
# GRAD-CAM
# ═════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self._gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            img = image.detach().requires_grad_(True)
            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            confidence = probs[0, target_class].item()
            self.model.zero_grad()
            logits[0, target_class].backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")
        acts = self._activations.squeeze(0)
        grads = self._gradients.squeeze(0)
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        h, w = image.shape[2], image.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        return cam, target_class, confidence

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ═════════════════════════════════════════════════════════════
# AUTO-DISCOVERY: Metrics & Models (NO CACHE — always fresh)
# ═════════════════════════════════════════════════════════════
_METRICS_DIR = _SCRIPT_DIR / "results" / "metrics"
_MODELS_DIR = _SCRIPT_DIR / "results" / "models"

def discover_metrics_files() -> List[Path]:
    """Find all metrics JSON files in results/metrics/. Reads fresh from disk every time."""
    if not _METRICS_DIR.exists():
        return []
    files = sorted(_METRICS_DIR.glob("metrics_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Verify each file actually exists and is readable
    valid = []
    for f in files:
        if f.exists() and f.is_file():
            try:
                with open(f) as fp:
                    json.load(fp)
                valid.append(f)
            except:
                pass
    return valid

def discover_eval_files() -> List[Path]:
    """Find all evaluation JSON files in results/metrics/. Reads fresh from disk every time."""
    if not _METRICS_DIR.exists():
        return []
    files = sorted(_METRICS_DIR.glob("eval*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    valid = []
    for f in files:
        if f.exists() and f.is_file():
            try:
                with open(f) as fp:
                    json.load(fp)
                valid.append(f)
            except:
                pass
    return valid

def discover_model_files() -> List[Path]:
    """Find all .pth checkpoints in results/models/. Reads fresh from disk every time."""
    if not _MODELS_DIR.exists():
        return []
    files = sorted(_MODELS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f for f in files if f.exists() and f.is_file()]


# ═════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS (from plot_results.py)
# ═════════════════════════════════════════════════════════════
def plot_training_curves(metrics: Dict, title: str = "Training History"):
    """Generate training curves figure."""
    _set_style()
    epochs = list(range(1, len(metrics["train_losses"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=PALETTE["navy"])

    ax = axes[0, 0]
    ax.plot(epochs, metrics["train_losses"], label="Train Loss", color=PALETTE["electric"], linewidth=2)
    ax.plot(epochs, metrics["val_losses"], label="Val Loss", color=PALETTE["error"], linewidth=2, linestyle="--")
    best_ep = metrics.get("best_epoch", np.argmax(metrics["val_accs"])) + 1
    ax.axvline(best_ep, color=PALETTE["success"], linestyle="--", linewidth=1, alpha=0.7, label=f"Best (ep {best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    train_pct = [a * 100 for a in metrics["train_accs"]]
    val_pct = [a * 100 for a in metrics["val_accs"]]
    ax.plot(epochs, train_pct, label="Train Acc", color=PALETTE["electric"], linewidth=2)
    ax.plot(epochs, val_pct, label="Val Acc", color=PALETTE["teal"], linewidth=2, linestyle="--")
    ax.axhline(max(val_pct), color=PALETTE["success"], linestyle=":", linewidth=1, alpha=0.8, label=f"Best: {max(val_pct):.1f}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(epochs, metrics["learning_rates"], color=PALETTE["gold"], linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log)"); ax.set_title("Learning Rate Schedule")

    ax = axes[1, 1]
    times = [t / 60 for t in metrics.get("epoch_times", [0] * len(epochs))]
    ax.bar(epochs, times, color=PALETTE["sky"], alpha=0.8, width=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Minutes"); ax.set_title("Epoch Time")

    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix, class_names: List[str], normalise: bool = True,
                          title: str = "Confusion Matrix"):
    """Generate confusion matrix heatmap figure."""
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    _set_style()
    cm = np.array(confusion_matrix)
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot, fmt, vmax = cm, "d", cm.max()

    fig, ax = plt.subplots(figsize=(12, 10))
    if sns:
        sns.heatmap(cm_plot, annot=True, fmt=fmt, xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", vmin=0, vmax=vmax, linewidths=0.5, ax=ax)
    else:
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm_plot[i, j]:{fmt}}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_per_class_metrics(per_class_metrics: Dict, class_names: List[str]):
    """Generate per-class metrics bar chart figure."""
    _set_style()
    precision = [per_class_metrics[c]["precision"] * 100 for c in class_names]
    recall = [per_class_metrics[c]["recall"] * 100 for c in class_names]
    f1 = [per_class_metrics[c]["f1"] * 100 for c in class_names]
    x = np.arange(len(class_names)); w = 0.28
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, precision, w, label="Precision", color=PALETTE["electric"], alpha=0.85)
    ax.bar(x, recall, w, label="Recall", color=PALETTE["teal"], alpha=0.85)
    ax.bar(x + w, f1, w, label="F1 Score", color=PALETTE["gold"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Per-Class Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115); ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_class_distribution(class_counts: Dict):
    """Generate class distribution bar chart figure."""
    _set_style()
    classes = list(class_counts.keys()); counts = list(class_counts.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(classes, counts, color=CLASS_COLORS, edgecolor="white")
    ax.set_xlabel("Number of Images")
    ax.set_title("EuroSAT Test Set — Class Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════
# STREAMLIT LAYOUT & CSS
# ═════════════════════════════════════════════════════════════
st.markdown("""<style>
.big-header{font-size:2.2rem;font-weight:800;color:#0D1B2A;border-bottom:3px solid #1565C0;padding-bottom:.4rem;margin-bottom:1rem}
.pred-box{border:2px solid #1565C0;border-radius:10px;padding:1.2rem;background:linear-gradient(135deg,#F0F4FA,#ffffff)}
.metric-card{background:#ffffff;border-radius:8px;padding:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);border-left:4px solid #1565C0}
.demo-badge{display:inline-block;background:#e3f2fd;color:#1565C0;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;margin-left:6px}
.run-card{background:#ffffff;border-radius:8px;padding:1rem;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,0.05);border-left:4px solid #2E7D32}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛰️ EuroSAT Classifier")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🔭 Image Classifier", "📊 Model Performance",
    "🗂️ Dataset Explorer", "🔍 Grad-CAM Gallery", "🧠 Techniques", "ℹ️ About"
], label_visibility="collapsed")

# ─── Architecture ───────────────────────────────────────────
model_choice = st.sidebar.selectbox(
    "🏗️ Architecture", SUPPORTED_MODELS, index=SUPPORTED_MODELS.index("efficientnet_b0")
)

# ─── Auto-discover checkpoints ──────────────────────────────
_discovered_models = discover_model_files()
_DEFAULT_MODEL = _discovered_models[0] if _discovered_models else (_MODELS_DIR / "best_model.pth")

model_path = st.sidebar.text_input("Checkpoint (.pth)", str(_DEFAULT_MODEL))
device_str = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Device: {device_str}")

# ─── Show discovered files in sidebar (ALWAYS FRESH) ──────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Discovered Files")

# Force fresh scan every time
metrics_files_now = discover_metrics_files()
eval_files_now = discover_eval_files()
model_files_now = discover_model_files()

st.sidebar.markdown(f"**Metrics:** {len(metrics_files_now)} file(s)")
for p in metrics_files_now[:5]:
    st.sidebar.markdown(f"<span style='font-size:0.7rem;color:#666'>• {p.name}</span>", unsafe_allow_html=True)

st.sidebar.markdown(f"**Evaluations:** {len(eval_files_now)} file(s)")
for p in eval_files_now[:5]:
    st.sidebar.markdown(f"<span style='font-size:0.7rem;color:#666'>• {p.name}</span>", unsafe_allow_html=True)

st.sidebar.markdown(f"**Models:** {len(model_files_now)} file(s)")
for p in model_files_now[:5]:
    st.sidebar.markdown(f"<span style='font-size:0.7rem;color:#666'>• {p.name}</span>", unsafe_allow_html=True)

if not metrics_files_now and not eval_files_now:
    st.sidebar.warning("No metrics/eval files found in results/metrics/")


@st.cache_resource(show_spinner=False)
def get_model(path, arch):
    dev = torch.device(device_str)
    if os.path.exists(path):
        try:
            m, _ = load_model(path, device=str(dev))
            if m.model_name != arch:
                st.sidebar.warning(f"Checkpoint is `{m.model_name}`, dropdown is `{arch}`. Using checkpoint.")
            return m, dev, f"Loaded: {Path(path).name}"
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")
    m = build_model(arch, num_classes=10, pretrained=True, device=str(dev))
    return m, dev, "Demo mode (untrained — random predictions)"


def classify(model, device, pil_img):
    t = get_inference_transforms()(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top5 = probs.argsort()[::-1][:5]
        return {
            "top_class": EUROSAT_CLASSES[top5[0]],
            "top_prob": float(probs[top5[0]]),
            "top5_cls": [EUROSAT_CLASSES[i] for i in top5],
            "top5_prob": probs[top5].tolist(),
            "all_probs": probs.tolist(),
            "tensor": t,
        }


# ═════════════════════════════════════════════════════════════
# PAGE 1 — Image Classifier
# ═════════════════════════════════════════════════════════════
if page == "🔭 Image Classifier":
    st.markdown('<div class="big-header">🔭 Satellite Image Classifier</div>', unsafe_allow_html=True)
    model, device, status = get_model(model_path, model_choice)
    st.sidebar.success(status)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Upload Image")
        f = st.file_uploader("Satellite image (jpg/png/tif)", type=["jpg", "jpeg", "png", "tif"])
        conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

        if f:
            if f.name.lower().endswith(".tif"):
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                img = tif_to_rgb(tmp_path)
                os.unlink(tmp_path)
            else:
                img = Image.open(f).convert("RGB")

            with st.spinner("Classifying..."):
                res = classify(model, device, img)

            badge_html = ""
            if "Demo mode" in status:
                badge_html += '<span class="demo-badge">DEMO</span>'

            st.markdown(
                f"""<div class="pred-box">
                       <div style="font-size:1.6rem;font-weight:700;color:#1565C0">{res["top_class"]}</div>
                       <div style="margin-top:4px">Confidence: <strong>{res["top_prob"]*100:.1f}%</strong>
                       {"⚠️ Uncertain" if res["top_prob"] < conf_thresh else "✅"}
                       {badge_html}</div>
                     </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("**Top-5 Predictions:**")
            for c, p in zip(res["top5_cls"], res["top5_prob"]):
                color = EUROSAT_COLORS[EUROSAT_CLASSES.index(c)]
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
                    f"<div style='width:12px;height:12px;border-radius:50%;background:{color}'></div>"
                    f"<div style='flex:1'>{c}</div>"
                    f"<div style='font-weight:600'>{p*100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
                st.progress(float(p), text="")
        else:
            st.info("👆 Upload a satellite image to classify it.")
            st.markdown("**Supported classes:**  " + ", ".join(
                f"<span style='color:{EUROSAT_COLORS[i]}'>●</span> {c}"
                for i, c in enumerate(EUROSAT_CLASSES)
            ), unsafe_allow_html=True)

    with col_right:
        st.subheader("Image Preview")
        if f:
            st.image(img, use_container_width=True)
            st.caption("Class Color Legend")
            legend_cols = st.columns(5)
            for i, (cls, col) in enumerate(zip(EUROSAT_CLASSES, EUROSAT_COLORS)):
                with legend_cols[i % 5]:
                    st.markdown(f"<div style='display:flex;align-items:center;gap:6px'>"
                                f"<div style='width:10px;height:10px;border-radius:2px;background:{col}'></div>"
                                f"<span style='font-size:0.75rem'>{cls}</span></div>",
                                unsafe_allow_html=True)
        else:
            st.info("Your uploaded image will appear here.")


# ═════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance (AUTO-LOAD + PLOTS)
# ═════════════════════════════════════════════════════════════
if page == "📊 Model Performance":
    st.markdown('<div class="big-header">📊 Model Performance</div>', unsafe_allow_html=True)

    # ─── ALWAYS FRESH scan (no caching) ───────────────────────
    metrics_files = discover_metrics_files()
    eval_files = discover_eval_files()

    if metrics_files:
        st.subheader("📁 Discovered Training Runs")
        st.markdown(f"Found **{len(metrics_files)}** metrics file(s) in `results/metrics/`")

        runs_data = []
        for p in metrics_files:
            try:
                with open(p) as f:
                    data = json.load(f)
                epochs = len(data.get("train_losses", []))
                best_acc = data.get("best_val_acc", 0) * 100
                best_ep = data.get("best_epoch", 0) + 1
                total_time = sum(data.get("epoch_times", [0])) / 3600
                runs_data.append({
                    "file": p.name,
                    "path": p,
                    "epochs": epochs,
                    "best_val_acc": best_acc,
                    "best_epoch": best_ep,
                    "total_time_h": total_time,
                    "data": data,
                })
            except Exception as e:
                st.warning(f"Could not read {p.name}: {e}")
                continue

        if not runs_data:
            st.error("No valid metrics files could be loaded.")
            st.stop()

        # Sort by best validation accuracy
        runs_data.sort(key=lambda x: x["best_val_acc"], reverse=True)

        # Best run highlight
        best = runs_data[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Best Val Acc", f"{best['best_val_acc']:.2f}%")
        c2.metric("Best Epoch", f"{best['best_epoch']}")
        c3.metric("Total Epochs", best["epochs"])
        c4.metric("Train Time", f"{best['total_time_h']:.1f}h")

        # Run selector
        st.markdown("---")
        selected_run_name = st.selectbox(
            "Select training run to visualize",
            options=[r["file"] for r in runs_data],
            index=0,
        )
        selected_run = next(r for r in runs_data if r["file"] == selected_run_name)
        m = selected_run["data"]

        # ─── Training Curves (matplotlib) ─────────────────────
        st.subheader("📈 Training Curves")
        fig = plot_training_curves(m, title=f"Training History — {selected_run_name.replace('.json', '')}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Val Acc", f"{max(m['val_accs'])*100:.2f}%")
        c2.metric("Best Train Acc", f"{max(m['train_accs'])*100:.2f}%")
        c3.metric("Total Epochs", len(m["train_accs"]))
        total_time = sum(m.get("epoch_times", [0]))
        c4.metric("Training Time", f"{total_time/3600:.1f}h" if total_time > 3600 else f"{total_time/60:.1f}min")

    # ─── Evaluation Results ───────────────────────────────────
    if eval_files:
        st.markdown("---")
        st.subheader("📊 Evaluation Results")

        selected_eval_name = st.selectbox(
            "Select evaluation file",
            options=[p.name for p in eval_files],
            index=0,
        )
        selected_eval_path = next(p for p in eval_files if p.name == selected_eval_name)

        try:
            with open(selected_eval_path) as f:
                r = json.load(f)
        except Exception as e:
            st.error(f"Could not read {selected_eval_name}: {e}")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{r['overall_accuracy']*100:.2f}%")
        c2.metric("Macro F1", f"{r['macro_f1']*100:.2f}%")
        c3.metric("Weighted F1", f"{r['weighted_f1']*100:.2f}%")
        c4.metric("Macro Recall", f"{r['macro_recall']*100:.2f}%")

        # Confusion Matrix
        if "confusion_matrix" in r:
            st.subheader("🔥 Confusion Matrix")
            fig = plot_confusion_matrix(r["confusion_matrix"], EUROSAT_CLASSES, normalise=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Per-Class Metrics
        if "per_class_metrics" in r:
            st.subheader("📊 Per-Class Metrics")
            fig = plot_per_class_metrics(r["per_class_metrics"], EUROSAT_CLASSES)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            import pandas as pd
            rows = [
                {
                    "Class": c,
                    "Precision": f"{v['precision']*100:.1f}%",
                    "Recall": f"{v['recall']*100:.1f}%",
                    "F1": f"{v['f1']*100:.1f}%",
                    "Support": v["support"],
                }
                for c, v in r["per_class_metrics"].items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Class Distribution
        if "per_class_metrics" in r:
            st.subheader("📊 Class Distribution")
            class_counts = {c: r["per_class_metrics"][c]["support"] for c in EUROSAT_CLASSES}
            fig = plot_class_distribution(class_counts)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if not metrics_files and not eval_files:
        st.warning("""
        No metrics or evaluation files found in `results/metrics/`.

        Expected paths:
        - `results/metrics/metrics_*.json` (from train.py)
        - `results/metrics/eval*.json` (from evaluate.py)

        Run training first:
        ```bash
        python train.py --model efficientnet_b0 --epochs 30
        python evaluate.py
        ```
        """)


# ═════════════════════════════════════════════════════════════
# PAGE 3 — Dataset Explorer
# ═════════════════════════════════════════════════════════════
if page == "🗂️ Dataset Explorer":
    st.markdown('<div class="big-header">🗂️ EuroSAT Dataset Explorer</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", "27,000")
    c2.metric("Classes", "10")
    c3.metric("Image Size", "64 × 64 px")
    c4.metric("Source", "Sentinel-2")
    st.markdown("---")

    class_info = {
        "AnnualCrop": "Fields with annual crops (wheat, corn, sugar beet).",
        "Forest": "Deciduous and coniferous forests.",
        "HerbaceousVegetation": "Natural grasslands and meadows.",
        "Highway": "Roads, motorways, and highway infrastructure.",
        "Industrial": "Factories, warehouses, and industrial estates.",
        "Pasture": "Permanent grazing land for livestock.",
        "PermanentCrop": "Orchards, vineyards, and permanent crop fields.",
        "Residential": "Urban housing and residential zones.",
        "River": "Rivers, streams, and waterways.",
        "SeaLake": "Coastal areas, lakes, and open water.",
    }

    for i, (cls, desc) in enumerate(class_info.items()):
        col = EUROSAT_COLORS[i]
        st.markdown(
            f'<div style="border-left:4px solid {col};padding:8px 14px;margin:6px 0;background:#ffffff;border-radius:0 6px 6px 0;box-shadow:0 1px 3px rgba(0,0,0,0.05)">'
            f'<strong style="color:#0D1B2A">{cls}</strong> — <span style="color:#546E7A">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Download")
    st.code('import torchvision.datasets as ds\nds.EuroSAT(root="./data", download=True)', language="python")


# ═════════════════════════════════════════════════════════════
# PAGE 4 — Grad-CAM Gallery
# ═════════════════════════════════════════════════════════════
if page == "🔍 Grad-CAM Gallery":
    st.markdown('<div class="big-header">🔍 Grad-CAM Attention Visualisation</div>', unsafe_allow_html=True)
    st.markdown("Grad-CAM shows where the model looks when classifying. **Note:** Grad-CAM requires convolutional layers; ViT uses attention instead.")

    model, device, status = get_model(model_path, model_choice)
    f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "tif"])

    if f:
        if f.name.lower().endswith(".tif"):
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            pil = tif_to_rgb(tmp_path)
            os.unlink(tmp_path)
        else:
            pil = Image.open(f).convert("RGB")
        tensor = get_inference_transforms()(pil).unsqueeze(0).to(device)

        col1, col2 = st.columns([1, 3])
        with col1:
            colormap = st.selectbox("Colormap", ["jet", "hot", "plasma", "viridis", "turbo"])
            alpha = st.slider("Opacity", 0.0, 1.0, 0.5)
            st.markdown("---")
            st.markdown("**Model Status:**")
            st.markdown(f"<span class='demo-badge'>{status}</span>", unsafe_allow_html=True)

        with col2:
            if model_choice == "vit_tiny":
                st.warning("⚠️ **ViT-Tiny** uses self-attention, not convolutions. Traditional Grad-CAM does not apply. Use attention-rollout instead.")
                st.image(pil, caption="Original Image", use_container_width=True)
            else:
                try:
                    target_layer = model.get_last_conv_layer()
                    gc = GradCAM(model, target_layer)
                    hm, pc, conf = gc(tensor)
                    gc.remove_hooks()

                    img_d = np.clip(
                        denormalize(tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy(), 0, 1
                    )
                    import matplotlib.cm as mplcm
                    cmap = mplcm.get_cmap(colormap)
                    hm_rgb = cmap(hm)[:, :, :3]
                    overlay = np.clip((1 - alpha) * img_d + alpha * hm_rgb, 0, 1)

                    c1, c2, c3 = st.columns(3)
                    c1.image(img_d, caption="Original", use_container_width=True)
                    c2.image(hm, caption="Heatmap", use_container_width=True, clamp=True)
                    c3.image(overlay, caption="Overlay", use_container_width=True)

                    color = EUROSAT_COLORS[pc]
                    st.markdown(
                        f"""<div style="margin-top:10px">
                            Predicted: <strong style="color:{color}">{EUROSAT_CLASSES[pc]}</strong>
                            <span style="font-size:1.2rem">({conf*100:.1f}%)</span>
                        </div>""", unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")
    else:
        st.info("Upload an image to see Grad-CAM. For best results, use a 64×64 EuroSAT sample.")


# ═════════════════════════════════════════════════════════════
# PAGE 5 — Techniques
# ═════════════════════════════════════════════════════════════
if page == "🧠 Techniques":
    st.markdown('<div class="big-header">🧠 Core Techniques & Architecture Details</div>', unsafe_allow_html=True)

    st.subheader(f"🏗️ Selected Architecture: {model_choice.replace('_', ' ').title()}")
    expected_acc, train_time = BENCHMARKS.get(model_choice, (0.0, 0.0))
    st.markdown(f"**Expected Val Accuracy:** ~{expected_acc*100:.0f}% | **Approx. Train Time:** ~{train_time}h (CPU)")

    if model_choice == "efficientnet_b0":
        st.markdown("""
        **EfficientNet-B0** uses compound scaling to balance depth, width, and resolution.
        Only **5.3M parameters** (vs 25M for ResNet50) with better ImageNet accuracy.
        """)
    elif model_choice == "resnet50":
        st.markdown("""
        **ResNet50** is a 25.6M parameter residual network. Skip connections allow gradients to flow through deep stacks.
        """)
    elif model_choice == "mobilenet_v3":
        st.markdown("""
        **MobileNetV3-Small** is optimized for edge devices with only **2.5M parameters**.
        Depthwise separable convolutions + squeeze-and-excitation blocks make it the fastest option.
        """)
    elif model_choice == "vit_tiny":
        st.markdown("""
        **ViT-Tiny** (Vision Transformer) uses self-attention patches instead of convolutions.
        Requires the `timm` library. Falls back to EfficientNet-B0 if unavailable.
        **Note:** Grad-CAM does not work on ViT; use attention-rollout instead.
        """)

    st.markdown("---")
    st.subheader("📈 Two-Phase Training Strategy")
    st.markdown("""
    | Phase | Epochs | Backbone | Learning Rate | Purpose |
    |-------|--------|----------|---------------|---------|
    | 1 — Warmup | 5 | ❄️ Frozen | 1e-4 | Stabilize new classification head |
    | 2 — Fine-tune | 25 | 🔥 Unfrozen | 1e-5 | Adapt pretrained features to satellite |

    **Discriminative Learning Rates:** backbone uses LR÷10, head uses full LR. This preserves low-level features while allowing the head to adapt quickly.
    """)


# ═════════════════════════════════════════════════════════════
# PAGE 6 — About
# ═════════════════════════════════════════════════════════════
if page == "ℹ️ About":
    st.markdown('<div class="big-header">ℹ️ About</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
**EuroSAT Land Use Classification**
MSc Space Engineering — ML Portfolio Project

Transfer learning on Sentinel-2 satellite imagery using PyTorch & torchvision.

| Architecture | Params | Expected Val Acc | CPU Train Time |
|-------------|--------|------------------|----------------|
| MobileNetV3 | 2.5M | ~86% | ~1.5h |
| EfficientNet-B0 | 5.3M | ~92% | ~2.5h |
| ResNet50 | 25.6M | ~90% | ~4h |
| ViT-Tiny | 5.7M | ~88% | ~5h |

**References:**
* Helber et al. 2019. EuroSAT dataset. IEEE J-STARS.
* Tan & Le. 2019. EfficientNet. ICML.
* Selvaraju et al. 2017. Grad-CAM. ICCV.
""")
    with col2:
        st.code("""
# Quick Start
pip install -r requirements.txt
python train.py --model efficientnet_b0 --epochs 30
python evaluate.py
streamlit run app.py
        """, language="bash")