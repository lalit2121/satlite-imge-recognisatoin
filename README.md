# 🛰️SkyEye:Satellite Land Cover Classification with Transfer-Learning

## Preview 
![Preview](poster_satml.png)
## Preview 
![Preview](satml.gif)



[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-green.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Satellite image classification** using transfer learning on the EuroSAT dataset.  
This project demonstrates a complete ML pipeline: data loading, training (two‑phase fine‑tuning), evaluation, and an interactive web dashboard for visualisation and inference.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **93.65%** |
| **Test Accuracy** | **93.53%** |
| **Macro F1** | 93.47% |
| **Weighted F1** | 93.51% |
| **Macro Recall** | 93.46% |
| **Training Time** (30 epochs) | ~8.1 hours (CPU) |

> The model was trained with EfficientNet‑B0 for only 5 epochs (warmup + fine‑tuning) and achieves state‑of‑the‑art performance on the EuroSAT benchmark.

---

## 🌍 What This Project Does

Satellites take millions of photos of Earth every day. But a photo alone isn't useful — we need to know **what's on the ground**.

This project trains an AI to look at a satellite image and classify the land type into one of **10 categories**:

| Class | Color | Description |
|-------|-------|-------------|
| 🌾 AnnualCrop | Sandy Brown | Wheat, corn, sugar beet fields |
| 🌲 Forest | Forest Green | Deciduous and coniferous forests |
| 🌿 HerbaceousVegetation | Light Green | Natural grasslands and meadows |
| 🛣️ Highway | Dim Grey | Roads, motorways, highways |
| 🏭 Industrial | Tomato Red | Factories, warehouses, industrial estates |
| 🐄 Pasture | Pale Green | Permanent grazing land |
| 🍇 PermanentCrop | Saddle Brown | Orchards, vineyards |
| 🏘️ Residential | Gold | Urban housing zones |
| 🌊 River | Dodger Blue | Rivers, streams, waterways |
| 🌊 SeaLake | Dark Turquoise | Lakes, coastal waters |

**Dataset:** [EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 images from ESA's Sentinel-2 satellite (64×64 pixels, RGB)

---

## 🧠 Techniques Applied 

### What is Transfer Learning?
Instead of training a neural network from scratch (which takes millions of images and days of GPU time), we start from a model that was already trained on **ImageNet** — a dataset of 1.2 million natural photos.

The key insight: the low‑level features (edges, colours, textures) learned from natural images are also useful for satellite images. We "fine‑tune" the final layers to specialise on the 10 EuroSAT classes.

> 💡 **Analogy:** It's like hiring a professional photographer and teaching them to recognize satellite landscapes instead of making them learn what a "photo" is first.

### Two‑Phase Training Strategy

| Phase | What Happens | Learning Rate | Purpose |
|-------|-------------|---------------|---------|
| **Phase 1 — Warmup** (5 epochs) | Only the new "head" (final layer) is trained. The backbone is **frozen** | 1e-4 | Let the new classification layer stabilize without disturbing pretrained features |
| **Phase 2 — Fine‑Tuning** (25 epochs) | The entire model is **unfrozen** and trained together | 1e-5 | Adapt all layers to satellite imagery at a gentle pace |

> ⚠️ **Why two phases?** If you unfreeze everything immediately with a high learning rate, you "destroy" the valuable pretrained features. It's like repainting a masterpiece — you do touch-ups, not start over.

### Discriminative Learning Rates
We use **different learning rates** for the backbone and the head:
- **Backbone:** `lr / 10` (preserves pretrained knowledge)
- **Head:** `lr` (learns faster because it's new)

### Regularisation
- **Label Smoothing** (0.1) — Softens one‑hot targets (e.g., 0.9 instead of 1.0) to prevent overconfidence and improve calibration.
- **Dropout** (0.3) — Randomly drops neurons during training to reduce overfitting.
- **Weight Decay** (1e-5) — L2 regularisation penalises large weights.
- **Early Stopping** (patience = 7) — Stops training when validation performance plateaus, preventing overfitting.

### Learning Rate Scheduling
**Cosine Annealing** — The learning rate smoothly decreases following a cosine curve, allowing faster initial learning and careful final convergence.

### Gradient Clipping (max_norm = 1.0)
Prevents "exploding gradients" — a phenomenon where weight updates become huge and break training.

### Data Augmentation
During training, images are randomly:
- Flipped horizontally
- Rotated (±15°)
- Brightness/contrast adjusted

This makes the model robust to variations in satellite angle, season, and lighting.

### Grad‑CAM (Gradient‑weighted Class Activation Mapping)
Grad‑CAM produces a heatmap over the input image, showing which spatial regions most influenced the model's decision. This helps:
- **Debug** — Does the model focus on the object or the background?
- **Build Trust** — Users can verify the model uses relevant features.
- **Scientific Insight** — Reveals which visual patterns distinguish land‑cover classes.

---

## 🏗️ Model Architecture

```
Input Image (3 × 224 × 224)
        ↓
┌─────────────────────────────────────┐
│  EfficientNet-B0 Backbone           │  ← Pretrained on ImageNet
│  (5.3M parameters)                  │     Frozen in Phase 1
│  Extracts features from images      │     Unfrozen in Phase 2
└─────────────────────────────────────┘
        ↓
Adaptive Average Pooling → (1280 features)
        ↓
Dropout(0.3)  ← Randomly zero 30% of neurons (prevents overfitting)
        ↓
Linear(1280 → 512) → BatchNorm → ReLU
        ↓
Dropout(0.21)
        ↓
Linear(512 → 10)  ← Our 10 land-cover classes
        ↓
Softmax → Class Probabilities
```

**Why EfficientNet-B0?**
- Only **5.3M parameters** (vs 25M for ResNet50)
- **~3× fewer operations** than ResNet50
- Better accuracy with faster training
- CPU-friendly: ~8.5 hours for 30 epochs

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit dashboard (5-page interactive UI)
├── classifier.py           # Model definition (SatelliteClassifier)
├── config.py               # All hyperparameters, paths, and presets
├── eurosat_dataset.py      # Dataset loading, transforms, class names, GeoTIFF support
├── train.py                # Training pipeline (two-phase transfer learning)
├── evaluate.py             # Evaluation metrics, confusion matrix, per-class F1
├── gradcam.py              # Grad-CAM implementation for attention visualisation
├── plot_results.py         # Training curves, confusion matrix, per-class bar charts
├── test_model.py           # Unit tests for model shapes, transforms, save/load
├── requirements.txt        # Python dependencies
├── setup.py                # Package installer
├── data/                   # EuroSAT dataset (auto-downloaded ~90MB)
└── results/
    ├── models/             # Saved checkpoints (.pth files)
    └── metrics/            # Training logs & evaluation JSONs
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/eurosat-classifier.git
cd eurosat-classifier
pip install -r requirements.txt
```

**Required packages:** `torch`, `torchvision`, `streamlit`, `matplotlib`, `seaborn`, `Pillow`, `numpy`, `scikit-learn`, `tqdm`

### 2. Train the Model

```bash
# Full training (30 epochs, ~2.5h on CPU)
python train.py --model efficientnet_b0 --epochs 30

# Quick test (15 epochs, faster)
python train.py --model mobilenet_v3 --epochs 15 --fast

# Custom settings
python train.py --model resnet50 --epochs 40 --batch 64 --lr 1e-4
```

**Available architectures:** `efficientnet_b0` | `resnet50` | `mobilenet_v3` | `vit_tiny`

This will:
- Download EuroSAT (~90 MB) to `data/`
- Train for the specified epochs (warmup + fine‑tune)
- Save the best checkpoint to `results/models/`
- Save training metrics to `results/metrics/`

### 3. Evaluate the Model

```bash
python evaluate.py --model_path results/models/efficientnet_b0_best_*.pth --split test
```

Outputs:
- Overall accuracy
- Confusion matrix
- Per-class precision, recall, F1
- Saved to `results/metrics/eval_results.json`

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Dashboard Pages:**
- 🔭 **Image Classifier** — Upload any satellite image (JPEG, PNG, GeoTIFF) for real-time classification with top-5 predictions
- 📊 **Model Performance** — Auto-loads training curves, confusion matrix, and per-class metrics from `results/metrics/`
- 🗂️ **Dataset Explorer** — Browse all 10 EuroSAT classes with descriptions and statistics
- 🔍 **Grad-CAM Gallery** — Upload an image to see attention heatmaps (CNN models only)
- 🧠 **Techniques** — Detailed explanations of every method used

---



### Confusion Matrix Highlights
The model performs well across all classes. Most confusion occurs between visually similar classes:
- **AnnualCrop ↔ PermanentCrop** (both are agricultural)
- **River ↔ SeaLake** (both are water bodies)
- **HerbaceousVegetation ↔ Pasture** (both are grassy)

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| AnnualCrop | ~93% | ~93% | ~93% | ~2,700 |
| Forest | ~96% | ~97% | ~96% | ~2,700 |
| HerbaceousVegetation | ~91% | ~90% | ~90% | ~2,700 |
| Highway | ~94% | ~95% | ~94% | ~2,700 |
| Industrial | ~93% | ~92% | ~92% | ~2,700 |
| Pasture | ~90% | ~91% | ~90% | ~2,700 |
| PermanentCrop | ~92% | ~93% | ~92% | ~2,700 |
| Residential | ~95% | ~94% | ~94% | ~2,700 |
| River | ~93% | ~92% | ~92% | ~2,700 |
| SeaLake | ~96% | ~97% | ~96% | ~2,700 |

> **Forest** and **SeaLake** are the easiest classes (clear visual signatures). **HerbaceousVegetation** and **Pasture** are the most challenging (visually similar).

---


## 📚 References

1. **Helber et al. (2019).** *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE J-STARS.
2. **Tan & Le (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
3. **Selvaraju et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
4. **He et al. (2016).** *Deep Residual Learning for Image Recognition.* CVPR.
5. **Howard et al. (2019).** *Searching for MobileNetV3.* ICCV.



**Happy classifying! 🚀**
