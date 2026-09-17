# 🌿 Branch: PyTorch Convolutional Neural Network (CNN) & HOG-CNN Hybrid

[![Branch](https://img.shields.io/badge/Git%20Branch-technique%2Fcnn--pytorch-blue.svg)](https://github.com/nirranjan121/Traffic_Sign_Prediction/tree/technique/cnn-pytorch)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)

This branch implements **PyTorch Deep Learning models**, featuring a multi-layer Convolutional Neural Network (CNN) and a **Hybrid HOG-CNN Fusion Network** for 43-class traffic sign classification.

---

## 🔬 Architecture Overview

### 1. PyTorch Baseline CNN (`Conv2d` + `MaxPool2d` + `ReLU`)
- **Conv Layer 1**: `3 -> 32` channels, kernel $3 \times 3$, padding 1 + MaxPool $2 \times 2$.
- **Conv Layer 2**: `32 -> 64` channels, kernel $3 \times 3$, padding 1 + MaxPool $2 \times 2$.
- **Dense Layers**: Linear classifier mapping flattened CNN spatial features to 43 output classes.

### 2. HOG-CNN Feature Fusion Network (`CNN_HOG_Model`)
- Extracts spatial deep learning features via CNN channels.
- Computes classical Histogram of Oriented Gradients (HOG) descriptor vector on the grayscale image input.
- **Fusion Layer**: Concatenates `[CNN_spatial_features, HOG_vector]` before dense classification layers (`Linear(cnn_dim + hog_dim, 256) -> Linear(256, 43)`).

---

## 📁 Key Branch Files

- **`cnn_pytorch_experiment.ipynb`**: Experiment notebook demonstrating custom `PyTorch Dataset`, `DataLoader`, GPU/CPU device assignment, loss computation (`CrossEntropyLoss`), Adam optimizer, and training loop.

---

## 🏃 Running This Branch

```bash
# 1. Switch to this branch
git checkout technique/cnn-pytorch

# 2. Install dependencies (including torch, torchvision)
pip install -r requirements.txt

# 3. Launch experiment notebook
jupyter notebook cnn_pytorch_experiment.ipynb
```
