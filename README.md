# Traffic Sign Prediction & Recognition 🚦

A comprehensive Machine Learning and Deep Learning system for classifying traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset (43 classes). Includes baseline models, feature extraction (HOG), ensemble architectures, deep learning models, and an interactive Flask web application.

---

## 🌟 Key Features

- **Multi-Model Machine Learning Experiments**:
  - **Logistic Regression**: Baseline classification and weather data augmentation evaluation.
  - **Support Vector Machine (SVM)**: Feature classification using Histogram of Oriented Gradients (HOG).
  - **k-Nearest Neighbors (k-NN)** & **Random Forest**: Supervised learning performance benchmarks.
  - **Stacking Ensemble**: Meta-classification combining Logistic Regression, Random Forest, and k-NN.
  - **PyTorch CNN & HOG-CNN Hybrid**: Deep learning architecture combining convolutional feature maps with classical HOG feature vectors.
- **Interactive Flask Web Application**: Simple, modern web interface to upload traffic sign images and view real-time class predictions.

---

## 🌿 Repository Branch Structure

This repository is organized into distinct feature branches representing different model techniques:

| Branch Name | Technique / Focus | Key Components |
| :--- | :--- | :--- |
| `main` | **Full Project Baseline & Web Application** | Clean baseline notebook, Flask app (`app.py`), model training & inference engine |
| `technique/logistic-regression` | **Logistic Regression & Weather Augmentations** | Weather effect functions (fog, rain, contrast) + linear classification |
| `technique/svm-hog` | **SVM with HOG Feature Extraction** | Histogram of Oriented Gradients descriptor + Support Vector Machine |
| `technique/knn-rf` | **k-NN & Random Forest Models** | Parameter comparison and decision tree ensemble classification |
| `technique/stacking-ensemble` | **Stacking Meta-Classifier** | Stacking classifier integrating LR, Random Forest, and k-NN base estimators |
| `technique/cnn-pytorch` | **PyTorch Deep Learning & HOG-CNN** | Multi-layer PyTorch CNN and hybrid CNN + HOG feature fusion pipeline |

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/nirranjan121/Traffic_Sign_Prediction.git
cd Traffic_Sign_Prediction
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset and extract it into a `GTSRB/` directory at the project root:

```text
Traffic_Sign_Prediction/
├── GTSRB/
│   ├── Train/
│   │   ├── 0/
│   │   ├── 1/
│   │   └── ... (0 to 42)
│   ├── Test/
│   └── Train.csv
```

### 3. Run Web Application
Train the model (optional if pre-trained joblib model exists) and launch the Flask server:

```bash
python train_model.py
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## 📊 Dataset Overview

- **Classes**: 43 distinct traffic sign categories (Speed limits, Yield, Stop, Warnings, Mandatory directions, etc.).
- **Total Images**: Over 50,000 images in varying lighting, scale, and environmental conditions.

---

## 📜 License
Distributed under the MIT License.
