# 🚦 Traffic Sign Recognition & Prediction System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning system for classifying road traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset across **43 distinct sign classes**.

This repository contains multiple machine learning model implementations, feature extraction techniques (HOG, raw pixels, spatial resizing), deep learning architectures (PyTorch CNN & HOG-CNN Hybrid), and a responsive Flask web application for real-time traffic sign image inference.

---

## 📋 Table of Contents

- [✨ Project Highlights](#-project-highlights)
- [🌿 Branch Directory & Guide](#-branch-directory--guide)
- [🧠 Machine Learning & Deep Learning Techniques](#-machine-learning--deep-learning-techniques)
- [📊 Performance Comparison](#-performance-comparison)
- [📂 Dataset Organization](#-dataset-organization)
- [🚀 Quick Start & Installation](#-quick-start--installation)
- [🌐 Flask Web Application](#-flask-web-application)
- [📁 Project Structure](#-project-structure)
- [📄 License](#-license)

---

## ✨ Project Highlights

- **43 Traffic Sign Classes**: Full classification coverage for speed limits, warnings, yields, stop signs, and mandatory road directions.
- **5 Isolated Technique Branches**: Dedicated git branches demonstrating different machine learning strategies from baseline linear models to deep neural networks.
- **Advanced Feature Engineering**: Histogram of Oriented Gradients (HOG) descriptor implementation paired with classical and ensemble models.
- **Weather Data Augmentations**: Evaluates model robustness under simulated weather perturbations (fog, rain, shadow, and lighting contrast shifts).
- **Glassmorphic Flask Web UI**: Clean, interactive web interface allowing users to upload sign photos and receive instant predictions.

---

## 🌿 Branch Directory & Guide

To explore specific machine learning models and experiments, switch between the dedicated feature branches:

```text
Traffic_Sign_Prediction (git branches)
├── 📌 main                             <-- Full Baseline & Web App UI
├── 🌿 technique/logistic-regression    <-- Linear Models & Weather Data Augmentation
├── 🌿 technique/svm-hog               <-- Support Vector Machine + HOG Feature Engineering
├── 🌿 technique/knn-rf                <-- k-Nearest Neighbors & Random Forest
├── 🌿 technique/stacking-ensemble     <-- Stacking Meta-Learning Ensemble
└── 🌿 technique/cnn-pytorch           <-- PyTorch Convolutional Neural Network & Hybrid Model
```

| Branch Name | Primary Focus | Key File | Switch Command |
| :--- | :--- | :--- | :--- |
| **`main`** | Baseline & Web UI | `app.py`, `trafficsignrecognition.ipynb` | `git checkout main` |
| **`technique/logistic-regression`** | Linear Classifier + Weather Augmentations | `logistic_regression_experiment.ipynb` | `git checkout technique/logistic-regression` |
| **`technique/svm-hog`** | SVM + HOG Features | `svm_hog_experiment.ipynb` | `git checkout technique/svm-hog` |
| **`technique/knn-rf`** | k-NN & Random Forest | `knn_rf_experiment.ipynb` | `git checkout technique/knn-rf` |
| **`technique/stacking-ensemble`** | Stacking Meta-Classifier | `stacking_ensemble_experiment.ipynb` | `git checkout technique/stacking-ensemble` |
| **`technique/cnn-pytorch`** | PyTorch CNN & HOG-CNN Hybrid | `cnn_pytorch_experiment.ipynb` | `git checkout technique/cnn-pytorch` |

---

## 🧠 Machine Learning & Deep Learning Techniques

### 1. Logistic Regression & Data Augmentation (`technique/logistic-regression`)
- Fast baseline linear classification on flattened pixel intensity features.
- Evaluates model performance on weather-perturbed traffic sign images (rain, fog, contrast alterations).

### 2. Support Vector Machine with HOG (`technique/svm-hog`)
- Computes **Histogram of Oriented Gradients (HOG)** to capture edge orientations and shape features independent of illumination.
- Trains an RBF/Linear Kernel **Support Vector Machine (SVM)** to achieve high precision edge classification.

### 3. k-Nearest Neighbors & Random Forest (`technique/knn-rf`)
- **k-NN**: Non-parametric distance-based classification evaluating $k=3, 5, 7$.
- **Random Forest**: Ensemble of decision trees trained with bootstrap aggregating on extracted HOG descriptors.

### 4. Stacking Classifier (`technique/stacking-ensemble`)
- Combines base estimators (**Logistic Regression**, **Random Forest**, **k-NN**) using a meta-classifier to leverage complimentary decision boundaries.

### 5. PyTorch CNN & HOG-CNN Hybrid (`technique/cnn-pytorch`)
- Multi-layer **Convolutional Neural Network (CNN)** learning visual features directly from RGB pixels.
- **Hybrid Architecture**: Concatenates deep CNN spatial features with classical HOG gradient vectors before the final fully-connected layers.

---

## 📊 Performance Comparison

| Model Architecture | Feature Type | Model Type | Key Strength |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Raw Pixels (64x128) | Linear Model | High speed, simple baseline |
| **HOG + SVM** | HOG Descriptors | Support Vector Classifier | Robust shape & boundary recognition |
| **Random Forest** | HOG Descriptors | Decision Tree Ensemble | Handles high feature dimensionality |
| **Stacking Ensemble** | Base Model Probabilities | Meta-Learning Ensemble | Superior generalization across sign classes |
| **PyTorch HOG-CNN** | RGB + HOG Hybrid | Deep Convolutional Network | High accuracy on complex image backgrounds |

---

## 📂 Dataset Organization

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

Download and structure the dataset in the project root folder:

```text
Traffic_Sign_Prediction/
├── GTSRB/
│   ├── Train/
│   │   ├── 0/   (Class 0: Speed limit 20km/h)
│   │   ├── 1/   (Class 1: Speed limit 30km/h)
│   │   ├── ...  
│   │   └── 42/  (Class 42: End of no passing)
│   ├── Test/
│   └── Train.csv
```

---

## 🚀 Quick Start & Installation

### 1. Clone Repository
```bash
git clone https://github.com/nirranjan121/Traffic_Sign_Prediction.git
cd Traffic_Sign_Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments (Jupyter Notebook)
Launch Jupyter Notebook to view and run the experimentation notebooks:
```bash
jupyter notebook trafficsignrecognition.ipynb
```

---

## 🌐 Flask Web Application

Launch the web application to test model inference interactively:

```bash
# 1. Train and save the model
python train_model.py

# 2. Start the web server
python app.py
```

Access the interface in your browser at `http://127.0.0.1:5000/`.

---

## 📁 Project Structure

```text
Traffic_Sign_Prediction/
├── app.py                         # Flask web application server
├── prediction.py                  # Model inference engine & HOG feature extraction
├── train_model.py                 # Automated training pipeline script
├── trafficsignrecognition.ipynb   # Main end-to-end experiment notebook
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules for datasets & model weights
├── README.md                      # Comprehensive project documentation
├── static/
│   ├── css/
│   │   └── style.css              # Glassmorphic UI stylesheet
│   └── uploads/                   # Uploaded image directory
└── templates/
    ├── index.html                 # Image upload view
    └── result.html                # Prediction result view
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
