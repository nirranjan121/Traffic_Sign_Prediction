# 🌿 Branch: Logistic Regression & Weather Data Augmentation

[![Branch](https://img.shields.io/badge/Git%20Branch-technique%2Flogistic--regression-blue.svg)](https://github.com/nirranjan121/Traffic_Sign_Prediction/tree/technique/logistic-regression)
[![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green.svg)](https://scikit-learn.org/)

This branch focuses on **Logistic Regression classification** and evaluating model robustness against **weather perturbations** (fog, rain, and lighting shifts) on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## 🔬 Technique Overview

- **Classifier**: `sklearn.linear_model.LogisticRegression`
- **Feature Representation**: Flattened pixel intensity vectors (resized to standard input resolution).
- **Weather Perturbations**: Custom OpenCV transformation pipeline simulating real-world environmental road conditions:
  - **Fog simulation**: Blending image channels with gaussian noise overlays.
  - **Rain simulation**: Linear stroke overlays simulating rainfall direction vectors.
  - **Contrast & Illumination**: Dynamic alpha and beta scaling.

---

## 📁 Key Branch Files

- **`logistic_regression_experiment.ipynb`**: Notebook containing data preprocessing, weather augmentation functions, model fitting, and evaluation matrices.
- **`trafficsignrecognition.ipynb`**: Baseline experiment notebook.

---

## 🏃 Running This Branch

```bash
# 1. Switch to this branch
git checkout technique/logistic-regression

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the experiment notebook
jupyter notebook logistic_regression_experiment.ipynb
```
