# 🌿 Branch: k-NN & Random Forest Classifiers

[![Branch](https://img.shields.io/badge/Git%20Branch-technique%2Fknn--rf-blue.svg)](https://github.com/nirranjan121/Traffic_Sign_Prediction/tree/technique/knn-rf)
[![Model](https://img.shields.io/badge/Model-k--NN%20%26%20Random%20Forest-green.svg)](https://scikit-learn.org/)

This branch explores non-parametric **k-Nearest Neighbors (k-NN)** distance classification and ensemble **Random Forest** tree models for traffic sign prediction.

---

## 🔬 Technique Overview

- **k-Nearest Neighbors (`KNeighborsClassifier`)**:
  - Distance metrics: Euclidean / Minkowski.
  - Hyperparameter evaluation across $k \in \{3, 5, 7\}$.
- **Random Forest (`RandomForestClassifier`)**:
  - Number of estimators: $n\_estimators = 100, 200$.
  - Feature inputs: Raw pixels vs extracted HOG descriptors.
  - Parallelized training utilizing multi-core execution (`n_jobs=-1`).

---

## 📁 Key Branch Files

- **`knn_rf_experiment.ipynb`**: Experiment notebook showcasing model fitting, hyperparameter comparison, classification reports, precision, recall, and F1 metrics.

---

## 🏃 Running This Branch

```bash
# 1. Switch to this branch
git checkout technique/knn-rf

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch experiment notebook
jupyter notebook knn_rf_experiment.ipynb
```
