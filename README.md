# 🌿 Branch: Support Vector Machine (SVM) with HOG Features

[![Branch](https://img.shields.io/badge/Git%20Branch-technique%2Fsvm--hog-blue.svg)](https://github.com/nirranjan121/Traffic_Sign_Prediction/tree/technique/svm-hog)
[![Model](https://img.shields.io/badge/Model-SVM%20%2B%20HOG-green.svg)](https://scikit-learn.org/)

This branch implements **Histogram of Oriented Gradients (HOG)** feature extraction coupled with a **Support Vector Machine (SVM)** classifier for robust traffic sign classification.

---

## 🔬 Technique Overview

- **Feature Descriptor**: `cv2.HOGDescriptor` / `skimage.feature.hog`
  - **Orientations**: 9 gradient bins
  - **Pixels per Cell**: (8, 8)
  - **Cells per Block**: (2, 2)
  - **Normalization**: L2-Hys block normalization
- **Classifier**: `sklearn.svm.SVC` (Support Vector Classifier)
- **Advantages**: Captures geometric structure and sign boundaries invariant to lighting variations.

---

## 📁 Key Branch Files

- **`svm_hog_experiment.ipynb`**: Experiment notebook demonstrating HOG feature computation, SVM training, classification metrics, and confusion matrix visualizations.
- **`prediction.py`**: Production inference engine utilizing the HOG + SVM pipeline.
- **`train_model.py`**: Standalone model training script generating `traffic_sign_model.joblib`.

---

## 🏃 Running This Branch

```bash
# 1. Switch to this branch
git checkout technique/svm-hog

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch experiment notebook
jupyter notebook svm_hog_experiment.ipynb
```
