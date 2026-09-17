# 🌿 Branch: Stacking Meta-Classifier Ensemble

[![Branch](https://img.shields.io/badge/Git%20Branch-technique%2Fstacking--ensemble-blue.svg)](https://github.com/nirranjan121/Traffic_Sign_Prediction/tree/technique/stacking-ensemble)
[![Model](https://img.shields.io/badge/Model-Stacking%20Ensemble-green.svg)](https://scikit-learn.org/)

This branch implements a **Stacking Classifier (`StackingClassifier`)** meta-learning architecture combining heterogeneous base classifiers to maximize overall traffic sign recognition accuracy.

---

## 🔬 Technique Overview

- **Base Estimators**:
  1. `LogisticRegression`: Linear decision boundaries.
  2. `RandomForestClassifier`: Non-linear decision trees.
  3. `KNeighborsClassifier`: Local distance-based voting.
- **Meta-Estimator**: `LogisticRegression` / `RandomForestClassifier` trained on the predicted class probabilities of base estimators.
- **Cross-Validation**: 5-fold internal cross-validation for out-of-fold prediction generation.

---

## 📁 Key Branch Files

- **`stacking_ensemble_experiment.ipynb`**: Experiment notebook demonstrating base model instantiation, stacking pipeline assembly, performance metrics, and confusion matrix visualizations.

---

## 🏃 Running This Branch

```bash
# 1. Switch to this branch
git checkout technique/stacking-ensemble

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch experiment notebook
jupyter notebook stacking_ensemble_experiment.ipynb
```
