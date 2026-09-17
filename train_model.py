import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------
# CONFIGURATION
# ---------------------------
# Make sure this path points to your GTSRB 'Train' folder
DATA_DIR = 'GTSRB/Train'
MODEL_FILENAME = "traffic_sign_model.joblib"

# ---------------------------
# SCRIPT START
# ---------------------------
print("Starting model training process...")

# Check if the dataset directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory not found at '{DATA_DIR}'")
    print("Please make sure the GTSRB 'Train' folder is in the correct location.")
    exit() # Exit the script if the data isn't there

# Initialize HOG descriptor and data lists
hog = cv2.HOGDescriptor()
X, y = [], []

# ---------------------------
# 1. LOAD AND PREPROCESS DATA
# ---------------------------
print("Loading and preprocessing images...")
class_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
class_folders = sorted(class_folders, key=int) # Sort folders numerically

for class_idx, class_name in enumerate(class_folders):
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        # Skip non-image files like .csv
        if img_name.endswith(('.png', '.jpg', '.jpeg', '.ppm')):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Preprocess: Resize and compute HOG features
            img_resized = cv2.resize(img, (64, 128))
            features = hog.compute(img_resized).flatten()
            X.append(features)
            y.append(class_idx)

X = np.array(X)
y = np.array(y)
print(f"Data loading complete. Total samples: {len(X)}")

# ---------------------------
# 2. TRAIN THE SVM MODEL
# ---------------------------
print("Training the SVM classifier...")
# Note: We are not splitting the data for validation, as the goal is just to produce the model file.
# We will train on all the data we loaded.
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)
print("Training complete.")

# ---------------------------
# 3. SAVE THE TRAINED MODEL
# ---------------------------
print(f"Saving the model to '{MODEL_FILENAME}'...")
joblib.dump(clf, MODEL_FILENAME)
print("Model saved successfully!")
print("\nYou can now use the 'traffic_sign_model.joblib' file in your Flask application.")