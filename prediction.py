import cv2
import numpy as np
import joblib

# Load the trained model
MODEL_PATH = "traffic_sign_model.joblib"
model = joblib.load(MODEL_PATH)

# Initialize HOG Descriptor
hog = cv2.HOGDescriptor()

# Define the class names in the correct order
CLASS_NAMES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
    'No passing', 'No passing for vehicles over 3.5 tons', 
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 
    'No vehicles', 'Vehicles over 3.5 tons prohibited', 'No entry', 'General caution', 
    'Dangerous curve to the left', 'Dangerous curve to the right', 
    'Double curve, first to the left', 'Bumpy road', 'Slippery road', 
    'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 
    'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 
    'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 
    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 
    'End of no passing by vehicles over 3.5 tons'
]

def predict_image(image_path):
    """
    This function takes an image path, preprocesses the image,
    and returns the predicted traffic sign.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Error: Could not read image."

    img_resized = cv2.resize(img, (64, 128))
    features = hog.compute(img_resized).flatten()
    prediction_index = model.predict([features])[0]
    
    return CLASS_NAMES[prediction_index]