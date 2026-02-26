import os
import cv2
import numpy as np
import joblib
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------
# Load model
# -------------------------

model = joblib.load("flower_ensemble.pkl")

label_encoder = joblib.load("label.pkl")

print("Model loaded ✅")


# -------------------------
# Load MobileNetV2 (เหมือนตอน train)
# -------------------------

base_model = tf.keras.applications.MobileNetV2(

    weights='imagenet',
    include_top=False,
    pooling='avg'

)


# -------------------------
# Feature extraction function
# -------------------------

def extract_feature(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img,(224,224))

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    feature = base_model.predict(img, verbose=0)

    return feature.flatten()


# -------------------------
# Load test dataset
# -------------------------

path = "./flowers/test"

data = []
labels = []


for folder in sorted(os.listdir(path)):

    folder_path = os.path.join(path, folder)

    for img in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img)

        feature = extract_feature(img_path)

        data.append(feature)

        labels.append(folder)


data = np.array(data)

labels_encoded = label_encoder.transform(labels)


# -------------------------
# Predict
# -------------------------

pred = model.predict(data)


# -------------------------
# Accuracy
# -------------------------

acc = accuracy_score(labels_encoded, pred)

print("\nAccuracy:", acc*100, "%")


# -------------------------
# Classification Report
# -------------------------

print("\nClassification Report:\n")

print(classification_report(labels_encoded, pred, target_names=label_encoder.classes_))


# -------------------------
# Confusion Matrix
# -------------------------

print("\nConfusion Matrix:\n")

print(confusion_matrix(labels_encoded, pred))