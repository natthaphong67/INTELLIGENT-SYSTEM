import os
import cv2
import numpy as np
import joblib
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# --------------------
# Load MobileNetV2
# --------------------

base_model = tf.keras.applications.MobileNetV2(

    weights='imagenet',
    include_top=False,
    pooling='avg'

)


# --------------------
# Feature extraction
# --------------------

def extract_feature(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img,(224,224))

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    feature = base_model.predict(img, verbose=0)

    return feature.flatten()


# --------------------
# Load train
# --------------------

X_train = []
y_train = []

train_path = "flowers/train"


for folder in sorted(os.listdir(train_path)):

    for img in os.listdir(train_path+"/"+folder):

        path = train_path+"/"+folder+"/"+img

        feature = extract_feature(path)

        X_train.append(feature)

        y_train.append(folder)


# --------------------
# Load test
# --------------------

X_test = []
y_test = []

test_path = "flowers/test"


for folder in sorted(os.listdir(test_path)):

    for img in os.listdir(test_path+"/"+folder):

        path = test_path+"/"+folder+"/"+img

        feature = extract_feature(path)

        X_test.append(feature)

        y_test.append(folder)


X_train = np.array(X_train)
X_test = np.array(X_test)


# --------------------
# Encode
# --------------------

le = LabelEncoder()

y_train = le.fit_transform(y_train)

y_test = le.transform(y_test)


# --------------------
# Train Ensemble
# --------------------

model = RandomForestClassifier(

    n_estimators=500,
    random_state=42

)

model.fit(X_train,y_train)


# --------------------
# Accuracy
# --------------------

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

print("TEST Accuracy:", acc*100, "%")


# --------------------
# Save
# --------------------

joblib.dump(model,"flower_ensemble.pkl")

joblib.dump(le,"label.pkl")

print("Saved model ✅")