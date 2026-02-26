import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf


# ====================================
# Load Models
# ====================================

# Neural Network
flower_nn = tf.keras.models.load_model("flower_nn.keras")

# Ensemble Model
flower_ensemble = joblib.load("flower_ensemble.pkl")

# Label encoder
label_encoder = joblib.load("label.pkl")

class_names = label_encoder.classes_


# MobileNetV2 for feature extraction (ใช้กับ Ensemble)
feature_model = tf.keras.applications.MobileNetV2(

    weights="imagenet",

    include_top=False,

    pooling="avg"

)


# ====================================
# Sidebar Menu
# ====================================

st.sidebar.title("🌸 Flower Classification App")

page = st.sidebar.radio(

    "Menu",

    (

        "Neural Network Info",

        "Ensemble Model Info",

        "Test Neural Network",

        "Test Ensemble Model"

    )

)


# ====================================
# Page 1: Neural Network Info
# ====================================

if page == "Neural Network Info":

    st.title("🧠 Neural Network Model")

    st.write("""

### Dataset

Flower dataset from Kaggle

5 classes:

- daisy

- dandelion

- rose

- sunflower

- tulip


---

### Data Preparation

- Resize image to 224x224

- Normalize image

- Data augmentation


---

### Algorithm

MobileNetV2

Convolutional Neural Network (CNN)

ใช้ Transfer Learning


---

### Development Steps

1. Load Dataset

2. Preprocess

3. Train Model

4. Fine Tuning

5. Save Model


---

### Accuracy

ประมาณ 85% - 95%


---

### Reference

https://www.kaggle.com

""")


# ====================================
# Page 2: Ensemble Info
# ====================================

elif page == "Ensemble Model Info":

    st.title("🌲 Ensemble Model")

    st.write("""

### Dataset

Flower Dataset


---

### Data Preparation

- Resize image

- Extract feature using MobileNetV2


---

### Algorithm

Machine Learning:

- Random Forest

- Voting


---

### Ensemble Concept

Combine multiple model

เพื่อลด error


---

### Development Steps

1. Extract Feature

2. Train Model

3. Save Model


---

### Accuracy

ประมาณ 85% - 95%


---

### Reference

https://www.kaggle.com

""")


# ====================================
# Page 3: Test Neural Network
# ====================================

elif page == "Test Neural Network":

    st.title("🔬 Test Neural Network")

    uploaded = st.file_uploader("Upload Flower Image")


    if uploaded is not None:


        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

        img = cv2.imdecode(file_bytes, 1)


        st.image(img, caption="Uploaded Image", width=300)


        img_resize = cv2.resize(img, (224,224))


        img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_resize)


        img_pre = np.expand_dims(img_pre, axis=0)


        pred = flower_nn.predict(img_pre, verbose=0)


        class_index = np.argmax(pred)


        confidence = np.max(pred)


        st.success(f"Prediction: {class_names[class_index]}")

        st.info(f"Confidence: {confidence*100:.2f}%")


# ====================================
# Page 4: Test Ensemble
# ====================================

elif page == "Test Ensemble Model":

    st.title("🔬 Test Ensemble Model")

    uploaded = st.file_uploader("Upload Flower Image")


    if uploaded is not None:


        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

        img = cv2.imdecode(file_bytes, 1)


        st.image(img, caption="Uploaded Image", width=300)


        img_resize = cv2.resize(img, (224,224))


        img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_resize)


        img_pre = np.expand_dims(img_pre, axis=0)


        feature = feature_model.predict(img_pre, verbose=0)


        pred = flower_ensemble.predict(feature)


        name = label_encoder.inverse_transform(pred)


        st.success(f"Prediction: {name[0]}")