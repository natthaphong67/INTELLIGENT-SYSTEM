import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
ensemble = joblib.load("ensemble_model.pkl")
nn = joblib.load("nn_model.pkl")

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

columns = joblib.load("columns.pkl")


# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("Menu")

page = st.sidebar.radio(

"Go to",

(

"Home",

"Ensemble Model",

"Neural Network",

"Predict"

)

)


# -------------------------
# Home
# -------------------------

if page == "Home":

    st.title("House Price Prediction Project")

    st.write("""

    Project นี้ใช้ Machine Learning และ Neural Network

    เพื่อทำนายราคาบ้าน

    Dataset: House Prices (Kaggle)

    Models:

    - Ensemble Model

    - Neural Network

    """)


# -------------------------
# Ensemble Page
# -------------------------

elif page == "Ensemble Model":

    st.title("Ensemble Model")

    st.write("""

    Ensemble Model คือการรวมหลาย Machine Learning Model

    Models used:

    - Random Forest

    - Gradient Boosting

    - Decision Tree

    ใช้ VotingRegressor ในการรวมผล

    """)


# -------------------------
# Neural Network Page
# -------------------------

elif page == "Neural Network":

    st.title("Neural Network")

    st.write("""

    Neural Network ใช้ MLPRegressor

    Structure:

    - Hidden Layers

    - ReLU activation

    - Adam optimizer

    ใช้ StandardScaler ในการ scale data

    """)


# -------------------------
# Predict Page
# -------------------------

elif page == "Predict":

    st.title("Predict House Price")

    uploaded = st.file_uploader(

    "Upload CSV",

    type=["csv"]

    )

    if uploaded:

        data = pd.read_csv(uploaded)

        if "Id" in data.columns:

            data = data.drop("Id", axis=1)

        data = data.fillna(0)

        data = pd.get_dummies(data)

        data = data.reindex(

        columns=columns,

        fill_value=0

        )


        # Ensemble

        if st.button("Predict Ensemble"):

            pred = ensemble.predict(data)

            st.write("Predicted Price:")

            st.write(pred)


        # Neural Network

        if st.button("Predict Neural Network"):

            X_scaled = scaler_X.transform(data)

            pred_scaled = nn.predict(X_scaled)

            pred_log = scaler_y.inverse_transform(

            pred_scaled.reshape(-1,1)

            )

            pred = np.expm1(pred_log)

            st.write("Predicted Price:")

            st.write(pred)