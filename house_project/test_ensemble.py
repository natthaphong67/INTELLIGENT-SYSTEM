import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------
# Load model
# ------------------------

model = joblib.load("ensemble_model.pkl")

columns = joblib.load("columns.pkl")

print("Ensemble model loaded ✅")


# ------------------------
# Load data
# ------------------------

df = pd.read_csv("train.csv")

df = df.drop("Id", axis=1)


# true value
y_true = df["SalePrice"]

X = df.drop("SalePrice", axis=1)


# ------------------------
# Preprocess
# ------------------------

X = X.fillna(0)

X = pd.get_dummies(X)

X = X.reindex(columns=columns, fill_value=0)


# ------------------------
# Predict
# ------------------------

pred = model.predict(X)


# ------------------------
# Metrics
# ------------------------

mse = mean_squared_error(y_true, pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_true, pred)

r2 = r2_score(y_true, pred)


print("\nRESULT")

print("RMSE:", rmse)

print("MAE :", mae)

print("R2 Score:", r2)