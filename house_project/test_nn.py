import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------
# Load model
# ------------------------

model = joblib.load("nn_model.pkl")

scaler_X = joblib.load("scaler_X.pkl")

scaler_y = joblib.load("scaler_y.pkl")

columns = joblib.load("columns.pkl")

print("Model loaded ✅")


# ------------------------
# Load data with true price
# ------------------------

df = pd.read_csv("train.csv")

df = df.drop("Id", axis=1)


# save true value
y_true_real = df["SalePrice"]


# preprocess same as training

df["SalePrice"] = np.log1p(df["SalePrice"])

X = df.drop("SalePrice", axis=1)

y = df["SalePrice"]


X = X.fillna(0)

X = pd.get_dummies(X)

X = X.reindex(columns=columns, fill_value=0)


# scale
X_scaled = scaler_X.transform(X)


# predict
pred_scaled = model.predict(X_scaled)

pred_log = scaler_y.inverse_transform(

pred_scaled.reshape(-1,1)

).ravel()

pred_real = np.expm1(pred_log)


# ------------------------
# Metrics
# ------------------------

mse = mean_squared_error(y_true_real, pred_real)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_true_real, pred_real)

r2 = r2_score(y_true_real, pred_real)


print("\nRESULT")

print("RMSE:", rmse)

print("MAE :", mae)

print("R2 Score:", r2)