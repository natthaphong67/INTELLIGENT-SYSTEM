import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("train.csv")

df = df.drop("Id", axis=1)

# log transform target
df["SalePrice"] = np.log1p(df["SalePrice"])

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Fill missing
X = X.fillna(0)

# Convert text
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Scale X
scaler_X = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# ✅ Scale y
scaler_y = StandardScaler()

y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()

# Neural Network (stable config)
nn = MLPRegressor(

hidden_layer_sizes=(128,64),

activation='relu',

solver='adam',

learning_rate_init=0.001,

max_iter=2000,

early_stopping=True,

random_state=42

)

# Train
nn.fit(X_train, y_train)

# Predict
pred_scaled = nn.predict(X_test)

# Convert back
pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()

pred = np.expm1(pred_log)

y_test_real = np.expm1(y_test)

# Evaluate
mse = mean_squared_error(y_test_real, pred)

print("NN MSE:", mse)

# Save
joblib.dump(nn, "nn_model.pkl")

joblib.dump(scaler_X, "scaler_X.pkl")

joblib.dump(scaler_y, "scaler_y.pkl")

joblib.dump(X.columns, "columns.pkl")