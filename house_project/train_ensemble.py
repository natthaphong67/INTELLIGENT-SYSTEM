import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("train.csv")

# Drop Id
df = df.drop("Id", axis=1)

# Separate X and y
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Fill missing values
X = X.fillna(0)

# Convert text to number
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
dt = DecisionTreeRegressor()

# Ensemble
ensemble = VotingRegressor([
    ('rf', rf),
    ('gb', gb),
    ('dt', dt)
])

# Train
ensemble.fit(X_train, y_train)

# Test
pred = ensemble.predict(X_test)

mse = mean_squared_error(y_test, pred)

print("Ensemble MSE:", mse)

# Save
joblib.dump(ensemble, "ensemble_model.pkl")

# Save columns
joblib.dump(X.columns, "columns.pkl")