"""
Retrain the car price prediction model with current scikit-learn version.
This resolves the '_drop_idx_after_grouping' AttributeError caused by
loading a model pickled with an older sklearn version.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Load dataset
df = pd.read_csv("final_scout_not_dummy.csv")
print(f"Dataset shape: {df.shape}")

# Use only the features available in the app
features = ["make_model", "hp_kW", "age", "km", "Gears", "Gearing_Type"]
target = "price"

df_model = df[features + [target]].dropna()
print(f"After dropna: {df_model.shape}")

X = df_model[features]
y = df_model[target]

# Define column types
categorical_cols = ["make_model", "Gearing_Type"]
numerical_cols = ["hp_kW", "age", "km", "Gears"]

# Build preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# Build full pipeline with GradientBoostingRegressor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])

# Train on full dataset (no train/test split — mirrors original approach)
pipeline.fit(X, y)
print("Model trained successfully!")

# Evaluate quickly
from sklearn.metrics import r2_score, mean_absolute_error
y_pred = pipeline.predict(X)
print(f"R² (train): {r2_score(y, y_pred):.4f}")
print(f"MAE (train): {mean_absolute_error(y, y_pred):.2f}")

# Save with current sklearn
with open("Auto_Price_Pred_Model", "wb") as f:
    pickle.dump(pipeline, f)
print("Model saved as 'Auto_Price_Pred_Model'")

# Quick sanity test
test_input = pd.DataFrame([{
    "make_model": "Audi A3",
    "hp_kW": 120,
    "age": 1,
    "km": 15000,
    "Gears": 6,
    "Gearing_Type": "Manual"
}])
result = pipeline.predict(test_input)
print(f"\nSanity check — predicted price for Audi A3: €{int(result[0]):,}")
