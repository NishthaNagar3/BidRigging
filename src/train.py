# src/train.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from preprocess import clean_data

df = pd.read_csv("data/raw/sample.csv")
df = clean_data(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_scaled)

# Save model and preprocessor
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/preprocessor.pkl")
