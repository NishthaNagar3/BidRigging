# src/predict.py
import joblib
import pandas as pd
from preprocess import clean_data

model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

def predict_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = clean_data(df)
    scaled = preprocessor.transform(clean_df)
    preds = model.predict(scaled)
    df["is_anomaly"] = (preds == -1).astype(int)
    return df
