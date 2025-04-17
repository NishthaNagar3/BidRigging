# src/preprocess.py
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.select_dtypes(include=["number"])
    return df
