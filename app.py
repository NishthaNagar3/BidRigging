import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_input

st.set_page_config(page_title="Bid Rigging Detection", layout="centered")

st.title("🕵️‍♂️ Bid Rigging Detection & Prediction System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model = joblib.load("models/model.pkl")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    processed = preprocess_input(df.copy())
    predictions = model.predict(processed)
    df['Rigging_Suspected'] = predictions
    df['Rigging_Suspected'] = df['Rigging_Suspected'].map({1: 'Likely Won', 0: 'Likely Lost'})
    
    st.success("Prediction Complete!")
    st.write(df)
    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
else:
    st.info("Please upload a valid CSV file with tender details.")
