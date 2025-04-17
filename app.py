import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("models/isolation_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🕵️ Bid Rigging Detection App")

st.markdown("Upload a procurement CSV file to detect suspicious tenders.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Features used in training
    features = ['bid_deviation', 'bid_ratio', 'num_bidders',
                'win_count_per_bidder', 'bidder_win_rate', 'avg_bid_per_tender']

    if all(col in df.columns for col in features):
        X_scaled = scaler.transform(df[features])
        df['anomaly_score'] = model.predict(X_scaled)
        df['is_suspicious'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

        st.write("### Predictions", df[['tender_id', 'bidder_id', 'is_suspicious']])
        st.success(f"🚨 Suspicious tenders detected: {df['is_suspicious'].sum()}")
        
        st.download_button("Download Results", df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.error("Uploaded file must contain the required features.")
