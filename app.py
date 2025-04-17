import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("models/isolation_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# App title and description
st.title("🕵️ Bid Rigging Detection App")
st.markdown("Upload a procurement CSV file to detect suspicious tenders.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Uploaded Data Preview", df.head())

    # Required features
    features = ['bid_deviation', 'bid_ratio', 'num_bidders',
                'win_count_per_bidder', 'bidder_win_rate', 'avg_bid_per_tender']

    if all(col in df.columns for col in features):
        # Preprocess
        X_scaled = scaler.transform(df[features])
        df['anomaly_score'] = model.predict(X_scaled)
        df['is_suspicious'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

        st.write("### ✅ Prediction Results", df[['tender_id', 'bidder_id', 'is_suspicious']])
        st.success(f"🚨 Suspicious tenders detected: {df['is_suspicious'].sum()}")

        # 📊 Visualizations
        st.write("### 📊 Suspicion Summary")

        # Bar chart for counts
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='is_suspicious', palette={0: 'green', 1: 'red'}, ax=ax1)
        ax1.set_xticklabels(['Not Suspicious', 'Suspicious'])
        ax1.set_ylabel("Count")
        ax1.set_title("Suspicious vs Normal Tenders")
        st.pyplot(fig1)

        # Distribution of suspicious bids across bidders
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        top_sus = df[df['is_suspicious'] == 1]['bidder_id'].value_counts().head(10)
        sns.barplot(x=top_sus.index.astype(str), y=top_sus.values, ax=ax2, palette='Reds')
        ax2.set_title("Top 10 Bidders with Most Suspicious Bids")
        ax2.set_xlabel("Bidder ID")
        ax2.set_ylabel("Count of Suspicious Bids")
        st.pyplot(fig2)

        # Correlation heatmap (optional)
        st.write("### 🔬 Feature Correlation")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        corr = df[features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
        ax3.set_title("Correlation Matrix of Features")
        st.pyplot(fig3)

        # Download button
        st.download_button("📥 Download Results", df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.error("Uploaded file must contain the required features: " + ", ".join(features))
