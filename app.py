import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_input

# Set the Streamlit page configuration
st.set_page_config(page_title="Bid Rigging Detection", layout="centered")

# Title of the app
st.title("🕵️‍♂️ Bid Rigging Detection & Prediction System")

# Model selection dropdown
model_choice = st.selectbox("Select Prediction Model", ["Logistic Regression", "Random Forest"])

# Load the selected model
if model_choice == "Logistic Regression":
    model = joblib.load("models/model.pkl")
else:
    model = joblib.load("models/random_forest_model.pkl")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Function to display bidder behavior visualizations
def plot_bidder_behavior(df):
    # Plot the distribution of bid values
    plt.figure(figsize=(10, 6))
    sns.histplot(df['bid_value'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Bid Values')
    plt.xlabel('Bid Value')
    plt.ylabel('Frequency')
    st.pyplot()

    # Boxplot of bid values by department to identify outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='department', y='bid_value', data=df)
    plt.title('Bid Values by Department')
    plt.xlabel('Department')
    plt.ylabel('Bid Value')
    st.pyplot()

    # Plot the correlation between tender value and bid value
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tender_value', y='bid_value', data=df, hue='award_status', palette='coolwarm')
    plt.title('Tender Value vs Bid Value')
    plt.xlabel('Tender Value')
    plt.ylabel('Bid Value')
    st.pyplot()

    # Visualize bidder behavior over time (submission date)
    df['submission_date'] = pd.to_datetime(df['submission_date'])
    df['month'] = df['submission_date'].dt.month
    plt.figure(figsize=(10, 6))
    sns.countplot(x='month', data=df, palette='viridis')
    plt.title('Bids Submitted by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Bids')
    st.pyplot()

# When the file is uploaded, run predictions and show visualizations
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocess the data and make predictions
    processed = preprocess_input(df.copy())
    predictions = model.predict(processed)
    df['Rigging_Suspected'] = predictions
    df['Rigging_Suspected'] = df['Rigging_Suspected'].map({1: 'Likely Won', 0: 'Likely Lost'})
    
    # Display prediction results
    st.success("Prediction Complete!")
    st.write(df)

    # Allow users to download the prediction results
    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

    # Visualize bidder behavior
    plot_bidder_behavior(df)

else:
    st.info("Please upload a valid CSV file with tender details.")
