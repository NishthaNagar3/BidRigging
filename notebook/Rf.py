import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('../data/bids.csv')

# Feature engineering
df['submission_date'] = pd.to_datetime(df['submission_date'])
df['year'] = df['submission_date'].dt.year
df['month'] = df['submission_date'].dt.month
df['day'] = df['submission_date'].dt.day
df['bid_ratio'] = df['bid_value'] / df['tender_value']

# Encode categorical variables
for col in ['department', 'location']:
    df[col] = LabelEncoder().fit_transform(df[col])

df['award_status'] = df['award_status'].map({'Won': 1, 'Lost': 0})

# Prepare features and target variable
features = ['bid_ratio', 'department', 'location', 'year', 'month', 'day']
X = df[features]
y = df['award_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_rf, '../models/random_forest_model.pkl')

# Check accuracy (optional)
accuracy = model_rf.score(X_test, y_test)
print(f'Accuracy of the Random Forest model: {accuracy:.4f}')
