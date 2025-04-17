import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

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

features = ['bid_ratio', 'department', 'location', 'year', 'month', 'day']
X = df[features]
y = df['award_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, '../models/model.pkl')
