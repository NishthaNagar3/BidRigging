import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(data):
    data['submission_date'] = pd.to_datetime(data['submission_date'])
    data['year'] = data['submission_date'].dt.year
    data['month'] = data['submission_date'].dt.month
    data['day'] = data['submission_date'].dt.day
    data['bid_ratio'] = data['bid_value'] / data['tender_value']

    encoders = {
        'department': LabelEncoder().fit(['Education', 'IT', 'Health', 'Infrastructure']),
        'location': LabelEncoder().fit(['Chennai', 'Kolkata', 'Bangalore']),
    }

    for col in ['department', 'location']:
        data[col] = encoders[col].transform(data[col])

    return data[['bid_ratio', 'department', 'location', 'year', 'month', 'day']]
