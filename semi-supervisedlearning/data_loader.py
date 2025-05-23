import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

FILE_PATH = "Filtered_Call_Records-cut (1).csv"  
target_col = "OverallCallQuality"

# Columns to ignore, Duy note: in case some columns are not in use but not remove all
cols_to_remove = [
    'SbcEstimatedGttCost', 'SbcEstimatedSoftnetCost', 'StreamQuality',
    'CallerConnectionType', 'CalleeConnectionType', 'CalleeLinkSpeed', 'CalleeTransportProtocol'
]

def load_call_quality_data():
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df = df.dropna(subset=[target_col])

    original_df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols_to_remove:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
        df = df.drop(columns=[col], errors='ignore')

    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    encoders = {}
    for col in categorical_cols + [target_col]:
        df = df.dropna(subset=[col])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[numeric_cols + categorical_cols].values
    y = df[target_col].values
    feature_names = numeric_cols + categorical_cols
    return X, y, feature_names, original_df, scaler
