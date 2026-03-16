import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Binary encoding for yes/no columns
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # Label encoding for furnishingstatus
    # furnished: 2, semi-furnished: 1, unfurnished: 0
    df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})
    
    # Features and Target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, columns = preprocess_data('data/housing.csv')
    print("Preprocessing complete.")
    print(f"Features: {list(columns)}")
    print(f"X_train shape: {X_train.shape}")
