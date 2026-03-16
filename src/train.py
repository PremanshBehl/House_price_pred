import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import preprocess_data

def train_model():
    # Load data
    X_train, X_test, y_train, y_test, columns = preprocess_data('data/housing.csv')
    
    # Initialize and train model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model and feature names
    joblib.dump(model, 'models/house_price_model.joblib')
    joblib.dump(list(columns), 'models/features.joblib')
    print("\nModel saved to models/house_price_model.joblib")

if __name__ == "__main__":
    train_model()
