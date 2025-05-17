import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from advanced_models import train_xgboost, train_lstm, save_models
import os

def main():
    # Load the preprocessed dataset
    print("Loading data...")
    df = pd.read_csv('final_data.csv')  # Since we're already in the correct directory
    
    # Prepare features
    features = ['temperature', 'humidity', 'windSpeed', 'pressure', 'precipIntensity', 'hour',
               'day_of_week', 'month', 'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter',
               'city_dallas', 'city_houston', 'city_la', 'city_nyc', 'city_philadelphia',
               'city_phoenix', 'city_san antonio', 'city_san diego', 'city_san jose', 'city_seattle']
    
    X = df[features]
    y = df['demand']
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model, y_pred_xgb = train_xgboost(X_train, y_train, X_test, y_test, features)
    
    # Train LSTM
    print("\nTraining LSTM model...")
    lstm_model, y_pred_lstm, feature_scaler, target_scaler = train_lstm(X_train, y_train, X_test, y_test, features)
    
    # Save all models
    print("\nSaving models...")
    save_models(rf_model, xgb_model, lstm_model, feature_scaler, target_scaler)
    
    # Save predictions for comparison
    print("\nSaving predictions...")
    # Get the minimum length among all prediction arrays
    min_length = min(len(y_test), len(y_pred_rf), len(y_pred_xgb), len(y_pred_lstm))
    
    # Create DataFrame with aligned predictions
    forecast_df = pd.DataFrame({
        'actual': y_test.values[:min_length],
        'rf_predicted': y_pred_rf[:min_length],
        'xgb_predicted': y_pred_xgb[:min_length],
        'lstm_predicted': y_pred_lstm.flatten()[:min_length]
    })
    
    # Print the lengths for debugging
    print(f"\nLengths of prediction arrays:")
    print(f"Actual values: {len(y_test)}")
    print(f"RF predictions: {len(y_pred_rf)}")
    print(f"XGBoost predictions: {len(y_pred_xgb)}")
    print(f"LSTM predictions: {len(y_pred_lstm)}")
    print(f"Final DataFrame length: {min_length}")
    
    forecast_df.to_csv('forecast_results_all_models.csv', index=False)
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main() 