import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt

def train_xgboost(X_train, y_train, X_test, y_test, features):
    """Train and evaluate XGBoost model"""
    # Create and train XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    print("XGBoost model trained.")

    # Make predictions
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluate model
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

    print("\nXGBoost Performance:")
    print(f"MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}, MAPE: {mape_xgb:.2f}%")

    # Plot feature importance
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=features)
    plt.figure(figsize=(10, 6))
    xgb_importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance in XGBoost Model')
    plt.xlabel('Importance')
    plt.savefig('xgb_feature_importance.png')
    plt.close()

    return xgb_model, y_pred_xgb

def create_sequences(X, y, time_steps=24):
    """Create sequences for LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train, y_train, X_test, y_test, features):
    """Train and evaluate LSTM model"""
    # Scale the features
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale the target
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    # Create sequences
    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

    # Build LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(time_steps, X_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    # Compile model
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train model
    history = lstm_model.fit(
        X_seq, y_seq,
        epochs=1,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Make predictions
    y_pred_lstm_scaled = lstm_model.predict(X_test_seq)
    y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled)
    y_test_lstm = target_scaler.inverse_transform(y_test_seq)

    # Evaluate model
    mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
    mape_lstm = np.mean(np.abs((y_test_lstm - y_pred_lstm) / y_test_lstm)) * 100

    print("\nLSTM Performance:")
    print(f"MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, MAPE: {mape_lstm:.2f}%")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_training_history.png')
    plt.close()

    # Save scalers
    joblib.dump(feature_scaler, 'feature_scaler.joblib')
    joblib.dump(target_scaler, 'target_scaler.joblib')

    return lstm_model, y_pred_lstm, feature_scaler, target_scaler

def save_models(rf_model, xgb_model, lstm_model, feature_scaler, target_scaler):
    """Save all models and scalers"""
    # Save Random Forest model
    joblib.dump(rf_model, 'rf_model.joblib')
    
    # Save XGBoost model
    joblib.dump(xgb_model, 'xgb_model.joblib')
    
    # Save LSTM model
    lstm_model.save('lstm_model.h5')
    
    print("All models saved successfully!")

def load_models():
    """Load all saved models"""
    try:
        rf_model = joblib.load('rf_model.joblib')
        xgb_model = joblib.load('xgb_model.joblib')
        lstm_model = load_model('lstm_model.h5')
        feature_scaler = joblib.load('feature_scaler.joblib')
        target_scaler = joblib.load('target_scaler.joblib')
        return rf_model, xgb_model, lstm_model, feature_scaler, target_scaler
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None, None 