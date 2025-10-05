import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load Boston Housing dataset from original source"""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Split into features and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def preprocess_data(df, target_col='MEDV', test_size=0.2, random_state=42):
    """Generic preprocessing function for regression tasks"""
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Data preprocessed - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(model, X_train, y_train):
    """Generic model training function"""
    logger.info(f"Training {type(model).__name__}...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Generic model evaluation function"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"{type(model).__name__} MSE: {mse:.4f}")
    return mse

def run_experiment(model, df, model_name="Model"):
    """Complete experiment pipeline"""
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    
    trained_model = train_model(model, X_train, y_train)
    mse = evaluate_model(trained_model, X_test, y_test)
    
    print(f"{model_name} - Mean Squared Error: {mse:.4f}")
    return mse, trained_model
