# disaster_prediction.py
import joblib
import pandas as pd
import numpy as np

# Load trained anomaly detection model
# Place your model file inside models/ folder
xgb_model = joblib.load("models/argo_anomaly_model.pkl")

def predict_disaster(float_data: dict):
    """
    Takes ARGO float data dict and predicts if anomaly (possible disaster).
    
    Parameters:
        float_data (dict): Keys must match training features
    
    Returns:
        dict: { "prediction": "Anomaly"/"Normal", "confidence": float }
    """
    # Convert input to DataFrame (single row)
    df = pd.DataFrame([float_data])
    
    # Predict class + probability
    pred = xgb_model.predict(df)[0]
    prob = xgb_model.predict_proba(df)[0]
    
    return {
        "prediction": "Anomaly" if pred == 1 else "Normal",
        "confidence": float(np.max(prob))
    }
