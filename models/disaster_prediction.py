# disaster_prediction.py
import joblib
import pandas as pd
import numpy as np

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to 'models' folder
model_path = os.path.join(BASE_DIR, "argo_anomaly_model.pkl")

xgb_model = joblib.load(model_path)


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