import joblib
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "models"


def load_model(model_path: str = None, scaler_path: str = None):
    model_path = model_path or BASE / "modelo.pkl"
    scaler_path = scaler_path or BASE / "scaler.pkl"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, X_scaled: np.ndarray) -> tuple:
    """
    Retorna (clase_predicha, probabilidad_bueno)
    """
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    return int(pred), float(proba)
