# Librerías necesarias para cargar el modelo y trabajar con arrays
import joblib
import numpy as np
from pathlib import Path

# Ruta base donde están guardados el modelo y el scaler
BASE = Path(__file__).parent.parent / "models"

# Función para cargar el modelo entrenado y el scaler
def load_model(model_path: str = None, scaler_path: str = None):

    # Si no se pasa una ruta personalizada,
    # usa los archivos por defecto del proyecto
    model_path = model_path or BASE / "modelo.pkl"
    scaler_path = scaler_path or BASE / "scaler.pkl"

    # Carga el modelo XGBoost entrenado
    model = joblib.load(model_path)
    # Carga el scaler usado durante el entrenamiento
    scaler = joblib.load(scaler_path)
     # Retorna ambos recursos para reutilizarlos en la app
    return model, scaler


# Función encargada de hacer predicciones
def predict(model, X_scaled: np.ndarray) -> tuple:
    """
    Retorna (clase_predicha, probabilidad_bueno)
    """
    # Predicción final de la clase (0 o 1)
    pred = model.predict(X_scaled)[0]
    # Predicción final de la clase (0 o 1)
    proba = model.predict_proba(X_scaled)[0][1]
    return int(pred), float(proba)
