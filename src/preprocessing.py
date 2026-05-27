import pandas as pd
import numpy as np

# Herramientas de Scikit-Learn para escalado y división del dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    # Lee el archivo y lo retorna como DataFrame
    return pd.read_csv(path)

# Función encargada de todo el preprocesamiento
def preprocess(df: pd.DataFrame):
    """
    Aplica el preprocesamiento completo del notebook:
    - Clasificación binaria (vino_bueno)
    - Drop de columnas irrelevantes
    - One-hot encoding de color
    - Split train/test
    - Escalado StandardScaler
    Retorna X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
     # Se hace copia para no modificar el DataFrame original
    df = df.copy()
    df["vino_bueno"] = (df["calidad"] >= 6).astype(int)
    df = df.drop(["botella_id", "calidad"], axis=1)
    df = pd.get_dummies(df, columns=["color"], drop_first=True)

    X = df.drop("vino_bueno", axis=1)
    y = df["vino_bueno"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

# Prepara un solo registro para hacer predicción desde la app
def prepare_single(input_dict: dict, scaler: StandardScaler, feature_names: list) -> np.ndarray:
    """
    Prepara un único registro para predicción.
    input_dict: valores del usuario (incluye 'color' como 'red'/'white')
    """
    color = input_dict.pop("color", "red")
    row = {f: input_dict.get(f, 0.0) for f in feature_names if f != "color_white"}
    row["color_white"] = 1 if color == "white" else 0

    df_row = pd.DataFrame([row])[feature_names]
    return scaler.transform(df_row)
