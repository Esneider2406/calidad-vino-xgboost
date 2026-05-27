# Predicción de calidad de vino con XGboost

Aplicación web desarrollada con Streamlit para predecir la calidad de un vino mediante técnicas de Machine Learning utilizando el algoritmo XGBoost.
El sistema permite ingresar características fisicoquímicas del vino y obtener una predicción en tiempo real sobre su calidad.

---

# Descripción del Proyecto

Este proyecto implementa un modelo de clasificación binaria entrenado sobre el dataset Wine Quality Prediction.

La aplicación permite:

- Realizar predicciones en tiempo real
- Visualizar probabilidades de clasificación
- Analizar métricas del modelo
- Explorar el dataset
- Representar visualmente la composición del vino mediante una botella dinámica SVG

---

# Tecnologías Utilizadas

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

# Estructura del Proyecto

```bash
calidad-vino-xgboost/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── calidad_de_vino.csv
│
├── models/
│   ├── modelo.pkl
│   ├── scaler.pkl
│   └── features.json
│
├── requirements.txt
│
└── README.md
```

---

# Dataset

El proyecto utiliza el dataset Wine Quality Prediction, el cual contiene información química de vinos tintos y blancos.

## Variables utilizadas

- Acidez fija
- Acidez volátil
- Ácido cítrico
- Azúcar residual
- Cloruros
- Dióxido de azufre libre
- Dióxido de azufre total
- Densidad
- pH
- Sulfatos
- Alcohol
- Tipo de vino

---

# Modelo de Machine Learning

Se implementó un modelo de clasificación utilizando XGBoost Classifier.

## Clasificación

| Clase | Descripción |
|---|---|
| 0 | Vino Malo |
| 1 | Vino Bueno |

La variable objetivo fue definida mediante:

```python
vino_bueno = calidad >= 6
```

---

# Funcionalidades

## Predicción de Calidad

La aplicación permite ingresar las características del vino y obtener:

- Predicción instantánea
- Nivel de confianza
- Probabilidad de clasificación
- Variables más importantes del modelo

---

## Visualización Dinámica

El sistema incluye una representación gráfica mediante SVG de una botella de vino.

Cada variable:

- Tiene un color asignado
- Representa un porcentaje dentro de la botella
- Cambia dinámicamente según los valores ingresados

---

## Métricas del Modelo

La aplicación muestra:

- Accuracy
- ROC-AUC
- Precision
- Recall
- Matriz de confusión
- Curva ROC
- Importancia de variables

---

## Análisis del Dataset

Incluye visualizaciones estadísticas como:

- Distribución de calidad
- Distribución por color
- Correlación entre variables
- Vista previa del dataset

---

# Instalación

## Clonar repositorio

```bash
git clone https://github.com/tu-usuario/calidad-vino-xgboost.git
```

---

## Acceder al proyecto

```bash
cd calidad-vino-xgboost
```

---

## Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución

```bash
streamlit run app/app.py
```

o

```bash
python -m streamlit run app/app.py
```

---

# Resultados del Modelo

| Métrica | Resultado aproximado |
|---|---|
| Accuracy | 84.7% |
| Modelo | XGBoost Classifier |

---

# Objetivo del Proyecto

El propósito de este proyecto es aplicar técnicas de Machine Learning y visualización de datos mediante una interfaz interactiva desarrollada en Streamlit.

El proyecto integra:

- Modelado predictivo
- Visualización interactiva
- Análisis exploratorio
- Desarrollo de dashboards

---

# Autores

Esneider Velasco Cruz y Sergio Alejandro Arias Romero

---

# Licencia

Proyecto desarrollado con fines educativos y académicos.

