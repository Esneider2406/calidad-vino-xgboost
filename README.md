# Predicción de calidad de vino con XGboost

## Problema

La evaluación de la calidad del vino normalmente requiere procesos de análisis químicos y pruebas sensoriales realizadas por expertos, lo que puede representar altos costos y tiempos de evaluación para productores y distribuidores.
Además, determinar si un vino será considerado de buena o mala calidad no siempre es sencillo únicamente mediante inspección manual, ya que múltiples variables fisicoquímicas influyen en el resultado final.
Ante esta situación, surge la necesidad de desarrollar una herramienta capaz de analizar automáticamente las características de un vino y predecir su calidad utilizando técnicas de Machine Learning.

Este proyecto propone una solución basada en una aplicación desarrollada en Streamlit implementando el algoritmo XGBoost, permitiendo clasificar vinos como buenos o malos a partir de variables químicas del producto, ofreciendo resultados rápidos, visuales e interactivos mediante una aplicación web desarrollada con Streamlit.

---

# Público Objetivo

Este proyecto está dirigido a:

- Estudiantes y personas interesadas en Machine Learning y Ciencia de Datos.
- Desarrolladores que deseen aprender sobre integración de modelos predictivos con Streamlit.
- Investigadores o analistas que trabajen con clasificación de datos.
- Productores o distribuidores de vino interesados en automatizar procesos básicos de evaluación.
- Usuarios que deseen explorar técnicas de análisis predictivo mediante aplicaciones interactivas.

La aplicación también puede ser utilizada como apoyo académico para demostrar el flujo completo de un proyecto de Machine Learning, desde el preprocesamiento de datos hasta el despliegue de una interfaz web funcional.

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

