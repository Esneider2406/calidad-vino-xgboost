# Agrega la ruta raíz del proyecto al PATH para poder importar
# archivos aunque app.py esté dentro de otra carpeta.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib

# Evita errores de renderizado de gráficos en Streamlit.
matplotlib.use("Agg")

import seaborn as sns

from pathlib import Path

# Métricas usadas para evaluar el modelo.

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ─────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODELS = BASE / "models"
DATA = BASE / "data" / "raw" / "calidad_de_vino.csv"

# CONFIGURACIÓN GENERAL DE STREAMLIT

st.set_page_config(
    page_title="Calidad de Vino · XGBoost",
    layout="wide"
)


# Todo este bloque es CSS personalizado para cambiar
# completamente la apariencia por defecto de Streamlit.

st.markdown("""
<style>
/* Oculta header superior blanco */
header {
    visibility: hidden;
}

/* Elimina espacio superior */
.block-container {
    padding-top: 1rem;
}

/* Oculta menú y footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
/* ===== Fondo general ===== */
[data-testid="stAppViewContainer"]{
    background: #081120;
    color: white;
}

[data-testid="stSidebar"]{
    background: #0d1726;
}

/* ===== Títulos ===== */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}

/* ===== Cards ===== */
.metric-card{
    background: linear-gradient(145deg, #111c2e, #162338);
    border: 1px solid #24344d;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}

.metric-card h2{
    color: #60a5fa;
    margin: 0;
}

.metric-card p{
    color: #d1d5db;
}

/* ===== Resultado bueno ===== */
.result-good{
    background: linear-gradient(145deg, #123524, #174a31);
    border: 1px solid #22c55e;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

/* ===== Resultado malo ===== */
.result-bad{
    background: linear-gradient(145deg, #3b1616, #4a1d1d);
    border: 1px solid #ef4444;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

/* ===== Textos resultado ===== */
.result-good h1,
.result-bad h1{
    color: white;
    font-size: 2.2rem;
}

/* ===== Inputs numéricos ===== */
.stNumberInput input{
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    font-size: 16px !important;
}

/* ===== Selectbox ===== */
.stSelectbox div[data-baseweb="select"] > div{
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* ===== Dropdown desplegado ===== */
div[role="listbox"]{
    background-color: #111827 !important;
    color: white !important;
}

div[role="option"]{
    background-color: #111827 !important;
    color: white !important;
}

div[role="option"]:hover{
    background-color: #2563eb !important;
    color: white !important;
}

/* ===== Labels ===== */
.stSelectbox label,
.stNumberInput label{
    color: white !important;
    font-weight: 600;
}

/* ===== Botones + y - ===== */
button[kind="secondary"]{
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
}

button[kind="secondary"]:hover{
    background-color: #2563eb !important;
    border: 1px solid #2563eb !important;
}

/* ===== Tabs / radio ===== */
.stRadio label{
    color: white !important;
}

/* ===== Separadores ===== */
hr{
    border-color: #24344d !important;
}

/* ===== Section title ===== */
.section-title{
    font-size: 1.2rem;
    font-weight: 700;
    color: #60a5fa !important;
    border-bottom: 1px solid #24344d;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* ===== Dataframe ===== */
[data-testid="stDataFrame"]{
    border-radius: 12px;
    overflow: hidden;
}

/* ===== Slider ===== */
.stSlider label{
    color: white !important;
}

.stSlider div[data-baseweb="slider"] span{
    background-color: #3b82f6 !important;
}

</style>

""", unsafe_allow_html=True)


# CARGA DE MODELO Y DATASET


# Cachea el modelo y archivos para no cargarlos
# cada vez que Streamlit se actualiza.

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODELS / "modelo.pkl")

    scaler = joblib.load(MODELS / "scaler.pkl")

    with open(MODELS / "features.json") as f:
        features = json.load(f)

    return model, scaler, features

# Cachea el dataset ya procesado.

@st.cache_data
def load_dataset():
    
    # Carga CSV original.
    df = pd.read_csv(DATA)

    # Convierte calidad en clasificación binaria.
    # >= 6 = vino bueno.
    df["vino_bueno"] = (df["calidad"] >= 6).astype(int)

     # Elimina columnas que no sirven para entrenamiento.
    df = df.drop(["botella_id", "calidad"], axis=1)

    # Convierte color en variable numérica.
    df = pd.get_dummies(df, columns=["color"], drop_first=True)

    return df

# Carga todos los recursos necesarios
model, scaler, FEATURES = load_artifacts()


# SIDEBAR


# Menú lateral principal de navegación
with st.sidebar:

    st.markdown("## Calidad de Vino")

    st.markdown("### XGBoost Classifier")

    st.markdown("---")

    # Selector de páginas de la aplicación.
    page = st.radio(
        "Navegación",
        [
            " Predicción",
            " Métricas del Modelo",
            " Análisis del Dataset"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Información rápida del modelo.
    st.markdown("""
    **Modelo:** XGBoost  
    **Accuracy:** ~84.7 %  
    **Target:** Clasificación binaria  

    `0` = Vino Malo  
    `1` = Vino Bueno
    """)


# PÁGINA DE PREDICCIÓN

# Verifica si el usuario está en la página de predicción
if page == " Predicción":

    st.title(" Predicción de Calidad de Vino")

    st.markdown(
        "Ajusta las características del vino y obtén una predicción instantánea."
    )

    # Divide pantalla en dos columnas
    # Izquierda: inputs
    # Derecha: resultados
    col_left, col_right = st.columns([2, 1], gap="large")


    # INPUTS DEL USUARIO
    
    with col_left:

        st.markdown(
            '<p class="section-title"> Características del Vino</p>',
            unsafe_allow_html=True
        )

        # Presets rápidos para llenar automáticamente inputs.
        preset = st.selectbox(
            " Configuración rápida",
            [
                "Personalizado",
                " Vino Excelente",
                " Vino Promedio",
                " Vino Malo"
            ]
        )

        # Valores de ejemplo para un vino bueno.
        if preset == " Vino Excelente":

            valores = {
                "acidez_fija": 7.0,
                "acidez_volatil": 0.25,
                "acido_citrico": 0.40,
                "azucar_residual": 2.0,
                "cloruros": 0.045,
                "so2_libre": 30.0,
                "so2_total": 90.0,
                "densidad": 0.9940,
                "ph": 3.30,
                "sulfatos": 0.75,
                "alcohol": 13.5,
                "color": "white"
            }

        # Valores de ejemplo para un vino malo
        elif preset == " Vino Malo":

            valores = {
                "acidez_fija": 15.0,
                "acidez_volatil": 1.4,
                "acido_citrico": 0.0,
                "azucar_residual": 12.0,
                "cloruros": 0.40,
                "so2_libre": 3.0,
                "so2_total": 15.0,
                "densidad": 1.0030,
                "ph": 2.8,
                "sulfatos": 0.35,
                "alcohol": 8.5,
                "color": "red"
            }

        # Valores promedio del dataset
        else:

            valores = {
                "acidez_fija": 8.3,
                "acidez_volatil": 0.53,
                "acido_citrico": 0.27,
                "azucar_residual": 2.5,
                "cloruros": 0.087,
                "so2_libre": 15.9,
                "so2_total": 46.5,
                "densidad": 0.9967,
                "ph": 3.31,
                "sulfatos": 0.66,
                "alcohol": 10.4,
                "color": "red"
            }
        # Divide inputs en tres columnas para organizar mejor
        c1, c2, c3 = st.columns(3)

        # Acá empiezan todos los inputs del usuario
        # Cada input representa una característica química del vino
        with c1:

            acidez_fija = st.number_input(
                "Acidez fija",
                4.0,
                16.0,
                valores["acidez_fija"],
                0.1
            )

            acido_citrico = st.number_input(
                "Ácido cítrico",
                0.0,
                1.0,
                valores["acido_citrico"],
                0.01
            )

            so2_libre = st.number_input(
                "SO₂ libre",
                1.0,
                70.0,
                valores["so2_libre"],
                0.5
            )

            ph = st.number_input(
                "pH",
                2.7,
                4.0,
                valores["ph"],
                0.01
            )

        with c2:

            acidez_volatil = st.number_input(
                "Acidez volátil",
                0.1,
                1.6,
                valores["acidez_volatil"],
                0.01
            )

            azucar_residual = st.number_input(
                "Azúcar residual",
                0.9,
                15.0,
                valores["azucar_residual"],
                0.1
            )

            so2_total = st.number_input(
                "SO₂ total",
                6.0,
                289.0,
                valores["so2_total"],
                1.0
            )

            sulfatos = st.number_input(
                "Sulfatos",
                0.33,
                2.0,
                valores["sulfatos"],
                0.01
            )

        with c3:

            cloruros = st.number_input(
                "Cloruros",
                0.01,
                0.6,
                valores["cloruros"],
                0.001,
                format="%.3f"
            )

            densidad = st.number_input(
                "Densidad",
                0.9900,
                1.0040,
                valores["densidad"],
                0.0001,
                format="%.4f"
            )

            alcohol = st.number_input(
                "Alcohol (%)",
                8.0,
                14.9,
                valores["alcohol"],
                0.1
            )

            color = st.selectbox(
                "Color del vino",
                ["red", "white"],
                index=0 if valores["color"] == "red" else 1
            )

    # ─────────────────────────────────────────
    # INPUT MODEL
    # ─────────────────────────────────────────
    input_vals = {
        "acidez fija": acidez_fija,
        "acidez volatil": acidez_volatil,
        "acido citrico": acido_citrico,
        "azucar residual": azucar_residual,
        "cloruros": cloruros,
        "dioxido de azufre libre": so2_libre,
        "dioxido de azufre total": so2_total,
        "densidad": densidad,
        "pH": ph,
        "sulfatos": sulfatos,
        "alcohol": alcohol,
        FEATURES[-1]: 1 if color == "red" else 0
    }

    row_df = pd.DataFrame([input_vals])[FEATURES]

    row_scaled = scaler.transform(row_df)

    pred = model.predict(row_scaled)[0]

    probas = model.predict_proba(row_scaled)[0]

    prob_bueno = probas[1]
    prob_malo = probas[0]

    
    # RESULTADOS
    
    with col_right:

        st.markdown(
            '<p class="section-title"> Resultado</p>',
            unsafe_allow_html=True
        )

        if pred == 1:

            st.markdown(f"""
            <div class="result-good">
                <h1> Vino Bueno</h1>
                <h2>{prob_bueno*100:.1f}% de confianza</h2>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-bad">
                <h1> Vino Malo</h1>
                <h2>{prob_malo*100:.1f}% de confianza</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###  Probabilidades")

        st.progress(float(prob_bueno))

        st.metric(
            " Probabilidad Bueno",
            f"{prob_bueno*100:.1f}%"
        )

        st.metric(
            " Probabilidad Malo",
            f"{prob_malo*100:.1f}%"
        )

        st.markdown("---")

        st.markdown("###  Variables más importantes")

        importances = model.feature_importances_

        feat_imp = pd.Series(
            importances,
            index=FEATURES
        ).sort_values(ascending=True).tail(5)

        fig_fi, ax_fi = plt.subplots(figsize=(5, 3))

        fig_fi.patch.set_facecolor("#0f172a")
        ax_fi.set_facecolor("#0f172a")

        ax_fi.barh(
            feat_imp.index,
            feat_imp.values,
            color="#3b82f6"
        )

        ax_fi.tick_params(colors="white")

        ax_fi.set_title(
            "Top Features",
            color="white"
        )

        for spine in ax_fi.spines.values():
            spine.set_edgecolor("#334155")

        st.pyplot(fig_fi)

        plt.close(fig_fi)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — MÉTRICAS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Métricas del Modelo":

    st.title(" Métricas del Modelo XGBoost")

    df_full = load_dataset()

    from sklearn.model_selection import train_test_split

    X = df_full.drop("vino_bueno", axis=1)[FEATURES]
    y = df_full["vino_bueno"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_test_scaled = scaler.transform(X_test)

    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    rep = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    # ── KPI Cards ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{acc*100:.1f}%</h2>
                <p>Accuracy</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{auc:.3f}</h2>
                <p>ROC-AUC</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{rep["1"]["precision"]*100:.1f}%</h2>
                <p>Precision (Bueno)</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{rep["1"]["recall"]*100:.1f}%</h2>
                <p>Recall (Bueno)</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("---")

    col_cm, col_roc = st.columns(2, gap="large")

    # ── MATRIZ DE CONFUSIÓN ────────────────────────────────────
    with col_cm:

        st.markdown(
            '<p class="section-title">Matriz de Confusión</p>',
            unsafe_allow_html=True
        )

        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

        fig_cm.patch.set_facecolor("#1a1025")
        ax_cm.set_facecolor("#1a1025")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=["Malo", "Bueno"],
            yticklabels=["Malo", "Bueno"],
            ax=ax_cm,
            linewidths=0.5,
            linecolor="#333",
            annot_kws={
                "color": "white",
                "size": 14
            }
        )

        ax_cm.set_xlabel(
            "Predicción",
            color="white"
        )

        ax_cm.set_ylabel(
            "Valor Real",
            color="white"
        )

        ax_cm.tick_params(colors="white")

        ax_cm.set_title(
            "Confusión",
            color="#c084fc"
        )

        st.pyplot(
            fig_cm,
            use_container_width=True
        )

        plt.close(fig_cm)

    # ── CURVA ROC ──────────────────────────────────────────────
    with col_roc:

        st.markdown(
            '<p class="section-title">Curva ROC</p>',
            unsafe_allow_html=True
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))

        fig_roc.patch.set_facecolor("#1a1025")
        ax_roc.set_facecolor("#1a1025")

        ax_roc.plot(
            fpr,
            tpr,
            color="#9333ea",
            lw=2,
            label=f"AUC = {auc:.3f}"
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            "w--",
            lw=1,
            alpha=0.4
        )

        ax_roc.set_xlabel(
            "Tasa Falsos Positivos",
            color="white"
        )

        ax_roc.set_ylabel(
            "Tasa Verdaderos Positivos",
            color="white"
        )

        ax_roc.set_title(
            "Curva ROC",
            color="#c084fc"
        )

        ax_roc.tick_params(colors="white")

        ax_roc.legend(
            labelcolor="white",
            framealpha=0
        )

        for spine in ax_roc.spines.values():
            spine.set_edgecolor("#555")

        st.pyplot(
            fig_roc,
            use_container_width=True
        )

        plt.close(fig_roc)

    st.markdown("---")

    # ── IMPORTANCIA DE VARIABLES ───────────────────────────────
    st.markdown(
        '<p class="section-title">Importancia de Variables</p>',
        unsafe_allow_html=True
    )

    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))

    fig_imp.patch.set_facecolor("#1a1025")
    ax_imp.set_facecolor("#1a1025")

    colors_bar = [
        "#9333ea" if i >= len(feat_df) - 3 else "#6b21a8"
        for i in range(len(feat_df))
    ]

    ax_imp.barh(
        feat_df["Feature"],
        feat_df["Importance"],
        color=colors_bar
    )

    ax_imp.set_xlabel(
        "Importancia",
        color="white"
    )

    ax_imp.set_title(
        "Feature Importance (XGBoost)",
        color="#c084fc"
    )

    ax_imp.tick_params(colors="white")

    for spine in ax_imp.spines.values():
        spine.set_edgecolor("#555")

    st.pyplot(
        fig_imp,
        use_container_width=True
    )

    plt.close(fig_imp)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — ANÁLISIS DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Análisis del Dataset":

    st.title(" Análisis Exploratorio del Dataset")

    df_raw = pd.read_csv(DATA)

    # ── Stats ──────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{len(df_raw):,}</h2>
                <p>Total muestras</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with s2:
        buenos = (df_raw["calidad"] >= 6).sum()

        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{buenos:,}</h2>
                <p>Vinos Buenos (≥6)</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with s3:
        malos = (df_raw["calidad"] < 6).sum()

        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{malos:,}</h2>
                <p>Vinos Malos (&lt;6)</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            f'''
            <div class="metric-card">
                <h2>{df_raw["calidad"].mean():.2f}</h2>
                <p>Calidad promedio</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("---")

    col_dist, col_color = st.columns(2, gap="large")

    # ── DISTRIBUCIÓN DE CALIDAD ────────────────────────────────
    with col_dist:

        st.markdown(
            '<p class="section-title">Distribución de Calidad</p>',
            unsafe_allow_html=True
        )

        fig_d, ax_d = plt.subplots(figsize=(5, 3.5))

        fig_d.patch.set_facecolor("#1a1025")
        ax_d.set_facecolor("#1a1025")

        vc = df_raw["calidad"].value_counts().sort_index()

        ax_d.bar(
            vc.index,
            vc.values,
            color="#9333ea",
            width=0.6
        )

        ax_d.set_xlabel(
            "Calidad",
            color="white"
        )

        ax_d.set_ylabel(
            "Frecuencia",
            color="white"
        )

        ax_d.set_title(
            "Distribución de Calidad",
            color="#c084fc"
        )

        ax_d.tick_params(colors="white")

        for spine in ax_d.spines.values():
            spine.set_edgecolor("#555")

        st.pyplot(
            fig_d,
            use_container_width=True
        )

        plt.close(fig_d)

    # ── DISTRIBUCIÓN POR COLOR ────────────────────────────────
    with col_color:

        st.markdown(
            '<p class="section-title">Distribución por Color</p>',
            unsafe_allow_html=True
        )

        fig_c, ax_c = plt.subplots(figsize=(5, 3.5))

        fig_c.patch.set_facecolor("#1a1025")
        ax_c.set_facecolor("none")

        color_counts = df_raw["color"].value_counts()

        wedge_colors = ["#9333ea", "#c084fc"]

        wedges, texts, autotexts = ax_c.pie(
            color_counts.values,
            labels=color_counts.index,
            autopct="%1.1f%%",
            colors=wedge_colors,
            textprops={"color": "white"},
        )

        for at in autotexts:
            at.set_color("white")

        ax_c.set_title(
            "Red vs White",
            color="#c084fc"
        )

        st.pyplot(
            fig_c,
            use_container_width=True
        )

        plt.close(fig_c)

    st.markdown("---")

    # ── CORRELACIÓN ────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">Correlación entre Variables</p>',
        unsafe_allow_html=True
    )

    df_num = df_raw.drop(
        ["botella_id", "color"],
        axis=1
    )

    corr = df_num.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))

    fig_corr.patch.set_facecolor("#1a1025")
    ax_corr.set_facecolor("#1a1025")

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="PRGn",
        center=0,
        ax=ax_corr,
        linewidths=0.3,
        linecolor="#333",
        annot_kws={
            "size": 7,
            "color": "white"
        }
    )

    ax_corr.tick_params(
        colors="white",
        labelsize=8
    )

    ax_corr.set_title(
        "Mapa de Correlación",
        color="#c084fc"
    )

    st.pyplot(
        fig_corr,
        use_container_width=True
    )

    plt.close(fig_corr)

    st.markdown("---")

    # ── DATAFRAME ──────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">Vista previa del Dataset</p>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df_raw.head(20),
        use_container_width=True,
        hide_index=True
    )

