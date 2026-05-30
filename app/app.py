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

/* ===== Títulos (Aislados para no romper los componentes de Streamlit) ===== */
h1, h2, h3, h4, h5, h6, .section-title {
    color: white !important;
}

/* ===== Cards de Métricas con Efecto Hover y Tooltip ===== */
.metric-card {
    background: linear-gradient(145deg, #111c2e, #162338);
    border: 1px solid #24344d;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    position: relative; /* Contexto de posicionamiento para el tooltip */
    cursor: help;
    transition: all 0.3s ease;
}

/* Efecto de elevación al pasar el cursor */
.metric-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
    z-index: 10 !important; /* Valor bajo para no tapar el botón de la barra lateral */
}

.metric-card h2 {
    color: #60a5fa !important;
    margin: 0;
}

.metric-card p {
    color: #d1d5db !important;
    margin: 0 !important; /* Evita que márgenes por defecto empujen el tooltip */
}

/* El cuadro de texto oculto por defecto - REESTRUCTURADO */
.metric-card .tooltip-text {
    visibility: hidden;
    width: 260px;
    background-color: #0b1320 !important;
    color: #e2e8f0 !important;
    text-align: left;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.85rem !important;
    line-height: 1.4 !important; /* Forzamos interlineado correcto para que no se encima el texto */
    display: block !important; /* Asegura que se comporte como un bloque sólido */
    
    /* Posicionamiento flotante absoluto */
    position: absolute;
    z-index: 100 !important; /* Rango seguro: mayor a las gráficas, menor a la barra lateral */
    top: 100%; /* Lo regresamos justo al límite inferior para mantener estabilidad */
    left: 50%;
    transform: translateX(-50%);
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
    opacity: 0;
    transition: opacity 0.2s ease;
}

/* Mostrar el cuadro flotante suavemente al hacer :hover */
.metric-card:hover .tooltip-text {
    visibility: visible;
    opacity: 1.0 !important;
}

/* Formato corregido para el título interno del tooltip */
.metric-card .tooltip-text b {
    color: #60a5fa !important;
    display: block !important; /* Obliga al texto descriptivo a empezar abajo, no al lado */
    margin-bottom: 6px !important; /* Separación limpia con la prosa */
    font-size: 0.9rem !important;
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

/* Estilo para los expanders de documentación técnica */
.stExpander {
    background-color: #111c2e !important;
    border: 1px solid #24344d !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}

</style>""", unsafe_allow_html=True)

# CARGA DE MODELO Y DATASET
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS / "modelo.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    with open(MODELS / "features.json") as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA)
    df["vino_bueno"] = (df["calidad"] >= 6).astype(int)
    df = df.drop(["botella_id", "calidad"], axis=1)
    df = pd.get_dummies(df, columns=["color"], drop_first=True)
    return df

model, scaler, FEATURES = load_artifacts()

# SIDEBAR (Diseño Premium con Título e Isotipo alineados)
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; padding-bottom: 10px; margin-top: -10px;">
        <svg width="45" height="45" viewBox="0 0 100 100" style="fill: #60a5fa; opacity: 0.95; transform: scaleY(-1);">
            <path d="M68.5,100 C73.1,100 75,98.1 75,93.5 L75,34.8 C75,30.3 72.4,26.4 69.1,23.5 L67.8,22.3 L67.8,7.3 C67.8,5.4 66.5,4.2 64.9,4.2 L59.1,4.2 C57.5,4.2 56.2,5.4 56.2,7.3 L56.2,22.3 L54.9,23.5 C51.6,26.4 49,30.3 49,34.8 L49,93.5 C49,98.1 50.9,100 55.5,100 Z M37.7,45.2 L27.3,45.2 C25.3,45.2 24.1,46.9 24.8,48.7 C26.7,53.8 28.1,61.4 31.2,67.6 C32.1,69.4 32.5,71.2 32.5,73.1 L32.5,89.6 L25.5,92.2 C24.1,92.7 24.7,94.8 26.2,94.8 L38.8,94.8 C40.3,94.8 40.9,92.7 39.5,92.2 L32.5,89.6 L32.5,73.1 C32.5,71.2 32.9,69.4 33.8,67.6 C36.9,61.4 38.3,53.8 40.2,48.7 C40.9,46.9 39.7,45.2 37.7,45.2 Z"/>
        </svg>
        <h2 style="margin: 0; font-size: 1.5rem; color: #ffffff; font-weight: 700; line-height: 1.2;">
            XGBoost Classifier
        </h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

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
    st.markdown("""
    **Modelo:** XGBoost  
    **Accuracy:** ~77.5 %  
    **Target:** Clasificación binaria  

    `0` = Vino Malo  
    `1` = Vino Bueno
    """)

# PÁGINA DE PREDICCIÓN
if page == " Predicción":

    st.title(" Predicción de Calidad de Vino Mediante XGBoost")
    st.info("Implementación de un modelo predictivo basado en el algoritmo XGBoost Classifier para la evaluación automatizada de muestras de vino. Al analizar variables críticas como los niveles de acidez, pH y balance alcohólico, el sistema identifica patrones complejos en la composición química para predecir la calificación del producto, ofreciendo un soporte analítico de alta precisión comparable con los estándares de catas de laboratorio tradicionales.")
    st.markdown("Ajusta las características del vino y obtén una predicción instantánea.")

    col_left, col_right = st.columns([2, 1], gap="large")

    # --- ENTRADAS DE USUARIO (COLUMNA IZQUIERDA) ---
    with col_left:
        st.markdown(
            '<p class="section-title"> Características del Vino</p>',
            unsafe_allow_html=True
        )

        preset = st.selectbox(
            " Configuración rápida",
            [
                "Personalizado",
                " Vino Excelente",
                " Vino Promedio",
                " Vino Malo"
            ]
        )

        if preset == " Vino Excelente":
            valores = {
                "acidez_fija": 7.0, "acidez_volatil": 0.25, "acido_citrico": 0.40,
                "azucar_residual": 2.0, "cloruros": 0.045, "so2_libre": 30.0,
                "so2_total": 90.0, "densidad": 0.9940, "ph": 3.30,
                "sulfatos": 0.75, "alcohol": 13.5, "color": "white"
            }
        elif preset == " Vino Malo":
            valores = {
                "acidez_fija": 15.0, "acidez_volatil": 1.4, "acido_citrico": 0.0,
                "azucar_residual": 12.0, "cloruros": 0.40, "so2_libre": 3.0,
                "so2_total": 15.0, "densidad": 1.0030, "ph": 2.8,
                "sulfatos": 0.35, "alcohol": 8.5, "color": "red"
            }
        else:
            valores = {
                "acidez_fija": 8.3, "acidez_volatil": 0.53, "acido_citrico": 0.27,
                "azucar_residual": 2.5, "cloruros": 0.087, "so2_libre": 15.9,
                "so2_total": 46.5, "densidad": 0.9967, "ph": 3.31,
                "sulfatos": 0.66, "alcohol": 10.4, "color": "red"
            }

        c1, c2, c3 = st.columns(3)

        with c1:
            acidez_fija = st.number_input("Acidez fija", 4.0, 16.0, valores["acidez_fija"], 0.1)
            acido_citrico = st.number_input("Ácido cítrico", 0.0, 1.0, valores["acido_citrico"], 0.01)
            so2_libre = st.number_input("SO₂ libre", 1.0, 70.0, valores["so2_libre"], 0.5)
            ph = st.number_input("pH", 2.7, 4.0, valores["ph"], 0.01)

        with c2:
            acidez_volatil = st.number_input("Acidez volátil", 0.1, 1.6, valores["acidez_volatil"], 0.01)
            azucar_residual = st.number_input("Azúcar residual", 0.9, 15.0, valores["azucar_residual"], 0.1)
            so2_total = st.number_input("SO₂ total", 6.0, 289.0, valores["so2_total"], 1.0)
            sulfatos = st.number_input("Sulfatos", 0.33, 2.0, valores["sulfatos"], 0.01)

        with c3:
            cloruros = st.number_input("Cloruros", 0.01, 0.6, valores["cloruros"], 0.001, format="%.3f")
            densidad = st.number_input("Densidad", 0.9900, 1.0040, valores["densidad"], 0.0001, format="%.4f")
            alcohol = st.number_input("Alcohol (%)", 8.0, 14.9, valores["alcohol"], 0.1)
            color = st.selectbox("Color del vino", ["red", "white"], index=0 if valores["color"] == "red" else 1)

    # ─────────────────────────────────────────
    # PROCESAMIENTO / LOGIC DEL MODELO
    # ─────────────────────────────────────────
    input_vals = {
        "acidez fija": acidez_fija, "acidez volatil": acidez_volatil, "acido citrico": acido_citrico,
        "azucar residual": azucar_residual, "cloruros": cloruros, "dioxido de azufre libre": so2_libre,
        "dioxido de azufre total": so2_total, "densidad": densidad, "pH": ph, "sulfatos": sulfatos,
        "alcohol": alcohol, FEATURES[-1]: 1 if color == "red" else 0
    }

    row_df = pd.DataFrame([input_vals])[FEATURES]
    row_scaled = scaler.transform(row_df)
    pred = model.predict(row_scaled)[0]
    probas = model.predict_proba(row_scaled)[0]

    prob_bueno = probas[1]
    prob_malo = probas[0]
    porcentaje_bueno = prob_bueno * 100

    if porcentaje_bueno >= 80.0:
        categoria_actual = "Excelente"
        color_alert = "#22c55e"
        descripcion_perfil = """
        **¿Qué significa este vino?** Los parámetros químicos ingresados corresponden a un **Vino de Alta Gama o Excelente**. 
        Presenta una armonía perfecta entre el grado alcohólico y la acidez. La baja acidez volátil garantiza 
        aromas limpios y complejos (frutales, florales o amaderados si tiene crianza), mientras que los niveles 
        de sulfatos aseguran una óptima preservación sin alterar el sabor.
        """
    elif 40.0 <= porcentaje_bueno < 80.0:
        categoria_actual = "Promedio"
        color_alert = "#eab308"
        descripcion_perfil = """
        **¿Qué significa este vino?** Los parámetros químicos corresponden a un **Vino Comercial Promedio**.  
        Es un vino equilibrado pero sin sobresalir. Sus componentes están en rangos aceptables: es frutal, 
        fresco y agradable para el consumo cotidiano. No posee defectos técnicos, pero carece de la concentración 
        o estructura requerida para competir en categorías premium.
        """
    else:
        categoria_actual = "Malo"
        color_alert = "#ef4444"
        descripcion_perfil = """
        **¿Qué significa este vino?** Los parámetros químicos indican un **Vino Defectuoso o Desequilibrado**.  
        Una probabilidad tan baja suele estar ligada a desbalances críticos: o bien la *acidez volátil* es muy alta 
        (lo que genera un sabor acético/avinagrado), los niveles de alcohol son insuficientes para sostener el cuerpo, 
        o la relación de dióxido de azufre es inadecuada, arriesgando la oxidación o generando olores reductores (químicos).
        """

    catalogo_vinos = {
        "Excelente": [
            {
                "nombre": "Catena Zapata Malbec Argentino", 
                "tipo": "Tinto (Mendoza, Argentina)", 
                "sabor": "Notas intensas de cassis, moca, especias dulces y un sutil toque floral de violetas.", 
                "textura": "Cuerpo pleno, asombrosamente denso pero con taninos sedosos y un final infinitamente largo."
            },
            {
                "nombre": "Cloudy Bay Sauvignon Blanc", 
                "tipo": "Blanco (Marlborough, Nueva Zelanda)", 
                "sabor": "Explosión frutal de maracuyá, lime madura, notas de pasto recién cortado y sutiles minerales.", 
                "textura": "Boca untuosa y concentrada, equilibrada por una acidez crujiente y vibrante que limpia el paladar."
            },
            {
                "nombre": "Marqués de Riscal Gran Reserva", 
                "tipo": "Tinto (Rioja, España)", 
                "sabor": "Frutas del bosque maduras entremezcladas con notas de cuero, tabaco, vainilla y maderas nobles.", 
                "textura": "Redondo, impecablemente estructurado, con un paso por boca aterciopelado y clásica elegancia."
            }
        ],
        "Promedio": [
            {
                "nombre": "Casillero del Diablo Cabernet", 
                "tipo": "Tinto (Valle Central, Chile)", 
                "sabor": "Frutas rojas frescas como ciruelas y cerezas, con toques muy ligeros de vainilla tostada.", 
                "textura": "Cuerpo medio, fácil de beber, con taninos amables y una acidez balanceada para consumo diario."
            },
            {
                "nombre": "Yellow Tail Chardonnay", 
                "tipo": "Blanco (Sureste de Australia)", 
                "sabor": "Notas directas de durazno, melón maduro y un fondo sutil a vainilla o mantequilla.", 
                "textura": "Fresco, de cuerpo ligero a medio, con un final ágil, limpio y ligeramente cremoso."
            },
            {
                "nombre": "Las Moras Syrah", 
                "tipo": "Tinto (San Juan, Argentina)", 
                "sabor": "Predominio de frutos negros frescos combinados con una nota especiada y pimienta negra sutil.", 
                "textura": "Entrada suave en boca, taninos muy maduros y un retrogusto corto pero muy agradable."
            }
        ],
        "Malo": [
            {
                "nombre": "Muestra con Picado Acético", 
                "tipo": "Defecto: Contaminación Bacteriana", 
                "sabor": "Fuerte presencia de ácido acético y acetato de etilo. Sabor punzante que recuerda al vinagre o quitaesmalte.", 
                "textura": "Áspero en la lengua, desvaído y con un ardor final desagradable debido al desbalance de la acidez."
            },
            {
                "nombre": "Muestra Oxidada (Aireada)", 
                "tipo": "Defecto: Pérdida de SO₂ / Exceso de Aire", 
                "sabor": "Ausencia total de fruta fresca. Notas a manzana rancia, frutos secos oxidados y un amargor plano.", 
                "textura": "Boca completamente plana, carente de frescura o vivacidad, dejando una sensación astringente y seca."
            },
            {
                "nombre": "Muestra con Quiebra Cuprosa", 
                "tipo": "Defecto: Exceso de Metales / Inestabilidad", 
                "sabor": "Sabor metálico persistente y notas químicas que enmascaran las propiedades organolépticas naturales.", 
                "textura": "Cuerpo acuoso o jaraboso artificial, acompañado frecuentemente de turbidez visual en la copa."
            }
        ]
    }

    # --- DESPLIEGUE EN PANEL CENTRAL (COL_LEFT) ---
    with col_left:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title"> Perfil Sensorial Estimado</p>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: #1e293b; border-left: 5px solid {color_alert}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="margin: 0; color: white;">Perfil Detectado en Laboratorio: <span style="color: {color_alert};">{categoria_actual}</span></h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(descripcion_perfil)
        st.markdown("---")
        
        st.markdown(f'###  Tu vino tiene un perfil ({categoria_actual}) y es muy similar a:')
        
        rec_cols = st.columns(3)
        for idx, vino in enumerate(catalogo_vinos[categoria_actual]):
            with rec_cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #111c2e, #162338); border: 1px solid #24344d; border-radius: 12px; padding: 15px; min-height: 220px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                    <h4 style="margin: 0 0 5px 0; color: #60a5fa;">{vino['nombre']}</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.85rem; color: #a1a1aa;"><i>Estilo: {vino['tipo']}</i></p>
                    <p style="margin: 0 0 6px 0; font-size: 0.88rem; color: #e2e8f0;"><b>Sabor:</b> {vino['sabor']}</p>
                    <p style="margin: 0; font-size: 0.88rem; color: #e2e8f0;"><b>Textura:</b> {vino['textura']}</p>
                </div>
                """, unsafe_allow_html=True)

        # =========================================================================
        #  GLOSARIO CIENTÍFICO DE LABORATORIO ENOLÓGICO
        # =========================================================================
        st.markdown("---")
        st.markdown('<p class="section-title">Glosario Científico y Métodos de Obtención</p>', unsafe_allow_html=True)
        st.markdown("Aprende qué mide cada variable química del panel superior y cómo se analiza en una bodega real:")

        g1, g2 = st.columns(2)

        with g1:
            with st.expander("Acidez Fija, Volátil y pH"):
                st.markdown("""
                * **Acidez Fija:** Mide los ácidos naturales de la uva (tartárico, málico). Aporta frescura.  
                  *↳ Obtención:* Titulación ácido-base en laboratorio con hidróxido de sodio ($NaOH$).
                * **Acidez Volátil:** Mide el ácido acético (vinagre). Un exceso indica daño bacteriano.  
                  *↳ Obtención:* Destilación por arrastre de vapor (Aparato Cash o Garcia-Tena).
                * **Ácido Cítrico:** Ácido orgánico nativo menor. Se usa para sutiles correcciones.  
                  *↳ Obtención:* Análisis espectrofotométrico mediante kits enzimáticos.
                * **pH:** Fuerza de la acidez libre en escala logarítmica. Protege al vino de bacterias.  
                  *↳ Obtención:* Lectura directa con un potenciómetro (pH-metro) calibrado.
                """)
            
            with st.expander(" Conservantes: SO₂ Libre y Total"):
                st.markdown("""
                * **SO₂ Libre:** Gas sulfito activo disuelto. Es el escudo antioxidante y antimicrobiano real del vino.  
                  *↳ Obtención:* Método Ripper clásico (titulación yodométrica directa).
                * **SO₂ Total:** Suma del sulfito libre y el ligado a moléculas. Controla restricciones legales.  
                  *↳ Obtención:* Acidificación e hidrólisis alcalina previa, seguida por el método Ripper.
                """)

        with g2:
            with st.expander(" Cuerpo: Azúcar, Densidad y Alcohol"):
                st.markdown("""
                * **Azúcar Residual:** Glucosa y fructosa remanentes que la levadura no fermentó.  
                  *↳ Obtención:* Métodos químicos reductores (Rebelein/Fehling) o análisis enzimáticos.
                * **Densidad:** Relación masa/volumen. Cae por el alcohol y sube por los azúcares.  
                  *↳ Obtención:* Densímetros de oscilación electrónica o hidrómetros de vidrio.
                * **Alcohol (%):** Concentración volumétrica de etanol purificado por fermentación.  
                  *↳ Obtención:* Destilación del vino seguida de alcoholimetría, o mediante sensores infrarrojos (FTIR).
                """)
            
            with st.expander("Composición Estructural y Visual"):
                st.markdown("""
                * **Cloruros:** Cantidad de sales minerales. Varía según el suelo del viñedo y vientos marinos.  
                  *↳ Obtención:* Titulación argentométrica mediante el método de Mohr (Nitrato de Plata).
                * **Sulfatos:** Sales de sulfato de potasio. Influyen en la estabilidad y retrogusto.  
                  *↳ Obtención:* Gravimetría por precipitación de sulfato de bario o turbidimetría.
                * **Color del Vino:** Tipo de vinificación en base al contacto con hollejos (pieles).  
                  *↳ Obtención:* Clasificación empírica inicial o medición espectrofotométrica de intensidad colorante ($420/520/620\text{ nm}$).
                """)


    # --- CONTENEDOR DE RESULTADOS TÉCNICOS (COLUMNA DERECHA) ---
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

        st.metric(" Probabilidad Bueno", f"{prob_bueno*100:.1f}%")
        st.metric(" Probabilidad Malo", f"{prob_malo*100:.1f}%")

        st.markdown("---")
        st.markdown("###  Variables más importantes")

        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True).tail(5)

        fig_fi, ax_fi = plt.subplots(figsize=(5, 3))
        fig_fi.patch.set_facecolor("#0f172a")
        ax_fi.set_facecolor("#0f172a")

        ax_fi.barh(feat_imp.index, feat_imp.values, color="#3b82f6")
        ax_fi.tick_params(colors="white")
        ax_fi.set_title("Top Variables", color="white")

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

    # Bloque de tarjetas interactivas con cuadro descriptivo flotante (Hover)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f'''
            <div class="metric-card">
                <h2>{acc*100:.1f}%</h2>
                <p>Accuracy</p>
                <div class="tooltip-text">
                    <b>Exactitud General:</b><br>
                    Porcentaje total de predicciones correctas (tanto aciertos de vinos buenos como malos) del algoritmo sobre el total de pruebas de laboratorio.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with k2:
        st.markdown(f'''
            <div class="metric-card">
                <h2>{auc:.3f}</h2>
                <p>ROC-AUC</p>
                <div class="tooltip-text">
                    <b>Área Bajo la Curva ROC:</b><br>
                    Mide el poder de separación del clasificador. Un valor de 1.0 indica un diagnóstico perfecto entre muestras excelentes y defectuosas.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with k3:
        st.markdown(f'''
            <div class="metric-card">
                <h2>{rep["1"]["precision"]*100:.1f}%</h2>
                <p>Precision (Bueno)</p>
                <div class="tooltip-text">
                    <b>Precisión de Calidad:</b><br>
                    De todas las muestras marcadas por el XGBoost como "Vino Bueno", este porcentaje representa las que realmente resultaron ser correctas. Evita falsas alarmas.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with k4:
        st.markdown(f'''
            <div class="metric-card">
                <h2>{rep["1"]["recall"]*100:.1f}%</h2>
                <p>Recall (Bueno)</p>
                <div class="tooltip-text">
                    <b>Sensibilidad Enológica:</b><br>
                    Mide la habilidad del modelo para detectar todos los vinos buenos reales disponibles dentro del dataset. Evita omitir lotes premium.
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    col_cm, col_roc = st.columns(2, gap="large")

    with col_cm:
        st.markdown('<p class="section-title">Matriz de Confusión</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#1a1025")
        ax_cm.set_facecolor("#1a1025")

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Malo", "Bueno"], yticklabels=["Malo", "Bueno"],
            ax=ax_cm, linewidths=0.5, linecolor="#333",
            annot_kws={"color": "white", "size": 14}
        )
        ax_cm.set_xlabel("Predicción", color="white")
        ax_cm.set_ylabel("Valor Real", color="white")
        ax_cm.tick_params(colors="white")
        ax_cm.set_title("Confusión", color="#ffffff")

        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col_roc:
        st.markdown('<p class="section-title">Curva ROC</p>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        fig_roc.patch.set_facecolor("#1a1025")
        ax_roc.set_facecolor("#1a1025")

        ax_roc.plot(fpr, tpr, color="#9333ea", lw=2, label=f"AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
        ax_roc.set_xlabel("Tasa Falsos Positivos", color="white")
        ax_roc.set_ylabel("Tasa Verdaderos Positivos", color="white")
        ax_roc.set_title("Curva ROC", color="#c084fc")
        ax_roc.tick_params(colors="white")
        ax_roc.legend(labelcolor="white", framealpha=0)

        for spine in ax_roc.spines.values():
            spine.set_edgecolor("#555")

        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    st.markdown("---")
    st.markdown('<p class="section-title">Importancia de Variables</p>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance", ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    fig_imp.patch.set_facecolor("#1a1025")
    ax_imp.set_facecolor("#1a1025")

    colors_bar = ["#9333ea" if i >= len(feat_df) - 3 else "#6b21a8" for i in range(len(feat_df))]
    ax_imp.barh(feat_df["Feature"], feat_df["Importance"], color=colors_bar)
    ax_imp.set_xlabel("Importancia", color="white")
    ax_imp.set_title("Importancia de variables", color="#c084fc")
    ax_imp.tick_params(colors="white")

    for spine in ax_imp.spines.values():
        spine.set_edgecolor("#555")

    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — ANÁLISIS DEL DATASET (REPARADA)
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Análisis del Dataset":

    st.title(" Análisis Exploratorio del Dataset")
    df_raw = pd.read_csv(DATA)

    # Tarjetas interactivas con Tooltips descriptivos aplicados a la metadata de exploración
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'''
            <div class="metric-card">
                <h2>{len(df_raw):,}</h2>
                <p>Total muestras</p>
                <div class="tooltip-text">
                    <b>Registros Totales:</b><br>
                    Volumen completo de corridas fisicoquímicas evaluadas en el conjunto de datos de origen.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    with s2:
        buenos = (df_raw["calidad"] >= 6).sum()
        st.markdown(f'''
            <div class="metric-card">
                <h2>{buenos:,}</h2>
                <p>Vinos Buenos (≥6)</p>
                <div class="tooltip-text">
                    <b>Clase Positiva (1):</b><br>
                    Muestras cuya evaluación sensorial experta otorgó una nota idónea e igual o mayor a 6 puntos.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with s3:
        malos = (df_raw["calicsad"] < 6).sum() if "calicsad" in df_raw else (df_raw["calidad"] < 6).sum()
        st.markdown(f'''
            <div class="metric-card">
                <h2>{malos:,}</h2>
                <p>Vinos Malos (&lt;6)</p>
                <div class="tooltip-text">
                    <b>Clase Negativa (0):</b><br>
                    Lotes comerciales que registraron fallas analíticas o baja puntuación organoléptica menor a 6.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with s4:
        st.markdown(f'''
            <div class="metric-card">
                <h2>{df_raw["calidad"].mean():.2f}</h2>
                <p>Calidad promedio</p>
                <div class="tooltip-text">
                    <b>Media del Target:</b><br>
                    Nota promedio ponderada de la base completa del viñedo en la escala sensorial clásica.
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    col_dist, col_color = st.columns(2, gap="large")

    with col_dist:
        st.markdown('<p class="section-title">Distribución de Calidad</p>', unsafe_allow_html=True)
        fig_d, ax_d = plt.subplots(figsize=(5, 3.5))
        fig_d.patch.set_facecolor("#1a1025")
        ax_d.set_facecolor("#1a1025")

        vc = df_raw["calidad"].value_counts().sort_index()
        ax_d.bar(vc.index, vc.values, color="#9333ea", width=0.6)
        ax_d.set_xlabel("Calidad", color="white")
        ax_d.set_ylabel("Frecuencia", color="white")
        ax_d.set_title("Distribución de Calidad", color="#c084fc")
        ax_d.tick_params(colors="white")

        for spine in ax_d.spines.values():
            spine.set_edgecolor("#555")

        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

    with col_color:
        st.markdown('<p class="section-title">Distribución por Color</p>', unsafe_allow_html=True)
        fig_c, ax_c = plt.subplots(figsize=(5, 3.5))
        fig_c.patch.set_facecolor("#1a1025")
        ax_c.set_facecolor("none")

        color_counts = df_raw["color"].value_counts()
        wedge_colors = ["#9333ea", "#c084fc"]

        wedges, texts, autotexts = ax_c.pie(
            color_counts.values, labels=color_counts.index, autopct="%1.1f%%",
            colors=wedge_colors, textprops={"color": "white"},
        )
        for at in autotexts:
            at.set_color("white")

        ax_c.set_title("Rojo Vs Blanco", color="#c084fc")
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    st.markdown("---")
    st.markdown('<p class="section-title">Correlación entre Variables</p>', unsafe_allow_html=True)
    df_num = df_raw.drop(["botella_id", "color"], axis=1)
    corr = df_num.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    fig_corr.patch.set_facecolor("#3B3363")
    ax_corr.set_facecolor("#ffffff")

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="PRGn", center=0, ax=ax_corr,
        linewidths=0.3, linecolor="#333", annot_kws={"size": 7, "color": "black"}
    )
    ax_corr.tick_params(colors="white", labelsize=8)
    ax_corr.set_title("Mapa de Correlación", color="#c084fc")

    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)

    st.markdown("---")
    st.markdown('<p class="section-title">Vista previa del Dataset</p>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)

