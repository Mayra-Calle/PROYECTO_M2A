import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
 
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, LabelEncoder, KBinsDiscretizer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
 
sns.set(style="whitegrid")
 
st.set_page_config(page_title="Preparación de Datos - Matriz CAAM", layout="wide")
 
st.title("🧠 Unidad Didáctica 2 — Preparación de Datos")
st.caption("Aplicado sobre la Matriz CAAM (Evaluación Cognitiva en Adultos Mayores)")
 
# ==============================================================================
# 2.1 RECOLECCIÓN Y ADQUISICIÓN DE DATOS
# ==============================================================================
st.header("2.1 Recolección y adquisición de datos")
 
RUTA_POR_DEFECTO = "MatrizCaam-Version2__2___1___1__.xlsx"
archivo_subido = st.file_uploader(
    "Sube el Excel de la Matriz CAAM (o deja vacío para usar el archivo de la carpeta)",
    type=["xlsx"],
)
 
@st.cache_data
def cargar_datos(fuente):
    return pd.read_excel(fuente, sheet_name="Matriz de evaluación")
 
if archivo_subido is not None:
    df_original = cargar_datos(archivo_subido)
elif os.path.exists(RUTA_POR_DEFECTO):
    df_original = cargar_datos(RUTA_POR_DEFECTO)
else:
    st.warning("⚠️ No se encontró el Excel. Súbelo arriba para continuar.")
    st.stop()
 
col1, col2 = st.columns(2)
col1.metric("Filas (sujetos)", df_original.shape[0])
col2.metric("Columnas originales", df_original.shape[1])
 
with st.expander("Ver primeras filas del dataset original"):
    st.dataframe(df_original.head())
 
columnas_trabajo = [
    "Nombres y Apellidos", "Edad", "Género", "Estado civil", "Lateralidad",
    "Años de estudio", "Déficit sensorial",
    "TOTAL PREGUNTAS BASICAS TAREA 1", "TOTAL PREGUNTAS MUSICA  TAREA 2",
    "TOTAL PREGUNTAS MEMORIA  TAREA 3",
    "TOTAL PREGUNTAS Tarea 4 (T4) -Llamada Teléfonica",
    "TOTAL DE COMANDOS BASICOS", "Total Escala de Depresión Geriátrica\n",
    "PUNTUACION TOTAL:Esacala de blessed", "MMSE", "TOTAL ACE-R", "TOTAL WAIS",
]
df = df_original[columnas_trabajo].copy()
df = df.rename(columns={
    "Nombres y Apellidos": "nombre", "Edad": "edad", "Género": "genero",
    "Estado civil": "estado_civil", "Lateralidad": "lateralidad",
    "Años de estudio": "anios_estudio", "Déficit sensorial": "deficit_sensorial",
    "TOTAL PREGUNTAS BASICAS TAREA 1": "tarea1_basicas",
    "TOTAL PREGUNTAS MUSICA  TAREA 2": "tarea2_musica",
    "TOTAL PREGUNTAS MEMORIA  TAREA 3": "tarea3_memoria",
    "TOTAL PREGUNTAS Tarea 4 (T4) -Llamada Teléfonica": "tarea4_llamada",
    "TOTAL DE COMANDOS BASICOS": "total_comandos",
    "Total Escala de Depresión Geriátrica\n": "edg_total",
    "PUNTUACION TOTAL:Esacala de blessed": "blessed_total",
    "MMSE": "mmse", "TOTAL ACE-R": "ace_r_total", "TOTAL WAIS": "wais_total",
})
 
# Limpieza de ruido de texto (espacios): "Femenino" vs "Femenino "
columnas_texto = ["genero", "estado_civil", "lateralidad", "deficit_sensorial"]
for c in columnas_texto:
    df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
 
st.success(f"Subconjunto de trabajo: {df.shape[0]} filas x {df.shape[1]} columnas")
with st.expander("Ver subconjunto de trabajo (columnas renombradas y texto limpio)"):
    st.dataframe(df)
 
columnas_numericas = [
    "edad", "tarea1_basicas", "tarea2_musica", "tarea3_memoria",
    "tarea4_llamada", "total_comandos", "edg_total", "blessed_total",
    "mmse", "ace_r_total", "wais_total",
]
 
# ==============================================================================
# 2.2.1 DETECCIÓN Y MANEJO DE VALORES ATÍPICOS
# ==============================================================================
st.header("2.2.1 Detección y manejo de valores atípicos (outliers)")
 
def detectar_outliers_iqr(serie, factor=1.5):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - factor * iqr, q3 + factor * iqr
    return (serie < lim_inf) | (serie > lim_sup), lim_inf, lim_sup
 
factor_iqr = st.slider("Factor IQR (sensibilidad del método)", 1.0, 3.0, 1.5, 0.1)
 
resumen_outliers = {}
for col in columnas_numericas:
    mask, li, ls = detectar_outliers_iqr(df[col], factor_iqr)
    resumen_outliers[col] = mask.sum()
st.write("Cantidad de outliers detectados por variable:")
st.dataframe(pd.Series(resumen_outliers, name="n_outliers").to_frame())
 
col_boxplot = st.selectbox("Variable a graficar (boxplot):", columnas_numericas, index=columnas_numericas.index("ace_r_total"))
fig, ax = plt.subplots(figsize=(4, 5))
sns.boxplot(y=df[col_boxplot], ax=ax, color="skyblue")
ax.set_title(f"Boxplot: {col_boxplot}")
st.pyplot(fig)
 
with st.expander("Ver boxplots de TODAS las variables numéricas"):
    fig2, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(columnas_numericas):
        sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col, fontsize=9)
    for j in range(len(columnas_numericas), len(axes)):
        fig2.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig2)
 
# Tratamiento: capping (winsorización) a los límites del IQR
df_sin_outliers = df.copy()
for col in columnas_numericas:
    mask, li, ls = detectar_outliers_iqr(df[col], factor_iqr)
    df_sin_outliers[col] = df_sin_outliers[col].clip(lower=li, upper=ls)
st.info("✅ Outliers tratados mediante **capping** (recorte a los límites del IQR), sin eliminar sujetos.")
 
# ==============================================================================
# 2.2.2 DATOS FALTANTES Y RUIDO
# ==============================================================================
st.header("2.2.2 Datos faltantes y ruido")
 
faltantes = df_sin_outliers.isna().sum()
porcentaje = (df_sin_outliers.isna().mean() * 100).round(2)
tabla_faltantes = pd.DataFrame({"n_faltantes": faltantes, "%_faltantes": porcentaje}).sort_values("%_faltantes", ascending=False)
st.dataframe(tabla_faltantes)
 
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.heatmap(df_sin_outliers.isna(), cbar=False, cmap="viridis", ax=ax3)
ax3.set_title("Mapa de datos faltantes")
st.pyplot(fig3)
 
df_limpio = df_sin_outliers.copy()
df_limpio[columnas_numericas] = SimpleImputer(strategy="median").fit_transform(df_limpio[columnas_numericas])
df_limpio[columnas_texto + ["anios_estudio"]] = SimpleImputer(strategy="most_frequent").fit_transform(
    df_limpio[columnas_texto + ["anios_estudio"]]
)
st.success("✅ Valores numéricos imputados con la **mediana**; categóricos con la **moda**.")
 
df_limpio["mmse_suavizado"] = pd.cut(df_limpio["mmse"], bins=5).apply(lambda i: i.mid).astype(float)
with st.expander("Ejemplo de suavizado de ruido (bin means) en 'mmse'"):
    st.dataframe(df_limpio[["mmse", "mmse_suavizado"]].head(10))
 
# ==============================================================================
# 2.3.1 NORMALIZACIÓN
# ==============================================================================
st.header("2.3.1 Normalización")
 
df_norm = df_limpio.copy()
columnas_minmax = [f"{c}_minmax" for c in columnas_numericas]
columnas_zscore = [f"{c}_zscore" for c in columnas_numericas]
df_norm[columnas_minmax] = MinMaxScaler().fit_transform(df_norm[columnas_numericas])
df_norm[columnas_zscore] = StandardScaler().fit_transform(df_norm[columnas_numericas])
 
st.write("Comparación Min-Max vs Z-score (ejemplo con 'edad' y 'ace_r_total'):")
st.dataframe(df_norm[["edad", "edad_minmax", "edad_zscore",
                       "ace_r_total", "ace_r_total_minmax", "ace_r_total_zscore"]].head(8))
 
# ==============================================================================
# 2.3.2 DISCRETIZACIÓN
# ==============================================================================
st.header("2.3.2 Discretización")
 
bins_edad = [0, 64, 74, 84, 200]
etiquetas_edad = ["Adulto (<65)", "Joven-mayor (65-74)", "Mayor (75-84)", "Muy mayor (85+)"]
df_norm["edad_categoria"] = pd.cut(df_norm["edad"], bins=bins_edad, labels=etiquetas_edad)
 
n_bins = st.slider("Número de bins para discretización automática", 2, 6, 4)
df_norm["mmse_bin_ancho_igual"] = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform").fit_transform(df_norm[["mmse"]])
df_norm["ace_r_bin_frecuencia_igual"] = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile").fit_transform(df_norm[["ace_r_total"]])
 
c1, c2, c3 = st.columns(3)
with c1:
    st.write("Por dominio (edad):")
    st.bar_chart(df_norm["edad_categoria"].value_counts())
with c2:
    st.write("Ancho igual (mmse):")
    st.bar_chart(df_norm["mmse_bin_ancho_igual"].value_counts().sort_index())
with c3:
    st.write("Frecuencia igual (ace_r):")
    st.bar_chart(df_norm["ace_r_bin_frecuencia_igual"].value_counts().sort_index())
 
# ==============================================================================
# 2.3.3 CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ==============================================================================
st.header("2.3.3 Codificación de variables categóricas")
 
le = LabelEncoder()
df_norm["genero_cod"] = le.fit_transform(df_norm["genero"].astype(str))
st.write("Label Encoding de 'genero':", dict(zip(le.classes_, le.transform(le.classes_))))
 
columnas_onehot = ["estado_civil", "lateralidad", "deficit_sensorial", "edad_categoria"]
df_onehot = pd.get_dummies(df_norm[columnas_onehot].astype(str), prefix=columnas_onehot)
df_norm = pd.concat([df_norm, df_onehot], axis=1)
st.write(f"One-Hot Encoding generó **{df_onehot.shape[1]} columnas nuevas**:")
st.code(", ".join(df_onehot.columns))
 
# ==============================================================================
# 2.4 SELECCIÓN DE CARACTERÍSTICAS RELEVANTES
# ==============================================================================
st.header("2.4 Selección de características relevantes")
 
variable_objetivo = "ace_r_total_zscore"
caracteristicas = [c for c in columnas_zscore if c != variable_objetivo]
X = df_norm[caracteristicas]
y = df_norm[variable_objetivo]
 
matriz_corr = df_norm[columnas_zscore].corr()
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax4)
st.pyplot(fig4)
 
corr_objetivo = matriz_corr[variable_objetivo].drop(variable_objetivo).sort_values(key=abs, ascending=False)
st.write("Correlación de cada variable con `ace_r_total`:")
st.dataframe(corr_objetivo)
 
sel_var = VarianceThreshold(threshold=0.01)
sel_var.fit(X)
baja_varianza = X.columns[~sel_var.get_support()].tolist()
if baja_varianza:
    st.warning(f"Variables con varianza casi nula (efecto techo, poco informativas): {baja_varianza}")
 
k = st.slider("Número de mejores características a seleccionar (k)", 2, 7, 5)
selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X, y)
puntajes = pd.Series(selector.scores_, index=caracteristicas).sort_values(ascending=False)
mejores_k = puntajes.head(k).index.tolist()
 
st.write("Puntaje F (SelectKBest) por variable:")
st.bar_chart(puntajes)
st.success(f"🏆 Las {k} características más relevantes para predecir `ace_r_total`: {mejores_k}")
 
# ==============================================================================
# DESCARGA DEL DATASET FINAL
# ==============================================================================
st.header("📥 Descargar dataset preparado")
import io
buffer = io.BytesIO()
df_norm.to_excel(buffer, index=False)
st.download_button(
    "Descargar matriz_caam_preparada.xlsx",
    data=buffer.getvalue(),
    file_name="matriz_caam_preparada.xlsx",
)