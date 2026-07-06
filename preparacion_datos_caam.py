import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

# Configuración de página
st.set_page_config(page_title="Análisis de Datos - Matriz CAAM", layout="wide")

st.title("🧠 Evaluación Cognitiva en Adultos Mayores (Matriz CAAM)")
st.caption("Proyecto de Carga, Exploración, Limpieza, Transformación y Visualización de Datos")

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
st.header("1. Carga de datos")

RUTA_POR_DEFECTO = "MatrizCaam-Version2.xlsx"
archivo_subido = st.file_uploader(
    "Sube el archivo de la Matriz CAAM (.xlsx)",
    type=["xlsx"],
)

@st.cache_data
def cargar_datos(fuente):
    # Carga inicial del archivo original
    return pd.read_excel(fuente, sheet_name="Matriz de evaluación")

if archivo_subido is not None:
    df_original = cargar_datos(archivo_subido)
elif os.path.exists(RUTA_POR_DEFECTO):
    df_original = cargar_datos(RUTA_POR_DEFECTO)
else:
    st.warning("⚠️ No se encontró el archivo por defecto. Súbelo para continuar.")
    st.stop()

# Mostrar número de registros y columnas
col1, col2 = st.columns(2)
col1.metric("Número de Registros (Filas)", df_original.shape[0])
col2.metric("Número de Columnas", df_original.shape[1])

st.subheader("DataFrame Original (Primeros registros):")
st.dataframe(df_original.head())

# Seleccionamos y renombramos un subconjunto para trabajar cómodamente en los siguientes puntos

# La columna de profesión puede venir con distinto nombre/tildes según el archivo
# (ej. "Profesion", "Profesión ", "PROFESION"), así que la buscamos por coincidencia
# en vez de asumir un nombre fijo, para no depender de la ortografía exacta.
def normalizar(texto):
    texto = str(texto).strip().lower()
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    for original, simple in reemplazos.items():
        texto = texto.replace(original, simple)
    return texto

columna_profesion_real = next(
    (c for c in df_original.columns if "profes" in normalizar(c)), None
)

if columna_profesion_real is None:
    st.warning(
        "⚠️ No se encontró una columna de 'Profesión' en el archivo cargado. "
        "El gráfico de profesión se mostrará como 'No especificado' para todos los registros. "
        f"Columnas disponibles en el archivo: {list(df_original.columns)}"
    )

columnas_trabajo = [
    "Nombres y Apellidos", "Edad", "Género", "Estado civil", "Lateralidad",
    "Años de estudio", "Déficit sensorial", "MMSE", "TOTAL ACE-R", "TOTAL WAIS"
]
if columna_profesion_real is not None:
    columnas_trabajo.append(columna_profesion_real)

df = df_original[columnas_trabajo].copy()
df = df.rename(columns={
    "Nombres y Apellidos": "nombre", "Edad": "edad", "Género": "genero",
    "Estado civil": "estado_civil", "Lateralidad": "lateralidad",
    "Años de estudio": "anios_estudio", "Déficit sensorial": "deficit_sensorial",
    "MMSE": "mmse", "TOTAL ACE-R": "ace_r_total", "TOTAL WAIS": "wais_total"
})
if columna_profesion_real is not None:
    df = df.rename(columns={columna_profesion_real: "profesion"})
else:
    df["profesion"] = np.nan

# ==============================================================================
# 2. EXPLORACIÓN DE DATOS
# ==============================================================================
st.header("2. Exploración de datos")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Primeros Registros", "Información e Inspección",
    "Estadísticas Descriptivas", "Valores Únicos", "Valores Nulos"
])

with tab1:
    st.write("**Primeros registros del dataset (`df.head()`):**")
    st.dataframe(df.head(10))

with tab2:
    st.write("**Tipos de datos por columna (`df.dtypes`):**")
    st.dataframe(df.dtypes.astype(str).to_frame(name="Tipo de Dato"))

    st.write("**Información General Estructurada (`df.info()`):**")
    buffer_info = io.StringIO()
    df.info(buf=buffer_info)
    st.text(buffer_info.getvalue())

with tab3:
    st.write("**Estadísticas descriptivas de variables numéricas (`df.describe()`):**")
    st.dataframe(df.describe())

with tab4:
    st.write("**Cantidad de valores únicos por columna (`df.nunique()`):**")
    st.dataframe(df.nunique().to_frame(name="Valores Únicos"))

with tab5:
    st.write("**Conteo de valores nulos por columna (`df.isnull().sum()`):**")
    st.dataframe(df.isnull().sum().to_frame(name="Valores Nulos"))

# ==============================================================================
# 3. LIMPIEZA DE DATOS
# ==============================================================================
st.header("3. Limpieza de datos")

df_limpio = df.copy()

# Técnica 1: Renombrar columnas (Ya realizada arriba de forma masiva con .rename())

# Técnica 2: Corregir tipos de datos y quitar espacios en blanco (Ruido de texto)
# Solo aplicamos strip/limpieza de texto a las columnas categóricas, NUNCA a
# las numéricas (ese era el bug: convertir todo el df a texto rompía edad,
# mmse, ace_r_total y wais_total para los gráficos y el groupby posteriores).
columnas_texto = ["genero", "estado_civil", "lateralidad", "deficit_sensorial", "profesion"]
for c in columnas_texto:
    df_limpio[c] = df_limpio[c].astype(str).str.strip().replace({"nan": np.nan})

# Aseguramos que las columnas numéricas realmente sean numéricas
# (por si vienen como texto desde Excel, ej. "24" en vez de 24)
# NOTA: "anios_estudio" se deja fuera de esta conversión forzada, porque en
# algunos archivos esa columna no trae un número de años sino texto descriptivo
# (ej. "Primaria", "Secundaria", "Superior"). Forzarla con pd.to_numeric
# convertía todo ese texto en NaN silenciosamente (por eso salía 100% "Sin Datos"
# en el gráfico de nivel educativo). Se limpia como texto más abajo y la
# clasificación de nivel educativo soporta ambos formatos (número o texto).
columnas_numericas = ["edad", "mmse", "ace_r_total", "wais_total"]
for c in columnas_numericas:
    df_limpio[c] = pd.to_numeric(df_limpio[c], errors="coerce")

df_limpio["anios_estudio"] = df_limpio["anios_estudio"].astype(str).str.strip().replace({"nan": np.nan})

with st.expander("🔍 Diagnóstico: valores originales de 'Años de estudio'"):
    st.write(
        "Estos son los valores únicos que trae la columna en tu archivo, "
        "para verificar si son números o texto descriptivo:"
    )
    st.write(sorted(df_limpio["anios_estudio"].dropna().unique().tolist(), key=str))

# Técnica 2b: Eliminar registros sin ninguna información relevante
# Hay filas que llegan completamente vacías (sin nombre, sin edad, sin ningún
# puntaje). No aportan nada al análisis, así que se eliminan aquí, ANTES de
# rellenar los categóricos con "No especificado" (si se hiciera después, esas
# filas ya no se verían "vacías" y no se podrían detectar).
columnas_informativas = [c for c in df_limpio.columns if c != "nombre"]
filas_antes_vacias = df_limpio.shape[0]
df_limpio = df_limpio.dropna(subset=columnas_informativas, how="all")
filas_vacias_eliminadas = filas_antes_vacias - df_limpio.shape[0]

# Técnica 3: Reemplazar / Imputar nulos (fillna)
# IMPORTANTE: las columnas NUMÉRICAS (edad, anios_estudio, mmse, ace_r_total,
# wais_total) NO se imputan con la media/mediana. Rellenar un puntaje de MMSE
# o de ACE-R que el paciente nunca respondió con un promedio inventa un dato
# clínico que no existe y distorsiona los gráficos (por eso antes aparecía una
# columna entera de puntos en MMSE=0 en la dispersión). Esos nulos se dejan
# como NaN: representan "no evaluado / sin dato", y los gráficos y agrupaciones
# los excluyen automáticamente en sus cálculos.
# Solo se imputan las columnas de TEXTO/categóricas, con una etiqueta explícita
# (no un valor inventado), para que puedan graficarse sin perder la categoría.
df_limpio["genero"] = df_limpio["genero"].fillna("No especificado")
df_limpio["estado_civil"] = df_limpio["estado_civil"].fillna("No especificado")
df_limpio["profesion"] = df_limpio["profesion"].fillna("No especificado")

# Técnica 4: Eliminar duplicados (drop_duplicates)
filas_antes = df_limpio.shape[0]
df_limpio = df_limpio.drop_duplicates()
filas_despues = df_limpio.shape[0]

# Técnica 5: Eliminar Outliers (Capping por IQR) sobre Edad
q1, q3 = df_limpio["edad"].quantile(0.25), df_limpio["edad"].quantile(0.75)
iqr = q3 - q1
df_limpio["edad"] = df_limpio["edad"].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

# NOTA: se eliminó la línea `df_limpio = df_limpio.astype(str).replace(...)`
# que existía en la versión original. Esa línea convertía TODO el DataFrame
# (incluidas las columnas numéricas) a texto, lo cual rompía silenciosamente
# el histograma de edad, el scatter de MMSE vs ACE-R y el groupby de la
# sección de Transformación (que tenía que "reparar" los tipos a último
# momento con pd.to_numeric). Con columnas_texto y columnas_numericas bien
# separadas desde el inicio, ya no hace falta ese parche.

st.success("✅ Se han aplicado 6 técnicas de limpieza (Renombrar, Corrección de tipos/stripping, "
           "Eliminación de registros sin ninguna información relevante, Etiquetado de nulos "
           "categóricos con `fillna('No especificado')` sin inventar valores numéricos, "
           "Eliminación de duplicados con `drop_duplicates()` y Control de outliers en Edad).")
st.write(f"Filas eliminadas por estar completamente vacías: {filas_vacias_eliminadas}")
st.write(f"Filas eliminadas por duplicación: {filas_antes - filas_despues}")

with st.expander("Ver Dataset Limpio"):
    st.dataframe(df_limpio)
    st.caption("Tipos de dato tras la limpieza:")
    st.dataframe(df_limpio.dtypes.astype(str).to_frame(name="Tipo de Dato"))

# ==============================================================================
# 4. TRANSFORMACIÓN DE DATOS
# ==============================================================================
st.header("4. Transformación de datos")

df_transformado = df_limpio.copy()

# Transformación 1: Crear nuevas columnas (Clasificación del MMSE de acuerdo al puntaje)
def clasificar_mmse(score):
    if pd.isna(score):
        return "Sin Datos"
    score = float(score)
    if score >= 24:
        return "Normal"
    elif score >= 20:
        return "Deterioro Cognitivo Leve"
    else:
        return "Deterioro Cognitivo Moderado/Severo"

df_transformado["estado_cognitivo"] = df_transformado["mmse"].apply(clasificar_mmse)

# Transformación 1b: Agrupar profesión en categorías amplias
# (Jubilado/a, Ama de casa, Profesiones varias) para que el gráfico de barras
# sea legible en vez de mostrar decenas de profesiones individuales distintas.
def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    for original, simple in reemplazos.items():
        texto = texto.replace(original, simple)
    return texto

def clasificar_profesion(valor):
    if pd.isna(valor) or normalizar_texto(valor) in ("no especificado", "nan", ""):
        return "No especificado"
    texto = normalizar_texto(valor)
    if "jubil" in texto or "pension" in texto or "retirad" in texto:
        return "Jubilado/a"
    elif "ama de casa" in texto or texto == "hogar" or "quehacer" in texto:
        return "Ama de casa"
    else:
        return "Profesiones varias"

df_transformado["profesion_agrupada"] = df_transformado["profesion"].apply(clasificar_profesion)

# Transformación 1c: Clasificar años de estudio en niveles educativos
# Soporta dos formatos posibles en el archivo original:
#  a) Numérico -> años de estudio (Básico: 0-10, Medio: 11-13, Superior: 14+),
#     siguiendo el esquema ecuatoriano (10 años de Educación General Básica +
#     3 años de Bachillerato).
#  b) Texto descriptivo -> ej. "Primaria", "Secundaria/Bachillerato", "Superior".
def clasificar_nivel_educativo(valor):
    if pd.isna(valor):
        return "Sin Datos"
    # 1) Intentar interpretarlo como número de años
    try:
        anios = float(valor)
        if anios <= 10:
            return "Básico"
        elif anios <= 13:
            return "Medio"
        else:
            return "Superior"
    except (ValueError, TypeError):
        pass
    # 2) Si no es numérico, interpretarlo como texto descriptivo del nivel
    texto = normalizar_texto(valor)
    if any(p in texto for p in ["primari", "basic", "inicial"]):
        return "Básico"
    elif any(p in texto for p in ["secundari", "bachiller", "medio"]):
        return "Medio"
    elif any(p in texto for p in ["superior", "universi", "tercer nivel", "cuarto nivel", "posgrado", "postgrado"]):
        return "Superior"
    else:
        return "Sin Datos"

df_transformado["nivel_educativo"] = df_transformado["anios_estudio"].apply(clasificar_nivel_educativo)

# Transformación 2: Ordenar registros (sort_values por Edad de forma descendente)
df_transformado = df_transformado.sort_values(by="edad", ascending=False)

# Transformación 3: Filtrar información (Filtro interactivo en Streamlit)
st.subheader("Filtrar registros por Género")
generos_disponibles = df_transformado["genero"].unique().tolist()
genero_seleccionado = st.selectbox("Selecciona un género para filtrar la tabla:", ["Todos"] + generos_disponibles)

if genero_seleccionado != "Todos":
    df_filtrado = df_transformado[df_transformado["genero"] == genero_seleccionado]
else:
    df_filtrado = df_transformado

st.dataframe(df_filtrado)

# Transformación 4: Agrupar datos (groupby por Estado Cognitivo recién creado)
st.subheader("Agrupación (`groupby`): Promedio de edad y puntajes por Estado Cognitivo")
# Ya no es necesario forzar pd.to_numeric aquí: las columnas ya vienen numéricas
# desde la sección de limpieza. Se deja el groupby directo.
df_agrupado = df_transformado.groupby("estado_cognitivo")[["edad", "mmse", "ace_r_total", "wais_total"]].mean().reset_index()
st.dataframe(df_agrupado)

# ==============================================================================
# 5. VISUALIZACIÓN
# ==============================================================================
st.header("5. Visualización de datos")

col_g1, col_g2 = st.columns(2)
col_g3, col_g4 = st.columns(2)

# Gráfico 1: Pastel de Estado Civil (Matplotlib)
with col_g1:
    st.subheader("1. Distribución de Estado Civil")
    conteo_estado_civil = df_transformado["estado_civil"].value_counts()
    paleta_estado_civil = sns.color_palette("Set3", n_colors=len(conteo_estado_civil))
    fig_pie_civil, ax_pie_civil = plt.subplots(figsize=(4, 4))
    ax_pie_civil.pie(
        conteo_estado_civil, labels=conteo_estado_civil.index,
        autopct='%1.1f%%', startangle=90, colors=paleta_estado_civil
    )
    ax_pie_civil.axis('equal')
    st.pyplot(fig_pie_civil)
    plt.close(fig_pie_civil)

# Gráfico 2: Histograma (Matplotlib/Seaborn)
with col_g2:
    st.subheader("2. Histograma: Distribución de la Edad")
    serie_edad = df_transformado["edad"].dropna()
    if serie_edad.empty:
        st.info("No hay datos disponibles de edad para graficar.")
    else:
        fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
        sns.histplot(serie_edad, bins=10, kde=True, color="teal", ax=ax_hist)
        ax_hist.set_xlabel("Edad")
        ax_hist.set_ylabel("Frecuencia")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

# Gráfico 3: Barras de Nivel Educativo (Básico / Medio / Superior)
with col_g3:
    st.subheader("3. Distribución de Nivel Educativo")
    orden_nivel = ["Básico", "Medio", "Superior", "Sin Datos"]
    conteo_nivel = df_transformado["nivel_educativo"].value_counts().reindex(
        [n for n in orden_nivel if n in df_transformado["nivel_educativo"].unique()]
    )
    fig_estudio, ax_estudio = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x=conteo_nivel.index, y=conteo_nivel.values, palette="crest", ax=ax_estudio)
    ax_estudio.set_xlabel("Nivel Educativo")
    ax_estudio.set_ylabel("Cantidad de personas")
    st.pyplot(fig_estudio)
    plt.close(fig_estudio)

# Gráfico 4: Pastel (Matplotlib)
with col_g4:
    st.subheader("4. Distribución de Género")
    conteo_genero = df_transformado["genero"].value_counts()
    paleta_genero = sns.color_palette("pastel", n_colors=len(conteo_genero))
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie(conteo_genero, labels=conteo_genero.index, autopct='%1.1f%%', startangle=90, colors=paleta_genero)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)
    plt.close(fig_pie)

# Gráfico 5: Barras horizontales de Profesión (agrupada)
st.subheader("5. Distribución de Profesión")
orden_profesion = ["Jubilado/a", "Ama de casa", "Profesiones varias", "No especificado"]
conteo_profesion = df_transformado["profesion_agrupada"].value_counts().reindex(
    [p for p in orden_profesion if p in df_transformado["profesion_agrupada"].unique()]
)
fig_prof, ax_prof = plt.subplots(figsize=(9, max(3, 0.4 * len(conteo_profesion))))
sns.barplot(x=conteo_profesion.values, y=conteo_profesion.index, palette="viridis", ax=ax_prof)
ax_prof.set_xlabel("Cantidad de personas")
ax_prof.set_ylabel("Profesión")
st.pyplot(fig_prof)
plt.close(fig_prof)

# ==============================================================================
# 6. EXPORTACIÓN DE DATOS
# ==============================================================================
st.header("6. Exportación de datos")

# Comando sugerido ejecutado localmente de fondo en el servidor
df_transformado.to_csv("datos_limpios.csv", index=False)

# Opción de descarga interactiva para el usuario en la interfaz de Streamlit
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df_transformado.to_excel(writer, sheet_name='Datos Limpios', index=False)

st.info("💾 El archivo `datos_limpios.csv` ha sido guardado localmente mediante el comando estricto de la rúbrica.")

st.download_button(
    label="📥 Descargar datos_limpios.xlsx (Formato Web)",
    data=buffer.getvalue(),
    file_name="datos_limpios.xlsx",
    mime="application/vnd.ms-excel"
)

# Nota de cumplimiento de interfaz
st.sidebar.title("Menú de Navegación")
st.sidebar.info("""
**Secciones del Proyecto:**
1. Carga de Datos
2. Exploración (EDA)
3. Limpieza de Datos
4. Transformación
5. Visualización Estructurada
6. Exportación final
""")