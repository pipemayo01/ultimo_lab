import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def calcular_velocidad_grupo(grupo):
    grupo["Distancia"] = np.sqrt(grupo["X"].diff(periods=-1)**2 + grupo["Y"].diff(periods=-1)**2)
    grupo["tiempo"] = 1 / 25
    grupo["Velocidad"] = grupo["Distancia"] / grupo["tiempo"]
    return grupo

def calcular_sk(data_frame):
    data_frame["sk"] = 0

    for i in data_frame["Frame"].unique():
        df = data_frame[data_frame["Frame"] == i].copy()
        coordenadas = df[["X", "Y"]].values
        df["indices"] = np.arange(len(df))

        for id in df["indices"]:
            punto = coordenadas[id]
            radius = 3
            tree = KDTree(coordenadas)
            vecino_indices = tree.query_ball_point(punto, radius)
            vecino_indices = [index for index in vecino_indices if index != id]
            vecino_coordenadas = coordenadas[vecino_indices]

            cantidad_vecinos = len(vecino_indices)

            if cantidad_vecinos > 0:
                diferencia = np.sqrt(np.sum((vecino_coordenadas - punto)**2, axis=1))
                promedio_sk = np.sum(diferencia) / cantidad_vecinos
            else:
                promedio_sk = 0

            indices_filtrados = df.index[df["indices"] == id].values
            data_frame.at[indices_filtrados[0], "sk"] = promedio_sk  # Acceder al primer elemento

# Cargar la base de datos inicial
data_frame_original = pd.read_csv("UNI_CORR_500_01.txt", delimiter="\t", header=0, skiprows=3)
data_frame = data_frame_original.copy()
df_velocidad_01 = data_frame_original.groupby("# PersID", group_keys=False).apply(calcular_velocidad_grupo)
calcular_sk(data_frame_original)

# Variable para mantener el estado de la base de datos
data_frame = data_frame_original.copy()

# Streamlit
st.title(":bar_chart: Laboratorio peatones")

st.write("""
A continuación se observarán gráficos con respecto al análisis del comportamiento de las personas en el interior de un pasillo dado ciertos tipos de entradas y salidas con diversas dimensiones.
""")


with st.sidebar:
    st.title("Opciones:")
    div = st.slider("Número de bins: ", 0, 130, 25)
    
with st.sidebar:
    st.write ("## Base de datos disponibles:")
    if st.button("UNI_CORR_500_08.txt"):
        data_frame = pd.read_csv("UNI_CORR_500_08.txt", delimiter="\t", header=0, skiprows=3)  
        df_velocidad_01 = data_frame.groupby("# PersID", group_keys=False).apply(calcular_velocidad_grupo)
        calcular_sk(data_frame)

    if st.button("UNI_CORR_500_01.txt"):
        data_frame = data_frame_original.copy()  
        df_velocidad_01 = data_frame.groupby("# PersID", group_keys=False).apply(calcular_velocidad_grupo)
        calcular_sk(data_frame)

st.write("### Primeros 5 valores de la base de datos")
st.write("Se pueden observar el ID de la persona con sus respectivas coordenadas en el eje X y Y por cada frame.")
st.table(data_frame.head())

st.write("### Grafico de dispersión")
st.write("En el gráfico entregado, se puede observar que las velocidades con Sk, al ser los puntos vecinos no siguen una linealidad mientras aumenten los puntos vecinos, sino que permanecen más constante de lo normal, esto se debe a que la caminata es fluida durante el transcurso del pasillo")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data_frame["sk"], df_velocidad_01["Velocidad"], label="Datos reales", color="lightcoral")

ax.set_xlabel("sk")
ax.set_ylabel("Velocidad real")
ax.set_title("sk vs. Velocidad real")

st.pyplot(fig)

st.write("### Histogramas de las coordenadas")
st.write("En el gráfico entregado, se pueden observar las coordenadas más frecuentes, tanto en el eje X como en el eje Y.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(data_frame["X"], bins=div, color="lightcoral")
ax[0].set_xlabel("Posición en metro")
ax[0].set_ylabel("Frecuencia")
ax[0].set_title("Histograma de posiciones en X")

ax[1].hist(data_frame["Y"], bins=div, color="indianred")
ax[1].set_xlabel("Posición en metro")
ax[1].set_ylabel("Frecuencia")
ax[1].set_title("Histograma de posiciones en Y")

st.pyplot(fig)

st.write("### Histogramas de las velocidades")
st.write("En el gráfico entregado, se pueden observar las velocidades más frecuentes.")
fig, ax = plt.subplots()
histograma = plt.hist(df_velocidad_01["Velocidad"], bins=div, color='indianred', alpha=0.7)

plt.title("Histograma de Velocidades")
plt.xlabel("Velocidades")
plt.ylabel("Frecuencia de personas")
st.pyplot(fig)

st.write("### Mapa de calor")
st.write("En el gráfico entregado, se pueden observar el mapa de calor en el cual hay mas flujo de gente por el transcurso del pasillo.")
fig, ax = plt.subplots()
calor = plt.hist2d(data_frame["X"], data_frame["Y"], bins=(30, 40), cmap=plt.cm.plasma)
plt.title("Gráfico 2D de Histograma")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Frecuencia")

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
