import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from transformers import pipeline


# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')


# Función de limpieza y preprocesamiento de texto
def clean_text_stemming(text):
    """
    Realiza la limpieza y preprocesamiento de un texto. Convierte el texto a minúsculas,
    elimina caracteres no alfabéticos, tokeniza, remueve stopwords y aplica stemming.

    :param text: El texto a limpiar y preprocesar.
    :return: Texto procesado y limpio.
    """
    text = text.lower()
    text = re.sub(r'[^a-zñáéíóú]', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    stemmer = SnowballStemmer('spanish')
    stemmed_text = ' '.join([stemmer.stem(word) for word in words if word not in stop_words])
    return stemmed_text


# Cargar modelos de Hugging Face para análisis de sentimientos y emociones
clasificador_sentimiento = pipeline('sentiment-analysis',
                                    model='citizenlab/twitter-xlm-roberta-base-sentiment-finetunned')
clasificador_emociones = pipeline("text-classification",
                                  model="maxpe/bertin-roberta-base-spanish_sem_eval_2018_task_1")


# Funciones para analizar sentimiento y emociones
def analizar_sentimiento(texto):
    """
    Analiza el sentimiento de un texto dado utilizando un modelo preentrenado de Hugging Face.

    :param texto: El texto a analizar.
    :return: La etiqueta del sentimiento detectado (ej. "Positive", "Negative", "Neutral").
             Retorna None si ocurre un error.
    """
    try:
        results = clasificador_sentimiento(texto, truncation=True, max_length=512)
        top_result = max(results, key=lambda x: x['score'])
        return top_result['label']
    except Exception as e:
        print(f"Error al procesar el texto: {e}")
        return None


def analizar_emociones(texto):
    """
    Analiza las emociones presentes en un texto utilizando un modelo preentrenado de Hugging Face.

    :param texto: El texto a analizar.
    :return: La etiqueta de la emoción detectada (ej. "alegría", "tristeza").
             Retorna None si ocurre un error.
    """
    try:
        results = clasificador_emociones(texto, truncation=True, max_length=512)
        top_result = max(results, key=lambda x: x['score'])
        return top_result['label']
    except Exception as e:
        print(f"Error al procesar el texto: {e}")
        return None


# Normalización de la longitud del texto
def normalizar_longitud(texto):
    """
    Normaliza la longitud de un texto dividiendo la longitud del texto por una longitud máxima observada.

    :param texto: El texto cuya longitud se va a normalizar.
    :return: Longitud normalizada del texto.
    """
    longitud_maxima_observada = 53352
    return len(texto) / longitud_maxima_observada


# Obtener características de polaridad y emociones
def obtener_caracteristicas_polaridad(texto):
    """
    Obtiene características de polaridad (positiva, negativa, neutral) de un texto analizando su sentimiento.

    :param texto: El texto a analizar.
    :return: Un diccionario con las características de polaridad en formato binario.
    """
    polaridad = analizar_sentimiento(texto)
    return {
        "polarity_Negative": 1 if polaridad == "Negative" else 0,
        "polarity_Neutral": 1 if polaridad == "Neutral" else 0,
        "polarity_Positive": 1 if polaridad == "Positive" else 0
    }


def obtener_caracteristicas_emociones(texto):
    """
    Obtiene características de emociones (ira, anticipación, miedo, alegría, tristeza) de un texto.

    :param texto: El texto a analizar.
    :return: Un diccionario con las características de emociones en formato binario.
    """
    emocion = analizar_emociones(texto)
    categorias = ["anger", "anticipation", "fear", "joy", "sadness"]
    return {f"emotions_{cat}": 1 if emocion == cat else 0 for cat in categorias}


# Función para realizar predicciones con información adicional
def predecir_suicidio_con_info(texto, modelo, tfidf_vectorizador):
    """
    Realiza la predicción de comportamiento suicida en un texto dado, utilizando un modelo
    de machine learning y un vectorizador TF-IDF. Incluye características adicionales
    como longitud del texto, polaridad y emociones.

    :param texto: Texto a analizar.
    :param modelo: Modelo de machine learning para realizar la predicción.
    :param tfidf_vectorizador: Vectorizador TF-IDF para procesar el texto.
    :return: Un diccionario con los resultados de la clasificación y análisis.
    """
    texto_limpio = clean_text_stemming(texto)

    # Obtener características
    longitud_normalizada = normalizar_longitud(texto)
    caracteristicas_polaridad = obtener_caracteristicas_polaridad(texto)
    caracteristicas_emociones = obtener_caracteristicas_emociones(texto)
    polaridad = analizar_sentimiento(texto)
    emocion = analizar_emociones(texto)

    # Vectorizar texto y combinar características
    texto_tfidf = tfidf_vectorizador.transform([texto_limpio]).toarray()
    caracteristicas_adicionales = np.array(
        [longitud_normalizada] + list(caracteristicas_polaridad.values()) + list(caracteristicas_emociones.values()))
    features = np.hstack((texto_tfidf, caracteristicas_adicionales.reshape(1, -1)))

    # Asegúrate de que el DataFrame tenga los mismos nombres de columnas que se usaron durante el entrenamiento
    column_names = tfidf_vectorizador.get_feature_names_out().tolist() + ['text_length', 'polarity_Negative',
                                                                          'polarity_Neutral', 'polarity_Positive',
                                                                          'emotions_anger', 'emotions_anticipation',
                                                                          'emotions_fear', 'emotions_joy',
                                                                          'emotions_sadness']
    features_df = pd.DataFrame(features, columns=column_names)

    # Realizar la predicción
    pred = modelo.predict(features_df)
    pred_proba = modelo.predict_proba(features_df)[0]

    # Confianza y nivel de riesgo
    confianza = abs(pred_proba[1] - pred_proba[0])
    nivel_riesgo = 'alto' if pred_proba[1] > 0.75 else 'moderado' if pred_proba[1] > 0.5 else 'bajo'
    accion = 'Se recomienda buscar ayuda profesional inmediatamente.' if nivel_riesgo == 'alto' \
        else 'Se sugiere monitorear los sentimientos y considerar hablar con un profesional.' \
        if nivel_riesgo == 'moderado' \
        else 'Probablemente no hay riesgo inmediato, pero mantén una actitud positiva.'

    # Traducciones de las etiquetas de polaridad y emociones al español
    traducciones_polaridad = {
        "Negative": "Negativa",
        "Neutral": "Neutral",
        "Positive": "Positiva"
    }
    traducciones_emociones = {
        "anger": "ira",
        "anticipation": "Anticipación",
        "fear": "Miedo",
        "joy": "Alegría",
        "sadness": "Tristeza"
    }

    # Traducir polaridad y emociones
    polaridad_traducida = traducciones_polaridad.get(polaridad, polaridad)
    emocion_traducida = traducciones_emociones.get(emocion, emocion)

    return {
        "clasificacion": 'suicidio' if pred[0] == 1 else 'no suicidio',
        "probabilidad_suicidio": pred_proba[1],
        "confianza": confianza,
        "nivel_riesgo": nivel_riesgo,
        "sugerencia_accion": accion,
        "polaridad": polaridad_traducida,
        "emocion": emocion_traducida
    }


def procesar_textos(modelo, tfidf_vectorizador):
    """
    Función principal para procesar textos en la aplicación Streamlit. Permite al usuario
    ingresar o subir un texto, y luego utiliza el modelo y el vectorizador para analizarlo.

    :param modelo: Modelo de machine learning para la predicción.
    :param tfidf_vectorizador: Vectorizador TF-IDF utilizado en el modelo.
    """
    st.title("Analizar Texto")

    # Opción para ingresar texto manualmente
    text_input = st.text_area("Ingrese su texto aquí:")

    # Opción para subir un archivo
    uploaded_file = st.file_uploader("O suba un archivo de texto:", type=["txt"])

    # Botón para procesar el texto
    if st.button("Procesar Texto"):
        text_to_process = ""
        if uploaded_file is not None:
            # Leer el archivo subido y almacenar su contenido
            text_to_process = uploaded_file.read().decode("utf-8")
        elif text_input:
            # Utilizar el texto ingresado manualmente
            text_to_process = text_input

        if text_to_process:
            # Llamar a la función de predicción con el texto procesado
            resultado = predecir_suicidio_con_info(text_to_process, modelo, tfidf_vectorizador)

            # Estilos personalizados según el nivel de riesgo
            if resultado['nivel_riesgo'] == 'alto':
                color = "red"
            elif resultado['nivel_riesgo'] == 'moderado':
                color = "orange"
            else:
                color = "green"

            # Mostrar los resultados con estilos
            st.markdown(f"<h2 style='color: {color};'>Resultado del Análisis:</h2>", unsafe_allow_html=True)
            st.markdown(f"<b>Nivel de Riesgo:</b> <span style='color: {color};'>{resultado['nivel_riesgo']}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<b>Polaridad:</b> {resultado['polaridad']}", unsafe_allow_html=True)
            st.markdown(f"<b>Emoción:</b> {resultado['emocion']}", unsafe_allow_html=True)
            st.markdown(f"<b>Clasificación:</b> {resultado['clasificacion']}", unsafe_allow_html=True)
            st.markdown(f"<b>Probabilidad de Suicidio:</b> {resultado['probabilidad_suicidio']:.4f}",
                        unsafe_allow_html=True)
            st.markdown(f"<b>Confianza:</b> {resultado['confianza']:.4f}", unsafe_allow_html=True)

            st.markdown(f"<b>Sugerencia de Acción:</b> {resultado['sugerencia_accion']}", unsafe_allow_html=True)
        else:
            st.warning("Por favor, ingrese texto o suba un archivo.")
