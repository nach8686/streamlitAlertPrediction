import streamlit as st
from transformers import pipeline
import nltk
from secciones.procesar_textos import procesar_textos  # Asegúrate de que esta ruta sea correcta
from secciones.home_page import home_page
from secciones.info_page import info_page


def main():
    """
    Función principal que ejecuta la aplicación Streamlit.

    Esta función se encarga de configurar la página, cargar el modelo y el vectorizador, y gestionar el menú de
    navegación de la aplicación. Dependiendo de la opción seleccionada en el menú, renderiza diferentes vistas como
    la página de inicio, la página de análisis de texto o la página de información.

    :return: None. Ejecuta y mantiene activa la aplicación Streamlit.
    """
    # Configuración de la página
    logo_log_bar = r'imagenes/Captura de pantalla 2024-01-06 a las 17.13.19.png'
    st.set_page_config(page_title='Análisis sentimientos', page_icon=logo_log_bar, layout="wide")

    # Cargar modelos de Hugging Face para análisis de sentimientos y emociones
    clasificador_sentimiento = pipeline('sentiment-analysis',
                                        model='citizenlab/twitter-xlm-roberta-base-sentiment-finetunned')
    clasificador_emociones = pipeline("text-classification",
                                      model="maxpe/bertin-roberta-base-spanish_sem_eval_2018_task_1")

    # Descargar recursos de NLTK
    @st.cache_data
    def descargar_recursos_nltk():
        """
        Descarga los recursos necesarios de NLTK.

        Esta función descarga los componentes 'punkt' y 'stopwords' de NLTK, que son necesarios para tokenizar y
        preprocesar el texto en las funciones de análisis.

        :return: None.
        """
        nltk.download('punkt')
        nltk.download('stopwords')

    descargar_recursos_nltk()

    # Menú de opciones
    with st.sidebar:
        # Imagen y título del menú centrados
        st.image("imagenes/uoc.png", width=290)  # Tamaño más grande para el logo
        st.markdown('<h1 style="text-align: center; color: black;">Menú de Navegación</h2>', unsafe_allow_html=True)

        # Opciones del menú con emojis como iconos
        opciones_menu = {
            "Home": "🏠 Home",
            "Analizar texto": "🔍 Analizar texto",
            "Info": "ℹ️ Info"
        }
        # Opción para ingresar texto manualmente
        text_input = st.text_area("Ingrese su texto aquí:")

        # Botón para procesar el texto
        if st.button("Analizar Texto"):
            if text_input:
                # Análisis de sentimiento
                resultado_sentimiento = clasificador_sentimiento(text_input, truncation=True, max_length=512)
                st.write("Resultado Sentimiento:", resultado_sentimiento)

                # Análisis de emociones
                resultado_emociones = clasificador_emociones(text_input, truncation=True, max_length=512)
                st.write("Resultado Emociones:", resultado_emociones)
            else:
                st.warning("Por favor, ingrese texto.")


if __name__ == "__main__":
    main()
