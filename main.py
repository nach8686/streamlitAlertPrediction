import streamlit as st
import joblib
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

    # Cargar el modelo y el vectorizador
    @st.cache_data
    def cargar_modelo_y_vectorizador():
        """
        Carga el modelo de aprendizaje automático y el vectorizador TF-IDF desde archivos almacenados.

        Esta función utiliza joblib para cargar un modelo preentrenado y un vectorizador TF-IDF. Estos componentes son
        esenciales para procesar y analizar el texto ingresado en la aplicación.

        :return: Tupla que contiene el modelo y el vectorizador.
        """
        return joblib.load('modelo_nn_g2.pkl'), joblib.load('tfidf_vectorizador2.pkl')

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

    modelo, tfidf_vectorizador = cargar_modelo_y_vectorizador()
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
        # Valor predeterminado para 'selected'
        if 'selected' not in st.session_state:
            st.session_state['selected'] = "Home"

        # Estilo personalizado para los botones
        btn_style = """
            <style>
                .css-2trqyj { 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    font-size: 18px; 
                    font-weight: bold;
                }
            </style>
        """
        st.markdown(btn_style, unsafe_allow_html=True)

        # Crear botones en la barra lateral para cada opción del menú
        for opcion, etiqueta in opciones_menu.items():
            if st.button(etiqueta, key=opcion, use_container_width=True):
                st.session_state['selected'] = opcion

    # Vista de información
    if st.session_state['selected'] == "Home":
        st.image("imagenes/uoc2.png", use_column_width=True)
        home_page()

    # Analizar texto
    elif st.session_state['selected'] == "Analizar texto":
        st.image("imagenes/uoc2.png", use_column_width=True)
        procesar_textos(modelo, tfidf_vectorizador)

    # Info
    elif st.session_state['selected'] == "Info":
        st.image("imagenes/uoc2.png", use_column_width=True)
        info_page()


if __name__ == "__main__":
    main()
