import streamlit as st


def home_page():
    """
    Muestra la página principal de la aplicación Streamlit.

    Esta función se encarga de renderizar la página de inicio de la aplicación de detección temprana de riesgo
    de suicidio.
    Incluye secciones que explican el propósito del proyecto, sus objetivos principales y cómo utilizar la herramienta.
    Cada sección está claramente definida y contiene información detallada y relevante para el usuario.

    La página de inicio se estructura en varias secciones, que incluyen:
    - Una introducción al proyecto y su relevancia.
    - Los objetivos principales del proyecto.
    - Instrucciones detalladas sobre cómo utilizar la herramienta.

    Las secciones están diseñadas para proporcionar a los usuarios una comprensión clara del propósito y
    la funcionalidad de la herramienta, así como guiarlos en su uso efectivo.

    :return: None. Renderiza los componentes de la página de inicio en la interfaz de usuario de Streamlit.
    """
    st.title("TFG: Detección Temprana de Riesgo de Suicidio Mediante Análisis de mensajes de texto")
    with st.container():
        st.write("""
            Este proyecto emplea técnicas avanzadas de Procesamiento de Lenguaje Natural (PLN) y análisis de datos para 
            explorar y detectar señales de comportamientos autolesivos en mensajes publicados en redes sociales. 
            La iniciativa busca abordar una de las principales causas de preocupación en la salud mental global, 
            ofreciendo una herramienta innovadora y valiosa para la identificación temprana de riesgos de suicidio.
        """)
    with st.container(border=True):
        st.header("Objetivos Principales")
        st.write("""
            - **Análisis Profundo de Comunicaciones Digitales**: Implementación de algoritmos de PLN para analizar 
            textos y detectar patrones de riesgo en comportamientos y estados emocionales.
            - **Prevención de Comportamientos de Riesgo**: Utilización de la herramienta como un medio para identificar 
            señales tempranas de crisis emocionales, proporcionando una base sólida para intervenciones oportunas y 
            medidas preventivas.
            - **Evaluación y Mejora de Estrategias de Prevención Actuales**: Contribución a la investigación y mejora de 
            estrategias de prevención del suicidio mediante la incorporación de tecnologías avanzadas de análisis de 
            datos.
        """)
    with st.container(border=True):
        st.header("Cómo Utilizar la Herramienta")
        st.write("""
            1. **Acceso a la herramienta de análisis**: Para acceder a esta propiedad tan solo hay que dirigirse
            al botón 'Analizar texto' que se encuentra en el menú de navegación.
        """)
        st.image("imagenes/Captura1.PNG", use_column_width="auto")
        st.write("""
            2. **Ingreso y procesamiento del texto**: Para llevar a cabo nuestro análisis tan solo se debe introducir 
            en el cuadro respectivo el texto a analizar. Haga clic en el botón 'Procesar texto' para que la herramienta 
            procese el texto. La herramienta utiliza modelos de PLN para analizar el contenido y la tonalidad 
            emocional del texto.
        """)
        st.write("""
            **Importante: Por ahora no esta permitido ingresar archivos de texto**
        """)
        st.image("imagenes/Captura2.PNG")
        st.image("imagenes/Captura3.PNG")
        st.write("""
            4. **Interpretación de resultados y acciones sugeridas**: Los resultados proporcionan insights 
            sobre el estado emocional del texto y posibles señales de riesgo. Estos incluyen análisis de sentimientos, 
            emociones y una evaluación de riesgo de suicidio. Basado en el nivel de riesgo detectado, 
            la herramienta ofrece recomendaciones sobre posibles pasos a seguir.
        """)
        st.image("imagenes/Captura4.PNG")
