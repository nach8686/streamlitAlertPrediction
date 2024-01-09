import joblib

# Variables globales para el modelo y el vectorizador
modelo_global = None
vectorizador_global = None


def cargar_modelo_y_vectorizador():
    global modelo_global
    global vectorizador_global

    if modelo_global is None or vectorizador_global is None:
        modelo_global = joblib.load('modelo_nn_g2.pkl')
        vectorizador_global = joblib.load('tfidf_vectorizador2.pkl')

    return modelo_global, vectorizador_global
