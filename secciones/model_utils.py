import joblib
import streamlit as st


# Cargar el modelo y el vectorizador
@st.cache_data
def cargar_modelo_y_vectorizador():
    modelo = joblib.load('modelo_nn_g2.pkl')
    tfidf_vectorizador = joblib.load('tfidf_vectorizador2.pkl')
    return modelo, tfidf_vectorizador
