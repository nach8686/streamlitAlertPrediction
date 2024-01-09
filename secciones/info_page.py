import streamlit as st


def info_page():

    with st.container():
        st.markdown("""
            <style>

                .logo {
                    display: inline-block;
                    margin-right: 5px;
                    width: 20px; /* Ajusta el tamaño del logo aquí */
                }
                .about-container{
                    text-align: center;
                }
            </style>
            <div class="about-container" >
                <h1 >Acerca de este TFG</h1>
                <p>
                    <h2>Ignacio Macías Martínez</h2><br>
                    <strong>Grado de ciencia de datos aplicada</strong>: Trabajo final de grado 22.536.<br>
                    <strong>Tutora TFG</strong>: Teresa Divorra Vallhonrat.<br>
                    <strong>Profesor responsable de la asignatura</strong>: David Merino Arranz.<br>
                    <strong>Contacto</strong>: /
                    <a href="https://github.com/imaciasm"><img class="logo" 
                    src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"/>GitHub</a> \\ /
                    <a href="https://www.linkedin.com/in/imaciasm/"><img class="logo" 
                    src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"/>LinkedIn</a> 
                    \\
                </p>
            </div>
        """, unsafe_allow_html=True)
