import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

st.set_page_config(
    page_title="TEAM STATS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)


import fonctions as f


# Charger les données
df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
         'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


    

# Remplir les trous dans ROUND
min_round, max_round = df["ROUND"].min(), df["ROUND"].max()

######################### PARAM
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 60px; /* Largeur minimale */
        max-width: 240px; /* Largeur maximale */
        width: 100px; /* Largeur fixe */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("SETTINGS")

selected_range = st.sidebar.slider(
    "Sélectionnez une plage de ROUND :",
    min_value=min_round,
    max_value=max_round,
    value=(min_round, max_round),
    step=1
)

# Filtrage dynamique des équipes

# Filtrage dynamique des équipes


zoom = st.sidebar.slider(
    "Choisissez une valeur de zoom pour les photos",
    min_value=0.3,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)

###################### PRINT

image_path = f"images/{competition}.png"  # Chemin vers l'image
col1, col2,col3 = st.columns([1.5, 3,1])

with col1 : 


    try:
        image = Image.open(image_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        max_width = 600
        image = image.resize((max_width, int(image.height * (max_width / image.width))))

        # Afficher l'image redimensionnée
        st.image(image)
    except FileNotFoundError:
        st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 


with col2 :
    taille_titre = 70*zoom
    st.markdown(
        f'''
        <p style="font-size:{int(taille_titre)}px; text-align: center; padding: 10pxs;">
            <b>PREVIEWS TOOLS - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
        

col1, col2,col3,col4,col5 = st.columns([3,1,2,1,3])

with col2:
    HOMETEAM = st.selectbox("HOME TEAM", options=sorted(df["TEAM"].unique()) if not df.empty else [],index = 0)
    HOMETEAM_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{HOMETEAM}.png")

    st.image(HOMETEAM_logo_path,  width=int(200*zoom))

with col3 :
    st.markdown(
            f'''
            <p style="font-size:{int(70*zoom)}px; text-align: center; color: black; padding: 10px;">
                <b>-</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with col4 :
    AWAYTEAM = st.selectbox("AWAY TEAM", options=sorted(df["TEAM"].unique()) if not df.empty else [],index = 1)
    AWAYTEAM_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{AWAYTEAM}.png")
    st.image(AWAYTEAM_logo_path,  width=int(200*zoom))
