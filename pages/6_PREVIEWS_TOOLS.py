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
st.sidebar.header("Paramètres")

selected_range = st.sidebar.slider(
    "Sélectionnez une plage de ROUND :",
    min_value=min_round,
    max_value=max_round,
    value=(min_round, max_round),
    step=1
)

# Filtrage dynamique des équipes
HOMETEAM = st.sidebar.selectbox("HOME TEAM", options=sorted(df["TEAM"].unique()) if not df.empty else [],index = 0)
HOMETEAM_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{HOMETEAM}.png")

# Filtrage dynamique des équipes
AWAYTEAM = st.sidebar.selectbox("AWAY TEAM", options=sorted(df["TEAM"].unique()) if not df.empty else [],index = 1)
AWAYTEAM_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{AWAYTEAM}.png")
