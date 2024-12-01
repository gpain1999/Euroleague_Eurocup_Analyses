import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import re
import math
season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory
import fonctions as f

st.set_page_config(
    page_title="GAME ANALYSIS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)



image_path = f"images/{competition}.png"  # Chemin vers l'image


col1, col2 = st.columns([1, 2])

with col1 : 
    try:
        image = Image.open(image_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        max_width = 400
        image = image.resize((max_width, int(image.height * (max_width / image.width))))

        # Afficher l'image redimensionnée
        st.image(image)
    except FileNotFoundError:
        st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

with col2 : 

    # Configuration de l'interface utilisateur
    st.title("Game Analysis - by gpain1999")


# SETTINGS interactifs
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 60px; /* Largeur minimale */
        max-width: 240px; /* Largeur maximale */
        width: 120px; /* Largeur fixe */
    }
    </style>
    """,
    unsafe_allow_html=True
)
###################### DATA INIT ################################

# Charger les données
df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
         'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

# Charger les données de jeux
gs = pd.read_csv(f"datas/{competition}_gs_{season}.csv")[["Gamecode", "Round", "local.club.code", "local.score", "road.club.code", "road.score"]]
gs.columns = ["GAMECODE", "ROUND", "LOCAL_TEAM", "LOCAL_SCORE", "ROAD_TEAM", "ROAD_SCORE"]
gs["GAME"] = gs.apply(lambda row: f"R{row['ROUND']} : {row['LOCAL_TEAM']} - {row['ROAD_TEAM']}", axis=1)


######################### PARAM
st.sidebar.header("SETTINGS")

zoom = st.sidebar.slider(
    "Choisissez une valeur de zoom pour les photos",
    min_value=0.3,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)
RECHERCHE = st.sidebar.selectbox("SEARCH", options=["ROUND","TEAM"], index=0)
if RECHERCHE == "ROUND" : 
    SUB = st.sidebar.number_input("ROUND", min_value=gs["ROUND"].min(),max_value=gs["ROUND"].max() ,value=gs["ROUND"].max())
else :
    SUB = st.sidebar.selectbox("TEAMS", options=sorted(df["TEAM"].unique()), index=0)

game_list = gs[gs['GAME'].str.contains(str(SUB), na=False)]['GAME'].tolist()
GAME = st.sidebar.selectbox("GAMES", options=game_list, index=len(game_list)-1)

# Utiliser une expression régulière pour extraire les valeurs
match = re.match(r"R(\d+)\s*:\s*([A-Za-z]+)\s*-\s*([A-Za-z]+)", GAME)


# Récupérer le nombre après le "R"
r = int(match.group(1))

# Récupérer les deux noms d'équipes
team_local = match.group(2)
team_road = match.group(3)

