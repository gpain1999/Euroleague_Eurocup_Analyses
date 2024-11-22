import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image


season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))

st.set_page_config(
    page_title="PLUS MINUS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)

import fonctions as f

players = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

df = f.analyse_io_2(data_dir = data_dir,
                   competition = competition,
                   season = season,
                   num_players = 3,
                    min_round = 1,
                    max_round = 34,
                   CODETEAM = [],
                   selected_players = [],
                   min_percent_in = 0)


image_path = f"images/{competition}.png"  # Chemin vers l'image


col1, col2 = st.columns([1, 2])

with col1 : 
    try:
        image = Image.open(image_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        max_width = 400
        image = image.resize((max_width, int(image.height * (max_width / image.width))))

        # Afficher l'image redimensionnée
        st.image(image, caption=f"Rapport pour {competition}")
    except FileNotFoundError:
        st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

with col2 : 

    # Configuration de l'interface utilisateur
    st.title("+/- Analysis - by gpain1999")

# Paramètres interactifs
st.sidebar.header("Paramètres")

num_players = st.sidebar.number_input("SOLO/DUO/TRIO", min_value=1,max_value=3 ,value=2)

min_round = st.sidebar.number_input("Round Minimum", min_value=1, value=1)
max_round = st.sidebar.number_input("Round Maximum", min_value=min_round, value=34 if not df.empty else 1)

# Filtrage dynamique des équipes
CODETEAM = st.sidebar.multiselect("Équipes Sélectionnées", options=sorted(df["TEAM"].unique()) if not df.empty else [])

# Mise à jour dynamique des joueurs en fonction des équipes sélectionnées
if CODETEAM:

    available_players = players[players["CODETEAM"].isin(CODETEAM)]["PLAYER"].unique()
else:
    available_players = players["PLAYER"].unique()

selected_players = st.sidebar.multiselect("Joueurs Sélectionnés", options=sorted(available_players))

min_percent_in = st.sidebar.slider("Time percent min", min_value=0, max_value=100, value=0)



# Mise à jour automatique des résultats
if not df.empty:
    result_df = f.analyse_io_2(data_dir = data_dir,
                   competition = competition,
                   season = season,
                   num_players = num_players,
                    min_round = min_round,
                    max_round = max_round,
                   CODETEAM = CODETEAM,
                   selected_players = selected_players,
                   min_percent_in = min_percent_in)

    st.write("### +/- RESULTS")
    st.dataframe(result_df, height=min(38*len(result_df),900),width=2000)  # Augmenter la hauteur du tableau
else:
    st.error("Les données ne sont pas chargées. Veuillez vérifier votre fichier source.")