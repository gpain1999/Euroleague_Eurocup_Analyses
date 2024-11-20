import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image


season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))


import fonctions as f

# Charger les données
df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
         'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

# Ajouter une image en haut à droite
image_path = f"images/{competition}.png"  # Chemin vers l'image

try:
    image = Image.open(image_path)
    # Redimensionner l'image (par exemple, largeur de 300 pixels)
    max_width = 600
    image = image.resize((max_width, int(image.height * (max_width / image.width))))

    # Afficher l'image redimensionnée
    st.image(image, caption=f"Rapport pour {competition}")
except FileNotFoundError:
    st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 
    
# Configuration de l'interface utilisateur
st.title("Stats PER - by gpain1999")

# Paramètres interactifs
st.sidebar.header("Paramètres")

selected_range = st.sidebar.slider(
    "Sélectionnez une plage de ROUND :",
    min_value=1,
    max_value=df["ROUND"].max(),
    value=(1, df["ROUND"].max()),
    step=1
)

min_round = selected_range[0]
max_round = selected_range[1]

# Filtrage dynamique des équipes
selected_teams = st.sidebar.multiselect("Équipes Sélectionnées", options=sorted(df["TEAM"].unique()) if not df.empty else [])

# Mise à jour dynamique des joueurs en fonction des équipes sélectionnées
if selected_teams:
    available_players = sorted(df[df["TEAM"].isin(selected_teams)]["PLAYER"].unique())
else:
    available_players = sorted(df["PLAYER"].unique())

selected_players = st.sidebar.multiselect("Joueurs Sélectionnés", options=available_players)

# Autres paramètres
selected_opponents = st.sidebar.multiselect("Adversaires Sélectionnés", options=sorted(df["OPPONENT"].unique()) if not df.empty else [])

selected_fields = st.sidebar.multiselect(
    "Champs Sélectionnés",
    options=["ROUND", "PLAYER", "TEAM", "OPPONENT"],
    default=["PLAYER", "TEAM"]
)

mode = st.sidebar.selectbox("Méthode d'Agrégation", options=["CUMULATED", "AVERAGE"], index=0)
percent = st.sidebar.selectbox("Affichage des tirs", options=["MADE", "PERCENT"], index=0)

I_MODE = st.sidebar.checkbox("Activer le mode I_stats", value=False)


stats_i = [ "PTS", "DR", "OR", "TR", "AS", "ST", "CO", "TO", "FP", "CF", "NCF"]

# Mise à jour automatique des résultats
if not df.empty:
    result_df = f.get_aggregated_data(
        df, min_round, max_round,
        selected_teams=selected_teams,
        selected_opponents=selected_opponents,
        selected_fields=selected_fields,
        selected_players=selected_players,
        mode=mode,
        percent=percent
    )

    result_df = result_df.loc[:, (result_df != "---").any(axis=0)]
    if I_MODE:
        for s in stats_i:
            if s in result_df.columns:  # Vérifie si la statistique existe dans df_grouped
                # Calcul des valeurs ajustées
                result_df[s] = (result_df[s] / (result_df["TIME_ON"] ** 0.5)).round(1)
                # Renommer la colonne
                result_df.rename(columns={s: f"I_{s}"}, inplace=True)
    


    st.write("### Tableau")
    st.dataframe(result_df, height=1000,width=2000)  # Augmenter la hauteur du tableau
else:
    st.error("Les données ne sont pas chargées. Veuillez vérifier votre fichier source.")