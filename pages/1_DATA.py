import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image
# Ajouter le chemin de la racine du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_authentication

# require_authentication()

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))


st.set_page_config(
    page_title="DATA",
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
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T','TO', 'FP', 'CF', 'NCF']]


# Ajouter une image en haut à droite
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
    taille_titre = 70*0.7
    st.markdown(
        f'''
        <p style="font-size:{int(taille_titre)}px; text-align: center; padding: 10pxs;">
            <b>DATA BRUT - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

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


col1,col2,col3,col4 = st.columns([1.5,1,1,1]) 

with col1 :
    selected_fields = st.multiselect(
        "GROUP BY",
        options=["ROUND", "PLAYER", "TEAM", "OPPONENT"],
        default=["PLAYER", "TEAM"]
    )

with col2 :

    mode = st.selectbox("Méthode d'Agrégation", options=["CUMULATED", "AVERAGE"], index=0)

with col3 :
    percent = st.selectbox("Affichage des tirs", options=["MADE", "PERCENT"], index=0)





st.sidebar.header("SETTINGS")

selected_range = st.sidebar.slider(
    "Select a ROUND range:",
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
    
    if 'NB_GAME' in result_df.columns and (result_df['NB_GAME'] == 1).all():
        result_df = result_df.drop(columns=['NB_GAME'])

    result_df = result_df.sort_values(by = "TIME_ON",ascending = False)

    st.dataframe(result_df, height=min(35 + 35*len(result_df),900),width=2000,hide_index=True)  # Augmenter la hauteur du tableau
else:
    st.error("Les données ne sont pas chargées. Veuillez vérifier votre fichier source.")
