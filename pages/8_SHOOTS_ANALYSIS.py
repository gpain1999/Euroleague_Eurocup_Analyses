import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math

# Ajouter le chemin de la racine du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_authentication

#require_authentication()


season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

st.set_page_config(
    page_title="SHOOTS ANALYSIS",
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


###########################EN TETE
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
            <b>SHOOTS ANALYSIS - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )



######################### PARAM
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

col1, col2,col3,col4,col5 = st.columns([2, 2,1,1,1])

with col1 : 
    mode = st.selectbox("Méthode d'Agrégation", options=["CUMULATED", "AVERAGE"], index=0)
with col2 : 

    gp = st.selectbox("GROUP BY", options=["PLAYER", "TEAM"], index=0)

with col3 : 
    M1 = st.checkbox("Stats 1P", value=True)
    M2 = st.checkbox("Stats 2P", value=True)
    M3 = st.checkbox("Stats 3P", value=True)

 


# Filtrage dynamique des équipes
selected_teams = st.sidebar.multiselect("Équipes Sélectionnées", options=sorted(df["TEAM"].unique()) if not df.empty else [])

# Mise à jour dynamique des joueurs en fonction des équipes sélectionnées
if selected_teams:
    available_players = sorted(df[df["TEAM"].isin(selected_teams)]["PLAYER"].unique())
else:
    available_players = sorted(df["PLAYER"].unique())

selected_players = st.sidebar.multiselect("Joueurs Sélectionnés", options=available_players)

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
######################### DATA


result_df = f.get_aggregated_data(
    df, min_round, max_round,
    selected_teams=selected_teams,
    selected_opponents=[],
    selected_fields=list(set(["TEAM"] + [gp])),
    selected_players=selected_players,
    mode=mode,
    percent="MADE"
)

col_to_return = ["NB_GAME","TEAM","WIN","#","PLAYER","TIME_ON"]
import numpy as np

if M1 : 
    result_df["1_P"] = (result_df["1_R"]/result_df["1_T"]*100).round(1)
    #result_df["INF_1P"] = (((result_df["1_R"] / result_df["1_T"]) - 0.674 * ((((result_df["1_R"] / result_df["1_T"])) * (1 - ((result_df["1_R"] / result_df["1_T"]))))/(result_df["1_T"]))**0.5)*100).round(1)
    result_df["INF_1P"] = (np.clip((result_df["1_R"] / result_df["1_T"]) - np.clip((1/3 - (result_df["1_T"])/300),0,0.5),0,1)*100).round(1)
    result_df["SUP_1P"] = (np.clip((result_df["1_R"] / result_df["1_T"]) + np.clip((1/3 - (result_df["1_T"])/300),0,0.5),0,1)*100).round(1)
    #result_df["SUP_1P"] = (((result_df["1_R"] / result_df["1_T"]) + 0.674 * ((((result_df["1_R"] / result_df["1_T"])) * (1 - ((result_df["1_R"] / result_df["1_T"]))))/(result_df["1_T"]))**0.5)*100).round(1)
    col_to_return = col_to_return + ["1_R","1_T","INF_1P","1_P","SUP_1P"]

if M2 :
    result_df["2_P"] = (result_df["2_R"]/result_df["2_T"]*100).round(1)
    # result_df["INF_2P"]= (((result_df["2_R"] / result_df["2_T"]) - 0.674 *((((result_df["2_R"] / result_df["2_T"])) * (1 - ((result_df["2_R"] / result_df["2_T"]))))/(result_df["2_T"]))**0.5)*100).round(1)
    # result_df["SUP_2P"]= (((result_df["2_R"] / result_df["2_T"]) + 0.674 *((((result_df["2_R"] / result_df["2_T"])) * (1 - ((result_df["2_R"] / result_df["2_T"]))))/(result_df["2_T"]))**0.5)*100).round(1)
    result_df["INF_2P"] = (np.clip((result_df["2_R"] / result_df["2_T"]) - np.clip((1/3 - (result_df["2_T"])/300),0,0.5),0,1)*100).round(1)
    result_df["SUP_2P"] = (np.clip((result_df["2_R"] / result_df["2_T"]) + np.clip((1/3 - (result_df["2_T"])/300),0,0.5),0,1)*100).round(1)

    col_to_return = col_to_return + ["2_R","2_T","INF_2P","2_P","SUP_2P"]

if M3 :
    result_df["3_P"] = (result_df["3_R"]/result_df["3_T"]*100).round(1)
    # result_df["INF_3P"]= (((result_df["3_R"] / result_df["3_T"]) - 0.674 *((((result_df["3_R"] / result_df["3_T"])) * (1 - ((result_df["3_R"] / result_df["3_T"]))))/(result_df["3_T"]))**0.5)*100).round(1)
    # result_df["SUP_3P"]= (((result_df["3_R"] / result_df["3_T"]) + 0.674 *((((result_df["3_R"] / result_df["3_T"])) * (1 - ((result_df["3_R"] / result_df["3_T"]))))/(result_df["3_T"]))**0.5)*100).round(1)
    result_df["INF_3P"] = (np.clip((result_df["3_R"] / result_df["3_T"]) - np.clip((1/3 - (result_df["3_T"])/300),0,0.5),0,1)*100).round(1)
    result_df["SUP_3P"] = (np.clip((result_df["3_R"] / result_df["3_T"]) + np.clip((1/3 - (result_df["3_T"])/300),0,0.5),0,1)*100).round(1)
    col_to_return = col_to_return + ["3_R","3_T","INF_3P","3_P","SUP_3P"]


result_df = result_df[col_to_return]
result_df = result_df.loc[:, (result_df != "---").any(axis=0)]

################### STYLES
###################### PRINT


st.dataframe(result_df, height=min(35 + 35*len(result_df),900),width=2000,hide_index=True)
