import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# Ajouter le chemin de la racine du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_authentication

#require_authentication()

season = 2025
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



# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("SETTINGS")


selected_range = st.sidebar.slider(
    "Select a ROUND range:",
    min_value=min_round,
    max_value=max_round,
    value=(min_round, max_round),
    step=1
)

######################DATA#############

team_detail_select = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)

opp_detail_select = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["OPPONENT"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)

result = pd.merge(
    team_detail_select,
    opp_detail_select,
    how="left",
    left_on=["TEAM"],
    right_on=["OPPONENT"],
    suffixes=('_team', '_opp')
)

result["PPS"] = ((result["2_R_team"]*2 + result["3_R_team"]*3)/(result["2_T_team"] + result["3_T_team"])).round(2)
PPS_MOYEN = (result["2_R_team"].sum()*2 + result["3_R_team"].sum()*3)/(result["2_T_team"].sum() + result["3_T_team"].sum())

result["NB_SHOOT"] = (result["2_T_team"] + result["3_T_team"]).round(1)

result["PPS_opp"] = ((result["2_R_opp"]*2 + result["3_R_opp"]*3)/(result["2_T_opp"] + result["3_T_opp"])).round(2)
result["NB_SHOOT_opp"] = (result["2_T_opp"] + result["3_T_opp"]).round(1)

result["PPS_DELTA"] = (result["PPS"] - result["PPS_opp"]).round(2)
result["NB_SHOOT_DELTA"] = (result["NB_SHOOT"] - result["NB_SHOOT_opp"]).round(1)

result["PP_FT"] = (result["1_R_team"] /result["1_T_team"] ).round(2)
result["NB_FT"] = (result["1_T_team"])

result["PP_FT_opp"] = (result["1_R_opp"] /result["1_T_opp"] ).round(2)
result["NB_FT_opp"] = (result["1_T_opp"])

result["NB_FT_DELTA"] = (result["NB_FT"] - result["NB_FT_opp"]).round(1)

result["BC"] = (100*(result["1_T_team"]*0.5 + result["2_T_team"] + result["3_T_team"])/(result["1_T_team"]*0.5 + result["2_T_team"] + result["3_T_team"] + result["TO_team"])).round(1)
BC_MOYEN = (100*(result["1_T_team"].sum()*0.5 + result["2_T_team"].sum() + result["3_T_team"].sum())/(result["1_T_team"].sum()*0.5 + result["2_T_team"].sum() + result["3_T_team"].sum() + result["TO_team"].sum())).round(1)
result["BC_opp"] = (100*(result["1_T_opp"]*0.5 + result["2_T_opp"] + result["3_T_opp"])/(result["1_T_opp"]*0.5 + result["2_T_opp"] + result["3_T_opp"] + result["TO_opp"])).round(1)

result["BC_DELTA"] = (result["BC"] - result["BC_opp"]).round(1)

result["PctDeReRec"] = (100*(result["DR_team"])/(result["DR_team"] + result["OR_opp"])).round(1)
PCT_DE_REB_MOYEN = (100*df["DR"].sum()/(df["DR"].sum() + df["OR"].sum())).round(1)
result["PctOfReRec"] = (100*(result["OR_team"])/(result["OR_team"] + result["DR_opp"])).round(1)


for side in ["team", "opp"]:
    PTS = 2*result[f"2_R_{side}"] + 3*result[f"3_R_{side}"] + result[f"1_R_{side}"]
    FGA = result[f"2_T_{side}"] + result[f"3_T_{side}"]
    FTA = result[f"1_T_{side}"]
    result[f"TS_{side}"] = (PTS * 100) / (2 * (FGA + 0.44 * FTA))

TS_MOYEN     = round((2*result["2_R_team"].sum() + 3*result["3_R_team"].sum() + result["1_R_team"].sum()) * 100 /
                     (2 * ((result["2_T_team"].sum() + result["3_T_team"].sum()) + 0.44 * result["1_T_team"].sum())), 1)

result["TS_team"] = result["TS_team"].round(2)
result["TS_opp"] = result["TS_opp"].round(2)


result["REB_PERF"] = ((result["PctDeReRec"]/70)/(result["PctDeReRec"]/70 + (100-result["PctDeReRec"])/30) + ((result["PctOfReRec"])/30)/((100 - result["PctOfReRec"])/70 + (result["PctOfReRec"])/30)).round(2)
result.rename(columns={"TEAM_team": "TEAM"}, inplace=True)
result.rename(columns={"NB_GAME_team": "NB_GAME"}, inplace=True)

result_rank = result.copy()

result_rank[["Rank_PctDeReRec", "Rank_PctOfReRec", "Rank_REB_PERF"]] = result[["PctDeReRec", "PctOfReRec", "REB_PERF"]].rank(ascending=False).astype(int)

result_rank = result_rank.sort_values(by = "Rank_REB_PERF").reset_index(drop = True)

result2 = result[["NB_GAME","TEAM","PPS","TS_team","NB_SHOOT","PPS_opp","TS_opp","NB_SHOOT_opp","PPS_DELTA","NB_SHOOT_DELTA","PP_FT","NB_FT","PP_FT_opp","NB_FT_opp","NB_FT_DELTA","BC","BC_opp","BC_DELTA","PctDeReRec","PctOfReRec","REB_PERF"]]

######################PRINT#############
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
    taille_titre = 60
    st.markdown(
        f'''
        <p style="font-size:{int(taille_titre)}px; text-align: center; padding: 10pxs;">
            <b>Teams DATA - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
        

st.dataframe(result2, height=min(35 + 35*len(result2),900),width=2000,hide_index=True)

col1, col2,col3 = st.columns([0.33,0.33,0.33])



with col1:
    st.header("True Shooting Efficiency")

    # Séparation des variables
    teams = result2["TEAM"]
    X = result2[["TS_team", "TS_opp"]]

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Nuage de points
    ax.scatter(X["TS_team"], X["TS_opp"], c='blue', alpha=0.7)

    # Ajout des logos ou noms des équipes
    for i, team in enumerate(teams):
        x, y = X.loc[i, "TS_team"], X.loc[i, "TS_opp"]
        logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team}.png")
        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            imagebox = OffsetImage(image, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.text(x, y, team, fontsize=9, ha='center', va='center')

    # Ajout des lignes pointillées pour les moyennes
    ax.axvline(TS_MOYEN, color='red', linestyle='--')
    ax.axhline(TS_MOYEN, color='green', linestyle='--')

    # Inversion de l’axe vertical pour que les meilleures défenses soient en haut
    ax.invert_yaxis()

    # Graduation tous les 0.02 sur les deux axes
    x_min, x_max = X["TS_team"].min(), X["TS_team"].max()
    y_min, y_max = X["TS_opp"].min(), X["TS_opp"].max()
    # Arrondir vers le bas pour le min, vers le haut pour le max
    x_start = math.floor(x_min)
    x_end = math.ceil(x_max)
    y_start = math.floor(y_min)
    y_end = math.ceil(y_max)

    # Définir les ticks tous les 1%
    ax.set_xticks(np.arange(x_start, x_end + 1, 1))
    ax.set_yticks(np.arange(y_start, y_end + 1, 1))

    # Mise en forme
    ax.set_xlabel("True Shooting %")
    ax.set_ylabel("Opponent True Shooting %")
    ax.set_title("Team vs Opponent True Shooting Efficiency")
    ax.grid(True, linestyle=':')

    # Sauvegarde et affichage dans Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.image(buf, caption="Team vs Opponent True Shooting Efficiency")



with col2:
    st.header("Ball Care")

    # Séparation des variables
    teams = result2["TEAM"]
    X = result2[["BC", "BC_opp"]]

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Nuage de points
    ax.scatter(X["BC"], X["BC_opp"], c='blue', alpha=0.7)

    # Ajout des logos ou noms des équipes
    for i, team in enumerate(teams):
        x, y = X.loc[i, "BC"], X.loc[i, "BC_opp"]
        logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team}.png")
        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            imagebox = OffsetImage(image, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.text(x, y, team, fontsize=9, ha='center', va='center')

    # Ajout des lignes pointillées pour les moyennes
    ax.axvline(BC_MOYEN, color='red', linestyle='--')
    ax.axhline(BC_MOYEN, color='green', linestyle='--')

    # Inversion de l’axe vertical pour que les meilleures défenses soient en haut
    ax.invert_yaxis()


    x_min, x_max = X["BC"].min(), X["BC"].max()
    y_min, y_max = X["BC_opp"].min(), X["BC_opp"].max()

    # Calculs pour que les axes commencent/finissent sur un multiple de 0.5
    x_start = math.floor(x_min * 2) / 2
    x_end = math.ceil(x_max * 2) / 2
    y_start = math.floor(y_min * 2) / 2
    y_end = math.ceil(y_max * 2) / 2

    # Définir les ticks tous les 0.5
    ax.set_xticks(np.arange(x_start, x_end + 0.5, 0.5))
    ax.set_yticks(np.arange(y_start, y_end + 0.5, 0.5))

    # Mise en forme
    ax.set_xlabel("Ball Care (%)")
    ax.set_ylabel("Opponent Ball Care (%)")
    ax.set_title("Ball Care vs Opponent Ball Care")
    ax.grid(True, linestyle=':')

    # Sauvegarde et affichage dans Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.image(buf, caption="Ball Care vs Opponent Ball Care")

with col3:
    st.header("Rebound Recovery")

    # Séparation des variables
    teams = result2["TEAM"]
    X = result2[["PctDeReRec", "PctOfReRec"]]

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Nuage de points
    ax.scatter(X["PctDeReRec"], X["PctOfReRec"], c='blue', alpha=0.7)

    # Ajout des logos ou noms des équipes
    for i, team in enumerate(teams):
        x, y = X.loc[i, "PctDeReRec"], X.loc[i, "PctOfReRec"]
        logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team}.png")
        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            imagebox = OffsetImage(image, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.text(x, y, team, fontsize=9, ha='center', va='center')

    # Ajout des lignes pointillées pour les moyennes
    ax.axvline(PCT_DE_REB_MOYEN, color='red', linestyle='--')
    ax.axhline(100-PCT_DE_REB_MOYEN, color='green', linestyle='--')



    x_min, x_max = X["PctDeReRec"].min(), X["PctDeReRec"].max()
    y_min, y_max = X["PctOfReRec"].min(), X["PctOfReRec"].max()

    # Arrondir vers le bas pour le min, vers le haut pour le max
    x_start = math.floor(x_min)
    x_end = math.ceil(x_max)
    y_start = math.floor(y_min)
    y_end = math.ceil(y_max)

    # Définir les ticks tous les 1%
    ax.set_xticks(np.arange(x_start, x_end + 1, 1))
    ax.set_yticks(np.arange(y_start, y_end + 1, 1))


    # Mise en forme
    ax.set_xlabel("Defensive Rebound Control (%)")
    ax.set_ylabel("Offensive Rebound Control (%)")
    ax.set_title("Defensive vs Offensive Rebound Control")
    ax.grid(True, linestyle=':')

    # Sauvegarde et affichage dans Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.image(buf, caption="Defensive vs Offensive Rebound Control")



