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
result["BC_opp"] = (100*(result["1_T_opp"]*0.5 + result["2_T_opp"] + result["3_T_opp"])/(result["1_T_opp"]*0.5 + result["2_T_opp"] + result["3_T_opp"] + result["TO_opp"])).round(1)

result["BC_DELTA"] = (result["BC"] - result["BC_opp"]).round(1)

result["PctDeReRec"] = (100*(result["DR_team"])/(result["DR_team"] + result["OR_opp"])).round(1)
result["PctOfReRec"] = (100*(result["OR_team"])/(result["OR_team"] + result["DR_opp"])).round(1)




result["REB_PERF"] = ((result["PctDeReRec"]/70)/(result["PctDeReRec"]/70 + (100-result["PctDeReRec"])/30) + ((result["PctOfReRec"])/30)/((100 - result["PctOfReRec"])/70 + (result["PctOfReRec"])/30)).round(2)
result.rename(columns={"TEAM_team": "TEAM"}, inplace=True)
result.rename(columns={"NB_GAME_team": "NB_GAME"}, inplace=True)

result_rank = result.copy()

result_rank[["Rank_PctDeReRec", "Rank_PctOfReRec", "Rank_REB_PERF"]] = result[["PctDeReRec", "PctOfReRec", "REB_PERF"]].rank(ascending=False).astype(int)

result_rank = result_rank.sort_values(by = "Rank_REB_PERF").reset_index(drop = True)

result2 = result[["NB_GAME","TEAM","PPS","NB_SHOOT","PPS_opp","NB_SHOOT_opp","PPS_DELTA","NB_SHOOT_DELTA","PP_FT","NB_FT","PP_FT_opp","NB_FT_opp","NB_FT_DELTA","BC","BC_opp","BC_DELTA","PctDeReRec","PctOfReRec","REB_PERF"]]

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



with col1 :
    st.header("ACP : OFFENSIVE")

    # Séparation de la colonne "TEAM" et des variables pour l'ACP
    teams = result2["TEAM"]
    X = result2[["PPS","NB_FT","BC","PctOfReRec"]]

    # Standardisation des données manuellement (en utilisant les moyennes et les écarts-types)
    X_standardized = (X - X.mean()) / X.std()

    # Application de l'ACP avec SVD (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

    # Les composantes principales sont dans la matrice Vt (les lignes sont les composantes)
    X_pca = U @ np.diag(S)  # Reconstituer les coordonnées des observations dans l'espace principal

    # Création d'un DataFrame pour les résultats de l'ACP
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["TEAM"] = teams

    # Visualisation du premier plan factoriel
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_df["PC1"], pca_df["PC2"], c='blue', alpha=0.7)

    for i, team in enumerate(teams):
        x, y = pca_df.loc[i, "PC1"], pca_df.loc[i, "PC2"]
        logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team}.png")
        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            imagebox = OffsetImage(image, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.text(x, y, team, fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Premier plan factoriel")
    ax.grid()

    # Sauvegarder et afficher l'image dans Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Premier plan factoriel")

    # Cercle des corrélations (utilisation de Vt pour les directions des axes)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 10))

    # Vt contient les directions des axes (composantes principales)
    for i in range(len(X.columns)):
        ax_corr.arrow(0, 0, Vt[0, i], Vt[1, i],
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax_corr.text(Vt[0, i] * 1.1, Vt[1, i] * 1.1, X.columns[i], color='black', ha='center', va='center')

    # Cercle des corrélations
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax_corr.add_artist(circle)
    ax_corr.set_xlim(-1.1, 1.1)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.set_xlabel("PC1")
    ax_corr.set_ylabel("PC2")
    ax_corr.set_title("Cercle des corrélations")
    ax_corr.grid()
    ax_corr.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_corr.axvline(0, color='black', linewidth=0.5, linestyle='--')

    buf_corr = BytesIO()
    fig_corr.savefig(buf_corr, format="png")
    buf_corr.seek(0)
    st.image(buf_corr, caption="Cercle des corrélations")
with col2 : 
    st.header("ACP : DEFENSIVE")

    # Séparation de la colonne "TEAM" et des variables pour l'ACP
    teams = result2["TEAM"]
    X = result2[["PPS_opp","NB_FT_opp","BC_opp","PctDeReRec"]]

    # Standardisation des données manuellement (en utilisant les moyennes et les écarts-types)
    X_standardized = (X - X.mean()) / X.std()

    # Application de l'ACP avec SVD (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

    # Les composantes principales sont dans la matrice Vt (les lignes sont les composantes)
    X_pca = U @ np.diag(S)  # Reconstituer les coordonnées des observations dans l'espace principal

    # Création d'un DataFrame pour les résultats de l'ACP
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["TEAM"] = teams

    # Visualisation du premier plan factoriel
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_df["PC1"], pca_df["PC2"], c='blue', alpha=0.7)

    for i, team in enumerate(teams):
        x, y = pca_df.loc[i, "PC1"], pca_df.loc[i, "PC2"]
        logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team}.png")
        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            imagebox = OffsetImage(image, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.text(x, y, team, fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Premier plan factoriel")
    ax.grid()

    # Sauvegarder et afficher l'image dans Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Premier plan factoriel")

    # Cercle des corrélations (utilisation de Vt pour les directions des axes)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 10))

    # Vt contient les directions des axes (composantes principales)
    for i in range(len(X.columns)):
        ax_corr.arrow(0, 0, Vt[0, i], Vt[1, i],
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax_corr.text(Vt[0, i] * 1.1, Vt[1, i] * 1.1, X.columns[i], color='black', ha='center', va='center')

    # Cercle des corrélations
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax_corr.add_artist(circle)
    ax_corr.set_xlim(-1.1, 1.1)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.set_xlabel("PC1")
    ax_corr.set_ylabel("PC2")
    ax_corr.set_title("Cercle des corrélations")
    ax_corr.grid()
    ax_corr.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_corr.axvline(0, color='black', linewidth=0.5, linestyle='--')

    buf_corr = BytesIO()
    fig_corr.savefig(buf_corr, format="png")
    buf_corr.seek(0)
    st.image(buf_corr, caption="Cercle des corrélations")
with col3 :
    st.header("ACP : DELTA")
    # Séparation de la colonne "TEAM" et des variables pour l'ACP
    teams = result2["TEAM"]
    X = result2[["PPS_DELTA","NB_SHOOT_DELTA","NB_FT_DELTA","BC_DELTA","REB_PERF"]]

    # Standardisation des données manuellement (en utilisant les moyennes et les écarts-types)
    X_standardized = (X - X.mean()) / X.std()

    # Application de l'ACP avec SVD (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

    # Les composantes principales sont dans la matrice Vt (les lignes sont les composantes)
    X_pca = U @ np.diag(S)  # Reconstituer les coordonnées des observations dans l'espace principal

    # Création d'un DataFrame pour les résultats de l'ACP
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["TEAM"] = teams

    # Visualisation du premier plan factoriel
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_df["PC1"], pca_df["PC2"], c='blue', alpha=0.7)

    for i, team in enumerate(teams):
        x, y = pca_df.loc[i, "PC1"], pca_df.loc[i, "PC2"]
        logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team}.png")
        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            imagebox = OffsetImage(image, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            ax.text(x, y, team, fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Premier plan factoriel")
    ax.grid()

    # Sauvegarder et afficher l'image dans Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Premier plan factoriel")

    # Cercle des corrélations (utilisation de Vt pour les directions des axes)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 10))

    # Vt contient les directions des axes (composantes principales)
    for i in range(len(X.columns)):
        ax_corr.arrow(0, 0, Vt[0, i], Vt[1, i],
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax_corr.text(Vt[0, i] * 1.1, Vt[1, i] * 1.1, X.columns[i], color='black', ha='center', va='center')

    # Cercle des corrélations
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax_corr.add_artist(circle)
    ax_corr.set_xlim(-1.1, 1.1)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.set_xlabel("PC1")
    ax_corr.set_ylabel("PC2")
    ax_corr.set_title("Cercle des corrélations")
    ax_corr.grid()
    ax_corr.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_corr.axvline(0, color='black', linewidth=0.5, linestyle='--')

    buf_corr = BytesIO()
    fig_corr.savefig(buf_corr, format="png")
    buf_corr.seek(0)
    st.image(buf_corr, caption="Cercle des corrélations")