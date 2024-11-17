import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory


import fonctions as f

image_path = f"images/{competition}.png"  # Chemin vers l'image

players = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

# Charger les données
df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
         'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


if os.path.exists(image_path):
    # Charger l'image
    img = Image.open(image_path)

    # Ajouter du CSS pour positionner l'image en haut à droite
    st.markdown(
        """
        <style>
            .top-right-image {
                position: absolute;
                top: 10px;
                right: 100px;
                z-index: 1;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Insérer l'image en utilisant HTML
    st.markdown(
        f"""
        <div class="top-right-image">
            <img src="data:image/png;base64,{st.image(image_path)}" alt="Logo" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning(f"L'image {image_path} est introuvable. Veuillez vérifier le chemin.")


# Titre de l'application
st.title("Diagramme : PER  et Minutes ")


# Remplir les trous dans ROUND
min_round, max_round = df["ROUND"].min(), df["ROUND"].max()

# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("Paramètres")

selected_range = st.sidebar.slider(
    "Sélectionnez une plage de ROUND :",
    min_value=min_round,
    max_value=max_round,
    value=(min_round, max_round),  # Valeurs initiales : tout afficher
    step=1
)


# Filtrage dynamique des équipes
CODETEAM = st.sidebar.selectbox("Équipe Sélectionnée", options=df["TEAM"].unique() if not df.empty else [])

# Mise à jour dynamique des joueurs en fonction des équipes sélectionnées
if CODETEAM:
    available_players = players[players["CODETEAM"].isin([CODETEAM])]["PLAYER"].unique()
else:
    available_players = players["PLAYER"].unique()

selected_players = st.sidebar.selectbox("Joueur Sélectionné", options=available_players)


filtered_df = f.get_aggregated_data(
    df = df, min_round = selected_range[0], max_round = selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["ROUND","PLAYER","TEAM"],
    selected_players=[selected_players],
    mode="CUMULATED",
    percent="PERCENT"
)

NAME_PLAYER = filtered_df["PLAYER"].to_list()[0]
NUMBER_PLAYER = filtered_df["#"].to_list()[0]
TEAM_PLAYER = filtered_df["TEAM"].to_list()[0]
PLAYER_ID = players[(players["CODETEAM"]==TEAM_PLAYER)&(players["PLAYER"]==NAME_PLAYER)]["PLAYER_ID"].to_list()[0]

filtered_df = filtered_df.drop(columns = ["NB_GAME","PLAYER","#","TEAM"] )

# Création du graphique avec Plotly Graph Objects
fig = go.Figure()

# Ajouter les barres pour PER avec des couleurs basées sur WIN
for i, row in filtered_df.iterrows():
    color = "green" if row["WIN"] == "YES" else "red"
    fig.add_trace(go.Bar(
        x=[row["ROUND"]],
        y=[row["PER"]],
        marker_color=color,
        yaxis="y1",
        showlegend=False  # Désactiver la légende pour les barres
    ))

# Ajouter la courbe pour Minutes
fig.add_trace(go.Scatter(
    x=filtered_df["ROUND"],
    y=filtered_df["TIME_ON"],
    mode="lines+markers",
    line=dict(color='blue', width=2),  # Courbe bleue
    marker=dict(size=8, symbol="circle", color="blue"),  # Points bleus
    yaxis="y2",
    showlegend=False  # Désactiver la légende pour la courbe
))

# Configuration des axes
fig.update_layout(
    title=f'#{NUMBER_PLAYER} {NAME_PLAYER} ({TEAM_PLAYER}) : GRAPH',
    xaxis=dict(
        title="ROUND",
        showgrid=False,  # Désactiver la grille sur l'axe X
        tickmode='linear',  # Garder des ticks réguliers
        tick0=min_round,  # Aligner sur le premier ROUND
    ),
    yaxis=dict(
        title="PER",
        titlefont=dict(color="grey"),
        tickfont=dict(color="grey"),
        showgrid=True,  # Grille légère pour Y1
        gridcolor="rgba(200,200,200,0.3)",  # Couleur claire pour la grille
    ),
    yaxis2=dict(
        title="Minutes",
        titlefont=dict(color="grey"),
        tickfont=dict(color="grey"),
        overlaying="y",
        side="right",
        showgrid=False  # Pas de grille pour Y2
    ),
    margin=dict(l=50, r=50, t=50, b=50),  # Marges autour du graphique
)

# Chemin vers la photo du joueur
player_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_PLAYER}_{PLAYER_ID}.png")
team_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_PLAYER}.png")

# Mise en page : Ajouter la photo du joueur à gauche
col1, col2, col3 = st.columns([1, 3, 2])  # Ajuster les colonnes pour inclure une photo à gauche

with col1:
    # Afficher la photo du joueur si elle existe
    if os.path.exists(player_image_path):
        st.image(player_image_path, caption=f"#{NUMBER_PLAYER} {NAME_PLAYER}", width=150)
    else:
        st.warning(f"Image introuvable pour le joueur : {NAME_PLAYER}")
    
    # Afficher le logo du club si le fichier existe
    if os.path.exists(team_logo_path):
        st.image(team_logo_path, caption=f"Équipe : {TEAM_PLAYER}", width=100)
    else:
        st.warning(f"Logo introuvable pour l'équipe : {TEAM_PLAYER}")

with col2:
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Afficher les données filtrées
    st.header("Données filtrées")
    st.dataframe(filtered_df)
