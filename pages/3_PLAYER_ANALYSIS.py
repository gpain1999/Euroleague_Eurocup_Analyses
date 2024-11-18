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
gs = pd.read_csv(os.path.join(data_dir, f"{competition}_gs_{season}.csv"))

gs = gs[["Round","local.club.code","local.score","road.score","road.club.code"]]
gs.columns = ["ROUND","LOCAL","SL","SR","ROAD"]
# Charger les données
df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
         'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

try:
    image = Image.open(image_path)
    # Redimensionner l'image (par exemple, largeur de 300 pixels)
    max_width = 600
    image = image.resize((max_width, int(image.height * (max_width / image.width))))

    # Afficher l'image redimensionnée
    st.image(image, caption=f"Rapport pour {competition}")
except FileNotFoundError:
    st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

    
st.title("Player Analysis - by gpain1999 ")

# Remplir les trous dans ROUND
min_round, max_round = df["ROUND"].min(), df["ROUND"].max()

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
CODETEAM = st.sidebar.selectbox("Équipe Sélectionnée", options=sorted(df["TEAM"].unique()) if not df.empty else [])

# Mise à jour dynamique des joueurs en fonction des équipes sélectionnées
if CODETEAM:
    available_players = players[players["CODETEAM"].isin([CODETEAM])]["PLAYER"].unique()
else:
    available_players = players["PLAYER"].unique()

selected_players = st.sidebar.selectbox("Joueur Sélectionné", options=sorted(available_players))

selected_stats = st.sidebar.selectbox("Stat Sélectionné", options=["PER","I_PER","PTS","TR","AS","PM_ON"])
window_size = st.sidebar.number_input("Moyenne glissante", min_value=1,max_value=max_round ,value=3)


filtered_df = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["ROUND", "PLAYER", "TEAM"],
    selected_players=[selected_players],
    mode="CUMULATED",
    percent="PERCENT"
)
gs_filtered = gs[(gs["LOCAL"]==CODETEAM)|(gs["ROAD"]==CODETEAM)].reset_index(drop = True)
gs_filtered['WIN'] = gs_filtered.apply(lambda row: 'YES' if 
                     (row['LOCAL'] == CODETEAM and row['SL'] > row['SR']) or 
                     (row['ROAD'] == CODETEAM and row['SL'] < row['SR']) 
                     else 'NO', axis=1)

gs_filtered['OPPONENT'] = gs_filtered.apply(lambda row: row['LOCAL'] if 
                     (row['ROAD'] == CODETEAM)
                     else row['ROAD'], axis=1)

gs_filtered['HOME'] = gs_filtered.apply(lambda row: 'NO' if 
                     (row['ROAD'] == CODETEAM)
                     else 'YES', axis=1)

NAME_PLAYER = filtered_df["PLAYER"].to_list()[0]
NUMBER_PLAYER = filtered_df["#"].to_list()[0]
TEAM_PLAYER = filtered_df["TEAM"].to_list()[0]
PLAYER_ID = players[(players["CODETEAM"] == TEAM_PLAYER) & (players["PLAYER"] == NAME_PLAYER)]["PLAYER_ID"].to_list()[0]

filtered_df = filtered_df.drop(columns=["NB_GAME", "PLAYER", "#", "TEAM"])


# Calcul de la moyenne glissante en tenant compte des rounds manquants
def calculate_moving_average(df, column, round_column, window_size):
    moving_avg = []
    for current_round in df[round_column]:
        # Filtrer les rounds à inclure dans la moyenne
        valid_rows = df[(df[round_column] <= current_round) & (df[round_column] > current_round - window_size)]
        moving_avg.append(valid_rows[column].mean())
    return moving_avg

# Ajout de la colonne "moving_avg"
filtered_df["moving_avg"] = calculate_moving_average(filtered_df, selected_stats, "ROUND",window_size)

# Création du graphique avec la nouvelle courbe de moyenne glissante
fig = go.Figure()

# Barres pour les stats
for i, row in filtered_df.iterrows():
    color = "green" if row["WIN"] == "YES" else "red"
    fig.add_trace(go.Bar(
        x=[row["ROUND"]],
        y=[row[selected_stats]],
        marker_color=color,
        yaxis="y1",
        showlegend=False
    ))

# Courbe pour les minutes
fig.add_trace(go.Scatter(
    x=filtered_df["ROUND"],
    y=filtered_df["TIME_ON"],
    mode="lines+markers",
    line=dict(color='blue', width=2),
    marker=dict(size=8, symbol="circle", color="blue"),
    yaxis="y2",
    showlegend=True,
    name="Minutes jouées"
))

# Nouvelle courbe pour la moyenne glissante
fig.add_trace(go.Scatter(
    x=filtered_df["ROUND"],
    y=filtered_df["moving_avg"],
    mode="lines+markers",
    line=dict(color='orange', width=2, dash="dot"),
    marker=dict(size=8, symbol="square", color="orange"),
    yaxis="y1",
    showlegend=True,
    name=f"Moyenne glissante ({window_size} derniers rounds)"
))

# Mise à jour du layout
fig.update_layout(
    title=f'#{NUMBER_PLAYER} {NAME_PLAYER} ({TEAM_PLAYER}) : {selected_stats} et Minutes',
    xaxis=dict(
        title="ROUND",
        showgrid=False,
        tickmode='linear',
        tick0=min(filtered_df["ROUND"]),
    ),
    yaxis=dict(
        title=selected_stats,
        titlefont=dict(color="grey"),
        tickfont=dict(color="grey"),
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
    ),
    yaxis2=dict(
        title="Minutes",
        titlefont=dict(color="grey"),
        tickfont=dict(color="grey"),
        overlaying="y",
        side="right",
        showgrid=False
    ),
    legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="top",
        y=-0.3,  # Position sous le graphique
        xanchor="center",
        x=0.5  # Centré horizontalement
    ),
    margin=dict(l=50, r=50, t=50, b=100),  # Augmenter la marge inférieure pour la légende
)


player_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_PLAYER}_{PLAYER_ID}.png")
team_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_PLAYER}.png")

col1, col2, col3,col4 = st.columns([1,2, 6, 4])

with col1:
    if os.path.exists(team_logo_path):
        st.image(team_logo_path, caption=f"Équipe : {TEAM_PLAYER}", width=100)
    else:
        st.warning(f"Logo introuvable pour l'équipe : {TEAM_PLAYER}")


with col2 :
    if os.path.exists(player_image_path):
        st.image(player_image_path, caption=f"#{NUMBER_PLAYER} {NAME_PLAYER}", width=250)
    else:
        st.warning(f"Image introuvable pour le joueur : {NAME_PLAYER}")


with col3:
    st.plotly_chart(fig, use_container_width=True)

with col4:
    # Création des boxplots avec Plotly Graph Objects
    box_fig = go.Figure()

    # Boxplot global pour PER
    box_fig.add_trace(go.Box(
        y=filtered_df[selected_stats],
        name="Global",
        marker_color="blue"
    ))

    # Boxplots pour chaque modalité de WIN
    for win_status in filtered_df["WIN"].unique():
        box_fig.add_trace(go.Box(
            y=filtered_df[filtered_df["WIN"] == win_status][selected_stats],
            name=f"WIN = {win_status}",
            marker_color="green" if win_status == "YES" else "red"
        ))

    # Configuration du layout des boxplots
    box_fig.update_layout(
        title=f"Distribution de {selected_stats} (Global et par resultat)",
        yaxis=dict(
            title=selected_stats,
            showgrid=True
        ),
        xaxis=dict(
            title="Catégories",
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Affichage des boxplots
    st.header(f"Boxplots de {selected_stats}")
    st.plotly_chart(box_fig, use_container_width=True)

def highlight_win(row):
    # Appliquer vert si WIN est "YES", rouge sinon
    color = 'background-color: #CCFFCC; color: black;' if row["WIN"] == "YES" else 'background-color: #FFCCCC; color: black;'
    return [color for _ in row.index]

# Fonction pour appliquer un format conditionnel par colonne
def format_columns(value, col):
    if isinstance(value, (int, float)):  # Vérifie si la valeur est numérique
        if col in ["PER", "I_PER", "TIME_ON"]:  # Colonnes avec 1 chiffre après la virgule
            return f"{value:.1f}"
        else:  # Autres colonnes numériques avec 0 chiffre après la virgule
            return f"{value:.0f}"
    return value  # Ne change pas les valeurs non numériques (str)

# Appliquer un format par colonne au DataFrame stylisé
def format_dataframe(df):
    return df.style.apply(highlight_win, axis=1).format(
        lambda x, col: format_columns(x, col),
        subset=df.columns
    )

filtered_df2 = filtered_df.drop(columns=["WIN","OPPONENT","HOME","moving_avg"])

# Jointure gauche
df_resultat = pd.merge(
    gs_filtered, 
    filtered_df2, 
    on='ROUND', 
    how='left'
)


colonnes_a_convertir = ['PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'TO', '1_T', '2_T', '3_T', 'CO', 'FP', 'CF', 'NCF']

# Conversion en type Int64 (supporte les NA)
df_resultat[colonnes_a_convertir] = df_resultat[colonnes_a_convertir].astype('Int64')


# Appliquer le style sur le DataFrame
styled_df = df_resultat.style.apply(highlight_win, axis=1).format(precision=1)  

# Afficher le tableau stylé dans Streamlit
st.header("Stats Joueur")
st.dataframe(styled_df, use_container_width=True)

