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

with_game = gs[gs["GAME"]==GAME].reset_index(drop = True)

# Utiliser une expression régulière pour extraire les valeurs
match = re.match(r"R(\d+)\s*:\s*([A-Za-z]+)\s*-\s*([A-Za-z]+)", GAME)


# Récupérer le nombre après le "R"
r = int(match.group(1))

# Récupérer les deux noms d'équipes
team_local = match.group(2)
team_road = match.group(3)

local_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team_local}.png")
road_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{team_road}.png")

player_stat = f.get_aggregated_data(
    df=df, min_round=r, max_round=r,
    selected_teams=[team_local,team_road],
    selected_opponents=[],
    selected_fields=["TEAM","PLAYER"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)
player_stat = player_stat.sort_values(by = "TIME_ON",ascending = False)

local_player_stat = player_stat[player_stat["TEAM"]==team_local]
road_player_stat = player_stat[player_stat["TEAM"]==team_road]

############################### DATA ###################################################

periode,cumul = f.team_evol_score(team_local,r,r,data_dir,competition,season,type = "MEAN")


############################ PRINT ########################################################

col1, col2,t1,t2 = st.columns([1, 2.5,0.5,0.5])

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

with t1 :
    if os.path.exists(local_logo_path):
        st.image(local_logo_path,  width=int(200*zoom))
    else:
        st.warning(f"Logo introuvable pour l'équipe : {team_local}")
    st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center;">
                <b>{with_game["LOCAL_SCORE"].to_list()[0]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    

with t2 :
    if os.path.exists(road_logo_path):
        st.image(road_logo_path,  width=int(200*zoom))
    else:
        st.warning(f"Logo introuvable pour l'équipe : {team_road}")
    st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center;">
                <b>{with_game["ROAD_SCORE"].to_list()[0]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

cola, colb = st.columns([1, 6])


with cola :
    type_delta = st.selectbox("DELTA", options=["CUMULATED","PER PERIODE"], index=0)

    if type_delta == "PER PERIODE" :
        data_delta = periode
    else : 
        data_delta = cumul
    # Noms des barres
    labels = [i * 2.5 for i in range(1, len(data_delta) + 1)]

    # Couleurs conditionnelles
    colors = ['green' if val > 0 else 'red' if val < 0 else 'yellow' for val in data_delta]

    # Création de la figure
    fig = go.Figure()

    # Ajout des barres
    fig.add_trace(
        go.Bar(
            x=labels,
            y=data_delta,
            marker=dict(color=colors),  # Couleurs dynamiques
        )
    )
    for i in range(3, len(labels)-1, 4):  # Indices toutes les 4 barres (commençant à 3)
        fig.add_shape(
            type="line",
            x0=labels[i] + 1.25,
            y0=min(data_delta) - 1,  # Commence un peu en dessous de la valeur min
            x1=labels[i] + 1.25,
            y1=max(data_delta) + 1,  # Va un peu au-dessus de la valeur max
            line=dict(color="grey", dash="dot", width=2)  # Ligne pointillée jaune
        )

    # Mise en page
    fig.update_layout(
        autosize=True,
        width=int(500*zoom),
        height=int(500*zoom),
        title=f"DELTA {type_delta}",
        xaxis_title="Minutes",
        yaxis_title="Delta",
        xaxis=dict(
            tickmode='linear',
            showticklabels=False  # Supprime les labels sur l'axe X
        ),
        plot_bgcolor="rgba(0,0,0,0)",  # Fond transparent
        paper_bgcolor="rgba(0,0,0,0)",  # Papier transparent (mode sombre Streamlit)
        font=dict(size=12),
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig)

    st.markdown(
        f'''
        <p style="font-size:{int(35*zoom)}px; text-align: center; padding: 10pxs;">
            <b>% SHOOTS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    shoot_local, shoot_road = st.columns([1, 1])

    with shoot_local :
        
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center;">
                <b>{team_local}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(local_player_stat["1_R"].sum()/local_player_stat["1_T"].sum() if local_player_stat["1_T"].sum() != 0 else 0,"1P",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(local_player_stat["2_R"].sum()/local_player_stat["2_T"].sum() if local_player_stat["2_T"].sum() != 0 else 0,"2P",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(local_player_stat["3_R"].sum()/local_player_stat["3_T"].sum() if local_player_stat["3_T"].sum() != 0 else 0,"3P",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    with shoot_road :
        
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center;">
                <b>{team_road}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(road_player_stat["1_R"].sum()/road_player_stat["1_T"].sum() if road_player_stat["1_T"].sum() != 0 else 0,"1P",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(road_player_stat["2_R"].sum()/road_player_stat["2_T"].sum() if road_player_stat["2_T"].sum() != 0 else 0,"2P",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(road_player_stat["3_R"].sum()/road_player_stat["3_T"].sum() if road_player_stat["3_T"].sum() != 0 else 0,"3P",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown(
        f'''
        <p style="font-size:{int(35*zoom)}px; text-align: center; padding: 10pxs;">
            <b>% REBONDS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    reb_local, reb_road = st.columns([1, 1])

    with reb_local :
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center;">
                <b> {team_local}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        fig2 = f.plot_semi_circular_chart(local_player_stat["DR"].sum()/(local_player_stat["DR"].sum() + road_player_stat["OR"].sum()),"DEF.", size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

        fig2 = f.plot_semi_circular_chart(local_player_stat["OR"].sum()/(local_player_stat["OR"].sum() + road_player_stat["DR"].sum()),"OFF.",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    with reb_road :
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center;">
                <b> {team_road}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        fig2 = f.plot_semi_circular_chart(road_player_stat["DR"].sum()/(road_player_stat["DR"].sum() + local_player_stat["OR"].sum()),"DEF.", size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

        fig2 = f.plot_semi_circular_chart(road_player_stat["OR"].sum()/(road_player_stat["OR"].sum() + local_player_stat["DR"].sum()),"OFF.",size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown(
        f'''
        <p style="font-size:{int(35*zoom)}px; text-align: center; padding: 10pxs;">
            <b>% BALL CARE </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    cb_local, cb_road = st.columns([1, 1])
    with cb_local :

        st.markdown(
            f'''
            <p style="font-size:{int(int(30*zoom))}px; text-align: center;">
                <b> {team_local}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        shot_T = local_player_stat["1_T"].sum()/2 + local_player_stat["2_T"].sum() + local_player_stat["3_T"].sum()
        TO = local_player_stat["TO"].sum()
        fig2 = f.plot_semi_circular_chart(shot_T/(shot_T+TO),"BC", size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    with cb_road :
        st.markdown(
            f'''
            <p style="font-size:{int(int(30*zoom))}px; text-align: center;">
                <b> {team_road}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        shot_T = road_player_stat["1_T"].sum()/2 + road_player_stat["2_T"].sum() + road_player_stat["3_T"].sum()
        TO = road_player_stat["TO"].sum()
        fig2 = f.plot_semi_circular_chart(shot_T/(shot_T+TO),"BC", size=int(85*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
with colb :
    t = st.selectbox("TEAM", options=[team_local,team_road], index=0)
    player_stat = player_stat[player_stat["TEAM"]==t].drop(columns = ["TEAM","ROUND","NB_GAME","OPPONENT","HOME","WIN"])
    player_stat = player_stat.sort_values(by = "TIME_ON",ascending = False)

    st.dataframe(player_stat, height=min(36 + 36*len(player_stat),13*36),width=2000,hide_index=True)