import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import re
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
import fonctions as f

st.set_page_config(
    page_title="ROUND ANALYSIS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)


image_path = f"images/{competition}.png"  # Chemin vers l'image




# SETTINGS interactifs



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

teams_color = pd.read_csv(f"datas/{competition}_{season}_teams_colors.csv",sep=";")
gs.columns = ["GAMECODE", "ROUND", "LOCAL_TEAM", "LOCAL_SCORE", "ROAD_TEAM", "ROAD_SCORE"]

players = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

######################### PARAM
st.sidebar.header("SETTINGS")

zoom = st.sidebar.slider(
    "Choose a zoom value for photos and graphs",
    min_value=0.3,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)

SUB = st.sidebar.number_input("ROUND", min_value=gs["ROUND"].min(),max_value=gs["ROUND"].max() ,value=gs["ROUND"].max())

############################ DATA ########################################################


df_round = df[df["ROUND"]==SUB]
gs_round = gs[gs["ROUND"]==SUB]

team_stat = f.get_aggregated_data(
    df=df, min_round=SUB, max_round=SUB,
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)

player_stat = f.get_aggregated_data(
    df=df, min_round=SUB, max_round=SUB,
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM","PLAYER"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)
player_stat = player_stat.sort_values(by=["I_PER","PER", "TIME_ON"], ascending=[False, False, True])
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

    taille_titre = 70*zoom
    st.markdown(
        f'''
        <p style="font-size:{int(taille_titre)}px; text-align: center; padding: 10pxs;">
            <b>Round Analysis - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )




st.markdown(
    f'''
    <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

games_part, _,players_part = st.columns([0.3, 0.1,0.6])


with games_part :
    PPS,BC,RP = st.columns([0.33, 0.33,0.33])

    with RP :
        st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: #FFED00;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #E2007A;">
                <b>Reb. Perf.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
 
    with BC :
        st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: #E2007A;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #009EE0;">
                <b>Ball Care</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    with PPS :
        st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: #009EE0;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #FFED00;">
                <b>PPS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

    st.markdown(
        f'''
        <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    for index, row in gs_round.iterrows():
        local_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{row['LOCAL_TEAM']}.png")
        road_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{row['ROAD_TEAM']}.png")
        if row["LOCAL_SCORE"]> row["ROAD_SCORE"] :
            cl1 = "#009036"
            cl2 = "#96BF0D"
            cr1 = "#E2001A"
            cr2 = "#E3004F"
        else : 
            cr1 = "#009036"
            cr2 = "#96BF0D"
            cl1 = "#E2001A"
            cl2 = "#E3004F"   

        local_team_stat = team_stat[team_stat["TEAM"]==row['LOCAL_TEAM']]
        road_team_stat = team_stat[team_stat["TEAM"]==row['ROAD_TEAM']]

        local_def_perc = local_team_stat["DR"].sum()/(local_team_stat["DR"].sum() + road_team_stat["OR"].sum())
        road_off_perc = 1 - local_def_perc
        road_def_perc = road_team_stat["DR"].sum()/(road_team_stat["DR"].sum() + local_team_stat["OR"].sum())
        local_off_perc = 1 - road_def_perc

        local_def = (local_def_perc/0.7)/(local_def_perc/0.7 + road_off_perc/0.3)
        road_off = (road_off_perc/0.3)/(local_def_perc/0.7 + road_off_perc/0.3)
        road_def = (road_def_perc/0.7)/( road_def_perc/0.7 + local_off_perc/0.3)
        local_off = (local_off_perc/0.3)/(road_def_perc/0.7 + local_off_perc/0.3)

        LD,LT,_,RT,RD = st.columns([0.20,0.30, 0.1,0.30,0.20])

        with LD :
            st.markdown(
                f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: #009EE0;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #FFED00;">
                    <b>{round((local_team_stat["2_R"].sum()*2+local_team_stat["3_R"].sum()*3)/(local_team_stat["2_T"].sum()+local_team_stat["3_T"].sum()),2):.2f}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
            shot_T = local_team_stat["1_T"].sum()/2 + local_team_stat["2_T"].sum() + local_team_stat["3_T"].sum()
            TO = local_team_stat["TO"].sum()

            st.markdown(
                f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: #E2007A;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #009EE0;">
                    <b>{round(100*shot_T/(shot_T+TO),1)} % </b>
                </p>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(
                f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: #FFED00;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #E2007A;">
                    <b>{round(local_def+local_off,2):.2f}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )

        with RD :
            st.markdown(
                f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: #009EE0;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #FFED00;">
                    <b>{round((road_team_stat["2_R"].sum()*2+road_team_stat["3_R"].sum()*3)/(road_team_stat["2_T"].sum()+road_team_stat["3_T"].sum()),2):.2f}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
            shot_T = road_team_stat["1_T"].sum()/2 + road_team_stat["2_T"].sum() + road_team_stat["3_T"].sum()
            TO = road_team_stat["TO"].sum()

            st.markdown(
                f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: #E2007A;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #009EE0;">
                    <b>{round(100*shot_T/(shot_T+TO),1)} % </b>
                </p>
                ''',
                unsafe_allow_html=True
            )
            st.markdown(
                f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: #FFED00;color: black; padding: 4px; border-radius: 5px;outline: 3px solid #E2007A;">
                    <b>{round(road_def+road_off,2):.2f}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
        with LT : 
            st.image(local_logo_path,  width=int(133*zoom))
            st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: {cl1};color: black; padding: 4px; border-radius: 5px;outline: 3px solid {cl2};">
                <b>{row["LOCAL_SCORE"]}</b>
            </p>
            ''',
            unsafe_allow_html=True
            )

        with RT : 
            st.image(road_logo_path,  width=int(133*zoom))
            st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: {cr1};color: black; padding: 4px; border-radius: 5px;outline: 3px solid {cr2};">
                <b>{row["ROAD_SCORE"]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )


        


        




        st.markdown(
        f'''
        <p style="font-size:{int(10*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
        )

with players_part :
    st.header(f"TEAMS PERFORMANCES : ")

    st.markdown(
        f'''
        <p style="font-size:{int(10*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
        )
    st.header(f"PLAYERS PERFORMANCES : ")

    st.markdown(
        f'''
        <p style="font-size:{int(10*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
        )
