import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
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
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 60px; /* Largeur minimale */
        max-width: 240px; /* Largeur maximale */
        width: 100px; /* Largeur fixe */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("SETTINGS")

selected_range = st.sidebar.slider(
    "Select a ROUND range:",
    min_value=min_round,
    max_value=max_round,
    value=(min_round, max_round),
    step=1
)

# Filtrage dynamique des équipes

# Filtrage dynamique des équipes


zoom = st.sidebar.slider(
    "Choisissez une valeur de zoom pour les photos",
    min_value=0.3,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)

###################### DATA

notation =f.analyse_per(data_dir,competition,season,R = [i for i in range(selected_range[0],selected_range[1]+1)],CODETEAM = [])

palette = [
    "#FF0000",  
    "#FF4C00",  
    "#FF7200",  
    "#FF9200",  
    "#FFAF00",  
    "#FFCA00", 
    "#FFE500",
    "#FFFF00",  
    "#D9F213",  
    "#B5E422", 
    "#93D52E",  
    "#72C637",  
    "#53B63E",  
    "#32A542",  
    "#009445"   
]


# Diviser les données en 7 classes égales en nombre
notation['CLASS'] = pd.qcut(notation['NOTE'], q=15, labels=range(15))

# Attribuer une couleur en fonction de la classe
notation['COULEUR'] = notation['CLASS'].map(lambda x: palette[int(x)])

notation = notation.sort_values(by = "NOTE",ascending = False)




###################### PRINT

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
    taille_titre = 70*zoom
    st.markdown(
        f'''
        <p style="font-size:{int(taille_titre)}px; text-align: center; padding: 10pxs;">
            <b>PREVIEWS TOOLS - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
        

col1, col2,col3,col4,col5 = st.columns([3,1,2,1,3])

with col2:
    LOCALTEAM = st.selectbox("HOME TEAM", options=sorted(df["TEAM"].unique()) if not df.empty else [],index = 0)
    LOCALTEAM_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{LOCALTEAM}.png")

    st.image(LOCALTEAM_logo_path,  width=int(200*zoom))


with col4 :
    ROADTEAM = st.selectbox("AWAY TEAM", options=sorted(df["TEAM"].unique()) if not df.empty else [],index = 1)
    ROADTEAM_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{ROADTEAM}.png")
    st.image(ROADTEAM_logo_path,  width=int(200*zoom))


######## PARAM COLOR ####################################
teams_color = pd.read_csv(f"datas/{competition}_{season}_teams_colors.csv",sep=";")


local_c1 = teams_color[teams_color["TEAM"]==LOCALTEAM]["COL1"].to_list()[0]
local_c2 = teams_color[teams_color["TEAM"]==LOCALTEAM]["COL2"].to_list()[0]

road_choix_1 = teams_color[teams_color["TEAM"]==ROADTEAM]["COL1"].to_list()[0]
road_choix_2 = teams_color[teams_color["TEAM"]==ROADTEAM]["COL2"].to_list()[0]

road_c1 = f.choisir_maillot(local_c1, [road_choix_1,road_choix_2])
road_c2 = road_choix_1 if road_c1 != road_choix_1 else road_choix_2

####################################
previews1,previews2,_,pronostic = st.columns([0.3,0.3,0.1,0.3])

with previews1 :
    st.markdown(
        f'''
        <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: white;color: black; padding: 4px; border-radius: 5px;outline: 3px solid black;">
            <b>AVERAGES STATS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    stat_name,loc_stat,color_best,road_stat = st.columns([0.4,0.25,0.1,0.25])


    with stat_name :
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>TEAMS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>NB SHOOTS (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>NB SHOOTS (DEF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>PPS (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>PPS (DEF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>% REB (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>% REB (DEF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>% ASSISTS (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>% ASSISTS (DEF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>BALL CARE (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>BALL CARE (DEF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    with loc_stat :
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b>{LOCALTEAM}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

    with road_stat :
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {road_c1};color: {road_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {road_c2};">
                <b>{ROADTEAM}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

    with color_best :
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: white; padding: 4px; border-radius: 5px;outline: 3px solid white;">
                <b>---</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
