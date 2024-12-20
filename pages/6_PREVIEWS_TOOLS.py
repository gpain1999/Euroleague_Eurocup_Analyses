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


################## PICKLE #########################

BCL = f.lire_fichier_pickle("./modeles/model_BCL.pkl")
BCR = f.lire_fichier_pickle("./modeles/model_BCR.pkl")
PPSL = f.lire_fichier_pickle("./modeles/model_PPSL.pkl")
PPSR = f.lire_fichier_pickle("./modeles/model_PPSR.pkl")
FTL = f.lire_fichier_pickle("./modeles/model_1TL.pkl")
FTR = f.lire_fichier_pickle("./modeles/model_1TR.pkl")

RDL = f.lire_fichier_pickle("./modeles/model_RDL.pkl")
RDR = f.lire_fichier_pickle("./modeles/model_RDR.pkl")

W = f.lire_fichier_pickle("./modeles/model_W.pkl")

#################################DATA############################################
local_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[LOCALTEAM],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
local_moyenne = local_moyenne.loc[:, (local_moyenne != "---").any(axis=0)]
local_moyenne = local_moyenne.drop(columns = ["TEAM","NB_GAME","HOME","TIME_ON"])

local_opp_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[LOCALTEAM],
    selected_fields=["OPPONENT"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
local_opp_moyenne = local_opp_moyenne.loc[:, (local_opp_moyenne != "---").any(axis=0)]
local_opp_moyenne = local_opp_moyenne.drop(columns = ["OPPONENT","NB_GAME","HOME","TIME_ON"])

road_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[ROADTEAM],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
road_moyenne = road_moyenne.loc[:, (road_moyenne != "---").any(axis=0)]
road_moyenne = road_moyenne.drop(columns = ["TEAM","NB_GAME","HOME","TIME_ON"])

road_opp_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[ROADTEAM],
    selected_fields=["OPPONENT"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
road_opp_moyenne = road_opp_moyenne.loc[:, (road_opp_moyenne != "---").any(axis=0)]
road_opp_moyenne = road_opp_moyenne.drop(columns = ["OPPONENT","NB_GAME","HOME","WIN","TIME_ON"])


####################################
previews1,pause1,previews2,pause2,pronostic = st.columns([0.25,0.1,0.25,0.15,0.25])

with pronostic :
    st.markdown(
        f'''
        <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: white;color: black; padding: 4px; border-radius: 5px;outline: 3px solid black;">
            <b>PRONOSTICS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    prono_name,loc_prono,road_prono = st.columns([0.46,0.27,0.27])
    ppsl_p,ppsr_p = f.predict_(PPSL,PPSR, LOCALTEAM, ROADTEAM, sorted(df["TEAM"].unique()))
    bcl_p,bcr_p = f.predict_(BCL,BCR, LOCALTEAM, ROADTEAM, sorted(df["TEAM"].unique()))
    ftl_p,ftr_p = f.predict_(FTL,FTR, LOCALTEAM, ROADTEAM, sorted(df["TEAM"].unique()))
    rdl_p,rdr_p = f.predict_(RDL,RDR, LOCALTEAM, ROADTEAM, sorted(df["TEAM"].unique()))
    w_p,_ = f.predict_(W,W, LOCALTEAM, ROADTEAM, sorted(df["TEAM"].unique()))

    local_def = (rdl_p/0.7)/(rdl_p/0.7 + (1-rdl_p)/0.3)
    road_off = ((1-rdl_p)/0.3)/(rdl_p/0.7 + (1-rdl_p)/0.3)
    road_def = ((rdr_p)/0.7)/( rdr_p/0.7 + (1-rdr_p)/0.3)
    local_off = ((1-rdr_p)/0.3)/(rdr_p/0.7 + (1-rdr_p)/0.3)

    with loc_prono :
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
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(ppsl_p, 2):.2f}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*bcl_p, 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(ftl_p, 0):.0f} </b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*rdl_p, 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(1-rdr_p), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(local_def+local_off,2):.2f}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: gold;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*w_p, 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    with road_prono :
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
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(ppsr_p, 2):.2f}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*bcr_p, 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(ftr_p):.0f} </b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*rdr_p, 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(1-rdl_p), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(road_def+road_off,2):.2f}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: gold;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(1-w_p), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    with prono_name :

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
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

        if ppsl_p > ppsr_p:
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>PTS PER SHOOT</b>
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

        if bcl_p > bcr_p:
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>BALL CARE</b>
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

        if ftl_p > ftr_p:
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>FT ATTEMPT.</b>
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

        if rdl_p > rdr_p:
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>% REB DEF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if (1-rdr_p )> (1 - rdl_p):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>% REB OFF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if (local_def+local_off )> (road_def+road_off):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>REB.PERF.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: gold;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if (w_p )> (0.5):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>WINNER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with previews1 :
    st.markdown(
        f'''
        <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: white;color: black; padding: 4px; border-radius: 5px;outline: 3px solid black;">
            <b>AVERAGES STATS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    stat_name,loc_stat,road_stat = st.columns([0.46,0.27,0.27])


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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: grey; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{local_moyenne["WIN"].to_list()[0]}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(local_moyenne["2_T"].sum()+local_moyenne["3_T"].sum(),1):.1f}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(local_opp_moyenne["2_T"].sum()+local_opp_moyenne["3_T"].sum(),1):.1f}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((local_moyenne['3_R'].sum()*3 + local_moyenne['2_R'].sum()*2)/(local_moyenne["2_T"].sum()+local_moyenne["3_T"].sum()), 2):.2f}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((local_opp_moyenne['3_R'].sum()*3 + local_opp_moyenne['2_R'].sum()*2)/(local_opp_moyenne["2_T"].sum()+local_opp_moyenne["3_T"].sum()), 2):.2f}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*local_moyenne['OR'].sum())/(local_opp_moyenne["DR"].sum()+local_moyenne['OR'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*local_moyenne['DR'].sum())/(local_opp_moyenne["OR"].sum()+local_moyenne['DR'].sum()), 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*local_moyenne['AS'].sum())/(local_moyenne["2_R"].sum()+local_moyenne['3_R'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*local_opp_moyenne['AS'].sum())/(local_opp_moyenne["2_R"].sum()+local_opp_moyenne['3_R'].sum()), 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(local_moyenne['1_T'].sum()*0.5 +local_moyenne['2_T'].sum() +local_moyenne['3_T'].sum()  )/(local_moyenne['1_T'].sum()*0.5 +local_moyenne['2_T'].sum() +local_moyenne['3_T'].sum() + local_moyenne['TO'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(local_opp_moyenne['1_T'].sum()*0.5 +local_opp_moyenne['2_T'].sum() +local_opp_moyenne['3_T'].sum()  )/(local_opp_moyenne['1_T'].sum()*0.5 +local_opp_moyenne['2_T'].sum() +local_opp_moyenne['3_T'].sum() + local_opp_moyenne['TO'].sum()), 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: grey;color: grey; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{road_moyenne["WIN"].to_list()[0]}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(road_moyenne["2_T"].sum()+road_moyenne["3_T"].sum(),1):.1f}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(road_opp_moyenne["2_T"].sum()+road_opp_moyenne["3_T"].sum(),1):.1f}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((road_moyenne['3_R'].sum()*3 + road_moyenne['2_R'].sum()*2)/(road_moyenne["2_T"].sum()+road_moyenne["3_T"].sum()), 2):.2f}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((road_opp_moyenne['3_R'].sum()*3 + road_opp_moyenne['2_R'].sum()*2)/(road_opp_moyenne["2_T"].sum()+road_opp_moyenne["3_T"].sum()), 2):.2f}</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*road_moyenne['OR'].sum())/(road_opp_moyenne["DR"].sum()+road_moyenne['OR'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*road_moyenne['DR'].sum())/(road_opp_moyenne["OR"].sum()+road_moyenne['DR'].sum()), 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*road_moyenne['AS'].sum())/(road_moyenne["2_R"].sum()+road_moyenne['3_R'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((100*road_opp_moyenne['AS'].sum())/(road_opp_moyenne["2_R"].sum()+road_opp_moyenne['3_R'].sum()), 1):.1f} %</b>
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
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(road_moyenne['1_T'].sum()*0.5 +road_moyenne['2_T'].sum() +road_moyenne['3_T'].sum()  )/(road_moyenne['1_T'].sum()*0.5 +road_moyenne['2_T'].sum() +road_moyenne['3_T'].sum() + road_moyenne['TO'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*(road_opp_moyenne['1_T'].sum()*0.5 +road_opp_moyenne['2_T'].sum() +road_opp_moyenne['3_T'].sum()  )/(road_opp_moyenne['1_T'].sum()*0.5 +road_opp_moyenne['2_T'].sum() +road_opp_moyenne['3_T'].sum() + road_opp_moyenne['TO'].sum()), 1):.1f} %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    with stat_name :

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: black;color: orange; padding: 4px; border-radius: 5px;outline: 3px solid orange;">
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


        if int(local_moyenne["WIN"].to_list()[0].strip('%')) > int(road_moyenne["WIN"].to_list()[0].strip('%')):
            c1 =local_c1
            c2 = local_c2
        elif int(local_moyenne["WIN"].to_list()[0].strip('%'))<  int(road_moyenne["WIN"].to_list()[0].strip('%')):
            c1 =road_c1
            c2 = road_c2
        else :
            c1 = "orange"
            c2 = "black"

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>% WIN</b>
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

        if (local_moyenne["2_T"].sum()+local_moyenne["3_T"].sum())>(road_moyenne["2_T"].sum()+road_moyenne["3_T"].sum()):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>NB SHOOTS (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if (local_opp_moyenne["2_T"].sum()+local_opp_moyenne["3_T"].sum())<(road_opp_moyenne["2_T"].sum()+road_opp_moyenne["3_T"].sum()):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
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

        if ((local_moyenne['3_R'].sum()*3 + local_moyenne['2_R'].sum()*2)/(local_moyenne["2_T"].sum()+local_moyenne["3_T"].sum()))>((road_moyenne['3_R'].sum()*3 + road_moyenne['2_R'].sum()*2)/(road_moyenne["2_T"].sum()+road_moyenne["3_T"].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>PPS (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if ((local_opp_moyenne['3_R'].sum()*3 + local_opp_moyenne['2_R'].sum()*2)/(local_opp_moyenne["2_T"].sum()+local_opp_moyenne["3_T"].sum()))<((road_opp_moyenne['3_R'].sum()*3 + road_opp_moyenne['2_R'].sum()*2)/(road_opp_moyenne["2_T"].sum()+road_opp_moyenne["3_T"].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
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
        if ((100*local_moyenne['OR'].sum())/(local_opp_moyenne["DR"].sum()+local_moyenne['OR'].sum())) > ((100*road_moyenne['OR'].sum())/(road_opp_moyenne["DR"].sum()+road_moyenne['OR'].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>% REB OFF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if ((100*local_moyenne['DR'].sum())/(local_opp_moyenne["OR"].sum()+local_moyenne['DR'].sum())) > ((100*road_moyenne['DR'].sum())/(road_opp_moyenne["OR"].sum()+road_moyenne['DR'].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>% REB DEF</b>
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
        if ((100*local_moyenne['AS'].sum())/(local_moyenne["2_R"].sum()+local_moyenne['3_R'].sum()))>((100*road_moyenne['AS'].sum())/(road_moyenne["2_R"].sum()+road_moyenne['3_R'].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>% ASSISTS (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if ((100*local_opp_moyenne['AS'].sum())/(local_opp_moyenne["2_R"].sum()+local_opp_moyenne['3_R'].sum()))<((100*road_opp_moyenne['AS'].sum())/(road_opp_moyenne["2_R"].sum()+road_opp_moyenne['3_R'].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
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
        if (100*(local_moyenne['1_T'].sum()*0.5 +local_moyenne['2_T'].sum() +local_moyenne['3_T'].sum()  )/(local_moyenne['1_T'].sum()*0.5 +local_moyenne['2_T'].sum() +local_moyenne['3_T'].sum() + local_moyenne['TO'].sum()))>(100*(road_moyenne['1_T'].sum()*0.5 +road_moyenne['2_T'].sum() +road_moyenne['3_T'].sum()  )/(road_moyenne['1_T'].sum()*0.5 +road_moyenne['2_T'].sum() +road_moyenne['3_T'].sum() + road_moyenne['TO'].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>BALL CARE (OFF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if (100*(local_opp_moyenne['1_T'].sum()*0.5 +local_opp_moyenne['2_T'].sum() +local_opp_moyenne['3_T'].sum()  )/(local_opp_moyenne['1_T'].sum()*0.5 +local_opp_moyenne['2_T'].sum() +local_opp_moyenne['3_T'].sum() + local_opp_moyenne['TO'].sum()))<(100*(road_opp_moyenne['1_T'].sum()*0.5 +road_opp_moyenne['2_T'].sum() +road_opp_moyenne['3_T'].sum()  )/(road_opp_moyenne['1_T'].sum()*0.5 +road_opp_moyenne['2_T'].sum() +road_opp_moyenne['3_T'].sum() + road_opp_moyenne['TO'].sum())):
            c1 =local_c1
            c2 = local_c2
        else :
            c1 =road_c1
            c2 = road_c2
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {c1};color: {c2}; padding: 4px; border-radius: 5px;outline: 3px solid {c2};">
                <b>BALL CARE (DEF)</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
