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

st.set_page_config(
    page_title="TEAM ANALYSIS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)


import fonctions as f

image_path = f"images/{competition}.png"  # Chemin vers l'image
col1, col2 = st.columns([1, 2])

with col1 : 


    try:
        image = Image.open(image_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        max_width = 400
        image = image.resize((max_width, int(image.height * (max_width / image.width))))

        # Afficher l'image redimensionnée
        st.image(image, caption=f"Rapport pour {competition}")
    except FileNotFoundError:
        st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 


with col2 : 
    st.title("Team Analysis - by gpain1999 ")


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
        min-width: 150px; /* Largeur minimale */
        max-width: 300px; /* Largeur maximale */
        width: 225px; /* Largeur fixe */
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
team_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{CODETEAM}.png")

zoom = st.sidebar.slider(
    "Choisissez une valeur de zoom pour les photos",
    min_value=0.4,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)

colonnes_a_convertir = ['PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'TO', '1_T', '2_T', '3_T',"1_R", "2_R", "3_R", 'CO', 'FP', 'CF', 'NCF']

######################### DATA

off_detail = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["ROUND","TEAM"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)
off_detail[colonnes_a_convertir] = off_detail[colonnes_a_convertir].astype('Int64')
off_detail = off_detail.sort_values(by = "ROUND",ascending=False)
off_detail = off_detail.loc[:, (off_detail != "---").any(axis=0)]
off_detail = off_detail.drop(columns = ["TEAM","NB_GAME"])
win_counts = off_detail["WIN"].value_counts().to_dict()


def_detail = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[CODETEAM],
    selected_fields=["ROUND","OPPONENT"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)
def_detail[colonnes_a_convertir] = def_detail[colonnes_a_convertir].astype('Int64')
def_detail = def_detail.sort_values(by = "ROUND",ascending=False)
def_detail = def_detail.loc[:, (def_detail != "---").any(axis=0)]
def_detail = def_detail.drop(columns = ["OPPONENT","NB_GAME"])

off_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
off_moyenne = off_moyenne.loc[:, (off_moyenne != "---").any(axis=0)]
off_moyenne = off_moyenne.drop(columns = ["TEAM","NB_GAME","HOME","WIN","TIME_ON"])

def_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[CODETEAM],
    selected_fields=["OPPONENT"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
def_moyenne = def_moyenne.loc[:, (def_moyenne != "---").any(axis=0)]
def_moyenne = def_moyenne.drop(columns = ["OPPONENT","NB_GAME","HOME","WIN","TIME_ON"])

################### STYLES

def highlight_win(row):
    # Appliquer vert si WIN est "YES", rouge sinon
    color = 'background-color: #CCFFCC; color: black;' if row["WIN"] == "YES" else 'background-color: #FFCCCC; color: black;'
    return [color for _ in row.index]

def highlight_win_o(row):
    # Appliquer vert si WIN est "NO", rouge sinon
    color = 'background-color: #FFCCCC; color: black;' if row["WIN"] == "YES" else 'background-color: #CCFFCC; color: black;'
    return [color for _ in row.index]
###################### PRINT



col1, col2 = st.columns([1, 7])

with col1 : 

    if os.path.exists(team_logo_path):
        st.image(team_logo_path, caption=f"Équipe : {CODETEAM}", width=int(200*zoom))
    else:
        st.warning(f"Logo introuvable pour l'équipe : {CODETEAM}")

    taille = int(25*zoom)
    # Pour les victoires (YES)
    st.markdown(
        f'''
        <p style="font-size:{taille}px; text-align: center; background-color: #CCFFCC; padding: 10px; border-radius: 5px;">
            <b>{win_counts["YES"]}&nbsp;WINS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )



    # Pour les défaites (NO)
    st.markdown(
        f'''
        <p style="font-size:{taille}px; text-align: center; background-color: #FFCCCC; padding: 10px; border-radius: 5px;">
            <b>{win_counts["NO"]}&nbsp;LOSES</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(taille*0.9)}px; text-align: center; padding: 10pxs;">
            <b>% SHOOTS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    cola, colb = st.columns([1, 1])
    


    with cola :
        
        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b> % {CODETEAM}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(off_detail["1_R"].sum()/off_detail["1_T"].sum() if off_detail["1_T"].sum() != 0 else 0,"1P",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(off_detail["2_R"].sum()/off_detail["2_T"].sum() if off_detail["2_T"].sum() != 0 else 0,"2P",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(off_detail["3_R"].sum()/off_detail["3_T"].sum() if off_detail["3_T"].sum() != 0 else 0,"3P",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    with colb : 
        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b> % OPPONENT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(def_detail["1_R"].sum()/def_detail["1_T"].sum() if def_detail["1_T"].sum() != 0 else 0,"1P",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(def_detail["2_R"].sum()/def_detail["2_T"].sum() if def_detail["2_T"].sum() != 0 else 0,"2P",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(def_detail["3_R"].sum()/def_detail["3_T"].sum() if def_detail["3_T"].sum() != 0 else 0,"3P",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown(
        f'''
        <p style="font-size:{int(taille*0.9)}px; text-align: center; padding: 10pxs;">
            <b>% REBONDS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    cola, colb = st.columns([1, 1])
    with cola :

        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b> DEF. REB.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        fig2 = f.plot_semi_circular_chart(off_moyenne["DR"].sum()/(off_moyenne["DR"].sum() + def_moyenne["OR"].sum()),"BLA", size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
    with colb :
        
        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b> OFF. REB.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        print(off_moyenne["OR"].sum()/(off_moyenne["OR"].sum() + def_moyenne["DR"].sum()))
        fig2 = f.plot_semi_circular_chart(off_moyenne["OR"].sum()/(off_moyenne["OR"].sum() + def_moyenne["DR"].sum()),"BLA",size=int(90*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

with col2 :
    st.header(f"Moyennes {CODETEAM}")
    st.dataframe(off_moyenne,height=60, use_container_width=True)

    st.header(f"Moyennes Adversaires")
    st.dataframe(def_moyenne,height=60, use_container_width=True)

    off_detail_2 = off_detail.style.apply(highlight_win, axis=1).format(precision=1) 
    def_detail_2 = def_detail.style.apply(highlight_win_o, axis=1).format(precision=1)  

    st.header(f"Stats {CODETEAM}")
    st.dataframe(off_detail_2,height=min(38*len(off_detail),650), use_container_width=True)
    st.header("Stats Adversaires")
    st.dataframe(def_detail_2,height=min(38*len(def_detail),650), use_container_width=True)
    

