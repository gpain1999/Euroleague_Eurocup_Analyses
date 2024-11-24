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

team_detail_select = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["ROUND","TEAM"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)

team_detail_select[colonnes_a_convertir] = team_detail_select[colonnes_a_convertir].astype('Int64')
team_detail_select = team_detail_select.sort_values(by = "ROUND",ascending=False)
team_detail_select = team_detail_select.loc[:, (team_detail_select != "---").any(axis=0)]
team_detail_select = team_detail_select.drop(columns = ["TEAM","NB_GAME","TIME_ON"])
win_counts = team_detail_select["WIN"].value_counts().to_dict()


opp_detail_select = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[CODETEAM],
    selected_fields=["ROUND","OPPONENT"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)

opp_detail_select[colonnes_a_convertir] = opp_detail_select[colonnes_a_convertir].astype('Int64')
opp_detail_select = opp_detail_select.sort_values(by = "ROUND",ascending=False)
opp_detail_select = opp_detail_select.loc[:, (opp_detail_select != "---").any(axis=0)]
opp_detail_select = opp_detail_select.drop(columns = ["OPPONENT","NB_GAME","TIME_ON"])


team_league_detail = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["ROUND","TEAM"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)

team_league_detail[colonnes_a_convertir] = team_league_detail[colonnes_a_convertir].astype('Int64')
team_league_detail = team_league_detail.sort_values(by = "ROUND",ascending=False)
team_league_detail = team_league_detail.loc[:, (team_league_detail != "---").any(axis=0)]
team_league_detail = team_league_detail.drop(columns = ["TEAM","NB_GAME","TIME_ON"])



team_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
team_moyenne = team_moyenne.loc[:, (team_moyenne != "---").any(axis=0)]
team_moyenne = team_moyenne.drop(columns = ["TEAM","NB_GAME","HOME","WIN","TIME_ON"])

opp_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[CODETEAM],
    selected_fields=["OPPONENT"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
opp_moyenne = opp_moyenne.loc[:, (opp_moyenne != "---").any(axis=0)]
opp_moyenne = opp_moyenne.drop(columns = ["OPPONENT","NB_GAME","HOME","WIN","TIME_ON"])

league_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)
league_moyenne = league_moyenne.loc[:, (league_moyenne != "---").any(axis=0)]
league_moyenne = league_moyenne.drop(columns = ["TEAM","NB_GAME","HOME","WIN","TIME_ON"])

c = (league_moyenne.columns)
for col in c :
    league_moyenne[c] = league_moyenne[c].mean().round(1)

delta_moyenne_opp = team_moyenne - opp_moyenne
delta_moyenne_opp = delta_moyenne_opp.drop(columns = ["PM_ON"])

delta_moyenne_league = team_moyenne - league_moyenne
delta_moyenne_league = delta_moyenne_league.drop(columns = ["PM_ON"])

################### STYLES

def highlight_win(row):
    # Appliquer vert si WIN est "YES", rouge sinon
    color = 'background-color: #CCFFCC; color: black;' if row["WIN"] == "YES" else 'background-color: #FFCCCC; color: black;'
    return [color for _ in row.index]

def highlight_win_o(row):
    # Appliquer vert si WIN est "NO", rouge sinon
    color = 'background-color: #FFCCCC; color: black;' if row["WIN"] == "YES" else 'background-color: #CCFFCC; color: black;'
    return [color for _ in row.index]
def color_delta(val, column_name, inverse_columns=None):
    """
    Applique une couleur au fond en fonction de la valeur et du nom de la colonne.
    - Vert pour les valeurs positives et rouge pour les négatives par défaut.
    - Logique inversée pour les colonnes spécifiées (inverse_columns).
    """
    if inverse_columns is None:
        inverse_columns = []

    if column_name in inverse_columns:
        # Logique inversée
        color = '#CCFFCC' if val < 0 else '#FFCCCC' if val > 0 else '#FFFFFF'
    else:
        # Logique par défaut
        color = '#CCFFCC' if val > 0 else '#FFCCCC' if val < 0 else '#FFFFFF'
    return f'background-color: {color}; color: black'

###################### PRINT

image_path = f"images/{competition}.png"  # Chemin vers l'image
col1, col2,col3 = st.columns([1, 3,1])

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
            <b>Team Stats - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
        

with col3 :
    colA, colB = st.columns([1, 1])


    taille = int(25*zoom)
    # Pour les victoires (YES)

    with colA :
        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: #CCFFCC;color: black; padding: 10px; border-radius: 5px;">
                <b>{win_counts["YES"]}&nbsp;WINS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        # Pour les défaites (NO)
        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: #FFCCCC;color: black; padding: 10px; border-radius: 5px;">
                <b>{win_counts["NO"]}&nbsp;LOSES</b>
            </p>
            ''',
            unsafe_allow_html=True
        )


    with colB :


        if os.path.exists(team_logo_path):
            st.image(team_logo_path,  width=int(200*zoom))
        else:
            st.warning(f"Logo introuvable pour l'équipe : {CODETEAM}")




col1,col2,col3,col4 = st.columns([1,1,1,4])

with col1:
    min_game = st.number_input("Minimum game played", min_value=1,max_value=selected_range[1] - selected_range[0] + 1 ,value=math.ceil((selected_range[1] - selected_range[0] + 1)*0.6))

with col2 :
    TOP = st.selectbox("TOP or FLOP", options=["TOP", "FLOP"], index=0)
    

with col3 :
    top = st.number_input(TOP, min_value=3,max_value=20 ,value=3)

col1, col2,col3,col4,col5,col6 = st.columns([1,1,1,1,1,1])


team_player_moyenne = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["TEAM","PLAYER"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)

team_player_moyenne= team_player_moyenne[team_player_moyenne["NB_GAME"]>=min_game]


with col1 :
    if TOP=="TOP" : 
        st.dataframe(team_player_moyenne[["PLAYER","PER"]].sort_values(by = "PER",ascending = False).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
    else : 
        st.dataframe(team_player_moyenne[["PLAYER","PER"]].sort_values(by = "PER",ascending = True).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)


with col2 :
    if TOP=="TOP" : 
        st.dataframe(team_player_moyenne[["PLAYER","PTS"]].sort_values(by = "PTS",ascending = False).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
    else : 
        st.dataframe(team_player_moyenne[["PLAYER","PTS"]].sort_values(by = "PTS",ascending = True).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)

with col3 :
    if TOP=="TOP" : 
        st.dataframe(team_player_moyenne[["PLAYER","TR"]].sort_values(by = "TR",ascending = False).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
    else : 
        st.dataframe(team_player_moyenne[["PLAYER","TR"]].sort_values(by = "TR",ascending = True).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
        
with col4 :
    if TOP=="TOP" : 
        st.dataframe(team_player_moyenne[["PLAYER","AS"]].sort_values(by = "AS",ascending = False).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
    else : 
        st.dataframe(team_player_moyenne[["PLAYER","AS"]].sort_values(by = "AS",ascending = True).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
with col5 :
    if TOP=="TOP" : 
        st.dataframe(team_player_moyenne[["PLAYER","CO"]].sort_values(by = "CO",ascending = False).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
    else : 
        st.dataframe(team_player_moyenne[["PLAYER","CO"]].sort_values(by = "CO",ascending = True).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
with col6 :
    if TOP=="TOP" : 
        st.dataframe(team_player_moyenne[["PLAYER","PM_ON"]].sort_values(by = "PM_ON",ascending = False).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
    
    else : 
        st.dataframe(team_player_moyenne[["PLAYER","PM_ON"]].sort_values(by = "PM_ON",ascending = True).head(top)
                    ,height=36*(top+1), 
                    use_container_width=True,hide_index=True)
        
col1, col2 = st.columns([1.5, 7])


with col2 :
    st.header(f"Averages : {CODETEAM}")
    st.dataframe(team_moyenne,height=60, use_container_width=True,hide_index=True)

    name, valu,_ = st.columns([2, 1,3])

    with valu :
        DELTA = st.selectbox("COMPARAISON TYPE", options=["LEAGUE", "OPPONENTS"], index=0)

    with name :
        st.header(f"Averages {DELTA}")


    if DELTA == "OPPONENTS" :

        st.dataframe(opp_moyenne,height=60, use_container_width=True,hide_index=True)
    else :
        st.dataframe(league_moyenne.head(1),height=60, use_container_width=True,hide_index=True)

    delta_moyenne_opp_2 = delta_moyenne_opp.style.apply(
        lambda x: [color_delta(val, col, ['TO', 'CF', 'NCF'] ) for val, col in zip(x, x.index)], axis=1
    ).format(precision=1)

    delta_moyenne_league_2 = delta_moyenne_league.head(1).style.apply(
        lambda x: [color_delta(val, col, ['TO', 'CF', 'NCF'] ) for val, col in zip(x, x.index)], axis=1
    ).format(precision=1)

    st.header(f"DELTA {CODETEAM} - {DELTA}")

    if DELTA == "OPPONENTS" :
        st.dataframe(delta_moyenne_opp_2,height=60, use_container_width=True,hide_index=True)
    else : 
        st.dataframe(delta_moyenne_league_2,height=60, use_container_width=True,hide_index=True)

with col1 : 

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

        fig2 = f.plot_semi_circular_chart(team_detail_select["1_R"].sum()/team_detail_select["1_T"].sum() if team_detail_select["1_T"].sum() != 0 else 0,"1P",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(team_detail_select["2_R"].sum()/team_detail_select["2_T"].sum() if team_detail_select["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(team_detail_select["3_R"].sum()/team_detail_select["3_T"].sum() if team_detail_select["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    with colb : 

        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b> % {DELTA}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if DELTA == "OPPONENTS" : 
            fig2 = f.plot_semi_circular_chart(opp_detail_select["1_R"].sum()/opp_detail_select["1_T"].sum() if opp_detail_select["1_T"].sum() != 0 else 0,"1P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
            fig2 = f.plot_semi_circular_chart(opp_detail_select["2_R"].sum()/opp_detail_select["2_T"].sum() if opp_detail_select["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
            fig2 = f.plot_semi_circular_chart(opp_detail_select["3_R"].sum()/opp_detail_select["3_T"].sum() if opp_detail_select["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
        else : 
            fig2 = f.plot_semi_circular_chart(team_league_detail["1_R"].sum()/team_league_detail["1_T"].sum() if team_league_detail["1_T"].sum() != 0 else 0,"1P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
            fig2 = f.plot_semi_circular_chart(team_league_detail["2_R"].sum()/team_league_detail["2_T"].sum() if team_league_detail["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
            fig2 = f.plot_semi_circular_chart(team_league_detail["3_R"].sum()/team_league_detail["3_T"].sum() if team_league_detail["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom), font_size=int(20*zoom))
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
                <b> {CODETEAM}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        fig2 = f.plot_semi_circular_chart(team_detail_select["DR"].sum()/(team_detail_select["DR"].sum() + opp_detail_select["OR"].sum()),"DEF. REB.", size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)


        fig2 = f.plot_semi_circular_chart(team_detail_select["OR"].sum()/(team_detail_select["OR"].sum() + opp_detail_select["DR"].sum()),"OFF. REB.",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)


    with colb :
        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b> {DELTA}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if DELTA == "OPPONENTS" :
            fig2 = f.plot_semi_circular_chart(opp_detail_select["DR"].sum()/(opp_detail_select["DR"].sum() + team_detail_select["OR"].sum()),"DEF. REB.", size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)


            fig2 = f.plot_semi_circular_chart(opp_detail_select["OR"].sum()/(opp_detail_select["OR"].sum() + team_detail_select["DR"].sum()),"OFF. REB.",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)       

        else  :
            fig2 = f.plot_semi_circular_chart(team_league_detail["DR"].sum()/(team_league_detail["DR"].sum() + team_league_detail["OR"].sum()),"DEF. REB.", size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)


            fig2 = f.plot_semi_circular_chart(team_league_detail["OR"].sum()/(team_league_detail["OR"].sum() + team_league_detail["DR"].sum()),"OFF. REB.",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)    

    
team_detail_select_2 = team_detail_select.style.apply(highlight_win, axis=1).format(precision=1) 
opp_detail_select_2 = opp_detail_select.style.apply(highlight_win_o, axis=1).format(precision=1)  
st.header(f"Stats GAME BY GAME : {CODETEAM}")
st.dataframe(team_detail_select_2,height=min(38*len(team_detail_select),650), use_container_width=True,hide_index=True)
st.header("Stats GAME BY GAME : Opponents")
st.dataframe(opp_detail_select_2,height=min(38*len(opp_detail_select),650), use_container_width=True,hide_index=True)


