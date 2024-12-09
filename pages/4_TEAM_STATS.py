import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

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

DELTA = st.sidebar.selectbox("COMPARAISON TYPE", options=["LEAGUE", "OPPONENTS"], index=0)

selected_range = st.sidebar.slider(
    "Select a ROUND range:",
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
    min_value=0.3,
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
if "YES" not in win_counts:
    win_counts["YES"] = 0
if "NO" not in win_counts:
    win_counts["NO"] = 0


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


team_def_perc = team_detail_select["DR"].sum()/(team_detail_select["DR"].sum() + opp_detail_select["OR"].sum())
opp_off_perc = 1 - team_def_perc
opp_def_perc = opp_detail_select["DR"].sum()/(opp_detail_select["DR"].sum() + team_detail_select["OR"].sum())
team_off_perc = 1 - opp_def_perc

team_def = (team_def_perc/0.7)/(team_def_perc/0.7 + opp_off_perc/0.3)
opp_off = (opp_off_perc/0.3)/(team_def_perc/0.7 + opp_off_perc/0.3)
opp_def = (opp_def_perc/0.7)/( opp_def_perc/0.7 + team_off_perc/0.3)
team_off = (team_off_perc/0.3)/(opp_def_perc/0.7 + team_off_perc/0.3)

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

team_top_values,team_bottom_values,opp_top_values,opp_bottom_values = f.stats_important_team(CODETEAM,selected_range[0],selected_range[1],df)

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

notation = notation[notation["CODETEAM"] == CODETEAM].sort_values(by = "IRS",ascending = False).reset_index(drop = True)

MVP_NOTE = notation["NOTE"].to_list()[0]
MVP_COLOR = notation["COULEUR"].to_list()[0]
NAME_MVP = notation["PLAYER"].to_list()[0]
NUMBER_MVP = notation["NUMBER"].to_list()[0]
MVP_ID = notation["PLAYER_ID"].to_list()[0]
mvp_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{CODETEAM}_{MVP_ID}.png")

mvp_stat = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["TEAM","ROUND","PLAYER"],
    selected_players=[NAME_MVP],
    mode="CUMULATED",
    percent="MADE"
)

teams_color = pd.read_csv(f"datas/{competition}_{season}_teams_colors.csv",sep=";")

teams_color[teams_color["TEAM"]==CODETEAM]["COL1"].to_list()[0]

local_c1 = teams_color[teams_color["TEAM"]==CODETEAM]["COL1"].to_list()[0]
local_c2 = teams_color[teams_color["TEAM"]==CODETEAM]["COL2"].to_list()[0]
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
            <p style="font-size:{taille}px; text-align: center; background-color: #00ff00;color: black; padding: 10px; border-radius: 5px;">
                <b>{win_counts["YES"]}&nbsp;WINS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        # Pour les défaites (NO)
        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: #ff0000;color: black; padding: 10px; border-radius: 5px;">
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



col1, col2 = st.columns([1.5, 7])

with col2 :
    

    col11,col21,col31,col41,col51 = st.columns([1,1,1,1,3])

    with col11:
        min_game = st.number_input("Minimum game played", min_value=1,max_value=selected_range[1] - selected_range[0] + 1 ,value=math.ceil((selected_range[1] - selected_range[0] + 1)*0.51))

    with col21 :
        TOP = st.selectbox("TOP or FLOP", options=["TOP", "FLOP"], index=0)
        
    with col31 :
        top = st.number_input(TOP, min_value=3,max_value=20 ,value=3)
    
    with col41 :
        pass
        


    col12, col22,col32,col42,col52= st.columns([1,1,1,1,1])


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


    with col12 :
        if TOP=="TOP" : 
            st.dataframe(team_player_moyenne[["PLAYER","PER"]].sort_values(by = "PER",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            
            st.dataframe(team_player_moyenne[["PLAYER","I_PER"]].sort_values(by = "I_PER",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)   
        else : 
            st.dataframe(team_player_moyenne[["PLAYER","PER"]].sort_values(by = "PER",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","I_PER"]].sort_values(by = "I_PER",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)

    with col22 :
        if TOP=="TOP" : 
            st.dataframe(team_player_moyenne[["PLAYER","PTS"]].sort_values(by = "PTS",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","PM_ON"]].sort_values(by = "PM_ON",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
                   
        else : 
            st.dataframe(team_player_moyenne[["PLAYER","PTS"]].sort_values(by = "PTS",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","PM_ON"]].sort_values(by = "PM_ON",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
    with col32 :
        if TOP=="TOP" : 
            st.dataframe(team_player_moyenne[["PLAYER","TR"]].sort_values(by = "TR",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","TIME_ON"]].sort_values(by = "TIME_ON",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)       
        else : 
            st.dataframe(team_player_moyenne[["PLAYER","TR"]].sort_values(by = "TR",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","TIME_ON"]].sort_values(by = "TIME_ON",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)   
    with col42 :
        if TOP=="TOP" : 
            st.dataframe(team_player_moyenne[["PLAYER","AS"]].sort_values(by = "AS",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","TO"]].sort_values(by = "TO",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
                   
        else : 
            st.dataframe(team_player_moyenne[["PLAYER","AS"]].sort_values(by = "AS",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","TO"]].sort_values(by = "TO",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
    with col52 :
        if TOP=="TOP" : 
            st.dataframe(team_player_moyenne[["PLAYER","CO"]].sort_values(by = "CO",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","ST"]].sort_values(by = "ST",ascending = False).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)       
        else : 
            st.dataframe(team_player_moyenne[["PLAYER","CO"]].sort_values(by = "CO",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)
            st.dataframe(team_player_moyenne[["PLAYER","ST"]].sort_values(by = "ST",ascending = True).head(top)
                        ,height=36*(top+1), 
                        use_container_width=True,hide_index=True)     

            


    st.header(f"Averages : {CODETEAM}")
    st.dataframe(team_moyenne,height=60, use_container_width=True,hide_index=True)

    name, valu,_ = st.columns([2, 1,3])

    with valu :
        pass

    with name :
        st.header(f"Averages : {DELTA}")


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
                <b>{CODETEAM}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(team_detail_select["1_R"].sum()/team_detail_select["1_T"].sum() if team_detail_select["1_T"].sum() != 0 else 0,"FT",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(team_detail_select["2_R"].sum()/team_detail_select["2_T"].sum() if team_detail_select["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)
        fig2 = f.plot_semi_circular_chart(team_detail_select["3_R"].sum()/team_detail_select["3_T"].sum() if team_detail_select["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)

    with colb : 

        st.markdown(
            f'''
            <p style="font-size:{int(taille*0.75)}px; text-align: center;">
                <b>{DELTA}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if DELTA == "OPPONENTS" : 
            fig2 = f.plot_semi_circular_chart(opp_detail_select["1_R"].sum()/opp_detail_select["1_T"].sum() if opp_detail_select["1_T"].sum() != 0 else 0,"FT",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
            fig2 = f.plot_semi_circular_chart(opp_detail_select["2_R"].sum()/opp_detail_select["2_T"].sum() if opp_detail_select["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
            fig2 = f.plot_semi_circular_chart(opp_detail_select["3_R"].sum()/opp_detail_select["3_T"].sum() if opp_detail_select["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom), font_size=int(20*zoom))
            st.plotly_chart(fig2,use_container_width=True)
        else : 
            fig2 = f.plot_semi_circular_chart(team_league_detail["1_R"].sum()/team_league_detail["1_T"].sum() if team_league_detail["1_T"].sum() != 0 else 0,"FT",size=int(120*zoom), font_size=int(20*zoom))
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
    
    st.markdown(
        f'''
        <p style="font-size:{int(taille*0.9)}px; text-align: center; padding: 10pxs;">
            <b>% BALL CARE </b>
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
        shot_T = team_detail_select["1_T"].sum()/2 + team_detail_select["2_T"].sum() + team_detail_select["3_T"].sum()
        TO = team_detail_select["TO"].sum()
        fig2 = f.plot_semi_circular_chart(shot_T/(shot_T+TO),"SHOOT/TO", size=int(120*zoom), font_size=int(20*zoom))
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
            shot_T = opp_detail_select["1_T"].sum()/2 + opp_detail_select["2_T"].sum() + opp_detail_select["3_T"].sum()
            TO = opp_detail_select["TO"].sum()

        else :
            shot_T = team_league_detail["1_T"].sum()/2 + team_league_detail["2_T"].sum() + team_league_detail["3_T"].sum()
            TO = team_league_detail["TO"].sum()

        fig2 = f.plot_semi_circular_chart(shot_T/(shot_T+TO),"SHOOT/TO", size=int(120*zoom), font_size=int(20*zoom))
        st.plotly_chart(fig2,use_container_width=True)


   

delta_graph,cola, colb,mvp,vide,colc = st.columns([1,0.666,0.666,0.9,0.1,0.666]) 





with delta_graph :
    c1,c2 = st.columns([1,1]) 

    with c1 :
        MM = st.selectbox("DELTA MODE", options=["MEAN", "MEDIAN"], index=1)
    with c2 :
        type_delta = st.selectbox("DELTA TYPE", options=["CUMULATED","PER PERIODE"], index=1)


    periode,cumul,_,_ = f.team_evol_score(CODETEAM,selected_range[0],selected_range[1],data_dir,competition,season,type = MM)
    periode = periode[:16]
    cumul = cumul[:16]

    if type_delta == "PER PERIODE" :
        data_delta = periode

    else : 
        data_delta = cumul
    # Noms des barres
    labels = [i * 2.5 for i in range(1, len(periode) + 1)]

    # Couleurs conditionnelles
    colors = [local_c1 if val > 0 else 'orange' if val < 0 else 'grey' for val in data_delta]
    contour = [local_c2 if val > 0 else "white" if val < 0 else 'grey' for val in data_delta]

    # Création de la figure
    fig = go.Figure()

    # Ajout des barres
    fig.add_trace(
        go.Bar(
            x=labels,
            y=data_delta,
            marker=dict(
                color=colors,  # Couleurs des barres
                line=dict(color=contour, width=2)  # Couleurs des contours dynamiques
            ),
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
        width=int(600*zoom),
        height=int(600*zoom),
        title=f"{MM} Delta score",
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
    st.plotly_chart(fig?use_container_width=True)




with cola :
        st.markdown(
                f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c2};">                    
                <b> SHOOTS REPART {CODETEAM} </b>
                </p>
                ''',
                unsafe_allow_html=True
            )
        a,b = st.columns([1,1])

        with a :


            st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: {local_c2}; color: {local_c1}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c1};">
                <b>{round(100*(team_moyenne["2_T"].mean()/(team_moyenne["2_T"].mean()+team_moyenne["3_T"].mean())),1)}&nbsp;% OF 2PTS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
            

        with b : 
        
            st.markdown(
                f'''
                <p style="font-size:{taille}px; text-align: center; background-color: {local_c2}; color: {local_c1}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c1};">
                    <b>{round(100*(team_moyenne["3_T"].mean()/(team_moyenne["2_T"].mean()+team_moyenne["3_T"].mean())),1)}&nbsp;% OF 3PTS</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
        st.markdown(
                f'''
                <p style="font-size:{taille}px; text-align: center; background-color: {local_c2}; color: {local_c1}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c1};">
                    <b>{round(((team_moyenne["3_R"].mean()*3 + team_moyenne["2_R"].mean()*2)/(team_moyenne["2_T"].mean()+team_moyenne["3_T"].mean())),2)}&nbsp;PTS PER SHOOT</b>
                </p>
                ''',
                unsafe_allow_html=True
            )  

        st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c2};">                    
                <b> RATIO AS/TO {CODETEAM} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: {local_c2}; color: {local_c1}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c1};">
                <b>{round(team_moyenne["AS"].mean()/team_moyenne["TO"].mean(),2)}&nbsp; OF AS/TO</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c2};">                    
                <b> SHOOTS PER GAME {CODETEAM} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: {local_c2}; color: {local_c1}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c1};">
                <b>{round(team_moyenne["2_T"].mean()+team_moyenne["3_T"].mean(),2)}&nbsp; SHOOTS + {round(team_moyenne["1_T"].mean(),1)}&nbsp; FT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c2};">                    
                <b> ASSISTED BUCKETS {CODETEAM} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: {local_c2}; color: {local_c1}; padding: 4px; border-radius: 5px; outline: 3px solid {local_c1};">
                <b>{round(100*team_moyenne["AS"].mean()/(team_moyenne["2_R"].mean()+team_moyenne["3_R"].mean()),1)}&nbsp; %</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with colb :
    
    st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: orange; color: white; padding: 4px; border-radius: 5px; outline: 3px solid white;">                    
                <b> {DELTA} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    if DELTA == "OPPONENTS" :
        data = opp_moyenne
    else :
        data = league_moyenne

    a,b = st.columns([1,1])
    with a :
        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: white; color: orange; padding: 4px; border-radius: 5px; outline: 3px solid orange;">
                <b>{round(100*(data["2_T"].mean()/(data["2_T"].mean()+data["3_T"].mean())),1)}&nbsp;% OF 2PTS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

    with b : 
        # Pour les défaites (NO)
        st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: white; color: orange; padding: 4px; border-radius: 5px; outline: 3px solid orange;">
                <b>{round(100*(data["3_T"].mean()/(data["2_T"].mean()+data["3_T"].mean())),1)}&nbsp;% OF 3PTS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
                f'''
                <p style="font-size:{taille}px; text-align: center; background-color: white; color: orange; padding: 4px; border-radius: 5px; outline: 3px solid orange;">
                    <b>{round(((data["3_R"].mean()*3 + data["2_R"].mean()*2)/(data["2_T"].mean()+data["3_T"].mean())),2)}&nbsp;PTS PER SHOOT</b>
                </p>
                ''',
                unsafe_allow_html=True
            ) 
    st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: orange; color: white; padding: 4px; border-radius: 5px; outline: 3px solid white;">                    
                <b> {DELTA} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: white; color: orange; padding: 4px; border-radius: 5px; outline: 3px solid orange;">
                <b>{round(data["AS"].mean()/data["TO"].mean(),2)}&nbsp; OF AS/TO</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: orange; color: white; padding: 4px; border-radius: 5px; outline: 3px solid white;">                    
                <b> {DELTA} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: white; color: orange; padding: 4px; border-radius: 5px; outline: 3px solid orange;">
                <b>{round(data["2_T"].mean()+data["3_T"].mean(),2)}&nbsp; SHOOTS + {round(data["1_T"].mean(),1)}&nbsp; FT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

    st.markdown(
            f'''
                <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: orange; color: white; padding: 4px; border-radius: 5px; outline: 3px solid white;">                    
                <b> {DELTA} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{taille}px; text-align: center; background-color: white; color: orange; padding: 4px; border-radius: 5px; outline: 3px solid orange;">
                <b>{round(100*data["AS"].mean()/(data["2_R"].mean()+data["3_R"].mean()),1)}&nbsp; %</b>
            </p>
            ''',
            unsafe_allow_html=True
    )
with mvp :
            # Intégrer la couleur dans le markdown
    st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: {MVP_COLOR};color: black; padding: 6px; border-radius: 5px;outline: 4px solid gold;">
                <b> BEST PLAYER NOTE : {round(MVP_NOTE)}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    mvp_photo, mvp_shoots = st.columns([1,0.7]) 

    with mvp_photo :



        if os.path.exists(mvp_image_path):
            st.image(mvp_image_path, caption=f"#{NUMBER_MVP} {NAME_MVP}", width=int(330*zoom))
        else:
            st.warning(f"Image introuvable pour le joueur : {NAME_MVP}")

    with mvp_shoots :
        fig2 = f.plot_semi_circular_chart(mvp_stat["1_R"].sum()/mvp_stat["1_T"].sum() if mvp_stat["1_T"].sum() != 0 else 0,"1P",size=int(120*zoom),font_size=int(20*zoom),m=False)
        st.plotly_chart(fig2)
        fig2 = f.plot_semi_circular_chart(mvp_stat["2_R"].sum()/mvp_stat["2_T"].sum() if mvp_stat["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom),font_size=int(20*zoom),m=False)
        st.plotly_chart(fig2)
        fig2 = f.plot_semi_circular_chart(mvp_stat["3_R"].sum()/mvp_stat["3_T"].sum() if mvp_stat["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom),font_size=int(20*zoom),m=False)
        st.plotly_chart(fig2)
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {MVP_COLOR};color: black; padding: 6px; border-radius: 5px;outline: 4px solid gold;">
                <b> {round(mvp_stat["PER"].mean(),1)} PER /G</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with colc :
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[1]};color: black; padding: 6px; border-radius: 4.5px;outline:4px solid silver;">
                <b> {notation["PLAYER"].to_list()[1]} : {round(notation["NOTE"].to_list()[1])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[2]};color: black; padding: 6px; border-radius: 4.5px;outline:4px solid #CD7F32";>
                <b> {notation["PLAYER"].to_list()[2]} : {round(notation["NOTE"].to_list()[2])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[3]};color: black; padding: 6px; border-radius: 4.5px;">
                <b> {notation["PLAYER"].to_list()[3]} : {round(notation["NOTE"].to_list()[3])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[4]};color: black; padding: 6px; border-radius: 4.5px;">
                <b> {notation["PLAYER"].to_list()[4]} : {round(notation["NOTE"].to_list()[4])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )    
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[5]};color: black; padding: 6px; border-radius: 4.5px;">
                <b> {notation["PLAYER"].to_list()[5]} : {round(notation["NOTE"].to_list()[5])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )  
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[6]};color: black; padding: 6px; border-radius: 4.5px;">
                <b> {notation["PLAYER"].to_list()[6]} : {round(notation["NOTE"].to_list()[6])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )  
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[7]};color: black; padding: 6px; border-radius: 4.5px;">
                <b> {notation["PLAYER"].to_list()[7]} : {round(notation["NOTE"].to_list()[7])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: {notation["COULEUR"].to_list()[8]};color: black; padding: 6px; border-radius: 4.5px;">
                <b> {notation["PLAYER"].to_list()[8]} : {round(notation["NOTE"].to_list()[8])}</b>
            </p>
            ''',
            unsafe_allow_html=True
        ) 
team_detail_select_2 = team_detail_select.style.apply(highlight_win, axis=1).format(precision=1) 
opp_detail_select_2 = opp_detail_select.style.apply(highlight_win_o, axis=1).format(precision=1)

good_bad_team,good_bad_opp = st.columns([1,1]) 


with good_bad_team :
    st.markdown(
        f'''
        <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2}; 
        padding: 4px; border-radius: 5px; outline: 3px solid {local_c2};">
            <b>{CODETEAM} - KEYS STATS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    team_df = pd.DataFrame({
        'POSITIVES': team_top_values,
        'NEGATIVES': team_bottom_values
    })

    good,bad,sho = st.columns([0.33,0.33,0.33])

    with good :
        st.markdown(
            f'''
            <p style="font-size:{int(23*zoom)}px; text-align: center; background-color: green;color: white; padding: 4px; border-radius: 5px;">
                <b>POSITIVES</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in team_df["POSITIVES"].to_list() :
            st.markdown(
                f'''
                <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: #E8FFD9;color: black; padding: 3px; border-radius: 5px;">
                    <b>{g}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )

    with bad :
        st.markdown(
            f'''
            <p style="font-size:{int(23*zoom)}px; text-align: center; background-color: red;color: white; padding: 4px; border-radius: 5px;">
                <b>NEGATIVES</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in team_df["NEGATIVES"].to_list() :
            st.markdown(
                f'''
                <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: #FFD3D3;color: black; padding: 3px; border-radius: 5px;">
                    <b>{g}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
    with sho : 

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(team_detail_select["2_T"].mean()+team_detail_select["3_T"].mean(),1)}&nbsp; SHOOTS + {round(team_detail_select["1_T"].mean(),1)} FT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round((team_detail_select["2_R"].sum()*2+team_detail_select["3_R"].sum()*3)/(team_detail_select["2_T"].sum()+team_detail_select["3_T"].sum()),2)}&nbsp; PTS PER SHOOT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(100*team_detail_select["AS"].sum()/(team_detail_select["2_R"].sum()+team_detail_select["3_R"].sum()),1)}&nbsp;% ASSISTS </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        shot_T = team_detail_select["1_T"].sum()/2 + team_detail_select["2_T"].sum() + team_detail_select["3_T"].sum()
        TO = team_detail_select["TO"].sum()

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(100*shot_T/(shot_T+TO),1)} % BALL CARE</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(team_def+team_off,2)} REB. PERF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )


        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(team_detail_select["PER"].mean(),1)} PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with good_bad_opp :
    st.markdown(
        f'''
        <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: orange; color: white; 
        padding: 4px; border-radius: 5px; outline: 3px solid white;">
            <b>OPPONENTS - KEYS STATS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    team_df = pd.DataFrame({
        'POSITIVES': opp_top_values,
        'NEGATIVES': opp_bottom_values
    })

    sho,good,bad = st.columns([0.33,0.33,0.33])

    with good :
        st.markdown(
            f'''
            <p style="font-size:{int(23*zoom)}px; text-align: center; background-color: green;color: white; padding: 4px; border-radius: 5px;">
                <b>POSITIVES</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in team_df["POSITIVES"].to_list() :
            st.markdown(
                f'''
                <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: #E8FFD9;color: black; padding: 3px; border-radius: 5px;">
                    <b>{g}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )

    with bad :
        st.markdown(
            f'''
            <p style="font-size:{int(23*zoom)}px; text-align: center; background-color: red;color: white; padding: 4px; border-radius: 5px;">
                <b>NEGATIVES</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in team_df["NEGATIVES"].to_list() :
            st.markdown(
                f'''
                <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: #FFD3D3;color: black; padding: 3px; border-radius: 5px;">
                    <b>{g}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )

    with sho : 

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: grey;color: black; padding: 6px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(opp_detail_select["2_T"].mean()+opp_detail_select["3_T"].mean(),1)}&nbsp; SHOOTS + {round(opp_detail_select["1_T"].mean(),1)} FT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: grey;color: black; padding: 6px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round((opp_detail_select["2_R"].sum()*2+opp_detail_select["3_R"].sum()*3)/(opp_detail_select["2_T"].sum()+opp_detail_select["3_T"].sum()),2)}&nbsp; PTS PER SHOOT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: grey;color: black; padding: 6px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*opp_detail_select["AS"].sum()/(opp_detail_select["2_R"].sum()+opp_detail_select["3_R"].sum()),1)}&nbsp;% ASSISTS </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        shot_T = opp_detail_select["1_T"].sum()/2 + opp_detail_select["2_T"].sum() + opp_detail_select["3_T"].sum()
        TO = opp_detail_select["TO"].sum()

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: grey;color: black; padding: 6px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(100*shot_T/(shot_T+TO),1)} % BALL CARE</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: grey;color: black; padding: 6px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(opp_def+opp_off,2)} REB. PERF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )


        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: grey;color: black; padding: 6px; border-radius: 5px;outline: 3px solid orange;">
                <b>{round(opp_detail_select["PER"].mean(),1)} PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

st.header(f"Stats GAME BY GAME : {CODETEAM}")
st.dataframe(team_detail_select_2,height=min(38*len(team_detail_select),650), use_container_width=True,hide_index=True)

st.header("Stats GAME BY GAME : OPPONENTS")
st.dataframe(opp_detail_select_2,height=min(38*len(opp_detail_select),650), use_container_width=True,hide_index=True)


