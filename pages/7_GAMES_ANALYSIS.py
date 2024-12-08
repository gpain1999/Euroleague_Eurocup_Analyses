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
gs["GAME"] = gs.apply(lambda row: f"R{row['ROUND']} : {row['LOCAL_TEAM']} - {row['ROAD_TEAM']}", axis=1)

players = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

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
    game_list = gs[gs['GAME'].str.contains("R"+str(SUB), na=False)]['GAME'].tolist()

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

############################### DATA ###################################################

periode,cumul,cumul_local,cumul_road = f.team_evol_score(team_local,r,r,data_dir,competition,season,type = "MEAN")

cumul_local = [int(x) for x in cumul_local if not math.isnan(x)]
cumul_road = [int(x) for x in cumul_road if not math.isnan(x)]

cumul_local = [cumul_local[i+1] for i in range(0, len(cumul_local), 2)]
cumul_road = [cumul_road[i+1] for i in range(0, len(cumul_road), 2)]

periode_local = [int(cumul_local[0])] + [int(cumul_local[i] - cumul_local[i-1]) for i in range(1, len(cumul_local))]
periode_road = [int(cumul_road[0])] + [int(cumul_road[i] - cumul_road[i-1]) for i in range(1, len(cumul_road))]

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

# Trier le DataFrame par "PER" et "I_PER"
df_sorted = player_stat.sort_values(by=["PER", "I_PER", "TIME_ON"], ascending=[False, False, True])

# Filtrer les lignes où WIN == "YES"
df_win_yes = df_sorted[df_sorted["WIN"] == "YES"]

# Sélectionner les 3 meilleurs PER et parmi eux, la ligne avec le meilleur I_PER
top_3_per = df_win_yes.head(3)
best_row = top_3_per.sort_values(by="I_PER", ascending=False).head(1)

# Supprimer la ligne sélectionnée du DataFrame trié
df_sorted = df_sorted.drop(best_row.index)

df_sorted = df_sorted.sort_values(by=["I_PER", "TIME_ON"], ascending=[False, True])
# Concatenation pour placer cette ligne en première position
player_stat = pd.concat([best_row, df_sorted], ignore_index=True).reset_index(drop=True)


local_player_stat = player_stat[player_stat["TEAM"]==team_local]
road_player_stat = player_stat[player_stat["TEAM"]==team_road]

### MVP ###

mvp_data = player_stat
NAME_MVP = mvp_data["PLAYER"].to_list()[0]
NUMBER_MVP = mvp_data["#"].to_list()[0]
TEAM_MVP = mvp_data["TEAM"].to_list()[0]
MVP_ID = players[(players["CODETEAM"] == TEAM_MVP) & (players["PLAYER"] == NAME_MVP)]["PLAYER_ID"].to_list()[0]
mvp_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_MVP}_{MVP_ID}.png")

BL_data = player_stat[player_stat["WIN"]=="NO"].sort_values(by = ["I_PER","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_BL = BL_data["PLAYER"].to_list()[0]
NUMBER_BL = BL_data["#"].to_list()[0]
TEAM_BL = BL_data["TEAM"].to_list()[0]
BL_ID = players[(players["CODETEAM"] == TEAM_BL) & (players["PLAYER"] == NAME_BL)]["PLAYER_ID"].to_list()[0]
BL_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_BL}_{BL_ID}.png")

### PTS ###

PTS_data = player_stat.sort_values(by = ["PTS","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_PTS = PTS_data["PLAYER"].to_list()[0]
NUMBER_PTS = PTS_data["#"].to_list()[0]
TEAM_PTS = PTS_data["TEAM"].to_list()[0]
PTS_ID = players[(players["CODETEAM"] == TEAM_PTS) & (players["PLAYER"] == NAME_PTS)]["PLAYER_ID"].to_list()[0]
PTS_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_PTS}_{PTS_ID}.png")

### PM_ON ###

PM_ON_data = player_stat.sort_values(by = ["PM_ON","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_PM_ON = PM_ON_data["PLAYER"].to_list()[0]
NUMBER_PM_ON = PM_ON_data["#"].to_list()[0]
TEAM_PM_ON = PM_ON_data["TEAM"].to_list()[0]
PM_ON_ID = players[(players["CODETEAM"] == TEAM_PM_ON) & (players["PLAYER"] == NAME_PM_ON)]["PLAYER_ID"].to_list()[0]
PM_ON_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_PM_ON}_{PM_ON_ID}.png")

### DR ###

DR_data = player_stat.sort_values(by = ["DR","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_DR = DR_data["PLAYER"].to_list()[0]
NUMBER_DR = DR_data["#"].to_list()[0]
TEAM_DR = DR_data["TEAM"].to_list()[0]
DR_ID = players[(players["CODETEAM"] == TEAM_DR) & (players["PLAYER"] == NAME_DR)]["PLAYER_ID"].to_list()[0]
DR_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_DR}_{DR_ID}.png")

### OR ###

OR_data = player_stat.sort_values(by = ["OR","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_OR = OR_data["PLAYER"].to_list()[0]
NUMBER_OR = OR_data["#"].to_list()[0]
TEAM_OR = OR_data["TEAM"].to_list()[0]
OR_ID = players[(players["CODETEAM"] == TEAM_OR) & (players["PLAYER"] == NAME_OR)]["PLAYER_ID"].to_list()[0]
OR_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_OR}_{OR_ID}.png")

### AS ###

AS_data = player_stat.sort_values(by = ["AS","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_AS = AS_data["PLAYER"].to_list()[0]
NUMBER_AS = AS_data["#"].to_list()[0]
TEAM_AS = AS_data["TEAM"].to_list()[0]
AS_ID = players[(players["CODETEAM"] == TEAM_AS) & (players["PLAYER"] == NAME_AS)]["PLAYER_ID"].to_list()[0]
AS_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_AS}_{AS_ID}.png")

### TO ###

TO_data = player_stat.sort_values(by = ["TO","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_TO = TO_data["PLAYER"].to_list()[0]
NUMBER_TO = TO_data["#"].to_list()[0]
TEAM_TO = TO_data["TEAM"].to_list()[0]
TO_ID = players[(players["CODETEAM"] == TEAM_TO) & (players["PLAYER"] == NAME_TO)]["PLAYER_ID"].to_list()[0]
TO_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_TO}_{TO_ID}.png")

### ST ###

ST_data = player_stat.sort_values(by = ["ST","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_ST = ST_data["PLAYER"].to_list()[0]
NUMBER_ST = ST_data["#"].to_list()[0]
TEAM_ST = ST_data["TEAM"].to_list()[0]
ST_ID = players[(players["CODETEAM"] == TEAM_ST) & (players["PLAYER"] == NAME_ST)]["PLAYER_ID"].to_list()[0]
ST_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_ST}_{ST_ID}.png")

### CO ###

CO_data = player_stat.sort_values(by = ["CO","TIME_ON"],ascending = [False,True]).reset_index(drop = True)
NAME_CO = CO_data["PLAYER"].to_list()[0]
NUMBER_CO = CO_data["#"].to_list()[0]
TEAM_CO = CO_data["TEAM"].to_list()[0]
CO_ID = players[(players["CODETEAM"] == TEAM_CO) & (players["PLAYER"] == NAME_CO)]["PLAYER_ID"].to_list()[0]
CO_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_CO}_{CO_ID}.png")


local_top_values,local_bottom_values,road_top_values,road_bottom_values = f.stats_important(r,team_local,team_road,df)
df_stats_important_players = f.stats_important_players(r,team_local,team_road,df)

local_def_perc = local_player_stat["DR"].sum()/(local_player_stat["DR"].sum() + road_player_stat["OR"].sum())
road_off_perc = 1 - local_def_perc
road_def_perc = road_player_stat["DR"].sum()/(road_player_stat["DR"].sum() + local_player_stat["OR"].sum())
local_off_perc = 1 - road_def_perc

local_def = (local_def_perc/0.7)/(local_def_perc/0.7 + road_off_perc/0.3)
road_off = (road_off_perc/0.3)/(local_def_perc/0.7 + road_off_perc/0.3)
road_def = (road_def_perc/0.7)/( road_def_perc/0.7 + local_off_perc/0.3)
local_off = (local_off_perc/0.3)/(road_def_perc/0.7 + local_off_perc/0.3)

############################# STYLE ####################################################




def highlight_columns(row, dataframe):
    styles = []
    for col in dataframe.columns:
        if col == "TEAM":
            # Pas de coloration pour TEAM
            styles.append("")
        else:
            max_val = dataframe[col].max()
            min_val = dataframe[col].min()
            if row[col] == max_val and row[col] == min_val:  # Égalité
                styles.append("background-color: yellow ; color: black")
            elif row[col] == max_val:
                styles.append("background-color: #CCFFCC ; color: black")
            elif row[col] == min_val:
                styles.append("background-color: #FFCCCC ; color: black")
            else:
                styles.append("")
    return styles

def colorize_pm_on(row):
    if row["PM_ON"] > 0:
        color = 'background-color: #CCFFCC; color: black'  # Vert clair
    elif row["PM_ON"] < 0:
        color = 'background-color: #FFCCCC; color: black'  # Rouge clair
    else:
        color = 'background-color: white; color: black'  # Blanc
    
    return [color for _ in row.index]

# Appliquer un style aux colonnes
def style_good(val):
    return 'background-color: #CCFFCC; font-weight: bold ; color: black'  # Colonne 'good' en vert et gras

def style_bad(val):
    return 'background-color: #FFCCCC; font-weight: bold ; color: black'  # Colonne 'bad' en rouge et gras


################################ COLOR TEAM ##############################################
def hex_to_rgb(hex_color):
    """Convertit une couleur hexadécimale en RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(rgb1, rgb2):
    """Calcule la distance euclidienne entre deux couleurs RGB."""
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)))

def choisir_maillot(couleur_domicile, couleurs_exterieur):
    """
    Sélectionne le maillot extérieur le plus éloigné de la couleur domicile.
    
    Args:
        couleur_domicile (str): Couleur hex de l'équipe à domicile (e.g., "#FF5733").
        couleurs_exterieur (list): Liste de couleurs hex pour l'équipe extérieure (e.g., ["#33FF57", "#3357FF"]).
    
    Returns:
        str: Couleur hex choisie pour l'équipe extérieure.
    """
    rgb_domicile = hex_to_rgb(couleur_domicile)
    distances = {
        couleur: color_distance(rgb_domicile, hex_to_rgb(couleur))
        for couleur in couleurs_exterieur
    }
    # Retourne la couleur avec la plus grande distance
    return max(distances, key=distances.get)



teams_color[teams_color["TEAM"]==team_local]["COL1"].to_list()[0]

local_c1 = teams_color[teams_color["TEAM"]==team_local]["COL1"].to_list()[0]
local_c2 = teams_color[teams_color["TEAM"]==team_local]["COL2"].to_list()[0]

road_choix_1 = teams_color[teams_color["TEAM"]==team_road]["COL1"].to_list()[0]
road_choix_2 = teams_color[teams_color["TEAM"]==team_road]["COL2"].to_list()[0]

road_c1 = choisir_maillot(local_c1, [road_choix_1,road_choix_2])
road_c2 = road_choix_1 if road_c1 != road_choix_1 else road_choix_2

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
            <b>Game Analysis - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


with t1 :
    if os.path.exists(local_logo_path):
        st.image(local_logo_path,  width=int(200*zoom))
    else:
        st.warning(f"Logo introuvable pour l'équipe : {team_local}")

    

with t2 :
    if os.path.exists(road_logo_path):
        st.image(road_logo_path,  width=int(200*zoom))
    else:
        st.warning(f"Logo introuvable pour l'équipe : {team_road}")


st.markdown(
    f'''
    <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

cola, good_bad,snum = st.columns([0.3, 0.3,0.4])

with cola :

    team1,team2 = st.columns([0.5,0.5])

    with team1 :
        st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid gold;">
                <b>{team_local} : {with_game["LOCAL_SCORE"].to_list()[0]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
    with team2 :
        st.markdown(
            f'''
            <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: {road_c1};color: {road_c2}; padding: 4px; border-radius: 5px;outline: 3px solid gold;">
                <b>{team_road} : {with_game["ROAD_SCORE"].to_list()[0]}</b>
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

    type_delta = st.selectbox("DELTA", options=["CUMULATED","PER PERIODE"], index=0)

    col_tab = ["TEAM","5","10","15","20","25","30","35","40","45","50","55","60"]

    if type_delta == "PER PERIODE" :
        data_delta = periode

        periode_local = [team_local] + periode_local
        periode_road = [team_road] + periode_road
        data_TABLEAU = pd.DataFrame([periode_local, periode_road], columns=col_tab[0:len(periode_local)])

    else : 
        data_delta = cumul
        cumul_local = [team_local] + cumul_local
        cumul_road = [team_road] + cumul_road
        data_TABLEAU = pd.DataFrame([cumul_local, cumul_road], columns=col_tab[0:len(cumul_local)])


    cols = st.columns([1.2] + [1] * (len(data_TABLEAU.columns)-1))  # Créer des colonnes dynamiques
    data_TABLEAU.columns = ["-"] + data_TABLEAU.columns[1:].tolist()

    for i, col in enumerate(cols):  # Itérer sur chaque colonne
        is_last_col = (i == len(cols) - 1)  # Vérifier si c'est la dernière colonne
        border_style = "border: 3px solid gold;" if is_last_col else "border: 3px solid black;"

        # Ajouter le titre de la colonne
        col.markdown(
            f'''
            <p style="font-size:{int(20*zoom)}px; text-align: center; background-color: #FAF0E6;color: black; padding: 2px; border-radius: 5px; {border_style}">
                <b>{data_TABLEAU.columns[i]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        # Contenu pour la première colonne
        if i == 0:  # Première colonne, pas de coloration spéciale
            col.markdown(
                f'''
                <p style="font-size:{int(20*zoom)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px; {border_style}">
                    <b>{data_TABLEAU[data_TABLEAU.columns[i]].to_list()[0]}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
            col.markdown(
                f'''
                <p style="font-size:{int(20*zoom)}px; text-align: center; background-color: {road_c1};color: {road_c2}; padding: 4px; border-radius: 5px; {border_style}">
                    <b>{data_TABLEAU[data_TABLEAU.columns[i]].to_list()[1]}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
        else:  # Autres colonnes avec coloration conditionnelle
            value1 = data_TABLEAU[data_TABLEAU.columns[i]].to_list()[0]
            value2 = data_TABLEAU[data_TABLEAU.columns[i]].to_list()[1]

            # Déterminer les couleurs
            if value1 > value2:
                color1, color2 = "green", "red"
            elif value1 < value2:
                color1, color2 = "red", "green"
            else:
                color1, color2 = "white", "white"

            # Ajouter les valeurs avec les couleurs appropriées
            col.markdown(
                f'''
                <p style="font-size:{int(20*zoom)}px; text-align: center; background-color: {color1};color: black; padding: 4px; border-radius: 5px; {border_style}">
                    <b>{value1}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
            col.markdown(
                f'''
                <p style="font-size:{int(20*zoom)}px; text-align: center; background-color: {color2};color: black; padding: 4px; border-radius: 5px; {border_style}">
                    <b>{value2}</b>
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
        
    # Noms des barres
    labels = [i * 2.5 for i in range(1, len(data_delta) + 1)]

    # Couleurs conditionnelles
    colors = [local_c1 if val > 0 else road_c1 if val < 0 else 'grey' for val in data_delta]
    contour = [local_c2 if val > 0 else road_c2 if val < 0 else 'grey' for val in data_delta]

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
        height=int(550*zoom),
        title=f"SCORE EVOL. DELTA {type_delta}",
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
    st.plotly_chart(fig,use_container_width=True)

    st.markdown(
        f'''
        <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: gold; color: black; 
        padding: 4px; border-radius: 5px; outline: 3px solid gold;">
            <b> PLAYERS KEYS STATS : </b>
        </p>
        ''',
        unsafe_allow_html=True
    )


with good_bad :
    st.markdown(
        f'''
        <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2}; 
        padding: 4px; border-radius: 5px; outline: 3px solid {local_c2};">
            <b>{team_local} - KEYS STATS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    local_df = pd.DataFrame({
        'BEST': local_top_values,
        'WORST': local_bottom_values
    })

    good,bad,shoot = st.columns([0.3,0.3,0.4])

    with good :
        st.markdown(
            f'''
            <p style="font-size:{int(23*zoom)}px; text-align: center; background-color: green;color: white; padding: 4px; border-radius: 5px;">
                <b>BEST</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in local_df["BEST"].to_list() :
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
                <b>WORST</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in local_df["WORST"].to_list() :
            st.markdown(
                f'''
                <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: #FFD3D3;color: black; padding: 3px; border-radius: 5px;">
                    <b>{g}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
    with shoot : 

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(local_player_stat["2_T"].sum()+local_player_stat["3_T"].sum(),2)}&nbsp; SHOOTS + {local_player_stat["1_T"].sum()} FT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{20*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round((local_player_stat["2_R"].sum()*2+local_player_stat["3_R"].sum()*3)/(local_player_stat["2_T"].sum()+local_player_stat["3_T"].sum()),2)}&nbsp; PTS PER SHOOT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        shot_T = local_player_stat["1_T"].sum()/2 + local_player_stat["2_T"].sum() + local_player_stat["3_T"].sum()
        TO = local_player_stat["TO"].sum()

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(100*shot_T/(shot_T+TO),1)} % BALL CARE</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(local_def+local_off,2)} REB. PERF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <p style="font-size:{20*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(100*local_player_stat["AS"].sum()/(local_player_stat["2_R"].sum()+local_player_stat["3_R"].sum()),1)}&nbsp;% ASSISTS </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {local_c1};">
                <b>{round(local_player_stat["PER"].sum(),1)} PER</b>
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

    st.markdown(
        f'''
        <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {road_c1};color: {road_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {road_c2};">
            <b>{team_road} - KEYS STATS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    road_df = pd.DataFrame({
        'BEST': road_top_values,
        'WORST': road_bottom_values
    })
    good,bad,shoot = st.columns([0.3,0.3,0.4])

    with good :
        st.markdown(
            f'''
            <p style="font-size:{int(23*zoom)}px; text-align: center; background-color: green;color: white; padding: 4px; border-radius: 5px;">
                <b>BEST</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in road_df["BEST"].to_list() :
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
                <b>WORST</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for g in road_df["WORST"].to_list() :
            st.markdown(
                f'''
                <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: #FFD3D3;color: black; padding: 3px; border-radius: 5px;">
                    <b>{g}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )
    with shoot :

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {road_c2};color: {road_c1}; padding: 6px; border-radius: 5px;outline: 3px solid {road_c1};">
                <b>{round(road_player_stat["2_T"].sum()+road_player_stat["3_T"].sum(),2)}&nbsp; SHOOTS + {road_player_stat["1_T"].sum()} FT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{20*zoom}px; text-align: center; background-color: {road_c2};color: {road_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {road_c1};">
                <b>{round((road_player_stat["2_R"].sum()*2+road_player_stat["3_R"].sum()*3)/(road_player_stat["2_T"].sum()+road_player_stat["3_T"].sum()),2)}&nbsp; PTS PER SHOOT</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        shot_T = road_player_stat["1_T"].sum()/2 + road_player_stat["2_T"].sum() + road_player_stat["3_T"].sum()
        TO = road_player_stat["TO"].sum()

        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {road_c2};color: {road_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {road_c1};">
                <b>{round(100*shot_T/(shot_T+TO),1)} % BALL CARE</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {road_c2};color: {road_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {road_c1};">
                <b>{round(road_def+road_off,2)} REB. PERF</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{20*zoom}px; text-align: center; background-color: {road_c2};color: {road_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {road_c1};">
                <b>{round(100*road_player_stat["AS"].sum()/(road_player_stat["2_R"].sum()+road_player_stat["3_R"].sum()),1)}&nbsp;% ASSISTS </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''
            <p style="font-size:{22*zoom}px; text-align: center; background-color: {road_c2};color: {road_c1}; padding: 8px; border-radius: 5px;outline: 3px solid {road_c1};">
                <b>{round(road_player_stat["PER"].sum(),1)} PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with snum :


    s1, s2,s3,s4 = st.columns([0.25, 0.25,0.25,0.25])


    with s1 :

        NAME_j = player_stat["PLAYER"].to_list()[0]
        TEAM_j = player_stat["TEAM"].to_list()[0]
        NUMBER_j = player_stat["#"].to_list()[0]
        STAT_j = player_stat["I_PER"].to_list()[0]
        ID_j = players[(players["CODETEAM"] == TEAM_j) & (players["PLAYER"] == NAME_j)]["PLAYER_ID"].to_list()[0]

        image_path_j = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_j}_{ID_j}.png")

        st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: gold;color: black; padding: 5px; border-radius: 5px;">
                <b>MVP : {STAT_j} I_PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(image_path_j):
            image = Image.open(image_path_j)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_j} {NAME_j}", width=int(250*zoom))        
        else:

            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_j}.png"), caption=f"#{NUMBER_j} {NAME_j}", width=int(200*zoom))

    with s2 :

        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{PTS_data["PTS"].to_list()[0]} POINTS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if os.path.exists(PTS_image_path):
            image = Image.open(PTS_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_PTS} {NAME_PTS}", width=int(250*zoom))
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_PTS}.png"), caption=f"#{NUMBER_PTS} {NAME_PTS}", width=int(200*zoom))




    with s3 :
        # Intégrer la couleur dans le markdown
        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{DR_data["DR"].to_list()[0]} DEF. REB.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )


        if os.path.exists(DR_image_path):
            image = Image.open(DR_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_DR} {NAME_DR}", width=int(250*zoom))
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_DR}.png"), caption=f"#{NUMBER_DR} {NAME_DR}", width=int(200*zoom))

            
    with s4 :
        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{OR_data["OR"].to_list()[0]} OFF. REB.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(OR_image_path):
            image = Image.open(OR_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_OR} {NAME_OR}", width=int(250*zoom)) 
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_OR}.png"), caption=f"#{NUMBER_OR} {NAME_OR}", width=int(200*zoom))



    st.markdown(
        f'''
        <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    s1, s2,s3,s4 = st.columns([0.25, 0.25,0.25,0.25])




    with s1 :
        # Intégrer la couleur dans le markdown

        NAME_j = player_stat["PLAYER"].to_list()[1]
        TEAM_j = player_stat["TEAM"].to_list()[1]
        NUMBER_j = player_stat["#"].to_list()[1]
        STAT_j = player_stat["I_PER"].to_list()[1]
        ID_j = players[(players["CODETEAM"] == TEAM_j) & (players["PLAYER"] == NAME_j)]["PLAYER_ID"].to_list()[0]

        image_path_j = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_j}_{ID_j}.png")

        st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: silver;color: black; padding: 5px; border-radius: 5px;">
                <b>2ND : {STAT_j} I_PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(image_path_j):
            image = Image.open(image_path_j)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_j} {NAME_j}", width=int(250*zoom))        
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_j}.png"), caption=f"#{NUMBER_j} {NAME_j}", width=int(200*zoom))


    with s2 :


        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{int(PM_ON_data["PM_ON"].to_list()[0])} +/-</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if os.path.exists(PM_ON_image_path):
            image = Image.open(PM_ON_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_PM_ON} {NAME_PM_ON}", width=int(250*zoom)) 
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_PM_ON}.png"), caption=f"#{NUMBER_PM_ON} {NAME_PM_ON}", width=int(200*zoom))



            
    with s3 :
        # Intégrer la couleur dans le markdown
        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{AS_data["AS"].to_list()[0]} ASSISTS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(AS_image_path):
            image = Image.open(AS_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_AS} {NAME_AS}", width=int(250*zoom))
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_AS}.png"), caption=f"#{NUMBER_AS} {NAME_AS}", width=int(200*zoom))


    st.markdown(
        f'''
        <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    with s4 :
        # Intégrer la couleur dans le markdown
        # Intégrer la couleur dans le markdown
        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #ff0000;color: black; padding: 5px; border-radius: 5px;">
                <b>{TO_data["TO"].to_list()[0]} TURNOVERS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(TO_image_path):
            image = Image.open(TO_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))
            st.image(image_cropped, caption=f"#{NUMBER_TO} {NAME_TO}", width=int(250*zoom))
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_TO}.png"), caption=f"#{NUMBER_TO} {NAME_TO}", width=int(200*zoom))


    s1, s2,s3,s4 = st.columns([0.25, 0.25,0.25,0.25])




    with s1 :
        NAME_j = player_stat["PLAYER"].to_list()[2]
        TEAM_j = player_stat["TEAM"].to_list()[2]
        NUMBER_j = player_stat["#"].to_list()[2]
        STAT_j = player_stat["I_PER"].to_list()[2]
        ID_j = players[(players["CODETEAM"] == TEAM_j) & (players["PLAYER"] == NAME_j)]["PLAYER_ID"].to_list()[0]

        image_path_j = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_j}_{ID_j}.png")

        st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: #CD7F32;color: white; padding: 5px; border-radius: 5px;">
                <b>3RD : {STAT_j} I_PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(image_path_j):
            image = Image.open(image_path_j)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_j} {NAME_j}", width=int(250*zoom))        
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_j}.png"), caption=f"#{NUMBER_j} {NAME_j}", width=int(200*zoom))



    with s2 :

        NAME_j = player_stat["PLAYER"].to_list()[3]
        TEAM_j = player_stat["TEAM"].to_list()[3]
        NUMBER_j = player_stat["#"].to_list()[3]
        STAT_j = player_stat["I_PER"].to_list()[3]
        ID_j = players[(players["CODETEAM"] == TEAM_j) & (players["PLAYER"] == NAME_j)]["PLAYER_ID"].to_list()[0]

        image_path_j = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_j}_{ID_j}.png")

        st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: #7B3F00;color: white; padding: 5px; border-radius: 5px;">
                <b>4TH : {STAT_j} I_PER</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(image_path_j):
            image = Image.open(image_path_j)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_j} {NAME_j}", width=int(250*zoom))        
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_j}.png"), caption=f"#{NUMBER_j} {NAME_j}", width=int(200*zoom))

            
    with s3 :

        st.markdown(
            f'''
            <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{ST_data["ST"].to_list()[0]} STEALS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        if os.path.exists(ST_image_path):
            # Charger l'image avec Pillow
            image = Image.open(ST_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_ST} {NAME_ST}", width=int(250*zoom))
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_ST}.png"), caption=f"#{NUMBER_ST} {NAME_ST}", width=int(200*zoom))
    with s4 :


        st.markdown(
            f'''
            <p style="font-size:{int(25*zoom)}px; text-align: center; background-color: #00ff00;color: black; padding: 5px; border-radius: 5px;">
                <b>{CO_data["CO"].to_list()[0]} BLOCKS</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        if os.path.exists(CO_image_path):
            image = Image.open(CO_image_path)

            # Calculer la hauteur des 75% supérieurs
            width, height = image.size
            cropped_height = int(0.66 * height)

            # Rogner l'image : garder seulement les 75% du haut
            image_cropped = image.crop((0, 0, width, cropped_height))

            # Afficher l'image rognée avec Streamlit
            st.image(image_cropped, caption=f"#{NUMBER_CO} {NAME_CO}", width=int(250*zoom))        
        else:
            st.image(os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_CO}.png"), caption=f"#{NUMBER_CO} {NAME_CO}", width=int(200*zoom))



st.markdown(
    f'''
    <p style="font-size:{int(27*zoom)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

sip1,sip2,sip3,sip4,sip5,sip6,sip7,sip8,sip9,sip10 = st.columns([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])



with sip1 :
    teamsip = df_stats_important_players["TEAM"].to_list()[0]
    playersip = df_stats_important_players["PLAYER"].to_list()[0]
    statssip = df_stats_important_players["VALUE"].to_list()[0]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with sip2:
    teamsip = df_stats_important_players["TEAM"].to_list()[1]
    playersip = df_stats_important_players["PLAYER"].to_list()[1]
    statssip = df_stats_important_players["VALUE"].to_list()[1]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip3 :
    teamsip = df_stats_important_players["TEAM"].to_list()[2]
    playersip = df_stats_important_players["PLAYER"].to_list()[2]
    statssip = df_stats_important_players["VALUE"].to_list()[2]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip4 :
    teamsip = df_stats_important_players["TEAM"].to_list()[3]
    playersip = df_stats_important_players["PLAYER"].to_list()[3]
    statssip = df_stats_important_players["VALUE"].to_list()[3]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip5 :
    teamsip = df_stats_important_players["TEAM"].to_list()[4]
    playersip = df_stats_important_players["PLAYER"].to_list()[4]
    statssip = df_stats_important_players["VALUE"].to_list()[4]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip6 :
    teamsip = df_stats_important_players["TEAM"].to_list()[5]
    playersip = df_stats_important_players["PLAYER"].to_list()[5]
    statssip = df_stats_important_players["VALUE"].to_list()[5]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip7 :
    teamsip = df_stats_important_players["TEAM"].to_list()[6]
    playersip = df_stats_important_players["PLAYER"].to_list()[6]
    statssip = df_stats_important_players["VALUE"].to_list()[6]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip8 :
    teamsip = df_stats_important_players["TEAM"].to_list()[7]
    playersip = df_stats_important_players["PLAYER"].to_list()[7]
    statssip = df_stats_important_players["VALUE"].to_list()[7]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with sip9 :
    teamsip = df_stats_important_players["TEAM"].to_list()[8]
    playersip = df_stats_important_players["PLAYER"].to_list()[8]
    statssip = df_stats_important_players["VALUE"].to_list()[8]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with sip10 :
    teamsip = df_stats_important_players["TEAM"].to_list()[9]
    playersip = df_stats_important_players["PLAYER"].to_list()[9]
    statssip = df_stats_important_players["VALUE"].to_list()[9]
    if teamsip == team_local :
        color = local_c1
        color_text = local_c2
    else :
        color = road_c1
        color_text = road_c2

    st.markdown(
        f'''
        <p style="font-size:{int(19*zoom)}px; text-align: center; background-color: {color};color: {color_text}; padding: 5px; border-radius: 5px;outline: 3px solid {color_text};">
            <b>{playersip}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(22*zoom)}px; text-align: center; background-color: {color_text};color: {color}; padding: 5px; border-radius: 5px;outline: 3px solid {color};">
            <b>{statssip}</b>
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

cola,colc = st.columns([0.2,0.80])

with cola :


    shoot_local, shoot_road = st.columns([1, 1])

    with shoot_local :
        
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center;background-color: {local_c1};color: {local_c2};padding: 4px; border-radius: 5px; outline: 3px solid gold;">
                <b>{team_local} PERC.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(local_player_stat["1_R"].sum()/local_player_stat["1_T"].sum() if local_player_stat["1_T"].sum() != 0 else 0,"FT",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "o")
        fig2 = f.plot_semi_circular_chart(local_player_stat["2_R"].sum()/local_player_stat["2_T"].sum() if local_player_stat["2_T"].sum() != 0 else 0,"2P",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "p")
        fig2 = f.plot_semi_circular_chart(local_player_stat["3_R"].sum()/local_player_stat["3_T"].sum() if local_player_stat["3_T"].sum() != 0 else 0,"3P",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "q")
        

        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {local_c1}; color: {local_c2};padding: 4px; border-radius: 5px; outline: 3px solid gold;">
                <b>{team_local} REB.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )


        fig2 = f.plot_semi_circular_chart(local_player_stat["DR"].sum()/(local_player_stat["DR"].sum() + road_player_stat["OR"].sum()),"DEF.", size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "a")

        fig2 = f.plot_semi_circular_chart(local_player_stat["OR"].sum()/(local_player_stat["OR"].sum() + road_player_stat["DR"].sum()),"OFF.",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "ba")

        fig2 = f.plot_semi_circular_chart(local_player_stat["TR"].sum()/(road_player_stat["TR"].sum() + local_player_stat["TR"].sum()),"TOT.",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "TT")








    with shoot_road :
        
        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {road_c1}; color: {road_c2};padding: 4px; border-radius: 5px; outline: 3px solid gold;">
                <b>{team_road} PERC. </b>
            </p>
            ''',
            unsafe_allow_html=True
        )


        fig2 = f.plot_semi_circular_chart(road_player_stat["1_R"].sum()/road_player_stat["1_T"].sum() if road_player_stat["1_T"].sum() != 0 else 0,"FT",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key="fig2_unique_key")

        fig2 = f.plot_semi_circular_chart(road_player_stat["2_R"].sum()/road_player_stat["2_T"].sum() if road_player_stat["2_T"].sum() != 0 else 0,"2P",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key="1")
        fig2 = f.plot_semi_circular_chart(road_player_stat["3_R"].sum()/road_player_stat["3_T"].sum() if road_player_stat["3_T"].sum() != 0 else 0,"3P",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key="2")


        st.markdown(
            f'''
            <p style="font-size:{int(30*zoom)}px; text-align: center; background-color: {road_c1}; color: {road_c2};padding: 4px; border-radius: 5px; outline: 3px solid gold;">
                <b>{team_road} REB.</b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        fig2 = f.plot_semi_circular_chart(road_player_stat["DR"].sum()/(road_player_stat["DR"].sum() + local_player_stat["OR"].sum()),"DEF.", size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "pio")

        fig2 = f.plot_semi_circular_chart(road_player_stat["OR"].sum()/(road_player_stat["OR"].sum() + local_player_stat["DR"].sum()),"OFF.",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "salut")

        fig2 = f.plot_semi_circular_chart(road_player_stat["TR"].sum()/(road_player_stat["TR"].sum() + local_player_stat["TR"].sum()),"TOT.",size=int(95*zoom), font_size=int(22*zoom))
        st.plotly_chart(fig2,use_container_width=True,key = "T")





with colc :
    C1, C2 = st.columns([1, 1])
    with C1 :
        t = st.selectbox("TEAM", options=[team_local,team_road], index=0)
    with C2 :
        affichage = st.selectbox("AFFICHAGE", options=["BOXESCORE","+/- DUO"], index=0)

    if affichage == "BOXESCORE" :
        data_aff = player_stat[player_stat["TEAM"]==t].drop(columns = ["TEAM","ROUND","NB_GAME","OPPONENT","HOME","WIN"]).sort_values(by = "TIME_ON",ascending = False)
        st.dataframe(data_aff, height=min(36 + 36*len(data_aff),16*36),hide_index=True,use_container_width=True)
    else :
        data_aff = f.analyse_io_2(data_dir = data_dir,
                    competition = competition,
                    season = season,
                    num_players = 2,
                        min_round = r,
                        max_round = r,
                    CODETEAM = [t],
                    selected_players = [],
                    min_percent_in = 0).drop(columns = ["TEAM"]).sort_values(by = "TIME_ON",ascending = False)[['P1', 'P2', 'PERCENT_ON', 'TIME_ON', 'PM_ON', 'OFF_ON_10', 'DEF_ON_10',
       'DELTA_ON', 'PERCENT_OFF', 'TIME_OFF', 'PM_OFF', 'OFF_OFF_10',
       'DEF_OFF_10', 'DELTA_OFF']]
        
        data_aff2 = data_aff.style.apply(colorize_pm_on, axis=1).format(precision=1)
        st.dataframe(data_aff2, height=min(36 + 36*len(data_aff),22*36),hide_index=True,use_container_width=True)