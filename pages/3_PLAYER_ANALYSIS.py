import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

st.set_page_config(
    page_title="PLAYER ANALYSIS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)


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



# Remplir les trous dans ROUND
min_round, max_round = df["ROUND"].min(), df["ROUND"].max()


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

# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("SETTINGS")

zoom = st.sidebar.slider(
    "Choisissez une valeur de zoom pour les photos",
    min_value=0.3,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)

selected_range = st.sidebar.slider(
    "Select a ROUND range:",
    min_value=min_round,
    max_value=max_round,
    value=(min_round, max_round),
    step=1
)
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


selected_stats = st.sidebar.selectbox("Stat Sélectionné", options=["PER","I_PER","PTS","TR","AS","PM_ON","ST"])

window_size = st.sidebar.number_input("Sliding average", min_value=1,max_value=max_round ,value=5)
# Calcul des min et max pour les axes
min_selected_stats = df[selected_stats].min()
max_selected_stats = df[selected_stats].max()

min_time_on = df["TIME_ON"].min()
max_time_on = df["TIME_ON"].max()
col1, col2,col3 = st.columns([1.2,2,1])

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
            <b>Player Analysis - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with col3 :
    # Filtrage dynamique des équipes
    CODETEAM = st.selectbox("Équipe Sélectionnée", options=sorted(df["TEAM"].unique()) if not df.empty else [],index=sorted(df["TEAM"].unique()).index(notation["CODETEAM"].to_list()[0]))

    # Mise à jour dynamique des joueurs en fonction des équipes sélectionnées
    if CODETEAM:
        available_players = players[players["CODETEAM"].isin([CODETEAM])]["PLAYER"].unique()
    else:
        available_players = players["PLAYER"].unique()
    if CODETEAM == notation["CODETEAM"].to_list()[0] :
        selected_players = st.selectbox("Joueur Sélectionné", options=sorted(available_players),index=sorted(available_players).index(notation["PLAYER"].to_list()[0]))
    else :
        selected_players = st.selectbox("Joueur Sélectionné", options=sorted(available_players))



filtered_df = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["ROUND", "PLAYER", "TEAM"],
    selected_players=[selected_players],
    mode="CUMULATED",
    percent="MADE"
)
gs_filtered = gs[(gs["LOCAL"]==CODETEAM)|(gs["ROAD"]==CODETEAM)].reset_index(drop = True)
gs_filtered = gs_filtered[(gs_filtered["ROUND"]>=selected_range[0])&(gs_filtered["ROUND"]<=selected_range[1])].reset_index(drop = True)

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

player_image_path = os.path.join(images_dir, f"{competition}_{season}_players/{TEAM_PLAYER}_{PLAYER_ID}.png")


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

def highlight_win(row):
    # Appliquer vert si WIN est "YES", rouge sinon
    color = 'background-color: #CCFFCC; color: black;' if row["WIN"] == "YES" else 'background-color: #FFCCCC; color: black;'
    return [color for _ in row.index]



filtered_df2 = filtered_df.drop(columns=["WIN","OPPONENT","HOME","moving_avg"])

# Jointure gauche
df_resultat = pd.merge(
    gs_filtered, 
    filtered_df2, 
    on='ROUND', 
    how='left'
)


colonnes_a_convertir = ['PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'TO', '1_T', '2_T', '3_T',"1_R", "2_R", "3_R", 'CO', 'FP', 'CF', 'NCF']

# Conversion en type Int64 (supporte les NA)
df_resultat[colonnes_a_convertir] = df_resultat[colonnes_a_convertir].astype('Int64')
df_resultat = df_resultat.sort_values(by = "ROUND",ascending=False)

# Appliquer le style sur le DataFrame
styled_df = df_resultat.style.apply(highlight_win, axis=1).format(precision=1)  





avg_data = f.get_aggregated_data(
    df, selected_range[0], selected_range[1],
    selected_teams=[CODETEAM],
    selected_opponents=[],
    selected_fields=["TEAM","PLAYER"],
    selected_players=[selected_players],
    mode="AVERAGE",
    percent="MADE"
)
avg_data = avg_data.loc[:, (avg_data != "---").any(axis=0)]

avg_data = avg_data.drop(columns = ["#","PLAYER","TEAM"])

def style_pm_on(value):
    if value > 0:
        return "background-color: #CCFFCC;color: black;"
    elif value < 0:
        return "background-color: #FFCCCC;color: black;"
    else:
        return "background-color: #FFFFFF;color: black;"





st.header(f"AVERAGES : {NAME_PLAYER} ")
st.dataframe(avg_data,height=60, use_container_width=True,hide_index=True)


col1,col_image, col2, col3 = st.columns([1.2,2, 6, 4])

with col1:
    cola, colb= st.columns([0.5,2])

    with colb : 


        team_logo_path = os.path.join(images_dir, f"{competition}_{season}_teams/{TEAM_PLAYER}.png")

        if os.path.exists(team_logo_path):
            st.image(team_logo_path, width=int(150*zoom))
        else:
            st.warning(f"Logo introuvable pour l'équipe : {TEAM_PLAYER}")



    fig2 = f.plot_semi_circular_chart(df_resultat["1_R"].sum()/df_resultat["1_T"].sum() if df_resultat["1_T"].sum() != 0 else 0,"FT",size=int(120*zoom),font_size=int(20*zoom),m=False)
    st.plotly_chart(fig2)
    fig2 = f.plot_semi_circular_chart(df_resultat["2_R"].sum()/df_resultat["2_T"].sum() if df_resultat["2_T"].sum() != 0 else 0,"2P",size=int(120*zoom),font_size=int(20*zoom),m=False)
    st.plotly_chart(fig2)
    fig2 = f.plot_semi_circular_chart(df_resultat["3_R"].sum()/df_resultat["3_T"].sum() if df_resultat["3_T"].sum() != 0 else 0,"3P",size=int(120*zoom),font_size=int(20*zoom),m=False)
    st.plotly_chart(fig2)

    st.markdown(
        f'''
        <p style="font-size:{int(20*zoom)}px; text-align: center; ">
            <b>SCORE PER 10 mins</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


        

with col_image :
    # Récupérer la note et la couleur associée pour le joueur
    player_note = notation.loc[(notation["CODETEAM"] == CODETEAM) & (notation["PLAYER"] == NAME_PLAYER), "NOTE"].to_list()[0]
    player_color = notation.loc[(notation["CODETEAM"] == CODETEAM) & (notation["PLAYER"] == NAME_PLAYER), "COULEUR"].to_list()[0]

    # Intégrer la couleur dans le markdown
    st.markdown(
        f'''
        <p style="font-size:{int(40*zoom)}px; text-align: center; background-color: {player_color};color: black; padding: 2px; border-radius: 5px;">
            <b>NOTE : {round(player_note)}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    if os.path.exists(player_image_path):
        st.image(player_image_path, caption=f"#{NUMBER_PLAYER} {NAME_PLAYER}", width=int(330*zoom))
    else:
        st.warning(f"Image introuvable pour le joueur : {NAME_PLAYER}")

    result_pm_solo = f.analyse_io_2(data_dir = data_dir,
                    competition = competition,
                    season = season,
                    num_players = 1,
                    min_round = min_round,
                    max_round = max_round,
                    CODETEAM = [CODETEAM],
                    selected_players = [selected_players],
                    min_percent_in = 0)

    st.markdown(
        f'''
    <p style="font-size:{int(35*zoom)}px; text-align: center; background-color: green;color: black; padding: 2px; border-radius: 5px;">
            <b style="margin-right: {int(24*zoom)}px;">ON :</b> 
            <span style="margin-right: {int(24*zoom)}px;">{round(result_pm_solo["OFF_ON_10"].to_list()[0],1)}</span> 
            <span style="margin-right: {int(24*zoom)}px;">-</span> 
            {round(result_pm_solo["DEF_ON_10"].to_list()[0],1)}
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
    <p style="font-size:{int(35*zoom)}px; text-align: center; background-color: red;color: black; padding: 2px; border-radius: 5px;">
            <b style="margin-right: {int(24*zoom)}px;">OFF :</b> 
            <span style="margin-right: {int(24*zoom)}px;">{round(result_pm_solo["OFF_OFF_10"].to_list()[0],1)}</span> 
            <span style="margin-right: {int(24*zoom)}px;">-</span> 
            {round(result_pm_solo["DEF_OFF_10"].to_list()[0],1)}
        </p>
        ''',
        unsafe_allow_html=True
    )
    

with col2:

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
        name=f"{selected_stats} Moyenne glissante ({window_size} derniers rounds)"
    ))

    # Mise à jour du layout
    fig.update_layout(
        autosize=True,
        title=f'#{NUMBER_PLAYER} {NAME_PLAYER} ({TEAM_PLAYER}) AVG: {avg_data[selected_stats].sum()} {selected_stats}  ON {avg_data["TIME_ON"].sum()} MIN ',
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
            range=[min_selected_stats, max_selected_stats],  # Fixation de la plage
        ),
        yaxis2=dict(
            title="Minutes",
            titlefont=dict(color="grey"),
            tickfont=dict(color="grey"),
            overlaying="y",
            side="right",
            showgrid=False,
            range=[min_time_on, max_time_on],  # Fixation de la plage
        ),
        legend=dict(
            orientation="h",  # Layout horizontal
            yanchor="top",
            y=-0.3,  # Position sous le graphique
            xanchor="center",
            x=0.5  # Centré horizontalement
        ),
        margin=dict(l=50, r=50, t=50, b=100),
        height=500,
        dragmode="pan"  
    )


    st.plotly_chart(fig, use_container_width=True,static_plot=True)


with col3:
    # Création des boxplots avec Plotly Graph Objects
    box_fig = go.Figure()

    # Boxplot global pour PER
    box_fig.add_trace(go.Box(
        y=filtered_df[selected_stats],
        name="Global",
        marker_color="blue"
    ))

    # Boxplots pour chaque modalité de WIN
    for win_status in reversed(sorted(filtered_df["WIN"].unique())):
        if win_status == "YES" :
            ecriture = "WIN"
        else :
            ecriture = "LOSE"

        box_fig.add_trace(go.Box(
            y=filtered_df[filtered_df["WIN"] == win_status][selected_stats],
            name=f"{ecriture}",
            marker_color="green" if win_status == "YES" else "red"
        ))

    # Configuration du layout des boxplots
    box_fig.update_layout(
        autosize=True,  # Le graphique s'ajuste automatiquement
        title=f"Distribution de {selected_stats} (Global et par résultat)",
        yaxis=dict(
            title=selected_stats,
            showgrid=True,
            range=[min_selected_stats, max_selected_stats]
        ),
        xaxis=dict(
            title="Catégories",
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        height=400,
        showlegend=False,
        dragmode="pan"
    )

    
    # Affichage des boxplots
    st.header(f"Boxplots de {selected_stats}")
    st.plotly_chart(box_fig, use_container_width=True)


# Afficher le tableau stylé dans Streamlit
st.header(f"Stats : {NAME_PLAYER}")
st.dataframe(styled_df,height=min(38*len(df_resultat),650), use_container_width=True,hide_index=True)


st.header(f"Duo +/- with {NAME_PLAYER}")

col1, col2,col3 = st.columns([1, 1,2])

with col1 :

    min_percent_in = st.slider("Minimum Time global percent together", min_value=0, max_value=100, value=0)
with col2 :
    min_percent_in_relative = st.slider("Minimum Time relative percent together", min_value=0, max_value=100, value=0)

result_pm = f.analyse_io_2(data_dir = data_dir,
                competition = competition,
                season = season,
                num_players = 2,
                min_round = min_round,
                max_round = max_round,
                CODETEAM = [CODETEAM],
                selected_players = [selected_players],
                min_percent_in = min_percent_in)



result_pm['P2'] = result_pm.apply(lambda row: row['P2'] if 
                     (row['P1'] == selected_players)
                     else row['P1'], axis=1)

result_pm = result_pm.drop(columns = ["P1","TEAM","TIME_TEAM","PM_TEAM","OFF_TEAM_10","DEF_TEAM_10"])

result_pm.insert(1,"PER_REL_ON", (result_pm["TIME_ON"]/(result_pm["TIME_ON"].sum()/4)*100).round(2) )

result_pm = result_pm[result_pm["PER_REL_ON"]>=min_percent_in_relative]
result_pm = result_pm.sort_values(by = ["TIME_ON"],ascending = False).reset_index(drop = True)
# Apply styling
styled_result_pm = result_pm.style.applymap(style_pm_on, subset=["PM_ON", "DELTA_ON","PM_OFF","DELTA_OFF"]).format(precision=2) 

st.dataframe(styled_result_pm,height=min(38*len(result_pm),650), use_container_width=True,hide_index=True)



