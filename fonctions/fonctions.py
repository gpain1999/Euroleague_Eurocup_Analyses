import pandas as pd
import os
import sys
from itertools import combinations
import itertools
import math
from euroleague_api.game_stats import GameStats
from euroleague_api.play_by_play_data import PlayByPlay
from euroleague_api.player_stats import PlayerStats
from euroleague_api.team_stats import TeamStats
import requests
import sqlite3
from PIL import Image
import os
import requests
from io import BytesIO
from PIL import Image, ImageOps,ImageDraw,ImageFont
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import threading
import streamlit as st

def plot_semi_circular_chart(value, t,width=600, height=300):
    """
    Create a semi-circular chart (top half only) with Plotly.
    The size can be controlled with the `width` and `height` parameters.
    """
    # Validate input
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1.")

    # Define chart data for the semi-circle
    value_percentage = value * 100
    filled_portion = value * 180  # Degrees for the filled portion
    empty_portion = 180 - filled_portion  # Degrees for the empty portion

    # Add a transparent placeholder to simulate the bottom half of the circle
    placeholder = 180

    # Create the semi-circular plot using a Pie chart
    fig = go.Figure()

    # Add the green (filled) and red (empty) portions
    fig.add_trace(go.Pie(
        values=[filled_portion, empty_portion, placeholder],
        labels=["Rempli", "Vide", ""],  # Empty label for the placeholder
        marker=dict(colors=["green", "red", "rgba(0,0,0,0)"]),  # Transparent for placeholder
        textinfo="none",  # Hide text on the slices
        hole=0.8,
        direction="clockwise",
        sort=False,
        showlegend=False
    ))

    # Add the percentage text in the center
    fig.update_layout(
        annotations=[
            dict(
                text=f"{t} : <b>{int(value_percentage)}%</b>",
                x=0.5, y=0.4,
                font_size=25,
                showarrow=False
            )
        ],
        # Adjust layout for the semi-circle
        margin=dict(t=0, b=0, l=0, r=0),
        width=width,  # Control the width of the graph
        height=height,  # Control the height of the graph
    )

    # Restrict to the top half of the circle
    fig.update_traces(rotation=270, pull=[0, 0, 0])  # Rotate to make green start on the top left

    return fig



def analyse_io_2(data_dir,competition,season,num_players,min_round,max_round,CODETEAM = [],selected_players = [],min_percent_in = 0) :
    conn = sqlite3.connect(':memory:')
    data_pm = pd.read_csv(os.path.join(data_dir, f"{competition}_io_{num_players}_{season}.csv"))


    if competition == "euroleague" :
        data_pm.insert(0,"R",(data_pm["Gamecode"] - 1 ) // 9 + 1 )
    else :
        data_pm.insert(0,"R",(data_pm["Gamecode"] - 1 ) // 10 + 1 )

    data_id = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

    data_pm.to_sql('data_pm', conn, index=False, if_exists='replace') 
    data_id.to_sql('data_id', conn, index=False, if_exists='replace')



    # Commencer à construire la requête SQL
    query = """
    SELECT t.CODETEAM as TEAM,
    """

    # Ajouter les jointures et les colonnes de joueurs dynamiquement
    joins = []
    for i in range(1, num_players + 1):
        query += f"t{i}.PLAYER as P{i},t{i}.PLAYER_ID as ID{i},\n"
        joins.append(f"LEFT JOIN data_id as t{i} on (t.CODETEAM = t{i}.CODETEAM) and (t.J{i} = t{i}.PLAYER_ID)")

    # Ajouter les agrégations
    query += """
    sum(t.TIME) as TIME_ON,
    sum(DELTA_SCORE) as PM_ON,
    ROUND(sum(OFF)/sum(TIME) *10,2) as OFF_ON_10,
    ROUND(sum(DEF)/sum(TIME) *10,2) as DEF_ON_10,

    ROUND(MAX(TEAM.TIME_TEAM) - sum(t.TIME),2) as TIME_OFF,
    MAX(TEAM.DELTA_SCORE_TEAM) - sum(DELTA_SCORE) as PM_OFF,
    ROUND((MAX(TEAM.OFF_TEAM) - sum(OFF)) / (MAX(TEAM.TIME_TEAM) - sum(t.TIME)) *10,2)  as OFF_OFF_10,
    ROUND((MAX(TEAM.DEF_TEAM) - sum(DEF)) / (MAX(TEAM.TIME_TEAM) - sum(t.TIME)) *10,2) as DEF_OFF_10,

    CAST(ROUND(MAX(TEAM.TIME_TEAM)) AS INTEGER) as TIME_TEAM,
    MAX(TEAM.DELTA_SCORE_TEAM) as PM_TEAM,
    ROUND(MAX(TEAM.OFF_TEAM)/MAX(TEAM.TIME_TEAM)*10,2) as OFF_TEAM_10,
    ROUND(MAX(TEAM.DEF_TEAM)/MAX(TEAM.TIME_TEAM)*10,2) as DEF_TEAM_10

    FROM data_pm as t
    """

    # Ajouter les jointures à la requête
    query += "\n".join(joins)

    # Ajouter le sous-requête pour les équipes
    query += f"""
    LEFT JOIN (SELECT CODETEAM,
    sum(TIME)/{math.comb(5, num_players)} as TIME_TEAM,
    sum(DELTA_SCORE)/{math.comb(5, num_players)} as DELTA_SCORE_TEAM,
    sum(OFF) /{math.comb(5, num_players)} as OFF_TEAM,
    sum(DEF) /{math.comb(5, num_players)} as DEF_TEAM

    from data_pm
    """

    query += f"WHERE R>= {min_round} and R <= {max_round}"

    query += f"""
    group by CODETEAM
    ORDER BY DELTA_SCORE_TEAM DESC) as TEAM on t.CODETEAM = TEAM.CODETEAM
    """

    # Terminer la requête avec les clauses GROUP BY et ORDER BY
    query += f"""
    WHERE 1=1
    """

    # Ajout conditionnel pour R
    R = [i for i in range(min_round,max_round +1)]
    if R:
        r_filter = ', '.join(map(str, R))
        query += f"AND t.R IN ({r_filter}) "

    # Ajout conditionnel pour CODETEAM
    if CODETEAM:
        code_filter = "', '".join(CODETEAM)  # Création de la chaîne pour la liste
        query += f"AND t.CODETEAM IN ('{code_filter}') "

    query += f"""
    GROUP BY t.CODETEAM, {', '.join([f't{i}.PLAYER' for i in range(1, num_players + 1)])}
    ORDER BY sum(DELTA_SCORE) DESC, sum(TIME)
    """

    # Exécution de la requête
    result = pd.read_sql_query(query, conn)
    conn.close()

    result = result.drop(columns=[f'ID{i}' for i in range(1,num_players+1)],axis = 1)

    result.insert(num_players+1,"PERCENT_ON",((result["TIME_ON"]/result["TIME_TEAM"]*100).round(1)))
    result.insert(num_players+6,"DELTA_ON",((result["OFF_ON_10"]-result["DEF_ON_10"]).round(2)))

    result.insert(num_players+7,"PERCENT_OFF",((result["TIME_OFF"]/result["TIME_TEAM"]*100).round(1)))
    result.insert(num_players+12,"DELTA_OFF",((result["OFF_OFF_10"]-result["DEF_OFF_10"]).round(2)))

    if selected_players != [] :
        result = result[result[[f"P{i+1}" for i in range(num_players)]].apply(lambda row: all(joueur in row.values for joueur in selected_players), axis=1)]

    result = result[result["PERCENT_ON"]>=min_percent_in]
    return result.reset_index(drop = True)


def afficher_dataframe(df):
    # Fonction pour trier les colonnes lors d'un clic sur l'en-tête
    def trier_colonne(col):
        nonlocal tri_inverse
        # Trier les données du DataFrame
        df.sort_values(by=col, ascending=not tri_inverse[col], inplace=True)
        tri_inverse[col] = not tri_inverse[col]  # Inverser l'ordre pour le prochain tri
        # Supprimer les anciennes données
        for row in tree.get_children():
            tree.delete(row)
        # Insérer les nouvelles données triées
        for row in df.itertuples(index=False):
            tree.insert("", "end", values=row)

    # Créer la fenêtre principale
    root = tk.Tk()
    root.title("Affichage du DataFrame")

    # Créer un cadre pour la table et la scrollbar
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    # Créer une Scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Créer un tableau (Treeview) pour afficher les données
    tree = ttk.Treeview(frame, yscrollcommand=scrollbar.set)

    # Définir les colonnes (sans l'index)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"  # Masquer l'index

    # Dictionnaire pour suivre si le tri est croissant ou décroissant
    tri_inverse = {col: False for col in df.columns}

    # Calcul de la largeur optimale pour chaque colonne et définir l'alignement
    for col in df.columns:
        # Calcul de la longueur maximale de la colonne, en gérant les valeurs manquantes
        if df[col].notna().any():  # Vérifie s'il existe au moins une valeur non nulle
            max_width = max(df[col].dropna().astype(str).map(len).max(), len(col)) * 10  # Ajustement pour l'affichage
        else:
            max_width = len(col) * 10  # Si la colonne est vide, on utilise la longueur du nom de la colonne

        # Centrer les colonnes numériques et aligner à gauche pour les autres
        align = "center" if pd.api.types.is_numeric_dtype(df[col]) else "w"
        tree.column(col, width=max_width, anchor=align)  # Définir la largeur et alignement
        tree.heading(col, text=col, command=lambda _col=col: trier_colonne(_col))

    # Insérer les données dans le tableau
    for row in df.itertuples(index=False):
        tree.insert("", "end", values=row)

    # Configurer la scrollbar pour contrôler le tableau
    scrollbar.config(command=tree.yview)

    # Ajouter le tableau dans la fenêtre
    tree.pack(fill='both', expand=True)

    # Lancer la fenêtre Tkinter
    root.mainloop()


def create_infographie(df,titre,season,competition,images_dir,infographie_dir,name_image,percent_time = False) : 
    nb_id = df.columns[df.columns.str.startswith('ID')].size
    # Define the size of the new image
    width, height = 2500  + 750 * nb_id, 1000 + len(df)*1000 + 50

    # Create a new image with a specific color background
    if competition == "euroleague" :
        background_color = "#FEB673"
        border_color = "#E9540D"  # Color code for the background
    else : 
        background_color = "#87C2FA"
        border_color = "#0971CE"

    white_image = Image.new("RGB", (width, height), background_color)

    # Path to the euroleague logo
    euroleague_image_path = os.path.join(images_dir, f"{competition}.png")  # Updated to use the images_dir
    # Load the euroleague logo image
    euroleague_image = Image.open(euroleague_image_path)
    new_size = (int(euroleague_image.width * 3), int(euroleague_image.height * 3))
    resized_euroleague_image = euroleague_image.resize(new_size, Image.Resampling.LANCZOS)
    # If the image has transparency, we can add a white background to make it visible
    euroleague_image = resized_euroleague_image.convert("RGBA")
    white_bg = Image.new("RGBA", euroleague_image.size, background_color)
    euroleague_image_with_bg = Image.alpha_composite(white_bg, euroleague_image)
    # Convert back to RGB to paste onto the white image
    euroleague_image_with_bg = euroleague_image_with_bg.convert("RGB")
    # Paste the logo at the top-left corner (coordinates 1115, 248)
    white_image.paste(euroleague_image_with_bg, (int((2500  + 750 * nb_id - 1800)/2), 125))


    font_path = "C:/Users/guill/OneDrive/Documents/euroleague_eurocup/infographie/police/GothamBookItalic.ttf"
    font_path_bold = "C:/Users/guill/OneDrive/Documents/euroleague_eurocup/infographie/police/GothamBoldItalic.ttf" 

    draw = ImageDraw.Draw(white_image)
    text_color = "black"  # Couleur du texte
    for index, row in df.iterrows():
        for j in range(nb_id) :

            # Path to the euroleague logo
            id_act = f'ID{j+1}'
            players_path = os.path.join(images_dir, f"{competition}_{season}_players/{row['TEAM']}_{row[id_act]}.png")  
            # Load the euroleague logo image
            players_image = Image.open(players_path)
            players_image = players_image.resize((750,1000), Image.Resampling.LANCZOS)
            # If the image has transparency, we can add a white background to make it visible
            players_image = players_image.convert("RGBA")
            white_bg = Image.new("RGBA", players_image.size, background_color)
            players_image_with_bg = Image.alpha_composite(white_bg, players_image)
            # Convert back to RGB to paste onto the white image
            players_image_with_bg = players_image_with_bg.convert("RGB")
            # Paste the logo at the top-left corner (coordinates 1115, 248)
            white_image.paste(players_image_with_bg, (2500  + j*750, 1000 + index*1000))



        # Path to the euroleague logo
        team_path = os.path.join(images_dir, f"{competition}_{season}_teams/{row['TEAM']}.png")  
        # Load the euroleague logo image
        team_image = Image.open(team_path)
        team_image = team_image.resize((400,400), Image.Resampling.LANCZOS)

        # If the image has transparency, we can add a white background to make it visible
        team_image = team_image.convert("RGBA")
        white_bg = Image.new("RGBA", team_image.size, background_color)
        team_image_with_bg = Image.alpha_composite(white_bg, team_image)
        # Convert back to RGB to paste onto the white image
        team_image_with_bg = team_image_with_bg.convert("RGB")
        # Paste the logo at the top-left corner (coordinates 1115, 248)
        white_image.paste(team_image_with_bg, (2100, 1600 + index*1000))

        # Texte à ajouter
        if row['PM_ON'] > 0 :
            text_bold = f"+{row['PM_ON']}"
        else :
            text_bold = f"{row['PM_ON']}"
        
        if percent_time :
            text_regular = f"ON court in {round(row['TIME_ON']/row['TIME_TEAM']*100)}% of mins"
            second_text = f"{round(row['TIME_OFF'] / row['TIME_TEAM'] * 100)}%"
        else : 
            text_regular = f"ON court in {row['TIME_ON']} mins"
            second_text = row['TIME_OFF']


        
        if row['PM_OFF'] > 0 :
            text_regular2 = f"+{row['PM_OFF']} OFF court in {second_text}"
        else :
            text_regular2 = f"{row['PM_OFF']} OFF court in {second_text}"

        # Calculer la largeur du texte en gras
        text_bold_width = draw.textlength(text_bold, font=ImageFont.truetype(font_path_bold, 300))

        # Positionner le texte
        text_position_x = 100  
        text_position_y = 1250  + index*1000

        # Ajouter le texte en gras
        draw.text((text_position_x, text_position_y), text_bold, fill=text_color, font=ImageFont.truetype(font_path_bold, 325))
        # Ajouter le texte normal juste après le texte en gras
        draw.text((text_position_x + text_bold_width + 50 , text_position_y + 130), text_regular, fill=text_color, font=ImageFont.truetype(font_path, 150))
        draw.text((text_position_x, text_position_y + 390), text_regular2, fill=text_color, font=ImageFont.truetype(font_path, 150))


    font_size = 200  # Ajustez la taille de la police si nécessaire
    font = ImageFont.truetype(font_path, font_size)

    # Créer un contexte de dessin


    # Texte à ajouter
    text = titre

    # Calculer la boîte englobante du texte
    bbox = draw.textbbox((0, 0), text, font=font)  # Obtenir les coordonnées de la boîte
    text_width = bbox[2] - bbox[0]  # Largeur = x_max - x_min
    text_height = bbox[3] - bbox[1]  # Hauteur = y_max - y_min

    # Calculer la position x pour centrer le texte
    text_position_x = (white_image.width - text_width) // 2  # Position x centrée
    text_position_y = 825  # Position y

    # Ajouter le texte à l'image
    draw.text((text_position_x, text_position_y), text, fill=text_color, font=font)


    white_image = ImageOps.expand(white_image, border= 50, fill=border_color)
    # Save the image as PNG
    white_image_path = os.path.join(infographie_dir, f"{name_image}.png")  # Save in the images directory
    white_image.save(white_image_path)

    print(f"Final image saved as: {white_image_path}")


def recuperation_team_photo(competition, season, image_dir):
    # Définir le chemin du dossier pour les images des équipes
    team_folder_path = os.path.join(image_dir, f'{competition}_{season}_teams')

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(team_folder_path):
        os.makedirs(team_folder_path)
        print(f"Folder '{team_folder_path}' created.")
    else:
        print(f"Folder '{team_folder_path}' already exists.")

    # Déterminer le code de compétition en fonction de la compétition
    if competition == "euroleague":
        competition_code = "E"
    else:
        competition_code = "U"

    # Récupérer les statistiques des équipes
    team_stats = TeamStats(competition_code)
    df_teams = team_stats.get_team_stats_single_season(
        endpoint="traditional", season=season, phase_type_code="RS", statistic_mode="PerGame"
    )

    # Extraire les colonnes nécessaires
    df_filtered_teams = df_teams[["team.code", "team.imageUrl"]]
    df_filtered_teams.columns = ["team_id", "team_image_url"]
    df_filtered_teams = df_filtered_teams[df_filtered_teams["team_image_url"].notna()]

    # Fonction pour vérifier si l'image de l'équipe existe déjà dans le dossier
    def image_not_exists(row):
        team_image_path = os.path.join(team_folder_path, f"{row['team_id']}.png")
        return not os.path.isfile(team_image_path)
    
    # Appliquer le filtre pour ne garder que les équipes dont l'image n'existe pas déjà
    df_new_teams = df_filtered_teams[df_filtered_teams.apply(image_not_exists, axis=1)]

    # Télécharger et sauvegarder les images pour chaque équipe filtrée
    for index, row in df_new_teams.iterrows():
        team_image_path = os.path.join(team_folder_path, f"{row['team_id']}.png")
        team_image_url = row["team_image_url"]
        response = requests.get(team_image_url)
        
        # Vérifier si la requête a réussi (code 200)
        if response.status_code == 200:
            # Ouvrir l'image depuis le contenu téléchargé
            image = Image.open(BytesIO(response.content))
            # Redimensionner l'image en 192x192 pixels
            image_resized = image.resize((192, 192), Image.Resampling.LANCZOS)

            # Enregistrer l'image redimensionnée sur le disque
            image_resized.save(team_image_path)
            print(f"Image for team {row['team_id']} downloaded and resized successfully.")
        else:
            print(f"Download failed for team {row['team_id']}. Error code:", response.status_code)


def convert_duration_to_minutes(duration_str):
    """
    Convertit une chaîne de caractères représentant une durée au format "0 days 00:06:020"
    en minutes.
    
    :param duration_str: Chaîne de caractères de la durée.
    :return: Durée en minutes.
    """
    # Extraire les jours, heures, minutes et secondes
    days, time = duration_str.split(' days ')
    hours, minutes, seconds = map(int, time.split(':'))

    # Convertir en minutes
    total_minutes = int(days) * 24 * 60 + hours * 60 + minutes + seconds / 60
    return total_minutes


def analyse_per(data_dir,competition,season,R = [],CODETEAM = []) :

    conn = sqlite3.connect(':memory:')
    data_per = pd.read_csv(os.path.join(data_dir, f'{competition}_PER_{season}_round.csv'))

    data_per.to_sql('data_per', conn, index=False, if_exists='replace') 

    query = """SELECT CODETEAM,MAX(PLAYER) as PLAYER,MAX(NUMBER) as NUMBER,PLAYER_ID,count(*) as NB_GAME, sum(TIME_ON) as TIME_ON,sum(PER) as PER
    FROM data_per
    """

    # Vérifier si R a des éléments pour ajuster la condition WHERE
    if R:
        r_filter = ', '.join(map(str, R))
        query += f"WHERE ROUND IN ({r_filter}) "
    else:
        query += "WHERE 1=1 "  # condition toujours vraie si R est vide


    query += "GROUP BY CODETEAM,PLAYER_ID"
    # Exécution de la requête
    df_result = pd.read_sql_query(query, conn)
    conn.close()
    
    #df_result['PER_MOYEN'] = (df_result["PER"] / df_result["PER"].max()) / (df_result["TIME_ON"] / df_result["TIME_ON"].max())**0.50
    df_result['IRS'] = (df_result["PER"]) / (df_result["TIME_ON"])**(0.5)

    # # Calculer la moyenne et l'écart type de la colonne actuelle
    moyenne_actuelle = df_result['IRS'].mean()
    ecart_type_actuel = df_result['IRS'].std()

    # Transformation des valeurs pour ajuster la moyenne à 75 et l'écart type à 10
    df_result['NOTE'] = 75 + 5 * ((df_result['IRS'] - moyenne_actuelle) / ecart_type_actuel)
    df_result['NOTE'] = np.clip(df_result['NOTE'], 50, 100)
    moyenne_actuelle = df_result['NOTE'].mean()
    ecart_type_actuel = df_result['NOTE'].std()
    df_result['NOTE'] = (75 + 5 * ((df_result['NOTE'] - moyenne_actuelle) / ecart_type_actuel)).round(1)
    df_result['NOTE'] = np.clip(df_result['NOTE'], 50, 100)
    df_result = df_result.sort_values(by = ["NOTE","PER"], ascending=[False,False]).reset_index(drop=True)
    df_result.insert(0,"RANK",range(1,len(df_result)+1))
    df_result["PER"] = df_result["PER"].round(2)
    df_result["IRS"] = df_result["IRS"].round(2)
    df_result["TIME_ON"] = df_result["TIME_ON"].round(2)
    df_result["NUMBER"] = df_result["NUMBER"].astype(int)
    #df_result = df_result.drop(["PER_MOYEN"],axis=1)

    if CODETEAM :
        df_result = df_result[df_result["CODETEAM"].isin(CODETEAM)].reset_index(drop=True)
    
    if len(R) == 1 :
        df_result = df_result.drop(columns=["NB_GAME"])

    return df_result


def calcul_per2(data_dir,season,competition) :
    competition_pbp = pd.read_csv(os.path.join(data_dir, f'{competition}_pbp_{season}.csv'))
    gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))
    competition_pbp = competition_pbp.merge(gs[['Gamecode', 'local.club.code', 'road.club.code','local.score','road.score']], 
                                  on='Gamecode', how='left')

    # Creating the 'opponent' column based on conditions
    competition_pbp['OPPONENT'] = competition_pbp.apply(
        lambda row: row['road.club.code'] if row['CODETEAM'] == row['local.club.code'] else row['local.club.code'], axis=1
    )

    competition_pbp['HOME'] = competition_pbp.apply(
        lambda row: 1 if row['CODETEAM'] == row['local.club.code'] else 0, axis=1
    )

    competition_pbp['WIN'] = competition_pbp.apply(lambda x: 1 if (x['local.score'] > x['road.score'] and x['HOME'] == 1) or
                     (x['local.score'] < x['road.score'] and x['HOME'] == 0) else 0, axis=1)




    competition_pbp = competition_pbp.drop(columns=['local.club.code', 'road.club.code','local.score','road.score'])

    # Dictionnaire des scores pour chaque PLAYTYPE
    playtype_scores = {
        'D': 0.75,
        '2FGA': -0.8,
        '2FGM': 1.5,
        'AS': 0.8,
        'RV': 0.5,
        'CM': -0.5,
        '3FGA': -0.6,
        'FTM': 0.75,
        '3FGM': 2.25,
        'O': 1.25,
        'TO': -1.33,
        'ST': 1.33,
        'FTA': -0.7,
        'FV': 0.5,
        'CMU': -1.5,
        'CMD': -1.25,
        'CMTI': -1.25,
        'CMT': -1.25,
        'OF': -1.33
    }

    if competition == "euroleague" : 
        competition_pbp.loc[:,"ROUND"] = (competition_pbp["Gamecode"] - 1) // 9 + 1
        competition_pbp = competition_pbp[competition_pbp["ROUND"]<= 34].reset_index(drop=True)
    else :
        competition_pbp.loc[:,"ROUND"] = (competition_pbp["Gamecode"] - 1) // 10 + 1
        competition_pbp = competition_pbp[competition_pbp["ROUND"]<= 18].reset_index(drop=True)

    # Filtrer le DataFrame pour garder seulement les PLAYTYPE présents dans playtype_scores
    #competition_pbp = competition_pbp[competition_pbp["PLAYER"]!=""].reset_index(drop=True)
    df_unique = competition_pbp[["ROUND", "CODETEAM", "OPPONENT", "HOME", "WIN", "DORSAL", "PLAYER","PLAYER_ID"]].drop_duplicates()
    df_unique = df_unique[~df_unique["PLAYER"].isna()]
    df_filtered = competition_pbp[competition_pbp['PLAYTYPE'].isin(playtype_scores.keys())]

    df_filtered = pd.merge(df_unique, df_filtered, 
                        on=["ROUND", "CODETEAM", "OPPONENT", "HOME", "WIN", "DORSAL", "PLAYER","PLAYER_ID"], 
                        how='left')
    df_filtered= df_filtered.fillna(0)
    # Grouper par 'ROUND', 'CODETEAM', 'PLAYER_ID', 'DORSAL' et compter les occurrences de chaque 'PLAYTYPE'
    grouped_df = df_filtered.groupby(['ROUND', 'CODETEAM','OPPONENT','HOME','WIN', 'PLAYER_ID','PLAYER', 'DORSAL', 'PLAYTYPE']).size().reset_index(name='count')

    # Pivot pour obtenir chaque modalité de 'PLAYTYPE' en colonne
    pivot_df = grouped_df.pivot_table(
        index=['ROUND', 'CODETEAM','OPPONENT','HOME','WIN', 'PLAYER_ID','PLAYER', 'DORSAL'],
        columns='PLAYTYPE',
        values='count',
        fill_value=0
    ).reset_index()
    playtype_columns = pivot_df.columns.difference(['ROUND', 'CODETEAM','OPPONENT', 'PLAYER_ID', 'PLAYER'])
    pivot_df[playtype_columns] = pivot_df[playtype_columns].astype(int)

    # Calcul de PER_V0 en multipliant chaque PLAYTYPE par son score et en faisant la somme
    pivot_df['PER_V0'] = sum(
        pivot_df[playtype] * score for playtype, score in playtype_scores.items() if playtype in pivot_df.columns
    )
    pivot_df = pivot_df.rename(columns={'DORSAL': 'NUMBER'})

    res_pm = pd.DataFrame(columns=[])

    for r in pivot_df["ROUND"].unique():
        res_ = analyse_io(data_dir = data_dir,
                    competition = competition,
                    season = season,
                    num_players = 1,
                    R = [r],
                    CODETEAM = [])
        res_["ROUND"] = r
        res_pm = pd.concat([res_pm,res_])


    res_pm["OFF_PER_0"] = (res_pm["TIME_ON"] * res_pm["OFF_ON_10"] / 10) * 0.05
    res_pm["DEF_PER_0"] = (res_pm["TIME_ON"] * res_pm["DEF_ON_10"] / 10) * 0.1
    res_pm = res_pm.rename(columns={'P1': 'PLAYER'})
    res_pm = res_pm.rename(columns={'ID1': 'PLAYER_ID'})
    res_pm = res_pm.rename(columns={'TEAM': 'CODETEAM'})

    res_pm2 = res_pm[["CODETEAM","PLAYER","PLAYER_ID","ROUND","TIME_ON","OFF_PER_0","DEF_PER_0","PM_ON"]]


    df_result = pd.merge(pivot_df, res_pm2, 
                        on=['CODETEAM', 'PLAYER_ID',"ROUND","PLAYER"], 
                        how='left')

    df_result = df_result.rename(columns={'CODETEAM': 'TEAM'})

    df_result["PER"] = df_result["PER_V0"] + df_result["OFF_PER_0"] - df_result["DEF_PER_0"]
    df_result["2_T"] = df_result["2FGA"] + df_result["2FGM"]
    df_result["3_T"] = df_result["3FGA"] + df_result["3FGM"]
    df_result["1_T"] = df_result["FTA"] + df_result["FTM"]
    df_result["2_R"] = df_result["2FGM"]
    df_result["3_R"] = df_result["3FGM"]
    df_result["1_R"] = df_result["FTM"]
    df_result["TO"] = df_result["TO"] + df_result["OF"]
    df_result["CF"] = df_result["CM"]
    df_result["NCF"] = df_result.filter(items=["CMU", "CMD", "CMTI", "CMT"]).sum(axis=1)
    df_result["OR"] = df_result["O"]
    df_result["DR"] = df_result["D"]
    df_result["TR"] = df_result["OR"] + df_result["DR"]
    df_result["FP"] = df_result["RV"]
    df_result["CO"] = df_result["FV"]
    df_result["PTS"] = df_result["FTM"]*1 + df_result["2FGM"]*2 + df_result["3FGM"]*3
    df_result["PER"] = (df_result["PER"]).round(1)

    #df_result.to_csv(os.path.join(data_dir, f'{competition}_PER_{season}_round.csv'),index = False)

    return df_result[['ROUND', 'TEAM','OPPONENT','HOME','WIN',"NUMBER","PLAYER","TIME_ON","PER",
                      "PM_ON","PTS","DR","OR","TR","AS","ST","CO","1_R",
                      "1_T","2_R","2_T","3_R","3_T","TO","FP","CF","NCF"]]

def calcul_per(data_dir,season,competition) : 
    competition_pbp = pd.read_csv(os.path.join(data_dir, f'{competition}_pbp_{season}.csv'))

    if competition == "euroleague" : 
        competition_pbp.loc[:,"ROUND"] = (competition_pbp["Gamecode"] - 1) // 9 + 1
        competition_pbp = competition_pbp[competition_pbp["ROUND"]<= 34].reset_index(drop=True)
    else :
        competition_pbp.loc[:,"ROUND"] = (competition_pbp["Gamecode"] - 1) // 10 + 1
        competition_pbp = competition_pbp[competition_pbp["ROUND"]<= 18].reset_index(drop=True)


    coefficients = {
        'D': 0.75,
        '2FGA': -0.8,
        '2FGM': 1.5,
        'AS': 0.8,
        'RV': 0.5,
        'CM': -0.5,
        '3FGA': -0.6,
        'FTM': 0.75,
        '3FGM': 2.25,
        'O': 1.25,
        'TO': -1.5,
        'ST': 1.5,
        'FTA': -0.7,
        'FV': 0.5,
        'CMU': -1.5,
        'CMD': -1.5,
        'CMTI': -1.5,
        'CMT': -1.5,
        'OF': -1.5
    }

    # Création de la nouvelle colonne 'Nouveau_Coeff' en utilisant map et en attribuant 0 par défaut si non trouvé
    competition_pbp['EVAL'] = competition_pbp['PLAYTYPE'].map(coefficients).fillna(0)
    competition_pbp = competition_pbp[~competition_pbp["PLAYER"].isna()]
    # Calcul de la somme de EVAL pour chaque joueur
    somme_eval_par_joueur = competition_pbp.groupby(['CODETEAM','PLAYER_ID','DORSAL','ROUND'])['EVAL'].sum().reset_index()

    # Renommage des colonnes
    somme_eval_par_joueur.columns = ['CODETEAM','PLAYER_ID','NUMBER','ROUND', 'PER_V0']


    res_pm = pd.DataFrame(columns=[])
    
    for r in somme_eval_par_joueur["ROUND"].unique():
        res_ = analyse_io(data_dir = data_dir,
                    competition = competition,
                    season = season,
                    num_players = 1,
                    R = [r],
                    CODETEAM = [])
        res_["ROUND"] = r
        res_pm = pd.concat([res_pm,res_])
    
    res_pm["OFF_PER_0"] = (res_pm["TIME_ON"] * res_pm["OFF_ON_10"] / 10) * 0.05
    res_pm["DEF_PER_0"] = (res_pm["TIME_ON"] * res_pm["DEF_ON_10"] / 10) * 0.1
    res_pm = res_pm.rename(columns={'P1': 'PLAYER'})
    res_pm = res_pm.rename(columns={'ID1': 'PLAYER_ID'})
    res_pm = res_pm.rename(columns={'TEAM': 'CODETEAM'})

    res_pm2 = res_pm[["CODETEAM","PLAYER","PLAYER_ID","ROUND","TIME_ON","OFF_PER_0","DEF_PER_0"]]

    df_result = pd.merge(somme_eval_par_joueur, res_pm2, 
                     on=['CODETEAM', 'PLAYER_ID',"ROUND"], 
                     how='left')
    
    df_result["PER"] = df_result["PER_V0"] + df_result["OFF_PER_0"] - df_result["DEF_PER_0"]
    
    df_result = df_result[['CODETEAM','NUMBER', 'PLAYER', 'PLAYER_ID',"ROUND","TIME_ON","PER"]]
    

    df_result.to_csv(os.path.join(data_dir, f'{competition}_PER_{season}_round.csv'),index = False)

def calcul_per_global(data_dir,competition,season) :
    df = analyse_per(data_dir,competition,season,R = [],CODETEAM = [])
    df.to_csv(os.path.join(data_dir, f'{competition}_PER_{season}.csv'),index = False)


def recup_io(data_dir,season,competition,nb_io) :

    competition_pbp = pd.read_csv(os.path.join(data_dir, f'{competition}_pbp_{season}.csv'))
    competition_gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))


    game_code_list = list(set(competition_gs["Gamecode"]))
    io_total = pd.DataFrame(columns=['GAMECODE', 'CODETEAM', f'{nb_io}-io', 'TIME', 'DELTA_SCORE',"OFF","DEF"])
    for gc in game_code_list :
        local_io_summary,road_io_summary = global_io_recuperation_data_game(gs_data = competition_gs,pbp_data = competition_pbp,gamecode = gc ,nb_io = nb_io)
        io_total = pd.concat([io_total,local_io_summary,road_io_summary]).reset_index(drop = True)


    io_total[[f'joueur_{i+1}' for i in range(nb_io)]] = pd.DataFrame(io_total[f'{nb_io}-io'].tolist(), index=io_total.index)
    io_total = io_total[["GAMECODE","CODETEAM"] +[f'joueur_{i+1}' for i in range(nb_io)] + ["TIME","DELTA_SCORE","OFF","DEF"]]

    io_total.to_csv(os.path.join(data_dir, f'{competition}_{nb_io}_io_{season}.csv'),index = False)

def process_game_data(data_dir, competition, season):
    # Charger les données
    competition_pbp = pd.read_csv(os.path.join(data_dir, f'{competition}_pbp_{season}.csv'))
    competition_gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))

    game_code_list = competition_gs["Gamecode"].unique().tolist()

    try:
        five_summary = pd.read_csv(os.path.join(data_dir, f'{competition}_five_evol_{season}.csv'))
        unique_game_codes = list(set(competition_gs["Gamecode"].unique()) - set(five_summary["Gamecode"].unique()))
    except FileNotFoundError:
        five_summary = pd.DataFrame(columns=["Gamecode", "CODETEAM","TIME", "DELTA_SCORE", "OFF", "DEF","P1","P2","P3","P4","P5"])
        unique_game_codes = game_code_list

    for gc in unique_game_codes:
        df_gs = competition_gs[competition_gs["Gamecode"] == gc]  # Correction ici, il faut utiliser gc
        df_pbp = competition_pbp[competition_pbp["Gamecode"] == gc]  # Correction ici, il faut utiliser gc
        df_pbp, code, score, game_duration = debroussallage_pbp_data(df_gs, df_pbp)

        evol_local = evolution_five_score_one_team(df=df_pbp, teamcode=code[0], score_final=score, game_duration=game_duration, local=True)
        evol_local.insert(0, "Gamecode", gc)

        evol_road = evolution_five_score_one_team(df=df_pbp, teamcode=code[1], score_final=score, game_duration=game_duration, local=False)
        evol_road.insert(0, "Gamecode", gc)
        evol_local[['P1', 'P2', 'P3', 'P4', 'P5']] = pd.DataFrame(evol_local['FIVE'].tolist(), index=evol_local.index)
        evol_road[['P1', 'P2', 'P3', 'P4', 'P5']] = pd.DataFrame(evol_road['FIVE'].tolist(), index=evol_road.index)
        evol_local.drop('FIVE', axis=1, inplace=True)
        evol_road.drop('FIVE', axis=1, inplace=True)
        evol_local['TIME'] = (evol_local['TIME'].dt.total_seconds() / 60).round(3)
        evol_road['TIME'] = (evol_road['TIME'].dt.total_seconds() / 60).round(3)


        five_summary = pd.concat([five_summary, evol_local, evol_road]).reset_index(drop=True)


    

    five_summary.to_csv(os.path.join(data_dir, f'{competition}_five_evol_{season}.csv'), index=False)


def recup_idplayers(data_dir,season,competition) : 
    _pbp = pd.read_csv(os.path.join(data_dir, f'{competition}_pbp_{season}.csv'))


    mapping_ = _pbp.dropna(subset=['PLAYER'])[['CODETEAM', 'PLAYER_ID','PLAYER']].drop_duplicates(subset=['CODETEAM', 'PLAYER_ID'], keep='first')[['CODETEAM', 'PLAYER_ID','PLAYER']]


    mapping_.to_csv(os.path.join(data_dir, f'{competition}_idplayers_{season}.csv'),index = False)

def recup_idteam(data_dir,season,competition) : 

    if competition == "euroleague" :

        ts = TeamStats("E")
    else :
        ts = TeamStats("U")

    df = ts.get_team_stats_single_season(
            endpoint="traditional", season=2024, phase_type_code="RS", statistic_mode="PerGame"
        )
    df = df[['team.code', 'team.name']].drop_duplicates()
    df.columns = ["CODETEAM","NAMETEAM"]

    df.to_csv(os.path.join(data_dir, f'{competition}_idteams_{season}.csv'),index = False)

def df_images_players(competition,season,round_,data_dir) :
    try :
        images_df = pd.read_csv(os.path.join(data_dir, f"{competition}_{season}_df_images_players.csv"))
    except :
        images_df = pd.DataFrame(columns=["TEAM","NAME", "PLAYER_ID", "NAT", "LINK"])

    if competition == "eurocup":
        gs_ = GameStats(competition="U")
        first_gamecode=10*(round_- 1) + 1 
        last_gamecode=10*(round_)
    else:
        gs_ = GameStats(competition="E")
        first_gamecode=9*(round_- 1) + 1
        last_gamecode=9*(round_)    

    for gc in range(first_gamecode,last_gamecode + 1) :
        try : 
            ggs = gs_.get_game_stats(season=season,game_code = gc)
            ggr = gs_.get_game_report(season=season,game_code = gc)

            for i in range(len(ggs["local.players"].to_list()[0])):
                try :
                    images = ggs["local.players"].to_list()[0][i]['player']['images']['action']
                    player_id = "P" + ggs["local.players"].to_list()[0][i]['player']['person']['code']
                    namep = ggs["local.players"].to_list()[0][i]['player']['person']['name']
                    nat = ggs["local.players"].to_list()[0][i]['player']['person']['country']['code']
                    teamcode = ggr['local.club.code'].to_list()[0]
                    images_df = pd.concat([images_df, pd.DataFrame([[teamcode, namep,player_id, nat, images]], columns=images_df.columns)], ignore_index=True)
                except :
                    print(namep)
            for i in range(len(ggs["road.players"].to_list()[0])):
                try : 
                    images = ggs["road.players"].to_list()[0][i]['player']['images']['action']
                    player_id = "P" + ggs["road.players"].to_list()[0][i]['player']['person']['code']
                    namep = ggs["road.players"].to_list()[0][i]['player']['person']['name']

                    nat = ggs["road.players"].to_list()[0][i]['player']['person']['country']['code']
                    teamcode = ggr['road.club.code'].to_list()[0]
                    images_df = pd.concat([images_df, pd.DataFrame([[teamcode, namep,player_id, nat, images]], columns=images_df.columns)], ignore_index=True)
                except :
                    print(namep)

        except :
            pass
    images_df = images_df.drop_duplicates().reset_index(drop = True)

    images_df.to_csv(os.path.join(data_dir, f"{competition}_{season}_df_images_players.csv"),index = False)

def recuperation_pbp(competition, season, data_dir, round_=None):
    # Chemin du fichier CSV
    file_path = os.path.join(data_dir, f'{competition}_pbp_{season}.csv')
    
    # Déterminer la compétition
    if competition == "eurocup":
        pbp_ = PlayByPlay(competition="U")
    else:
        pbp_ = PlayByPlay(competition="E")
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(file_path):
        # Charger le fichier existant
        _pbp = pd.read_csv(file_path)
        
        # Si first_gamecode et last_gamecode sont fournis, itérer sur la plage de gamecodes
        if round_ is not None :
            if competition == "eurocup":
                first_gamecode=10*(round_- 1) + 1 
                last_gamecode=10*(round_)
            else :
                first_gamecode=9*(round_- 1) + 1
                last_gamecode=9*(round_) 
            
            for gc in range(first_gamecode, last_gamecode + 1):
                # Vérifier si _pbp contient déjà des lignes pour ce gamecode
                if not (_pbp['Gamecode'] == gc).any():
                    # Récupérer les données pour le gamecode spécifique
                    try : 
                        sub = pbp_.get_game_play_by_play_data(season=season, gamecode=gc)
                        if "End Game" in sub["PLAYINFO"].values:
                            # Concaténer les nouvelles données avec _pbp
                            _pbp = pd.concat([_pbp, sub], ignore_index=True)
                    except :
                        pass
                else :
                    # Récupérer les données pour le gamecode spécifique
                    try : 
                        _pbp = _pbp[_pbp["Gamecode"]!=gc]
                        sub = pbp_.get_game_play_by_play_data(season=season, gamecode=gc)
                        # Concaténer les nouvelles données avec _pbp
                        _pbp = pd.concat([_pbp, sub], ignore_index=True)
                    except :
                        pass
        
        if competition == "euroleague" : 
            _pbp.loc[:,"Round"] = (_pbp["Gamecode"] - 1) // 9 + 1
        else :
             _pbp.loc[:,"Round"] = (_pbp["Gamecode"] - 1) // 10 + 1

        # Sauvegarder les données mises à jour dans le fichier CSV
        _pbp.to_csv(file_path, index=False)

    else:
        # Récupérer les données en utilisant PlayByPlay
        _pbp = pbp_.get_game_play_by_play_data_single_season(season=season)

        if competition == "euroleague" : 
            _pbp.loc[:,"Round"] =(_pbp["Gamecode"] - 1) // 9 + 1
        else :
             _pbp.loc[:,"Round"] = (_pbp["Gamecode"] - 1) // 10 + 1
        # Sauvegarder les données dans un fichier CSV
        _pbp.to_csv(file_path, index=False)

def recuperation_players_photo2(competition,season,image_dir,data_dir) :
    folder_link = os.path.join(image_dir, f'{competition}_{season}_players')

    if not os.path.exists(folder_link):
        os.makedirs(folder_link)
        print(f"Folder '{folder_link}' created.")
    else:
        print(f"Folder '{folder_link}' already exists.")

    pdf = pd.read_csv(os.path.join(data_dir, f"{competition}_{season}_df_images_players.csv"))
    def image_not_exists(row):
        player_image_path = os.path.join(folder_link, f"{row['TEAM']}_{row['PLAYER_ID']}.png")
        return not os.path.isfile(player_image_path)   
     
    df_filtered = pdf[pdf.apply(image_not_exists, axis=1)]

    for index, row in df_filtered.iterrows():

        image_player_link = os.path.join(image_dir, rf'{competition}_{season}_players\{row["TEAM"]}_{row["PLAYER_ID"]}.png')
        url_player = row["LINK"]
        response = requests.get(url_player)
        # Vérifier que la requête est réussie (code 200)
        if response.status_code == 200:
            # Enregistrer l'image sur le disque
            with open(image_player_link, "wb") as file:
                file.write(response.content)
            print("Image téléchargée avec succès.")
        else:
            print("Échec du téléchargement. Code d'erreur :", response.status_code)


def recuperation_players_photo(competition,season,image_dir) :
    folder_link = os.path.join(image_dir, f'{competition}_{season}_players')

    if not os.path.exists(folder_link):
        os.makedirs(folder_link)
        print(f"Folder '{folder_link}' created.")
    else:
        print(f"Folder '{folder_link}' already exists.")

    
    if competition == "euroleague" :
        competition_code = "E"
    else :
        competition_code = "U"

    ps = PlayerStats(competition_code)

    df = ps.get_player_stats_single_season(endpoint = "traditional",season =  season ,phase_type_code = "RS",statistic_mode = "Accumulated")

    df_utile = df[["player.code","player.imageUrl","player.team.code","player.team.imageUrl"]]

    df_utile.columns = ["ID_PLAYERS","URL_PLAYERS","ID_TEAM","URL_TEAM"]
    df_utile = df_utile[df_utile["URL_PLAYERS"].notna()]


    # Filter out rows where the player's image already exists in the folder
    def image_not_exists(row):
        player_image_path = os.path.join(folder_link, f"P{row['ID_PLAYERS']}.png")
        return not os.path.isfile(player_image_path)
    
    # Apply the filter
    df_filtered = df_utile[df_utile.apply(image_not_exists, axis=1)]

    for index, row in df_filtered.iterrows():

        image_player_link = os.path.join(image_dir, f'{competition}_{season}_players\P{row["ID_PLAYERS"]}.png')
        url_player = row["URL_PLAYERS"]
        response = requests.get(url_player)
        # Vérifier que la requête est réussie (code 200)
        if response.status_code == 200:
            # Enregistrer l'image sur le disque
            with open(image_player_link, "wb") as file:
                file.write(response.content)
            print("Image téléchargée avec succès.")
        else:
            print("Échec du téléchargement. Code d'erreur :", response.status_code)

def recuperation_gs(competition, season, data_dir, round_ = None ):
    # Chemin du fichier CSV
    file_path = os.path.join(data_dir, f'{competition}_gs_{season}.csv')
    
    # Déterminer la compétition
    if competition == "eurocup":
        gs_ = GameStats(competition="U")
    else:
        gs_ = GameStats(competition="E")
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(file_path):
        # Charger le fichier existant
        _gs = pd.read_csv(file_path)
        
        # Si first_gamecode et last_gamecode sont fournis, itérer sur la plage de gamecodes
        if round_ is not None :
            if competition == "eurocup":
                first_gamecode=10*(round_- 1) + 1 
                last_gamecode=10*(round_)
            else :
                first_gamecode=9*(round_- 1) + 1
                last_gamecode=9*(round_)


            for gc in range(first_gamecode, last_gamecode + 1):
                # Vérifier si _gs contient déjà des lignes pour ce gamecode
                if not (_gs['Gamecode'] == gc).any():
                    # Récupérer les données pour le gamecode spécifique
                    try : 
                        sub = gs_.get_game_report(season=season, game_code=gc)
                        if sub["local.score"].to_list()[0] == 0 :

                            pass
                        else :
                        # Concaténer les nouvelles données avec _gs
                            _gs = pd.concat([_gs, sub], ignore_index=True)
                    except :
                        pass
        
        # Sauvegarder les données mises à jour dans le fichier CSV
        _gs.to_csv(file_path, index=False)
        
    else:
        # Récupérer les données en utilisant GameStats pour toute la saison
        _gs = gs_.get_game_reports_single_season(season=season)

        # Sauvegarder les données dans un fichier CSV
        _gs.to_csv(file_path, index=False)

def debroussallage_pbp_data(df_gs,df_pbp): 
    """
    Traite les données de jeu pour un match de basket afin de préparer 
    un DataFrame d'événements de jeu et d'identifier les joueurs titulaires.

    Cette fonction prend en entrée deux DataFrames : `df_gs`, contenant les 
    informations générales sur le match (y compris les scores et les codes d'équipe), 
    et `df_pbp`, qui contient les événements de jeu (play-by-play). Elle effectue 
    plusieurs transformations et nettoyages sur les données pour faciliter 
    l'analyse des performances des joueurs et des équipes.

    Paramètres
    ----------
    df_gs : pandas.DataFrame
        Un DataFrame contenant les informations générales sur le match. 
        Il doit inclure les colonnes suivantes :
        - "local.club.code" : Code de l'équipe locale.
        - "road.club.code" : Code de l'équipe visiteuse.
        - "local.score" : Score de l'équipe locale.
        - "road.score" : Score de l'équipe visiteuse.

    df_pbp : pandas.DataFrame
        Un DataFrame contenant les événements de jeu. 
        Il doit inclure les colonnes suivantes :
        - "CODETEAM" : Identifiant de l'équipe.
        - "PLAYER_ID" : Identifiant du joueur.
        - "PLAYTYPE" : Type d'événement (par exemple, "IN", "OUT").
        - "MARKERTIME" : Temps marqué de l'événement.
        - "POINTS_A" : Points de l'équipe A.
        - "POINTS_B" : Points de l'équipe B.
        - "PERIOD" : Période du match.

    Retourne
    -------
    df_pbp : pandas.DataFrame
        Un DataFrame filtré et nettoyé contenant les événements de jeu,
        incluant les joueurs titulaires avec des valeurs de points 
        initialisées à zéro et un temps marqué à zéro.

    code : list
        Une liste contenant les codes des équipes locales et visiteuses.

    score : list
        Une liste contenant les scores des équipes locales et visiteuses.

    game_duration : pandas.Timedelta
        Durée totale du match calculée à partir des périodes et des minutes.
    """
    df_pbp.loc[:, "NUMBEROFPLAY"] = range(len(df_pbp))


    code = [df_gs["local.club.code"].to_list()[0],df_gs["road.club.code"].to_list()[0]]
    score = [df_gs["local.score"].to_list()[0],df_gs["road.score"].to_list()[0]]



    df_pbp['PERIOD'] = df_pbp['PERIOD'].apply(lambda x: x if x in [1, 2, 3, 4] else 4 + 0.5 * (x - 4)).astype(float)

    game_duration = pd.to_timedelta(max(df_pbp['PERIOD']) *10,unit = "minutes")
    
    df_pbp = df_pbp[
        df_pbp['POINTS_A'].notna() |
        df_pbp['POINTS_B'].notna() |
        (df_pbp['PLAYINFO'] == 'Out') |
        (df_pbp['PLAYINFO'] == 'In')
    ].reset_index(drop = True)

    df_pbp['MARKERTIME'] = pd.to_timedelta('00:' + df_pbp['MARKERTIME'])

    df_pbp['MARKERTIME'] = pd.to_timedelta(10,unit = "minutes") - df_pbp['MARKERTIME'] + pd.to_timedelta(10 * (df_pbp['PERIOD']-1), unit='minutes')

    
    ##### POINTS_A
    # Remplacer les NaN au début de la colonne par 0
    first_valid_index = df_pbp['POINTS_A'].first_valid_index()  # Trouver le premier index valide
    if first_valid_index is not None:  # S'il y a au moins un valide
        df_pbp.loc[:first_valid_index-1, 'POINTS_A'] = df_pbp.loc[:first_valid_index-1, 'POINTS_A'].fillna(0)

    # Remplacer les autres NaN par la dernière valeur non NaN
    df_pbp['POINTS_A'] = df_pbp['POINTS_A'].ffill().astype(int)


    ##### POINTS_B
    # Remplacer les NaN au début de la colonne par 0
    first_valid_index = df_pbp['POINTS_B'].first_valid_index()  # Trouver le premier index valide
    if first_valid_index is not None:  # S'il y a au moins un valide
        df_pbp.loc[:first_valid_index-1, 'POINTS_B'] = df_pbp.loc[:first_valid_index-1, 'POINTS_B'].fillna(0)

    # Remplacer les autres NaN par la dernière valeur non NaN
    df_pbp['POINTS_B'] = df_pbp['POINTS_B'].ffill().astype(int)

    # Extraire tous les couples uniques de PLAYER_ID et CODETEAM
    unique_players = df_pbp[['PLAYER_ID', 'CODETEAM']].drop_duplicates()



    # Filtrer les événements "IN" et "OUT"
    in_events = df_pbp[df_pbp['PLAYTYPE'] == 'IN']
    out_events = df_pbp[df_pbp['PLAYTYPE'] == 'OUT']

    # Trouver le premier IN pour chaque joueur/équipe
    first_in = in_events.groupby(['PLAYER_ID', 'CODETEAM']).first().reset_index()
    first_in = first_in[['PLAYER_ID', 'CODETEAM', 'NUMBEROFPLAY']].rename(columns={'NUMBEROFPLAY': 'FIRST_IN'})

    # Trouver le premier OUT pour chaque joueur/équipe
    first_out = out_events.groupby(['PLAYER_ID', 'CODETEAM']).first().reset_index()
    first_out = first_out[['PLAYER_ID', 'CODETEAM', 'NUMBEROFPLAY']].rename(columns={'NUMBEROFPLAY': 'FIRST_OUT'})

    # Fusionner les premiers événements IN et OUT avec tous les joueurs
    result = pd.merge(unique_players, first_in, on=['PLAYER_ID', 'CODETEAM'], how='left')
    result = pd.merge(result, first_out, on=['PLAYER_ID', 'CODETEAM'], how='left')


    # Filtrer le DataFrame
    df_pbp = df_pbp[
        (df_pbp['PLAYINFO'] == 'Out') |
        (df_pbp['PLAYINFO'] == 'In')
    ].reset_index(drop = True)

    df_pbp = df_pbp[["CODETEAM","PLAYER_ID","PLAYTYPE","MARKERTIME","POINTS_A","POINTS_B"]]



    result["STARTER"] = (result['FIRST_IN'] > result['FIRST_OUT']) | (result['FIRST_IN'].isna() & result['FIRST_OUT'].isna())| (result['FIRST_IN'].isna())
    starter = result[result["STARTER"]].reset_index(drop=True)



    
    # Créer le DataFrame des titulaires
    df_pbp_start = pd.DataFrame({
        "CODETEAM": starter["CODETEAM"].to_list(),
        "PLAYER_ID": starter["PLAYER_ID"].to_list(),
        "PLAYTYPE": ["IN"] * len(starter),
        "MARKERTIME": [pd.Timedelta('0 days 00:00:00')] * len(starter),
        "POINTS_A": [0] * len(starter),
        "POINTS_B": [0] * len(starter)
    })


    df_pbp = pd.concat([df_pbp_start, df_pbp], ignore_index=True)

    return df_pbp,code,score,game_duration

def io_summary(evol,nbg) :
    """
    Résume les combinaisons de joueurs ("in-out") sur une période donnée en agrégeant 
    le temps et le score associés pour chaque combinaison unique.

    Cette fonction prend un DataFrame `evol` contenant des informations sur les équipes et 
    les joueurs, et génère un résumé des combinaisons de joueurs sur le terrain. Chaque combinaison 
    est triée dans l'ordre croissant et agrégée par le temps total joué et le score associé 
    à cette combinaison.

    Paramètres
    ----------
    evol : pandas.DataFrame
        Un DataFrame contenant les informations sur les équipes et les joueurs. 
        Il doit inclure au moins les colonnes suivantes :
        - "CODETEAM" : Identifiant de l'équipe.
        - "FIVE" : Tuple contenant les joueurs actifs à un instant donné.
        - "TIME" : Durée pendant laquelle cette combinaison de joueurs était sur le terrain.
        - "DELTA_SCORE" : Score associé à cette combinaison pendant cette période.

    nbg : int
        Le nombre de joueurs pour lesquels on veut calculer les combinaisons ("in-out"). 
        Par exemple, pour des trios, `nbg` sera égal à 3.

    Retourne
    -------
    df_io : pandas.DataFrame
        Un DataFrame contenant les combinaisons triées de joueurs ("in-out") pour chaque équipe, 
        avec le temps total et le score associé à chaque combinaison. Les colonnes sont :
        - "CODETEAM" : Identifiant de l'équipe.
        - f"{nbg}-io" : Tuple contenant la combinaison de `nbg` joueurs triés.
        - "TIME" : Temps total joué par cette combinaison.
        - "DELTA_SCORE" : Score total accumulé par cette combinaison.
    """
    df_io = pd.DataFrame(columns=["CODETEAM",f"{nbg}-io","TIME","DELTA_SCORE","OFF","DEF"])
    for index, row in evol.iterrows():
        # Boucler sur tous les trios uniques
        for io in itertools.combinations(row["FIVE"], nbg):
            io = tuple(sorted(io))
                # Vérifier si le io existe déjà dans io_df
            if io in df_io[f"{nbg}-io"].to_list():
                # MAJ de la ligne
                df_io.loc[df_io[f"{nbg}-io"] == io, "TIME"] += row["TIME"]
                df_io.loc[df_io[f"{nbg}-io"] == io, "DELTA_SCORE"] += row["DELTA_SCORE"]
                df_io.loc[df_io[f"{nbg}-io"] == io, "OFF"] += row["OFF"]
                df_io.loc[df_io[f"{nbg}-io"] == io, "DEF"] += row["DEF"]
            else:
                # Ajouter une nouvelle ligne
                new_io = pd.DataFrame({
                    "CODETEAM": row["CODETEAM"],
                    f"{nbg}-io": [io],
                    "TIME": row["TIME"],
                    "DELTA_SCORE": row["DELTA_SCORE"],
                    "OFF": row["OFF"],
                    "DEF": row["DEF"]
                })
                
                new_io = new_io.dropna(axis=1, how='all')
                df_io = pd.concat([df_io, new_io], ignore_index=True)

    df_io = df_io.sort_values(by="TIME", ascending=False).reset_index(drop=True)
    return df_io



def evolution_five_score_one_team(df: pd.DataFrame, teamcode: str, score_final: list, game_duration: pd.Timedelta, local: bool) -> pd.DataFrame:
    """
    Calcule l'évolution du score d'une équipe et des cinq joueurs présents.

    Parameters:
    df (pd.DataFrame): DataFrame contenant les données de jeu.
    teamcode (str): Code de l'équipe à analyser.
    score_final (list): Liste contenant le score final [POINTS_A, POINTS_B].
    game_duration (pd.Timedelta): Durée du jeu.
    local (bool): Indique si l'équipe est à domicile ou non.

    Returns:
    pd.DataFrame: DataFrame contenant les résultats calculés.
    """
    one_team_df = df[df["CODETEAM"] == teamcode].reset_index(drop=True)
    actual_team = set()  # Utiliser un set pour une gestion plus rapide
    historique_team = []

    for index, row in one_team_df.iterrows():
        player_id = row["PLAYER_ID"]
        if row["PLAYTYPE"] == "IN":
            actual_team.add(player_id)  # Ajouter le joueur
        elif row["PLAYTYPE"] == "OUT":
            actual_team.discard(player_id)  # Retirer le joueur si présent

        historique_team.append(sorted(actual_team))  # Append une liste triée des joueurs

    one_team_df["FIVE"] = historique_team

    # Filtrer pour garder seulement les lignes avec 5 éléments dans la colonne FIVE
    one_team_df = one_team_df[one_team_df["FIVE"].apply(lambda x: len(x) == 5)]

    # Garder une ligne par MARKERTIME avec l'index le plus élevé
    one_team_df = one_team_df.loc[one_team_df.groupby("MARKERTIME").tail(1).index].reset_index(drop=True)[["CODETEAM", "MARKERTIME", "POINTS_A", "POINTS_B", "FIVE"]]

    f = pd.DataFrame({
        "CODETEAM": [teamcode],
        "MARKERTIME": [game_duration],
        "POINTS_A": [score_final[0]],
        "POINTS_B": [score_final[1]],
        "FIVE": [sorted(actual_team)]
    })

    one_team_df = pd.concat([one_team_df, f], ignore_index=True).reset_index(drop=True)

    one_team_df['TIME'] = one_team_df['MARKERTIME'].shift(-1) - one_team_df['MARKERTIME']

    # Calculer DELTA_SCORE
    if local:
        one_team_df['DELTA_SCORE'] = (one_team_df['POINTS_A'].shift(-1) - one_team_df['POINTS_A']) - (one_team_df['POINTS_B'].shift(-1) - one_team_df['POINTS_B'])
        one_team_df['OFF'] = (one_team_df['POINTS_A'].shift(-1) - one_team_df['POINTS_A'])
        one_team_df['DEF'] = (one_team_df['POINTS_B'].shift(-1) - one_team_df['POINTS_B'])
    else:
        one_team_df['DELTA_SCORE'] = (one_team_df['POINTS_B'].shift(-1) - one_team_df['POINTS_B']) - (one_team_df['POINTS_A'].shift(-1) - one_team_df['POINTS_A'])
        one_team_df['OFF'] = (one_team_df['POINTS_B'].shift(-1) - one_team_df['POINTS_B'])
        one_team_df['DEF'] = (one_team_df['POINTS_A'].shift(-1) - one_team_df['POINTS_A'])

    one_team_df = one_team_df.drop(one_team_df.index[-1]).reset_index(drop=True)  # Suppression de la dernière ligne

    one_team_df['DELTA_SCORE'] = one_team_df['DELTA_SCORE'].astype(int)
    one_team_df['OFF'] = one_team_df['OFF'].astype(int)
    one_team_df['DEF'] = one_team_df['DEF'].astype(int)

    one_team_df["FIVE"] = one_team_df["FIVE"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    one_team_df['FIVE'] = one_team_df['FIVE'].apply(lambda x: tuple(sorted(x)))
    # Groupby avec somme pour TIME et DELTA_SCORE
    one_team_df = one_team_df.groupby(["CODETEAM", "FIVE"]).agg({
        "TIME": lambda x: x.sum(),
        "DELTA_SCORE": "sum",
        "OFF": "sum",
        "DEF": "sum",
    }).sort_values(by="TIME", ascending=False).reset_index()

    one_team_df['FIVE'] = one_team_df['FIVE'].apply(lambda x: tuple(sorted(x)))


    return one_team_df



def global_io_recuperation_data_game(gs_data,pbp_data,gamecode,nb_io) : 
    df_gs = gs_data[gs_data["Gamecode"]==gamecode].reset_index(drop = True)
    df_pbp = pbp_data[pbp_data["Gamecode"]==gamecode].reset_index(drop = True)

    
    df_pbp,code,score,game_duration = debroussallage_pbp_data(df_gs,df_pbp)
    evol_local = evolution_five_score_one_team(df = df_pbp, teamcode = code[0], score_final = score, game_duration = game_duration, local = True)
    evol_road = evolution_five_score_one_team(df = df_pbp, teamcode = code[1], score_final = score, game_duration = game_duration, local = False)

    local_io_summary = io_summary(evol = evol_local,nbg = nb_io)
    road_io_summary = io_summary(evol = evol_road,nbg = nb_io)

    local_io_summary.insert(0,"GAMECODE",gamecode)
    road_io_summary.insert(0,"GAMECODE",gamecode)

    return local_io_summary,road_io_summary

def analyse_io(data_dir,competition,season,num_players,R = [],CODETEAM = []) :
    conn = sqlite3.connect(':memory:')
    data_pm = pd.read_csv(os.path.join(data_dir, f"{competition}_io_{num_players}_{season}.csv"))


    if competition == "euroleague" :
        data_pm.insert(0,"R",(data_pm["Gamecode"] - 1 ) // 9 + 1 )
    else :
        data_pm.insert(0,"R",(data_pm["Gamecode"] - 1 ) // 10 + 1 )

    data_id = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

    data_pm.to_sql('data_pm', conn, index=False, if_exists='replace') 
    data_id.to_sql('data_id', conn, index=False, if_exists='replace')



    # Commencer à construire la requête SQL
    query = """
    SELECT t.CODETEAM as TEAM,
    """

    # Ajouter les jointures et les colonnes de joueurs dynamiquement
    joins = []
    for i in range(1, num_players + 1):
        query += f"t{i}.PLAYER as P{i},t{i}.PLAYER_ID as ID{i},\n"
        joins.append(f"LEFT JOIN data_id as t{i} on (t.CODETEAM = t{i}.CODETEAM) and (t.J{i} = t{i}.PLAYER_ID)")

    # Ajouter les agrégations
    query += """
    sum(t.TIME) as TIME_ON,
    sum(DELTA_SCORE) as PM_ON,
    ROUND(sum(OFF)/sum(TIME) *10,2) as OFF_ON_10,
    ROUND(sum(DEF)/sum(TIME) *10,2) as DEF_ON_10,

    ROUND(MAX(TEAM.TIME_TEAM) - sum(t.TIME),2) as TIME_OFF,
    MAX(TEAM.DELTA_SCORE_TEAM) - sum(DELTA_SCORE) as PM_OFF,
    ROUND((MAX(TEAM.OFF_TEAM) - sum(OFF)) / (MAX(TEAM.TIME_TEAM) - sum(t.TIME)) *10,2)  as OFF_OFF_10,
    ROUND((MAX(TEAM.DEF_TEAM) - sum(DEF)) / (MAX(TEAM.TIME_TEAM) - sum(t.TIME)) *10,2) as DEF_OFF_10,

    CAST(ROUND(MAX(TEAM.TIME_TEAM)) AS INTEGER) as TIME_TEAM,
    MAX(TEAM.DELTA_SCORE_TEAM) as PM_TEAM,
    ROUND(MAX(TEAM.OFF_TEAM)/MAX(TEAM.TIME_TEAM)*10,2) as OFF_TEAM_10,
    ROUND(MAX(TEAM.DEF_TEAM)/MAX(TEAM.TIME_TEAM)*10,2) as DEF_TEAM_10

    FROM data_pm as t
    """

    # Ajouter les jointures à la requête
    query += "\n".join(joins)

    # Ajouter le sous-requête pour les équipes
    query += f"""
    LEFT JOIN (SELECT CODETEAM,
    sum(TIME)/{math.comb(5, num_players)} as TIME_TEAM,
    sum(DELTA_SCORE)/{math.comb(5, num_players)} as DELTA_SCORE_TEAM,
    sum(OFF) /{math.comb(5, num_players)} as OFF_TEAM,
    sum(DEF) /{math.comb(5, num_players)} as DEF_TEAM

    from data_pm
    """

    # Vérifier si R a des éléments pour ajuster la condition WHERE
    if R:
        r_filter = ', '.join(map(str, R))
        query += f"WHERE R IN ({r_filter}) "
    else:
        query += "WHERE 1=1 "  # condition toujours vraie si R est vide

    query += f"""
    group by CODETEAM
    ORDER BY DELTA_SCORE_TEAM DESC) as TEAM on t.CODETEAM = TEAM.CODETEAM
    """

    # Terminer la requête avec les clauses GROUP BY et ORDER BY
    query += f"""
    WHERE 1=1
    """

    # Ajout conditionnel pour R
    if R:
        r_filter = ', '.join(map(str, R))
        query += f"AND t.R IN ({r_filter}) "

    # Ajout conditionnel pour CODETEAM
    if CODETEAM:
        code_filter = "', '".join(CODETEAM)  # Création de la chaîne pour la liste
        query += f"AND t.CODETEAM IN ('{code_filter}') "

    query += f"""
    GROUP BY t.CODETEAM, {', '.join([f't{i}.PLAYER' for i in range(1, num_players + 1)])}
    ORDER BY sum(DELTA_SCORE) DESC, sum(TIME)
    """

    # Exécution de la requête
    result = pd.read_sql_query(query, conn)
    conn.close()


    return result

def io_plus_minus_analysis(data_dir,season,competition,io,sortby = "PM",teamcode = "",round = "") :


    if sortby == "PM" :
        sortby = f"PM_IN" 
    data_io = pd.read_csv(os.path.join(data_dir, f'{competition}_{io}_io_{season}.csv'))
    data_id = pd.read_csv(os.path.join(data_dir, f'{competition}_idplayers_{season}.csv'))

    if teamcode != "" :
        data_io = data_io[data_io["CODETEAM"]== teamcode] 
        data_id = data_id[data_id["CODETEAM"]== teamcode]

    if round != "" :
        if competition == "euroleague" :
            data_io = data_io[(data_io["GAMECODE"]>= (9*round - 8 ))&(data_io["GAMECODE"]<= (9*round))]
        else  :
            data_io = data_io[(data_io["GAMECODE"]>= (10*round - 9 ))&(data_io["GAMECODE"]<= (10*round))]

    # Fonction pour convertir le temps en format mm:ss avec timedelta
    def convert_to_minutes(time_str):
        # Convertir le temps str en timedelta
        time_delta = pd.to_timedelta(time_str)
        
        # Retourner au format mm:ss
        return time_delta# Appliquer la fonction à la colonne TIME
    data_io['TIME'] = data_io['TIME'].apply(convert_to_minutes)
    if competition == "eurocup" : 
        data_io = data_io[data_io["GAMECODE"]<=18*10]
    elif competition =="euroleague" :
        data_io = data_io[data_io["GAMECODE"]<=34*9]

    for i in range(1,io+1):
        data_io = pd.merge(left=data_io,right=data_id,left_on=["CODETEAM",f"joueur_{i}"],right_on=["CODETEAM","PLAYER_ID"],how="left")

    data_io.columns = ['GAMECODE', 'CODETEAM'] + [f'ID{i}' for i in range(1, io+1)] + ['TIME', 'PM','OFF','DEF'] + [item for i in range(1, io+1) for item in [f'ID{i}b', f'PLAYER{i}']]

    data_io = data_io[['GAMECODE', 'CODETEAM'] +  [item for i in range(1, io+1) for item in [f'ID{i}', f'PLAYER{i}']] + ['TIME',
        'PM','OFF','DEF']]

    gb = data_io.groupby(['CODETEAM'] + [item for i in range(1, io+1) for item in [f'ID{i}', f'PLAYER{i}']]).agg({
        "TIME" : "sum","PM" : "sum","OFF" : "sum","DEF" : "sum"
    }).reset_index()


    gb_team = data_io.groupby(['CODETEAM']).agg({
        "TIME" : "sum","PM" : "sum","OFF" : "sum","DEF" : "sum"
    }).reset_index()

    gb_team["TIME"] /= math.comb(5, io)
    gb_team["PM"] /= math.comb(5, io)
    gb_team["OFF"] /= math.comb(5, io)
    gb_team["DEF"] /= math.comb(5, io)

    gb2 = pd.merge(left=gb,right=gb_team,left_on=["CODETEAM"],right_on=["CODETEAM"],how="left",suffixes=[f"_{io}IO","_TEAM"])
    gb2.rename(columns={f'PM_{io}IO': 'PM_IN'}, inplace=True)
    gb2[f"TIME_{io}IO"] = gb2[f"TIME_{io}IO"].dt.total_seconds() / 60
    gb2[f"TIME_TEAM"] = gb2[f"TIME_TEAM"].dt.total_seconds() / 60
    gb2[f"TIME_OUT_{io}IO"] = gb2[f"TIME_TEAM"] - gb2[f"TIME_{io}IO"]

    gb2["PM_OUT"] = (gb2["PM_TEAM"] - gb2["PM_IN"]).astype(int)
    gb2[f"PERCTIME_IN"] = (gb2[f"TIME_{io}IO"]/gb2["TIME_TEAM"]*100).round(1)
    gb2[f"PERCTIME_OUT"] = (gb2[f"TIME_OUT_{io}IO"]/gb2["TIME_TEAM"]*100).round(1)
    gb2[f"QT_PLAY"] = (gb2[f"TIME_{io}IO"] / 10 ).round(1)


    gb2["OFF_QT_IN"] = (gb2[f"OFF_{io}IO"] / gb2[f"TIME_{io}IO"] * 10).round(1)
    gb2["DEF_QT_IN"] = (gb2[f"DEF_{io}IO"] / gb2[f"TIME_{io}IO"] * 10).round(1)

    gb2["OFF_QT_OUT"] = ((gb2["OFF_TEAM"] - gb2[f"OFF_{io}IO"]) / gb2[f"TIME_OUT_{io}IO"] * 10).round(1)
    gb2["DEF_QT_OUT"] = ((gb2["DEF_TEAM"] - gb2[f"DEF_{io}IO"] )/ gb2[f"TIME_OUT_{io}IO"] * 10).round(1)

    gb2["DIF_QT"] = (gb2["OFF_QT_IN"] + gb2["DEF_QT_OUT"] - gb2["DEF_QT_IN"] - gb2["OFF_QT_OUT"]).round(1)


    gb3 = gb2.drop(columns=["TIME_TEAM","PM_TEAM",f"TIME_{io}IO",f"TIME_OUT_{io}IO"]+[f"OFF_{io}IO",f"DEF_{io}IO","OFF_TEAM","DEF_TEAM"])
    gb4 = gb3.sort_values(by = sortby,ascending=False).reset_index(drop=True)

    gb5 = gb4[["CODETEAM"]+[f"ID{i}" for i in range(1,io+1)]+ [f"PLAYER{i}" for i in range(1,io+1)] + ["QT_PLAY","PM_IN","PERCTIME_IN","OFF_QT_IN","DEF_QT_IN","PM_OUT","PERCTIME_OUT","OFF_QT_OUT","DEF_QT_OUT","DIF_QT"]]
    return gb5



def process_dataframe(data_dir, competition, season, nb_groupe):
    """
    Traite les données des joueurs pour la compétition et la saison données,
    crée des combinaisons de joueurs, agrège les résultats et sauvegarde dans un CSV.
    
    :param data_dir: Répertoire contenant les fichiers de données.
    :param competition: Nom de la compétition.
    :param season: Saison pour laquelle traiter les données.
    :param nb_groupe: Nombre de colonnes à combiner (de 1 à 5).
    """

    # Chargement des données
    five_summary = pd.read_csv(os.path.join(data_dir, f'{competition}_five_evol_{season}.csv'))

    # Tentative de chargement des données existantes
    already_exist_path = os.path.join(data_dir, f'{competition}_io_{nb_groupe}_{season}.csv')
    already_exist = pd.read_csv(already_exist_path) if os.path.exists(already_exist_path) else pd.DataFrame(
        columns=['Gamecode', 'CODETEAM'] + [f"J{i+1}" for i in range(nb_groupe)] + ["TIME", "DELTA_SCORE", "OFF", "DEF"]
    )
    
    # Calcul de la liste des Gamecode
    gc_list = list(set(five_summary["Gamecode"].unique()) - set(already_exist["Gamecode"].unique()))

    # Validation de nb_groupe
    if not (1 <= nb_groupe <= 5):
        raise ValueError(f"nb_groupe doit être compris entre 1 et 5.")

    # Filtrage des données
    df = five_summary[five_summary["Gamecode"].isin(gc_list)]
    new_rows = []

    # Création des combinaisons de colonnes P1 à P5
    columns_to_combine = ['P1', 'P2', 'P3', 'P4', 'P5']
    for index, row in df.iterrows():
        for combo in combinations(columns_to_combine, nb_groupe):
            new_row = {f'J{i+1}': row[col] for i, col in enumerate(combo)}
            # Ajouter les autres colonnes
            for col in df.columns:
                if col not in columns_to_combine:
                    new_row[col] = row[col]
            new_rows.append(new_row)

    # Création d'un DataFrame avec les nouvelles lignes
    new_df = pd.DataFrame(new_rows)

    # Agrégation des données
    group_columns = ['Gamecode', 'CODETEAM'] + [col for col in new_df.columns if col.startswith('J')]
    aggregated_df = new_df.groupby(group_columns).agg({
        'TIME': 'sum',
        'DELTA_SCORE': 'sum',
        'OFF': 'sum',
        'DEF': 'sum'
    }).reset_index()

    # Concaténation des données existantes et nouvelles
    aggregated_df["TIME"] = (aggregated_df["TIME"]).round(3)
    existing_filtered = already_exist.dropna(axis=1, how='all')  # Supprimer les colonnes vides
    aggregated_filtered = aggregated_df.dropna(axis=1, how='all')  # Supprimer les colonnes vides

    # Concaténer les DataFrames en ignorant les index
    new_exist = pd.concat([existing_filtered, aggregated_filtered], ignore_index=True)


    # Sauvegarde dans un fichier CSV
    new_exist.to_csv(already_exist_path, index=False)


def get_aggregated_data(df, 
                        min_round, 
                        max_round, 
                        selected_teams = [],
                        selected_opponents=[],
                        selected_fields = [],
                        selected_players = [],
                        mode="CUMULATED",
                        percent = "MADE",
                        ranking_mode = False):
    selected_fields2 = selected_fields

    if selected_fields == [] :
        selected_fields2 = ["PLAYER","ROUND","TEAM","OPPONENT"]

    df = df.drop(columns=["I_PER"])
    df_filtered = df[(df['ROUND'] >= min_round) & (df['ROUND'] <= max_round)].reset_index(drop=True)
    
    if selected_opponents != []:
        df_filtered = df_filtered[df_filtered['OPPONENT'].isin(selected_opponents)]

    if selected_players != []:
        df_filtered = df_filtered[df_filtered['PLAYER'].isin(selected_players)]

    if selected_teams != []:
        df_filtered = df_filtered[df_filtered['TEAM'].isin(selected_teams)]
    
    stats = ["TIME_ON"   , "PER" , "PM_ON","PTS",   "DR" ,  "OR" ,  "TR"  , "AS"  , "ST" , "CO" , "1_R" , "1_T" , "2_R" , "2_T" , "3_R" , "3_T" ,  "TO" ,  "FP"  , "CF" , "NCF"]
    stats_i = ["PER" ,"PTS",   "DR" ,  "OR" ,  "TR"  , "AS"  , "ST" , "CO" ,  "TO" ,  "FP"  , "CF" , "NCF"]

    aggregation_functions = {
        'TIME_ON': 'sum', 'PER': 'sum', 'PM_ON': 'sum', 'PTS': 'sum', 'DR': 'sum', 'OR': 'sum', 'TR': 'sum',
        'AS': 'sum', 'ST': 'sum', 'CO': 'sum', '1_R': 'sum', '1_T': 'sum', '2_R': 'sum', '2_T': 'sum',
        '3_R': 'sum', '3_T': 'sum', 'TO': 'sum', 'FP': 'sum', 'CF': 'sum', 'NCF': 'sum'
    }

    df_team = df_filtered.groupby(['ROUND','TEAM', 'OPPONENT', 'HOME', 'WIN']).agg(aggregation_functions).reset_index()
    df_team["TIME_ON"] =( (df_team["TIME_ON"]).round(0)).astype(int)
    df_team["PM_ON"] = (df_team["PM_ON"]/5).astype(int)


    if selected_fields2 == ["ROUND"] : 

        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values / 2).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/10).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/10).round(0)).astype(int)


    if selected_fields2 == ["TEAM"] :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if selected_fields2 == ["OPPONENT"] :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if selected_fields2 == ["PLAYER"] :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","TEAM"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["OPPONENT"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"OPPONENT","TEAM"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"ROUND","OPPONENT"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["TEAM"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"ROUND","PLAYER"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["OPPONENT","NUMBER","TEAM"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)


    if set(selected_fields2) == {"TEAM","PLAYER"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)


    if set(selected_fields2) == {"OPPONENT","PLAYER"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","PLAYER","TEAM"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2+["OPPONENT","NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","PLAYER","OPPONENT"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2+["TEAM","NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"TEAM","PLAYER","OPPONENT"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","OPPONENT","TEAM"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"ROUND","PLAYER","TEAM","OPPONENT"} and len(selected_fields2) == 4 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if mode == "AVERAGE" : 
        for s in stats :
            df_grouped[s] = (df_grouped[s]/df_grouped["NB_GAME"]).round(1)

    # if mode == "I_MODE" :
    #     for s in stats_i :
    #         df_grouped[s] = (df_grouped[s]/(df_grouped["TIME_ON"]**0.5)).round(1)

    if (df_grouped['NB_GAME'] == 1).all():
        # If NB_GAME is entirely filled with 1, map HOME and WIN to 'YES'/'NO'
        df_grouped['HOME'] = df_grouped['HOME'].map({1: 'YES', 0: 'NO'})
        df_grouped['WIN'] = df_grouped['WIN'].map({1: 'YES', 0: 'NO'})
    else:
        # Otherwise, convert HOME and WIN to percentages
        df_grouped["HOME"] = (df_grouped["HOME"] * 100).round(0).astype(int).astype(str) + '%'
        df_grouped["WIN"] = (df_grouped["WIN"] * 100).round(0).astype(int).astype(str) + '%'


    df_grouped["I_PER"] = (df_grouped["PER"] / df_grouped["TIME_ON"].pow(0.5)).round(1).fillna(0)
    df_grouped["PER"] = (df_grouped["PER"]).round(1)

    # Define the required columns
    required_columns = [
        'ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
        'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF'
    ]

    # Adding missing columns and filling with "---"
    for col in required_columns:
        if col not in df_grouped.columns:
            df_grouped[col] = "---"

    if percent == "PERCENT" : 
        df_grouped["1_R"] = (df_grouped["1_R"] / df_grouped["1_T"] * 100).round(1).fillna(0).astype(float)
        df_grouped["2_R"] = (df_grouped["2_R"] / df_grouped["2_T"] * 100).round(1).fillna(0).astype(float)
        df_grouped["3_R"] = (df_grouped["3_R"] / df_grouped["3_T"] * 100).round(1).fillna(0).astype(float)

        df_grouped.rename(columns={"1_R": '1_P', "2_R": '2_P', "3_R": '3_P'}, inplace=True)

        df_grouped["PER"] = (df_grouped["PER"]).round(1)
        df_grouped["I_PER"] = (df_grouped["I_PER"]).round(1)
        df_grouped["TIME_ON"] = (df_grouped["TIME_ON"]).round(2)
        df_grouped.rename(columns={"NUMBER": '#'}, inplace=True)
        return df_grouped[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', '#', 'PLAYER',
        'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_P', '1_T', '2_P', '2_T', '3_P', '3_T', 'TO', 'FP', 'CF', 'NCF']]
    
    else :
        df_grouped["PER"] = (df_grouped["PER"]).round(1)
        df_grouped["I_PER"] = (df_grouped["I_PER"]).round(1)
        df_grouped["TIME_ON"] = (df_grouped["TIME_ON"]).round(2)
        df_grouped.rename(columns={"NUMBER": '#'}, inplace=True)

        return df_grouped[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', '#', 'PLAYER',
        'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

