import pandas as pd
import os
import sys
import itertools
import math
from euroleague_api.game_stats import GameStats
from euroleague_api.play_by_play_data import PlayByPlay
from euroleague_api.player_stats import PlayerStats
import requests

def recup_io(data_dir,season,competition,nb_io) :

    competition_pbp = pd.read_csv(os.path.join(data_dir, f'{competition}_pbp_{season}.csv'))
    competition_gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))

    # try :
    #     io_total = pd.DataFrame(os.path.join(data_dir, f'{competition}_{nb_io}_io_{season}.csv'),index = False)
    #     game_code_list = list(set(competition_pbp["Gamecode"]) - set(io_total["GAMECODE"]))
    # except :
    #     io_total = pd.DataFrame(columns=['GAMECODE', 'CODETEAM', f'{nb_io}-io', 'TIME', 'DELTA_SCORE',"OFF","DEF"])
    #     game_code_list = list(set(competition_pbp["Gamecode"]))

    game_code_list = list(set(competition_pbp["Gamecode"]))
    io_total = pd.DataFrame(columns=['GAMECODE', 'CODETEAM', f'{nb_io}-io', 'TIME', 'DELTA_SCORE',"OFF","DEF"])

    for gc in game_code_list :
        local_io_summary,road_io_summary = global_io_recuperation_data_game(gs_data = competition_gs,pbp_data = competition_pbp,gamecode = gc ,nb_io = nb_io)
        io_total = pd.concat([io_total,local_io_summary,road_io_summary]).reset_index(drop = True)


    io_total[[f'joueur_{i+1}' for i in range(nb_io)]] = pd.DataFrame(io_total[f'{nb_io}-io'].tolist(), index=io_total.index)
    io_total = io_total[["GAMECODE","CODETEAM"] +[f'joueur_{i+1}' for i in range(nb_io)] + ["TIME","DELTA_SCORE","OFF","DEF"]]

    io_total.to_csv(os.path.join(data_dir, f'{competition}_{nb_io}_io_{season}.csv'),index = False)

def recup_idplayers(data_dir,season,competiton) : 
    _pbp = pd.read_csv(os.path.join(data_dir, f'{competiton}_pbp_{season}.csv'))


    mapping_ = _pbp.dropna(subset=['PLAYER'])[['CODETEAM', 'PLAYER_ID','PLAYER']].drop_duplicates(subset=['CODETEAM', 'PLAYER_ID'], keep='first')[['CODETEAM', 'PLAYER_ID','PLAYER']]


    mapping_.to_csv(os.path.join(data_dir, f'{competiton}_idplayers_{season}.csv'),index = False)

def recuperation_pbp(competition, season, data_dir, first_gamecode=None, last_gamecode=None):
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
        if first_gamecode is not None and last_gamecode is not None:
            for gc in range(first_gamecode, last_gamecode + 1):
                print(gc)
                # Vérifier si _pbp contient déjà des lignes pour ce gamecode
                if not (_pbp['Gamecode'] == gc).any():
                    # Récupérer les données pour le gamecode spécifique
                    try : 
                        sub = pbp_.get_game_play_by_play_data(season=season, gamecode=gc)
                        
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
        
        # Sauvegarder les données mises à jour dans le fichier CSV
        _pbp.to_csv(file_path, index=False)
        
    else:
        # Récupérer les données en utilisant PlayByPlay
        _pbp = pbp_.get_game_play_by_play_data_single_season(season=season)

        # Sauvegarder les données dans un fichier CSV
        _pbp.to_csv(file_path, index=False)

def recuperation_players_photo(competition,season,image_dir) :
    folder_link = os.path.join(image_dir, f'{competition}_{season}')

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

    df = ps.get_player_stats_single_season(endpoint = "traditional",season =  season ,phase_type_code = "RS",statistic_mode = "PerGame")

    df_utile = df[["player.code","player.imageUrl","player.team.code","player.team.imageUrl"]]

    df_utile.columns = ["ID_PLAYERS","URL_PLAYERS","ID_TEAM","URL_TEAM"]
    df_utile = df_utile[df_utile["URL_PLAYERS"].notna()]


    # Filter out rows where the player's image already exists in the folder
    def image_not_exists(row):
        player_image_path = os.path.join(folder_link, f"{row['ID_PLAYERS']}.png")
        return not os.path.isfile(player_image_path)
    
    # Apply the filter
    df_filtered = df_utile[df_utile.apply(image_not_exists, axis=1)]

    for index, row in df_filtered.iterrows():

        image_player_link = os.path.join(image_dir, f'{competition}_{season}\{row["ID_PLAYERS"]}.png')
        print(image_player_link)
        url_player = row["URL_PLAYERS"]
        print(url_player)
        response = requests.get(url_player)
        # Vérifier que la requête est réussie (code 200)
        if response.status_code == 200:
            # Enregistrer l'image sur le disque
            with open(image_player_link, "wb") as file:
                file.write(response.content)
            print("Image téléchargée avec succès.")
        else:
            print("Échec du téléchargement. Code d'erreur :", response.status_code)

def recuperation_gs(competition, season, data_dir, first_gamecode=None, last_gamecode=None):
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
        if first_gamecode is not None and last_gamecode is not None:
            for gc in range(first_gamecode, last_gamecode + 1):
                # Vérifier si _gs contient déjà des lignes pour ce gamecode
                if not (_gs['Gamecode'] == gc).any():
                    # Récupérer les données pour le gamecode spécifique
                    try : 
                        sub = gs_.get_game_report(season=season, gamecode=gc)
                        
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
    df_pbp["NUMBEROFPLAY"] = [i for i in range(len(df_pbp))]


    code = [df_gs["local.club.code"].to_list()[0],df_gs["road.club.code"].to_list()[0]]
    score = [df_gs["local.score"].to_list()[0],df_gs["road.score"].to_list()[0]]



    df_pbp['PERIOD'] = df_pbp['PERIOD'].apply(lambda x: x if x in [1, 2, 3, 4] else 4 + 0.5 * (x - 4))

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
    if len(starter) != 10 :
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


    
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


def io_plus_minus_analysis(data_dir,season,competition,io,sortby = "PM",teamcode = "") :


    if sortby == "PM" :
        sortby = f"PM_IN" 
    data_io = pd.read_csv(os.path.join(data_dir, f'{competition}_{io}_io_{season}.csv'))
    data_id = pd.read_csv(os.path.join(data_dir, f'{competition}_idplayers_{season}.csv'))

    if teamcode != "" :
        data_io = data_io[data_io["CODETEAM"]== teamcode] 
        data_id = data_id[data_id["CODETEAM"]== teamcode] 

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

    gb2["PM_OUT"] = (gb2["PM_TEAM"] - gb2["PM_IN"]).astype(int)
    gb2[f"PERC_TIME_{io}IO"] = round(gb2[f"TIME_{io}IO"]/gb2["TIME_TEAM"]*100,1)
    gb2[f"PERC_OFF"] = round(gb2[f"OFF_{io}IO"]/gb2["OFF_TEAM"]*100,1)
    gb2[f"PERC_DEF"] = round(gb2[f"DEF_{io}IO"]/gb2["DEF_TEAM"]*100,1)
    gb2["DIF_OFF"] = gb2["PERC_OFF"] - gb2[f"PERC_TIME_{io}IO"]
    gb2["DIF_DEF"] = gb2[f"PERC_TIME_{io}IO"] - gb2["PERC_DEF"]
    gb2["DIF"] = gb2["DIF_OFF"] + gb2["DIF_DEF"]

    gb2["PM_THEO"] = round(gb2[f"TIME_{io}IO"]/gb2["TIME_TEAM"]*gb2["PM_TEAM"],1)
    gb2["PREF_PM"] = gb2[f"PM_IN"] - gb2["PM_THEO"]
   


    gb3 = gb2.drop(columns=[f"ID{i}" for i in range(1,io+1)]+["TIME_TEAM","PM_TEAM",f"TIME_{io}IO"]+[f"OFF_{io}IO",f"DEF_{io}IO","OFF_TEAM","DEF_TEAM"])

    gb4 = gb3.sort_values(by = sortby,ascending=False).reset_index(drop=True)
    return gb4