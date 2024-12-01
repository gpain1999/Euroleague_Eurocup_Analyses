import pandas as pd
import sys
import os
import numpy as np

# Charger les modules nécessaires
season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))
import fonctions as f

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
gs.columns = ["GAMECODE", "ROUND", "LOCAL_TEAM", "LOCAL_SCORE", "ROAD_TEAM", "ROAD_SCORE"]
gs["GAME"] = gs.apply(lambda row: f"R{row['ROUND']} : {row['LOCAL_TEAM']} - {row['ROAD_TEAM']}", axis=1)



###################################################################################################################################

# Récupérer les statistiques agrégées pour l'équipe locale et l'équipe visiteuse
def get_aggregated_stats(df, round_value, team, opponent=None):
    return f.get_aggregated_data(
        df, round_value, round_value,
        selected_teams=[team],
        selected_opponents=opponent if opponent else [],
        selected_fields=["TEAM", "ROUND"],
        selected_players=[],
        mode="CUMULATED",
        percent="MADE"
    )
# Créer les colonnes '1_L', '2_L', '3_L' pour local_stats et road_stats
def add_delta_columns(stats_df):
    for col in ["1", "2", "3"]:
        stats_df[f"{col}_L"] = stats_df[f"{col}_T"] - stats_df[f"{col}_R"]
    return stats_df

# Fonction pour appliquer les coefficients
def apply_coefficients(stats_df, coefficients):
    for column, coef in coefficients.items():
        stats_df[column] = (stats_df[column] * coef).round(3)
    return stats_df

# Création des colonnes '1', '2', '3' et suppression des colonnes intermédiaires
def process_dataframes(df_list):
    for df in df_list:
        for col in ['1', '2', '3']:
            df[col] = df[f"{col}_R"] + df[f"{col}_L"]
        df.drop(columns=[f"{col}_R" for col in ['1', '2', '3']] + [f"{col}_L" for col in ['1', '2', '3']], inplace=True)

# Fonction pour extraire les noms des colonnes 'top' et 'bottom' selon les critères
def get_top_and_bottom_column_names(df, n=4):
    top_values = df.unstack()
    top_values = top_values[top_values > 0].nlargest(n)
    top_columns = [col for col, _ in top_values.index]
    
    bottom_values = df.unstack()
    bottom_values = bottom_values[bottom_values < 0].nsmallest(n)
    bottom_columns = [col for col, _ in bottom_values.index]
    
    return top_columns, bottom_columns

# Fonction pour récupérer les valeurs avec un formatage spécial pour '1', '2', '3'
def extract_column_values(columns, stats_df):
    extracted_values = []
    for column in columns:
        if column in ['1', '2', '3']:
            extracted_values.append(f"{stats_df.loc[0, f'{column}_R']}/{stats_df.loc[0, f'{column}_T']} {column}-PTS")

        elif column == "PTS_C" :
            extracted_values.append(f"{stats_df.loc[0, column]} PTS CONCEDED")
        else:
            extracted_values.append(f"{stats_df.loc[0, column]} {column}")
    return extracted_values

###################################################################################################################################
row = gs.iloc[105]
###################################################################################################################################

local_stats = get_aggregated_stats(df, row["ROUND"], row["LOCAL_TEAM"])
road_stats = get_aggregated_stats(df, row["ROUND"], row["ROAD_TEAM"])



local_stats = add_delta_columns(local_stats)
road_stats = add_delta_columns(road_stats)

# Mettre à jour la colonne "PTS_C" dans local_stats et road_stats
local_stats["PTS_C"] = road_stats["PTS"]
road_stats["PTS_C"] = local_stats["PTS"]

# Calculer les moyennes pour mean_stats
mean_stats = f.get_aggregated_data(
    df, gs["ROUND"].min(), gs["ROUND"].max(),
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM", "ROUND"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)

# Ajouter les colonnes '1_L', '2_L', '3_L' à mean_stats
mean_stats = add_delta_columns(mean_stats)
mean_stats["PTS_C"] = mean_stats["PTS"]

# Liste des colonnes à traiter
columns_to_average = ["PTS_C", "DR", "OR", "AS", "ST", "CO", 
                      "1_R", "1_L", "2_R", "2_L", "3_R", "3_L", 
                      "TO", "FP", "CF", "NCF"]

# Calcul des moyennes arrondies et ajout dans un DataFrame
mean_stats = mean_stats[columns_to_average].mean().round(3).to_frame().T

# Coefficients pour appliquer aux colonnes
coefficients = {
    'DR': 0.85, 'OR': 1.35, 'AS': 0.8, 'ST': 1.33, 'CO': 0.6,
    '1_R': 1, '2_R': 2, '3_R': 3, '1_L': -0.9, '2_L': -0.75, '3_L': -0.5,
    'TO': -1.25, 'PTS_C': -0.5, 'FP': 0.5, 'CF': -0.5, 'NCF': -1.25
}



local_coeff = apply_coefficients(local_stats.copy()[columns_to_average], coefficients)
road_coeff = apply_coefficients(road_stats.copy()[columns_to_average], coefficients)
mean_coeff = apply_coefficients(mean_stats.copy()[columns_to_average], coefficients)

# Calcul des deltas
local_delta = (local_coeff - mean_coeff).round(3)
road_delta = (road_coeff - mean_coeff).round(3)

# Liste des DataFrames à traiter
dataframes = [local_coeff, road_coeff, mean_coeff, local_delta, road_delta]


process_dataframes(dataframes)



# Extraire les colonnes top et bottom pour local_delta et road_delta
local_top_columns, local_bottom_columns = get_top_and_bottom_column_names(local_delta)
road_top_columns, road_bottom_columns = get_top_and_bottom_column_names(road_delta)



# Extraire les valeurs formatées
local_top_values = extract_column_values(local_top_columns, local_stats)
local_bottom_values = extract_column_values(local_bottom_columns, local_stats)
road_top_values = extract_column_values(road_top_columns, road_stats)
road_bottom_values = extract_column_values(road_bottom_columns, road_stats)

# Afficher les résultats
print(f"TOP {row['LOCAL_TEAM']} :", local_top_values)
print(f"FLOP {row['LOCAL_TEAM']} :", local_bottom_values)
print(f"TOP {row['ROAD_TEAM']} :", road_top_values)
print(f"FLOP {row['ROAD_TEAM']} :", road_bottom_values)
