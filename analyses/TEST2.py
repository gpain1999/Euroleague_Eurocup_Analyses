import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate

season = 2024
competition = "euroleague"
round_to = 8
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f

df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
         'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
         '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]
r = 14

team_local = 'TEL'
team_road = 'BAR'

def stats_important_players(r,team_local,team_road,df) : 
    local_stats = get_aggregated_stats_players(df, r, team_local)
    road_stats = get_aggregated_stats_players(df, r, team_road)


    local_stats = add_delta_columns(local_stats)
    road_stats = add_delta_columns(road_stats)


    columns_to_average = ["DR", "OR", "AS", "ST", "CO", 
                        "1_R", "1_L", "2_R", "2_L", "3_R", "3_L", 
                        "TO", "FP", "CF", "NCF"]

    coefficients = {
        'DR': 0.85, 'OR': 1.35, 'AS': 0.8, 'ST': 1.33, 'CO': 0.6,
        '1_R': 1, '2_R': 2, '3_R': 3, '1_L': -0.6, '2_L': -0.75, '3_L': -0.5,
        'TO': -1.25, 'FP': 0.5, 'CF': -0.5, 'NCF': -1.25
    }

    local_coeff = apply_coefficients(local_stats.copy()[columns_to_average], coefficients)
    local_coeff = process_dataframes(local_coeff)
    local_coeff.index = local_stats[["TEAM","PLAYER"]]


    road_coeff = apply_coefficients(road_stats.copy()[columns_to_average], coefficients)
    road_coeff = process_dataframes(road_coeff)
    road_coeff.index = road_stats[["TEAM","PLAYER"]]

    coeff = pd.concat([local_coeff,road_coeff])
    stats = pd.concat([local_stats,road_stats])

    largest_values = coeff.stack().nlargest(8)

    result = largest_values.reset_index()

    result[['TEAM', 'PLAYER']] = pd.DataFrame(result['level_0'].tolist(), index=result.index)
    result = result.drop(columns=['level_0'])

    result.columns = ["STAT","VALUE",'TEAM', 'PLAYER']

    VALUE_0 = []
    for s,p in zip(result["STAT"].to_list(),result["PLAYER"].to_list()):
        stats_df = stats[stats["PLAYER"]==p].reset_index(drop = True)
        print(stats_df)
        if s in ['2', '3']:
            VALUE_0.append(f"{stats_df.loc[0, f'{s}_R']}/{stats_df.loc[0, f'{s}_T']} {s}-PTS")
        elif s in ['1']:
            VALUE_0.append(f"{stats_df.loc[0, f'{s}_R']}/{stats_df.loc[0, f'{s}_T']} FT")
        else :
            VALUE_0.append(f"{stats_df.loc[0, s]} {s}")

    result["VALUE"] = VALUE_0
    result = result.drop(columns=['STAT'])
    result = result["TEAM","PLAYER","VALUE"]

    return result
