import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image


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


def get_aggregated_data(df, min_round, max_round, selected_teams = [],selected_opponents=[],selected_fields = [],selected_players = [],mode="CUMULATED",percent = "MADE"):
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
    
    stats = ["TIME_ON", "PER" , "PM_ON","PTS",   "DR" ,  "OR" ,  "TR"  , "AS"  , "ST" , "CO" , "1_R" , "1_T" , "2_R" , "2_T" , "3_R" , "3_T" ,  "TO" ,  "FP"  , "CF" , "NCF"]
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

    if mode == "I_MODE" :
        for s in stats_i :
            df_grouped[s] = (df_grouped[s]/(df_grouped["TIME_ON"]**0.5)).round(1)

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


result_df = get_aggregated_data(
    df, min_round=1, max_round=10,
    selected_teams=["MCO"],
    selected_opponents=[],
    selected_fields=[],
    selected_players=[],
    mode="I_MODE",
    percent="PERCENT"
)