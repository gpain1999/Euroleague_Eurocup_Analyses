import pandas as pd
import sys
import os
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import webbrowser

season = 2024
competition = "euroleague"
round_to = 8
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))
import fonctions as f



df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
'1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

df1 = f.calcul_per2(data_dir, season-1, competition)
df1.insert(6, "I_PER", df1["PER"] / df1["TIME_ON"] ** 0.5)
df1["I_PER"] = df1["I_PER"].round(2)
df1["NB_GAME"] = 1
df1 = df1[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
'1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


### COLECTIV STATS ###
df2 = f.get_aggregated_data(df, 1, round_to, selected_opponents=None,selected_fields = ["TEAM","ROUND"],mode="CUMULATED",percent = "MADE")
df2.insert(0, "SEASON", season)

df2b = f.get_aggregated_data(df1, 1, 34, selected_opponents=None,selected_fields = ["TEAM","ROUND"],mode="CUMULATED",percent = "MADE")
df2b.insert(0, "SEASON", season-1)


df2 = pd.concat([df2,df2b])

df2["INF_1P"] = (df2["1_R"] / df2["1_T"]) - 0.674 * ((((df2["1_R"] / df2["1_T"])) * (1 - ((df2["1_R"] / df2["1_T"]))))/(df2["1_T"]))**0.5
df2["SUP_1P"] = (df2["1_R"] / df2["1_T"]) + 0.674 * ((((df2["1_R"] / df2["1_T"])) * (1 - ((df2["1_R"] / df2["1_T"]))))/(df2["1_T"]))**0.5

df2["INF_2P"]= (df2["2_R"] / df2["2_T"]) - 0.674 *((((df2["2_R"] / df2["2_T"])) * (1 - ((df2["2_R"] / df2["2_T"]))))/(df2["2_T"]))**0.5
df2["SUP_2P"]= (df2["2_R"] / df2["2_T"]) + 0.674 *((((df2["2_R"] / df2["2_T"])) * (1 - ((df2["2_R"] / df2["2_T"]))))/(df2["2_T"]))**0.5

df2["INF_3P"]= (df2["3_R"] / df2["3_T"]) - 0.674 *((((df2["3_R"] / df2["3_T"])) * (1 - ((df2["3_R"] / df2["3_T"]))))/(df2["3_T"]))**0.5
df2["SUP_3P"]= (df2["3_R"] / df2["3_T"]) + 0.674 *((((df2["3_R"] / df2["3_T"])) * (1 - ((df2["3_R"] / df2["3_T"]))))/(df2["3_T"]))**0.5


df2 = df2.drop(["NB_GAME","NUMBER","PLAYER","TIME_ON","I_PER","PM_ON","NCF","OPPONENT","HOME","WIN"],axis = 1)

df3 = df2.copy()
# Select columns to transform into percentiles
columns_to_percentile = ['PER', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO', '1_R', '1_T', 
                         '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'INF_1P', 'INF_2P', 'INF_3P','SUP_1P', 'SUP_2P', 'SUP_3P']

# Apply percentile rank transformation to each specified column
for column in columns_to_percentile:
    df3[column] = (df3[column].rank(pct=True) * 100).round(1)

df3["TO"] = 100 - df3["TO"]
df3["CF"] = 100 - df3["CF"]

columns_b = ['DR', 'OR', 'TR', 'AS', 'ST', 'CO',  'TO', 'FP', 'CF', 'INF_1P', 'INF_2P', 'INF_3P']

columns_w = ['DR', 'OR', 'TR', 'AS', 'ST', 'CO',  'TO', 'FP', 'CF','SUP_1P', 'SUP_2P', 'SUP_3P']

df3['BEST_TO'] = df3[columns_b].apply(
    lambda row: row.nlargest(4).index.tolist(), axis=1
)

df3['WORST_TO'] = df3[columns_w].apply(
    lambda row: row.nsmallest(4).index.tolist(), axis=1
)



df_result = pd.merge(df2[(df2["ROUND"]==round_to) & (df2["SEASON"]==season)], df3[(df3["ROUND"]==round_to) & (df3["SEASON"]==season)][["ROUND", "TEAM","BEST_TO","WORST_TO"]], on=["ROUND", "TEAM"], how='inner')



df_result["PER"] = df_result["PER"].astype(str) + " PERV2"
df_result["PTS"] = df_result["PTS"].astype(str) + " PTS"
df_result["DR"] = df_result["DR"].astype(str) + " DEF. REB."
df_result["OR"] = df_result["OR"].astype(str) + " OFF. REB."
df_result["TR"] = df_result["TR"].astype(str) + " TOT. REB."
df_result["AS"] = np.where(df_result["AS"].isin([0, 1]),
                           df_result["AS"].astype(str) + " ASSIST",
                           df_result["AS"].astype(str) + " ASSISTS")
df_result["ST"] = np.where(df_result["ST"].isin([0, 1]),
                           df_result["ST"].astype(str) + " STEAL",
                           df_result["ST"].astype(str) + " STEALS")

df_result["CO"] = np.where(df_result["CO"].isin([0, 1]),
                           df_result["CO"].astype(str) + " BLOCK",
                           df_result["CO"].astype(str) + " BLOCKS")

df_result["TO"] = df_result["TO"].astype(str) + " TO/OF"
df_result["FP"] = np.where(df_result["FP"].isin([0, 1]),
                           df_result["FP"].astype(str) + " FOUL REC",
                           df_result["FP"].astype(str) + " FOULS REC")

df_result["CF"] = df_result["CF"].astype(str) + " DEF.FAULTS"
df_result["INF_1P"] = df_result["1_R"].astype(str) + "/" + df_result["1_T"].astype(str) + " FT"
df_result["SUP_1P"] = df_result["1_R"].astype(str) + "/" + df_result["1_T"].astype(str) + " FT"
df_result["INF_2P"] = df_result["2_R"].astype(str) + "/" + df_result["2_T"].astype(str) + " 2 PTS"
df_result["SUP_2P"] = df_result["2_R"].astype(str) + "/" + df_result["2_T"].astype(str) + " 2 PTS"
df_result["INF_3P"] = df_result["3_R"].astype(str) + "/" + df_result["3_T"].astype(str) + " 3 PTS"
df_result["SUP_3P"] = df_result["3_R"].astype(str) + "/" + df_result["3_T"].astype(str) + " 3 PTS"

# Création des colonnes BEST_1, BEST_2, BEST_3, BEST_4
for i in range(4):
    # Extraction de la colonne à partir de BEST_TO et assignation de la valeur correspondante
    df_result[f'BEST_{i+1}'] = df_result.apply(lambda row: row[row['BEST_TO'][i]], axis=1)

# Création des colonnes WORST_1, WORST_2, WORST_3, WORST_4
for i in range(4):
    # Extraction de la colonne à partir de WORST_TO et assignation de la valeur correspondante
    df_result[f'WORST_{i+1}'] = df_result.apply(lambda row: row[row['WORST_TO'][i]], axis=1)

df_result_2 = df_result[["ROUND","TEAM","PER","BEST_1","BEST_2","BEST_3","BEST_4","WORST_1","WORST_2","WORST_3","WORST_4"]]

print(df_result_2)

gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))
gs.loc[:,"Round"] = (gs["Gamecode"] - 1) // 9 + 1
gs = gs[gs["Round"] == round_to].reset_index(drop = True)
gs = gs[['Round','local.club.code',"local.club.abbreviatedName",'local.score','road.club.code','road.club.abbreviatedName','road.score']]
gs.columns = ["ROUND","LOCAL_TEAM","LOCAL_NAME","LOCAL_SCORE","ROAD_TEAM","ROAD_NAME","ROAD_SCORE"]


gs = pd.merge(gs, df_result_2, left_on=["ROUND", "LOCAL_TEAM"], right_on=["ROUND", "TEAM"], how='inner')

for i in range(4) :
    gs.rename(columns={f'BEST_{i+1}': f'BEST_{i+1}_LOCAL'}, inplace=True)
    gs.rename(columns={f'WORST_{i+1}': f'WORST_{i+1}_LOCAL'}, inplace=True)

gs = pd.merge(gs, df_result_2, left_on=["ROUND","ROAD_TEAM"],right_on= ["ROUND","TEAM"], how='inner')

for i in range(4) :
    gs.rename(columns={f'BEST_{i+1}': f'BEST_{i+1}_ROAD'}, inplace=True)
    gs.rename(columns={f'WORST_{i+1}': f'WORST_{i+1}_ROAD'}, inplace=True)


####FIND PLAYER STATS #####

df_p_m1 = f.get_aggregated_data(df1, 1, 34, selected_opponents=None,selected_fields = ["TEAM","PLAYER","ROUND"],mode="CUMULATED",percent = "MADE")
df_p_m1.insert(0, "SEASON", season-1)

df_p = f.get_aggregated_data(df, 1, round_to, selected_opponents=None,selected_fields = ["TEAM","PLAYER","ROUND"],mode="CUMULATED",percent = "MADE")
df_p.insert(0, "SEASON", season)

df2 = pd.concat([df_p,df_p_m1])

df2["INF_1P"] = (df2["1_R"] / df2["1_T"]) - 0.674 * ((((df2["1_R"] / df2["1_T"])) * (1 - ((df2["1_R"] / df2["1_T"]))))/(df2["1_T"]))**0.5
df2["SUP_1P"] = (df2["1_R"] / df2["1_T"]) + 0.674 * ((((df2["1_R"] / df2["1_T"])) * (1 - ((df2["1_R"] / df2["1_T"]))))/(df2["1_T"]))**0.5

df2["INF_2P"]= (df2["2_R"] / df2["2_T"]) - 0.674 *((((df2["2_R"] / df2["2_T"])) * (1 - ((df2["2_R"] / df2["2_T"]))))/(df2["2_T"]))**0.5
df2["SUP_2P"]= (df2["2_R"] / df2["2_T"]) + 0.674 *((((df2["2_R"] / df2["2_T"])) * (1 - ((df2["2_R"] / df2["2_T"]))))/(df2["2_T"]))**0.5

df2["INF_3P"]= (df2["3_R"] / df2["3_T"]) - 0.674 *((((df2["3_R"] / df2["3_T"])) * (1 - ((df2["3_R"] / df2["3_T"]))))/(df2["3_T"]))**0.5
df2["SUP_3P"]= (df2["3_R"] / df2["3_T"]) + 0.674 *((((df2["3_R"] / df2["3_T"])) * (1 - ((df2["3_R"] / df2["3_T"]))))/(df2["3_T"]))**0.5



df2 = df2.drop(["NB_GAME","NUMBER","OPPONENT","HOME"],axis = 1)

df3 = df2.copy()
df3["NCF"] = df3["NCF"].replace(0, None)

# Select columns to transform into percentiles
columns_to_percentile = ["PM_ON","I_PER",'PER', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO', '1_R', '1_T', 
                         '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF' ,'INF_1P', 'INF_2P', 'INF_3P','SUP_1P', 'SUP_2P', 'SUP_3P']


# Apply percentile rank transformation to each specified column
for column in columns_to_percentile:
    df3[column] = (df3[column].rank(pct=True) * 100).round(1)

df3["TO"] = 100 - df3["TO"]
df3["CF"] = 100 - df3["CF"]
df3["NCF"] = 100 - df3["NCF"]

columns_b = ["PM_ON",'DR', 'OR', 'TR', 'AS', 'ST', 'CO',  'TO', 'FP', 'INF_1P', 'INF_2P', 'INF_3P']

columns_w = ["PM_ON",'TR', 'AS', 'ST', 'CO',  'TO', 'FP', 'CF',"NCF",'SUP_1P', 'SUP_2P', 'SUP_3P']

df3['BEST_TO'] = df3[columns_b].apply(
    lambda row: row.nlargest(3).index.tolist(), axis=1
)

df3['WORST_TO'] = df3[columns_w].apply(
    lambda row: row.nsmallest(3).index.tolist(), axis=1
)

df_result = pd.merge(df2[(df2["ROUND"]==round_to) & (df2["SEASON"]==season)], df3[(df3["ROUND"]==round_to) & (df3["SEASON"]==season)][["ROUND","TEAM","PLAYER","BEST_TO","WORST_TO"]], on=["ROUND", "TEAM",'PLAYER'], how='inner')
df_result["PER_OFF"] = df_result["PER"]
df_result["I_PER_OFF"] = df_result["I_PER"]
df_result["TIME_ON_OFF"] = df_result["TIME_ON"]

df_result["PER"] = df_result["PER"].astype(str) + " PERV2"
df_result["I_PER"] = df_result["I_PER"].astype(str) + " I_PERV2"

df_result["PTS"] = df_result["PTS"].astype(str) + " PTS"
df_result["DR"] = df_result["DR"].astype(str) + " DEF. REB."
df_result["OR"] = df_result["OR"].astype(str) + " OFF. REB."
df_result["TR"] = df_result["TR"].astype(str) + " TOT. REB."
df_result["AS"] = np.where(df_result["AS"].isin([0, 1]),
                           df_result["AS"].astype(str) + " ASSIST",
                           df_result["AS"].astype(str) + " ASSISTS")
df_result["ST"] = np.where(df_result["ST"].isin([0, 1]),
                           df_result["ST"].astype(str) + " STEAL",
                           df_result["ST"].astype(str) + " STEALS")

df_result["CO"] = np.where(df_result["CO"].isin([0, 1]),
                           df_result["CO"].astype(str) + " BLOCK",
                           df_result["CO"].astype(str) + " BLOCKS")

df_result["TO"] = df_result["TO"].astype(str) + " TO/OF"
df_result["FP"] = np.where(df_result["FP"].isin([0, 1]),
                           df_result["FP"].astype(str) + " F. REC",
                           df_result["FP"].astype(str) + " F. REC")

df_result["CF"] = df_result["CF"].astype(str) + " DEF.FAULTS"
df_result["NCF"] = df_result["NCF"].astype(str) + " NO.CLASSIC.FOULS."

df_result["INF_1P"] = df_result["1_R"].astype(str) + "/" + df_result["1_T"].astype(str) + " FT"
df_result["SUP_1P"] = df_result["1_R"].astype(str) + "/" + df_result["1_T"].astype(str) + " FT"
df_result["INF_2P"] = df_result["2_R"].astype(str) + "/" + df_result["2_T"].astype(str) + " 2 PTS"
df_result["SUP_2P"] = df_result["2_R"].astype(str) + "/" + df_result["2_T"].astype(str) + " 2 PTS"
df_result["INF_3P"] = df_result["3_R"].astype(str) + "/" + df_result["3_T"].astype(str) + " 3 PTS"
df_result["SUP_3P"] = df_result["3_R"].astype(str) + "/" + df_result["3_T"].astype(str) + " 3 PTS"
df_result["TIME_ON"] = df_result["TIME_ON"].round(0).astype(int).astype(str) + " MINS"
df_result["PM_ON"] = df_result["PM_ON"].apply(lambda x: f"+{x}" if x > 0 else str(x)) + " +/-"


# Création des colonnes BEST_1, BEST_2, BEST_3, BEST_4
for i in range(3):
    # Extraction de la colonne à partir de BEST_TO et assignation de la valeur correspondante
    df_result[f'BEST_{i+1}'] = df_result.apply(lambda row: row[row['BEST_TO'][i]], axis=1)

# Création des colonnes WORST_1, WORST_2, WORST_3, WORST_4
for i in range(3):
    # Extraction de la colonne à partir de WORST_TO et assignation de la valeur correspondante
    df_result[f'WORST_{i+1}'] = df_result.apply(lambda row: row[row['WORST_TO'][i]], axis=1)

df_result_2 = df_result[["ROUND","TEAM","PLAYER","WIN","TIME_ON_OFF","I_PER_OFF","PER_OFF","TIME_ON","PER","PTS","BEST_1","BEST_2","BEST_3","WORST_1","WORST_2","WORST_3"]]

df_win = df_result_2[df_result_2["WIN"]=="YES"]

# Étape 1 : Trier par ROUND, TEAM et PER_OFF en ordre décroissant
df_sorted_MVP = df_result_2.sort_values(by=["ROUND", "TEAM", "PER_OFF"], ascending=[True, True, False])

# Étape 2 : Garder les 3 lignes avec les plus grandes valeurs de PER_OFF pour chaque ROUND et TEAM
top_3_per_off_MVP = df_sorted_MVP.groupby(['ROUND', 'TEAM']).head(1)

# Étape 3 : Parmi ces 3 lignes, sélectionner celle avec le plus grand I_PER_OFF
result_MVP = top_3_per_off_MVP.sort_values(by="I_PER_OFF", ascending=False).drop_duplicates(subset=["ROUND", "TEAM"])


print(result_MVP[["ROUND","TEAM","PLAYER","TIME_ON","PER","PTS","BEST_1","BEST_2","BEST_3"]])



df_l = df_result_2[(df_result_2["WIN"]=="NO")& (df_result_2["TIME_ON_OFF"]>=11)]




# Étape 1 : Trier par ROUND, TEAM et PER_OFF en ordre décroissant
df_sorted_l = df_l.sort_values(by=["ROUND", "TEAM", "PER_OFF"], ascending=[False, False, True])

# Étape 2 : Garder les 3 lignes avec les plus grandes valeurs de PER_OFF pour chaque ROUND et TEAM
top_3_per_off_l = df_sorted_l.groupby(['ROUND', 'TEAM']).head(2)

# Étape 3 : Parmi ces 3 lignes, sélectionner celle avec le plus grand I_PER_OFF
result_l = top_3_per_off_l.sort_values(by="I_PER_OFF", ascending=True).drop_duplicates(subset=["ROUND", "TEAM"])

print(result_l[["ROUND","TEAM","PLAYER","TIME_ON","PER","PTS","WORST_1","WORST_2","WORST_3"]])


# Étape 1 : Trier par ROUND, TEAM et PER_OFF en ordre décroissant

df_fact_x = df_result_2[(df_result_2["WIN"]=="YES")&(~df_result_2["PLAYER"].isin(list(result_MVP["PLAYER"])))]

df_sorted_x = df_fact_x.sort_values(by=["ROUND", "TEAM", "PER_OFF"], ascending=[True, True, False])



# Étape 2 : Garder les 3 lignes avec les plus grandes valeurs de PER_OFF pour chaque ROUND et TEAM
top_3_per_off_x = df_sorted_x.groupby(['ROUND', 'TEAM']).head(2)

# Étape 3 : Parmi ces 3 lignes, sélectionner celle avec le plus grand I_PER_OFF
result_x = top_3_per_off_x.sort_values(by="I_PER_OFF", ascending=False).drop_duplicates(subset=["ROUND", "TEAM"])


print(result_x[["ROUND","TEAM","PLAYER","TIME_ON","PER","PTS","BEST_1","BEST_2","BEST_3"]])


