import pandas as pd
import sys
import os
import numpy as np

import plotly
print(plotly.__version__)


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



###################################################################################################################################
row = gs.iloc[105]
###################################################################################################################################

selected_range = [1,17]

team_detail_select = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM","ROUND"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)

opp_detail_select = f.get_aggregated_data(
    df=df, min_round=selected_range[0], max_round=selected_range[1],
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["OPPONENT","ROUND"],
    selected_players=[],
    mode="AVERAGE",
    percent="MADE"
)

result = pd.merge(
    team_detail_select,
    opp_detail_select,
    how="left",
    left_on=["TEAM","ROUND"],
    right_on=["OPPONENT","ROUND"],
    suffixes=('_team', '_opp')
)

result["PPS"] = ((result["2_R_team"]*2 + result["3_R_team"]*3)/(result["2_T_team"] + result["3_T_team"]))
result["PPS_opp"] = ((result["2_R_opp"]*2 + result["3_R_opp"]*3)/(result["2_T_opp"] + result["3_T_opp"]))

result["BC"] = (100*(result["1_T_team"]*0.5 + result["2_T_team"] + result["3_T_team"])/(result["1_T_team"]*0.5 + result["2_T_team"] + result["3_T_team"] + result["TO_team"]))
result["BC_opp"] = (100*(result["1_T_opp"]*0.5 + result["2_T_opp"] + result["3_T_opp"])/(result["1_T_opp"]*0.5 + result["2_T_opp"] + result["3_T_opp"] + result["TO_opp"]))

result["PctDeReRec"] = (100*(result["DR_team"])/(result["DR_team"] + result["OR_opp"]))
result["PctOfReRec"] = (100*(result["OR_team"])/(result["OR_team"] + result["DR_opp"]))

result["REB_PERF"] = ((result["PctDeReRec"]/70)/(result["PctDeReRec"]/70 + (100-result["PctDeReRec"])/30) + ((result["PctOfReRec"])/30)/((100 - result["PctOfReRec"])/70 + (result["PctOfReRec"])/30))

#X
result["Battle_SHOOTING"] = (result["PPS"]  > result["PPS_opp"]).astype(int)
result["Battle_BC"] = (result["BC"]  > result["BC_opp"]).astype(int)
result["Battle_REB"] = (result["REB_PERF"]  > 1).astype(int)
result["Battle_Agr"] = (result["FP_team"]  > result["FP_opp"]).astype(int)
#Y
result["RES"] = (result["WIN_team"]  == "YES").astype(int)


from sklearn.tree import DecisionTreeClassifier, export_text

X = result[["Battle_SHOOTING", "Battle_BC", "Battle_REB", "Battle_Agr"]]
y = result["RES"]
# Initialiser un arbre de décision avec une profondeur maximale de 2
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Entraîner l'arbre de décision
clf.fit(X, y)

# Obtenir les probabilités prédites pour chaque feuille
node_indicator = clf.decision_path(X)
leaf_ids = clf.apply(X)
leaf_counts = np.bincount(leaf_ids)
leaf_probs = clf.predict_proba(X)

# Afficher le pourcentage de y=1 pour chaque feuille
leaf_y1_percentage = {}
for leaf in np.unique(leaf_ids):
    leaf_samples = (leaf_ids == leaf)
    y1_percentage = y[leaf_samples].mean() * 100
    leaf_y1_percentage[leaf] = y1_percentage

print("Pourcentage de y=1 pour chaque feuille :")
for leaf, percentage in leaf_y1_percentage.items():
    print(f"Feuille {leaf}: {percentage:.2f}%")

# Exporter la structure de l'arbre sous forme de texte
arbre_texte = export_text(clf, feature_names=X.columns.tolist())
print("Structure de l'arbre de décision :")
print(arbre_texte)