import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from statsmodels.api import OLS


season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f

from statsmodels.api import add_constant


df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
        'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

r = 14

def modele_PPS(r,df) : 

        team_detail_select = f.get_aggregated_data(
        df=df, min_round=1, max_round=r,
        selected_teams=[],
        selected_opponents=[],
        selected_fields=["ROUND","TEAM"],
        selected_players=[],
        mode="CUMULATED",
        percent="MADE"
        )[["ROUND","HOME",'TEAM', 'OPPONENT','2_R','2_T','3_R','3_T']]


        team_detail_select["PPS"] = ((team_detail_select["2_R"]*2 + team_detail_select["3_R"]*3)/(team_detail_select["2_T"] + team_detail_select["3_T"])).round(4)



        # Filtrer selon la condition HOME
        t1 = team_detail_select[team_detail_select["HOME"] == "YES"]
        t2 = team_detail_select[team_detail_select["HOME"] == "NO"]

        # Effectuer le LEFT JOIN sur les conditions spécifiées
        result = pd.merge(
        t1,
        t2,
        how="left",
        left_on=["TEAM", "OPPONENT", "ROUND"],
        right_on=["OPPONENT", "TEAM", "ROUND"],
        suffixes=('_local', '_road')
        )

        # Sélectionner et renommer les colonnes pour correspondre à la sortie SQL
        result = result[["ROUND", "TEAM_local", "TEAM_road", "PPS_local", "PPS_road"]]
        result.columns = ["ROUND", "local.club.code", "road.club.code", "PPSL", "PPSR"]

        clubs = pd.unique(result[["local.club.code", "road.club.code"]].values.ravel())

        # Ajouter une colonne pour chaque club
        for club in clubs:
                result[club] = result.apply(
                        lambda row: 1 if row["local.club.code"] == club else 
                                -1 if row["road.club.code"] == club else 0, 
                        axis=1
                )

        X = result[clubs]
        X = add_constant(X)
        weights = result['ROUND'] + (result["ROUND"].max()+4)/3

        ################################## PPSL ########################################################""
        Y = result['PPSL']

        model_PPSL = OLS(Y, X)
        results_PPSL = model_PPSL.fit()

        ################################## PPSR ########################################################""
        Y = result['PPSR']

        model_PPSR = OLS(Y, X)
        results_PPSR = model_PPSR.fit()

        # Enregistrer le modèle BCL
        with open('./modeles/model_PPSL.pkl', 'wb') as f:
                pickle.dump(results_PPSL, f)

        # Enregistrer le modèle BCR
        with open('./modeles/model_PPSR.pkl', 'wb') as f:
                pickle.dump(results_PPSR, f)


# Résultats

# prediction_BCL,prediction_BCR = predict_ball_care(results_BCL = results_BCL,results_BCR = results_BCR, local_team = "BER", road_team = "PAR", clubs = clubs)

# print(prediction_BCL,prediction_BCR)