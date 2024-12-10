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

f.modele_PPS(r,df)
f.modele_BC(r,df)


# Résultats

# prediction_BCL,prediction_BCR = predict_ball_care(results_BCL = results_BCL,results_BCR = results_BCR, local_team = "BER", road_team = "PAR", clubs = clubs)

# print(prediction_BCL,prediction_BCR)