import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from statsmodels.api import add_constant
import pickle

from statsmodels.api import GLM, families
from statsmodels.genmod.families.links import logit

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f



def predict_ball_care(results_BCL,results_BCR, local_team, road_team, clubs):
    """
    Prédire la variable BCL pour un match entre une équipe locale et une équipe adverse.
    
    Parameters:
    - model: le modèle GLM ajusté (modèle BCL).
    - local_team: le code du club local (par exemple, un code de club).
    - road_team: le code du club adverse (par exemple, un code de club).
    - clubs: la liste des codes de clubs dans les données.
    
    Returns:
    - La prédiction du modèle pour la variable BCL.
    """
    # Créer une nouvelle ligne de données pour ce match
    row = {club: 0 for club in clubs}  # Initialisation avec des valeurs 0
    row[local_team] = 1  # L'équipe locale reçoit la valeur 1
    row[road_team] = -1  # L'équipe adverse reçoit la valeur -1
    
    # Convertir la ligne en DataFrame
    match_data = pd.DataFrame(row, index=[0])
    
    # Ajouter la constante, mais vérifier que toutes les colonnes sont présentes
    match_data = add_constant(match_data, has_constant='add')  # Ajouter une constante correctement
    # S'assurer que les colonnes du match_data sont dans le même ordre que celles de X
    
    # Effectuer la prédiction avec le modèle GLM
    prediction_BCL = results_BCL.predict(match_data)[0]  
    prediction_BCR = results_BCR.predict(match_data)[0]  

    return prediction_BCL,prediction_BCR

# Résultats

# prediction_BCL,prediction_BCR = predict_ball_care(results_BCL = results_BCL,results_BCR = results_BCR, local_team = "BER", road_team = "PAR", clubs = clubs)

# print(prediction_BCL,prediction_BCR)