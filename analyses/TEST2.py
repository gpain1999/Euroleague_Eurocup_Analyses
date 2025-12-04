import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f

# df = f.calcul_per2(data_dir, season, competition)
# df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
# df["I_PER"] = df["I_PER"].round(2)
# df["NB_GAME"] = 1
# df = df[['ROUND', 'NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
#          'TIME_ON', "I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
#          '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


# team = 'ASV'
# min_round = 1
# max_round = 14

# print(f.stats_important_team(team,min_round,max_round,df))


gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))[["Round",'local.club.code','road.club.code','local.score','road.score']]
gs["W"] = (gs["local.score"]>gs["road.score"]).astype(int)
gs["weights"] = gs["Round"] + (gs["Round"].max()+4)/3
# Obtenir la liste unique des clubs
clubs = pd.unique(gs[["local.club.code", "road.club.code"]].values.ravel())

# Ajouter une colonne pour chaque club
for club in clubs:
    gs[club] = gs.apply(
        lambda row: 1 if row["local.club.code"] == club else 
                    -1 if row["road.club.code"] == club else 0, 
        axis=1
    )
gs = gs.drop(columns =["Round",'local.club.code','road.club.code','local.score','road.score'])

# Variables explicatives (colonnes des clubs) et cible
X = gs[clubs]
y = gs["W"]
# Utiliser 'Round' comme poids
weights = gs["weights"]

# Régression logistique avec poids
model = LogisticRegression()
model.fit(X, y, sample_weight=weights)

# Récupérer les coefficients
coefficients = pd.DataFrame({
    "Club": clubs,
    "Coefficient": model.coef_[0]
})

# Trier par coefficient (du plus fort au plus faible)
coefficients_sorted = coefficients.sort_values(by="Coefficient", ascending=False).reset_index(drop = True)

def predict_victory_probability(model, local_team, road_team, clubs):
    """
    Prédit la probabilité que l'équipe locale gagne (W=1).
    
    Arguments :
    - model : modèle de régression logistique entraîné.
    - local_team : nom de l'équipe locale (str).
    - road_team : nom de l'équipe visiteuse (str).
    - clubs : liste des noms de clubs utilisés dans le modèle.
    
    Retourne :
    - probabilité (float) que W=1.
    """
    # Créer une ligne avec les caractéristiques pour chaque club
    features = {club: 0 for club in clubs}  # Initialiser avec 0
    features[local_team] = 1  # Local team
    features[road_team] = -1  # Road team
    
    # Convertir en DataFrame pour compatibilité avec le modèle
    input_data = pd.DataFrame([features])
    
    # Prédire la probabilité
    proba = model.predict_proba(input_data)[0][1]  # Proba de W=1
    return proba

print(coefficients_sorted)

local_team = "VIR"
road_team = "TEL"
proba = predict_victory_probability(model, local_team = local_team, road_team = road_team, clubs = clubs)

print(f"Probabilité que {local_team} gagne contre {road_team} : {proba:.2%}")