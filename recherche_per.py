import pandas as pd
import sys
import os
import numpy as np

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



result_df = f.get_aggregated_data(
    df, 1, 34,
    selected_teams=[],
    selected_opponents=[],
    selected_fields=["TEAM","ROUND","PLAYER"],
    selected_players=[],
    mode="CUMULATED",
    percent="MADE"
)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Supposons que 'df' est ton DataFrame contenant les données
# 'WIN' est la variable cible, et les autres sont les variables explicatives

# Convertir WIN en variable binaire (1 pour "YES", 0 pour "NO")
result_df['WIN'] = result_df['WIN'].apply(lambda x: 1 if x == 'YES' else 0)

# Sélectionner les variables explicatives
X = result_df[['DR', 'OR', 'AS', 'ST', 'CO', '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]

# Variable cible
y = result_df['WIN']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Créer un modèle de régression logistique
log_reg = LogisticRegression()

# Entraîner le modèle
log_reg.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = log_reg.predict(X_test)

# Afficher les coefficients du modèle
coefficients = pd.DataFrame(log_reg.coef_[0], X.columns, columns=['Coefficient'])

# Trier les coefficients du plus grand au plus petit
coefficients_sorted = coefficients.sort_values(by='Coefficient', ascending=False)

print(coefficients_sorted)


