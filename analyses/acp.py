import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate

season = 2023
competition = "euroleague"
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
import fonctions as f
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
'1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


off = f.get_aggregated_data(df, 1, 34, selected_teams=[],selected_opponents=[],selected_players=[],selected_fields = ["TEAM","ROUND"],mode="CUMULATED",percent = "MADE")
deff = f.get_aggregated_data(df, 1, 34, selected_teams=[],selected_opponents=[],selected_players=[],selected_fields = ["OPPONENT","ROUND"],mode="CUMULATED",percent = "MADE")

off = off.drop(columns = ["OPPONENT","HOME","NUMBER","PLAYER",'TIME_ON',"I_PER","PER","PM_ON","NB_GAME","TR","1_R","2_R","3_R"],axis = 1)
deff = deff[['OPPONENT',"ROUND","PTS"]]
tot = pd.merge(off, deff, left_on=['TEAM',"ROUND"], right_on=['OPPONENT',"ROUND"], suffixes=('_OFF', '_DEF'))
tot = tot.drop(columns = ["OPPONENT","ROUND","TEAM"],axis=1)

tot['WIN'] = tot['WIN'].map({'YES': 1, 'NO': 0})

Y = tot["WIN"]
X = tot.drop(columns=["WIN"])

# Performing logistic regression without a constant
logit_model = sm.Logit(Y, X)
result = logit_model.fit()

# Displaying the results
print(result.summary())

# def convert_win(value):
#     try:
#         # Check if the value ends with '%'
#         if value.endswith('%'):
#             return float(value.strip('%')) / 100
#         else:
#             # Handle non-percentage cases (e.g., 'exce') by returning NaN or 0
#             return 0.0
#     except:
#         return 0.0  # Default fallback for any errors

# # Apply the function to the WIN column
# tot['WIN'] = tot['WIN'].apply(convert_win)

# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt



# # Sélection des colonnes statistiques pour l'ACP
# X = tot.drop(columns=['TEAM', 'WIN'])  # Exclude TEAM and WIN


# y = tot['WIN']
# teams = tot['TEAM']

# # Standardisation des données
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Réalisation de l'ACP
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Réalisation de l'ACP (4 composantes pour visualiser les variables)
# pca_full = PCA(n_components=4)
# X_pca_full = pca_full.fit_transform(X_scaled)

# # Grap# Graphique des variables (biplot) avec flèches moins imposantes
# plt.figure(figsize=(12, 6))

# # Subplot pour les variables
# plt.subplot(1, 2, 1)
# for i, feature in enumerate(X.columns):
#     plt.arrow(0, 0,
#               pca_full.components_[0, i],
#               pca_full.components_[1, i],
#               color='r',
#               alpha=0.5,            # Augmentation de la transparence
#               head_width=0.02,      # Réduction de la taille de la tête de flèche
#               head_length=0.02,
#               linewidth=1)          # Réduction de l'épaisseur de la ligne
#     plt.text(pca_full.components_[0, i] * 1.15,
#              pca_full.components_[1, i] * 1.15,
#              feature,
#              color='g',
#              ha='center',
#              va='center',
#              fontsize=10)
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
# plt.xlabel('Dim 1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
# plt.ylabel('Dim 2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
# plt.title('Variables factor map (PCA)')
# plt.grid(alpha=0.3)
# plt.axis('equal')

# # Subplot pour les individus (inchangé)
# plt.subplot(1, 2, 2)
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
# plt.colorbar(scatter, label='WIN (0=Bleu, 1=Rouge)')
# for i, team in enumerate(teams):
#     plt.text(X_pca[i, 0] + 0.1, X_pca[i, 1], team, fontsize=9, ha='left', va='center')
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
# plt.xlabel('Dim 1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
# plt.ylabel('Dim 2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
# plt.title('Individuals factor map (PCA)')
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()