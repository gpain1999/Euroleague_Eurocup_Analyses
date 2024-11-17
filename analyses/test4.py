import pandas as pd

# Exemple de DataFrame avec des colonnes supplémentaires
data = {
    'ID': [1, 2, 3, 4],
    'P1': ['Gui', 'Alice', 'Leon', 'Gui'],
    'P2': ['Bob', 'Leon', 'Gui', 'Charlie'],
    'P3': ['Charlie', 'Gui', 'Alice', 'Leon'],
    'Score': [10, 20, 15, 30]
}
df = pd.DataFrame(data)

# Liste des joueurs à rechercher
joueurs = ["Gui", "Leon"]

# Colonnes à analyser
colonnes_cibles = ['P1', 'P2', 'P3']

# Filtrer les lignes où au moins un des joueurs est présent dans les colonnes spécifiées
df_filtre = df[df[colonnes_cibles].apply(lambda row: all(joueur in row.values for joueur in joueurs), axis=1)]

# Résultat
print(df_filtre)
