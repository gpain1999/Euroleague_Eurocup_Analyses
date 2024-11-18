import pandas as pd

# Exemple de données pour les rounds manquants
data = {
    "ROUND": [1, 2, 3, 4, 6],  # Rounds avec un gap (pas de round 5)
    "selected_stats": [10, 15, 20, 25, 30],
    "WIN": ["YES", "NO", "YES", "NO", "YES"],
    "TIME_ON": [5, 10, 15, 20, 25]
}
filtered_df = pd.DataFrame(data)

# Calcul de la moyenne glissante en tenant compte des rounds manquants
def calculate_moving_average(df, column, round_column, window_size=3):
    moving_avg = []
    for current_round in df[round_column]:
        # Filtrer les rounds à inclure dans la moyenne
        valid_rows = df[(df[round_column] <= current_round) & (df[round_column] > current_round - window_size)]
        moving_avg.append(valid_rows[column].mean())
    return moving_avg

# Ajout de la colonne "moving_avg"
filtered_df["moving_avg"] = calculate_moving_average(filtered_df, "selected_stats", "ROUND")

print(filtered_df)