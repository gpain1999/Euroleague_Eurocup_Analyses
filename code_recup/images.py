from euroleague_api.player_stats   import PlayerStats
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

image_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
competition = "euroleague"
season = 2024

if competition == "euroleague" :
    competition_code = "E"
else :
    competition_code = "U"

ps = PlayerStats(competition_code)

df = ps.get_player_stats_single_season(endpoint = "traditional",season =  season ,phase_type_code = "RS",statistic_mode = "PerGame")

df_utile = df[["player.code","player.imageUrl","player.team.code","player.team.imageUrl"]]

df_utile.columns = ["ID_PLAYERS","URL_PLAYERS","ID_TEAM","URL_TEAM"]




for index, row in df_utile.iterrows():

    image_player_link = os.path.join(image_dir, f'{competition}_{season}\{row["ID_PLAYERS"]}.png')
    url_player = row["URL_PLAYERS"]
    response = requests.get(url_player)
    # Vérifier que la requête est réussie (code 200)
    if response.status_code == 200:
        # Enregistrer l'image sur le disque
        with open(image_player_link, "wb") as file:
            file.write(response.content)
        print("Image téléchargée avec succès.")
    else:
        print("Échec du téléchargement. Code d'erreur :", response.status_code)
