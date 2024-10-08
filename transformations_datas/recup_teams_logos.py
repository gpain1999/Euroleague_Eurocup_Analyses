import pandas as pd
import os
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')

season = 2024
competition = "euroleague"

_gs = pd.read_csv(os.path.join(data_dir, f'{competition}_gs_{season}.csv'))

#EUROCUP

road_ = _gs[["road.club.code","road.club.images.crest"]].drop_duplicates().reset_index(drop=True)
road_.columns = ["code","link"]

local_ = _gs[["local.club.code","local.club.images.crest"]].drop_duplicates().reset_index(drop=True)
local_.columns = ["code","link"]

ec = pd.concat([road_,local_])[["code","link"]].drop_duplicates().reset_index(drop=True)

# Chemin du dossier à créer
directory = f"../images/logo_{competition}_{season}"

# Créer le dossier s'il n'existe pas déjà
if not os.path.exists(directory):
    os.makedirs(directory)