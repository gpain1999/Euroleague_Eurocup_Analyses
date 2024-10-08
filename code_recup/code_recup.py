import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
image_dir = os.path.join(os.path.dirname(__file__), '..', 'images')


season = 2024
competition = "eurocup"


f.recuperation_gs(competition,season,data_dir,first_gamecode=21, last_gamecode=30)
f.recuperation_pbp(competition,season,data_dir,first_gamecode=21, last_gamecode=30)
f.recuperation_players_photo(competition,season,image_dir)