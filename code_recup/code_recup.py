import pandas as pd
import os
import sys
from euroleague_api.game_stats import GameStats

sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
image_dir = os.path.join(os.path.dirname(__file__), '..', 'images')

season = 2024
competition = "euroleague"
round_ = 13



f.recuperation_gs(competition,season,data_dir,round_=round_)
f.recuperation_pbp(competition,season,data_dir,round_=round_)
f.recup_idplayers(data_dir,season,competition)
f.recuperation_team_photo(competition,season,image_dir)
f.recup_idteam(data_dir,season,competition)
f.process_game_data(data_dir, competition, season)
f.process_dataframe(data_dir = data_dir, competition = competition, season = season, nb_groupe = 1)
f.process_dataframe(data_dir = data_dir, competition = competition, season = season, nb_groupe = 2)
f.process_dataframe(data_dir = data_dir, competition = competition, season = season, nb_groupe = 3)
f.calcul_per(data_dir = data_dir, season = season , competition = competition)
f.calcul_per_global(data_dir = data_dir, season = season , competition = competition)
f.df_images_players(competition,season,round_,data_dir)
f.recuperation_players_photo2(competition,season,image_dir,data_dir)
f.evol_score(data_dir,competition,season)

