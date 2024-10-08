#################################################
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#################################################
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competition = "euroleague"
#################################################
f.recup_io(data_dir = data_dir,season = season,competition = competition,nb_io = 1)
f.recup_io(data_dir = data_dir,season = season,competition = competition,nb_io = 2)
f.recup_io(data_dir = data_dir,season = season,competition = competition,nb_io = 3)

