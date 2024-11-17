import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')

f.recup_idplayers(data_dir,season,competition)
print("ok")
f.recup_io(data_dir = data_dir,season = season,competition = competition,nb_io = 1)
print("ok")
f.recup_io(data_dir = data_dir,season = season,competition = competition,nb_io = 2)
print("ok")
f.recup_io(data_dir = data_dir,season = season,competition = competition,nb_io = 3)