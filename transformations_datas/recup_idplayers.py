import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f

data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competiton = "euroleague"

f.recup_idplayers(data_dir,season,competiton)


