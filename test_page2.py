import pandas as pd
import sys
import os
import fonctions as f
import math
import sqlite3

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))




res = f.analyse_io_2(data_dir = data_dir,
                   competition = competition,
                   season = season,
                   num_players = 2,
                    min_round = 1,
                    max_round = 10,
                   CODETEAM = ["MCO"],
                   selected_players = ["JAMES, MIKE"],
                   min_percent_in = 10)

print(res)