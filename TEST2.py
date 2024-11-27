import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate
import math
import itertools

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))
import fonctions as f

team = "MCO"
min_round = 1
max_round = 11



print(team_evol_score(team,min_round,max_round))