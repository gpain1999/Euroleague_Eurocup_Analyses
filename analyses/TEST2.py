import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate

season = 2024
competition = "euroleague"
round_to = 8
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))
import fonctions as f


f.analyse_per(data_dir,competition,season,R = [i for i in range(1,13)],CODETEAM = [])