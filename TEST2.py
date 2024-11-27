import pandas as pd
import sys
import os
import numpy as np
from tabulate import tabulate
import math

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))
import fonctions as f




evol_score(data_dir,competition,season)