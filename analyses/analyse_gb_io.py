import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competition = "eurocup"
io = 3

res = f.io_plus_minus_analysis(data_dir = data_dir,season = season,competition = competition,io = io,sortby ="PM_IN")

print(res.head(50))
print(res.tail(50))

#meta base