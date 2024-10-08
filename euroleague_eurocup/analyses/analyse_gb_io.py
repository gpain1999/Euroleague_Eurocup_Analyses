import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competition = "euroleague"
io = 2

res = f.io_plus_minus_analysis(data_dir = data_dir,season = season,competition = competition,io = io,sortby ="PM_IN",teamcode = "MCO")

print(res.head(20))
print(res.tail(20))

#meta base