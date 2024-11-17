import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f


data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competition = "eurocup"
nb_round = 4

R = [i+1 for i in range(nb_round)]
CODETEAM = []
num_players = 2
top = 10
t = "DIFF_" + "_".join(CODETEAM)
t = "top"

res = f.analyse_io(data_dir = data_dir,competition = competition,season = season,num_players = num_players,R = R,CODETEAM = CODETEAM)
res = res[res["TIME_ON"]>=15]
res = res.sort_values(by = "PM_ON",ascending=False).reset_index(drop = True)
print(res.columns)
df = res.head(top)

titre = f"BEST +/-  after {nb_round} ROUNDS (MIN 15MIN)"
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory
infographie_dir = os.path.join(os.path.dirname(__file__), '..', 'infographie')

f.create_infographie(df,titre,season,competition,images_dir,infographie_dir,name_image = f"{competition}/{t}_{top}_pm_{competition}_{season}_round_{nb_round}_IO_{num_players}")