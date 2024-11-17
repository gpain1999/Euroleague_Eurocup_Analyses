import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f

data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competition = "euroleague"

R = []
CODETEAM = []
num_players = 2
top = 10

res = f.analyse_io(data_dir = data_dir,
                   competition = competition,
                   season = season,
                   num_players = num_players,
                   R = R,
                   CODETEAM = CODETEAM)

# res = res[res["TIME_ON"]>=15].reset_index(drop = True)
res = res.sort_values(by = "PM_ON",ascending=False).reset_index(drop = True)
df = res.head(top).reset_index(drop = True)
titre = "BEST DUOS  +/- AFTER 10 ROUNDS"
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory
infographie_dir = os.path.join(os.path.dirname(__file__), '..', 'infographie')

f.create_infographie(df,titre,season,competition,images_dir,infographie_dir,name_image = f"BEST_PM_DUO_R1_R10",percent_time = True)

