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



df = f.calcul_per2(data_dir, season, competition)
df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
df["I_PER"] = df["I_PER"].round(2)
df["NB_GAME"] = 1
df = df[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
'1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


d = f.get_aggregated_data(df, 1, 10, selected_opponents=None,selected_fields = ["OPPONENT"],mode="AVERAGE",percent = "PERCENT")


# print(d[(d["WIN"]>="60%")&(d["NB_GAME"]>=6)].sort_values(by="I_PER",ascending = False).head(10).reset_index(drop = True))

#d1 =  d[d["TEAM"].isin(["PAR","MIL"])]
f.afficher_dataframe(d)


#print(d.sort_values(by="I_PER",ascending = False).head(50).reset_index(drop = True))


# for t in ["ASV","PRS","MCO","BAR","MAD",'BAS',"MUN","BER","VIR","MIL","OLY","PAN","RED","PAR","ZAL","IST","ULK","TEL"] :

#     d1 = d[d["TEAM"].isin([t])]
#     print(t)
#     print("---")
#     print(d1.sort_values(by="I_PER",ascending = False).head(20).reset_index(drop = True))
#     #print(d1[(d1["WIN"]>="60%")&(d1["NB_GAME"]>=6)].sort_values(by="I_PER",ascending = False).head(3).reset_index(drop = True))
#     print()
#     print("-----------------------------------------------------------")
#     print()

