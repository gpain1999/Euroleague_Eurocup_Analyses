import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
import fonctions as f

from tabulate import tabulate

data_dir = os.path.join(os.path.dirname(__file__), '..', 'datas')
season = 2024
competition = "euroleague"
R = []
CODETEAM = []
num_players = 2


# ['TEAM', 'P1', 'ID1', 'TIME_ON', 'PM_ON', 'OFF_ON_10', 'DEF_ON_10',
#        'TIME_OFF', 'PM_OFF', 'OFF_OFF_10', 'DEF_OFF_10', 'TIME_TEAM',
#        'PM_TEAM', 'OFF_TEAM_10', 'DEF_TEAM_10', 'TIME_PERC']

res = f.analyse_io(data_dir = data_dir,
                   competition = competition,
                   season = season,
                   num_players = num_players,
                   R = R,
                   CODETEAM = CODETEAM)

res["TIME_PERC_ON"] =( res["TIME_ON"] *100 / res["TIME_TEAM"]).round(1)
res["TIME_PERC_OFF"] =( res["TIME_OFF"] *100 / res["TIME_TEAM"]).round(1)
res["TIME_ON"] = (res["TIME_ON"]).round(1)

res = res.drop(columns = ["ID1","ID2",'TIME_TEAM'],axis = 1)
res = res[res["TIME_PERC_ON"]>=10]
f.afficher_dataframe(res)

# res = res[["TEAM","P1","P2","TIME_PERC_ON","PM_ON","TIME_PERC_OFF",'PM_OFF']].sort_values(by = ["PM_ON"],ascending = [False]).reset_index(drop = True)

# per = f.analyse_per(data_dir = data_dir,
#                     competition = competition,
#                     season = season,
#                     R = R,
#                     CODETEAM = CODETEAM)



# df = f.calcul_per2(data_dir, season, competition)
# df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
# df["I_PER"] = df["I_PER"].round(2)
# df["NB_GAME"] = 1
# df = df[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
# 'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
# '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]


# d = f.get_aggregated_data(df, 9, 9, selected_opponents=None,selected_fields = ["TEAM","PLAYER"],mode="CUMULATED",percent = "MADE")

