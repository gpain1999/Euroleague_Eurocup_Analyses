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

def analyse_io_2(data_dir,competition,season,num_players,min_round,max_round,CODETEAM = [],selected_players = [],min_percent_in = 0) :
    conn = sqlite3.connect(':memory:')
    data_pm = pd.read_csv(os.path.join(data_dir, f"{competition}_io_{num_players}_{season}.csv"))


    if competition == "euroleague" :
        data_pm.insert(0,"R",(data_pm["Gamecode"] - 1 ) // 9 + 1 )
    else :
        data_pm.insert(0,"R",(data_pm["Gamecode"] - 1 ) // 10 + 1 )

    data_id = pd.read_csv(os.path.join(data_dir, f"{competition}_idplayers_{season}.csv"))

    data_pm.to_sql('data_pm', conn, index=False, if_exists='replace') 
    data_id.to_sql('data_id', conn, index=False, if_exists='replace')



    # Commencer à construire la requête SQL
    query = """
    SELECT t.CODETEAM as TEAM,
    """

    # Ajouter les jointures et les colonnes de joueurs dynamiquement
    joins = []
    for i in range(1, num_players + 1):
        query += f"t{i}.PLAYER as P{i},t{i}.PLAYER_ID as ID{i},\n"
        joins.append(f"LEFT JOIN data_id as t{i} on (t.CODETEAM = t{i}.CODETEAM) and (t.J{i} = t{i}.PLAYER_ID)")

    # Ajouter les agrégations
    query += """
    sum(t.TIME) as TIME_ON,
    sum(DELTA_SCORE) as PM_ON,
    ROUND(sum(OFF)/sum(TIME) *10,2) as OFF_ON_10,
    ROUND(sum(DEF)/sum(TIME) *10,2) as DEF_ON_10,

    ROUND(MAX(TEAM.TIME_TEAM) - sum(t.TIME),2) as TIME_OFF,
    MAX(TEAM.DELTA_SCORE_TEAM) - sum(DELTA_SCORE) as PM_OFF,
    ROUND((MAX(TEAM.OFF_TEAM) - sum(OFF)) / (MAX(TEAM.TIME_TEAM) - sum(t.TIME)) *10,2)  as OFF_OFF_10,
    ROUND((MAX(TEAM.DEF_TEAM) - sum(DEF)) / (MAX(TEAM.TIME_TEAM) - sum(t.TIME)) *10,2) as DEF_OFF_10,

    CAST(ROUND(MAX(TEAM.TIME_TEAM)) AS INTEGER) as TIME_TEAM,
    MAX(TEAM.DELTA_SCORE_TEAM) as PM_TEAM,
    ROUND(MAX(TEAM.OFF_TEAM)/MAX(TEAM.TIME_TEAM)*10,2) as OFF_TEAM_10,
    ROUND(MAX(TEAM.DEF_TEAM)/MAX(TEAM.TIME_TEAM)*10,2) as DEF_TEAM_10

    FROM data_pm as t
    """

    # Ajouter les jointures à la requête
    query += "\n".join(joins)

    # Ajouter le sous-requête pour les équipes
    query += f"""
    LEFT JOIN (SELECT CODETEAM,
    sum(TIME)/{math.comb(5, num_players)} as TIME_TEAM,
    sum(DELTA_SCORE)/{math.comb(5, num_players)} as DELTA_SCORE_TEAM,
    sum(OFF) /{math.comb(5, num_players)} as OFF_TEAM,
    sum(DEF) /{math.comb(5, num_players)} as DEF_TEAM

    from data_pm
    """

    query += f"WHERE R>= {min_round} and R <= {max_round}"

    query += f"""
    group by CODETEAM
    ORDER BY DELTA_SCORE_TEAM DESC) as TEAM on t.CODETEAM = TEAM.CODETEAM
    """

    # Terminer la requête avec les clauses GROUP BY et ORDER BY
    query += f"""
    WHERE 1=1
    """

    # Ajout conditionnel pour R
    R = [i for i in range(min_round,max_round +1)]
    if R:
        r_filter = ', '.join(map(str, R))
        query += f"AND t.R IN ({r_filter}) "

    # Ajout conditionnel pour CODETEAM
    if CODETEAM:
        code_filter = "', '".join(CODETEAM)  # Création de la chaîne pour la liste
        query += f"AND t.CODETEAM IN ('{code_filter}') "

    query += f"""
    GROUP BY t.CODETEAM, {', '.join([f't{i}.PLAYER' for i in range(1, num_players + 1)])}
    ORDER BY sum(DELTA_SCORE) DESC, sum(TIME)
    """

    # Exécution de la requête
    result = pd.read_sql_query(query, conn)
    conn.close()

    result = result.drop(columns=[f'ID{i}' for i in range(1,num_players+1)],axis = 1)

    result.insert(num_players+1,"PERCENT_IN",((result["TIME_ON"]/result["TIME_TEAM"]*100).round(0)).astype(int))
    result.insert(num_players+6,"PERCENT_OUT",((result["TIME_OFF"]/result["TIME_TEAM"]*100).round(0)).astype(int))

    if selected_players != [] :
        result = result[result[[f"P{i+1}" for i in range(num_players)]].apply(lambda row: all(joueur in row.values for joueur in selected_players), axis=1)]

    result = result[result["PERCENT_IN"]>=min_percent_in]
    return result.reset_index(drop = True)



res = analyse_io_2(data_dir = data_dir,
                   competition = competition,
                   season = season,
                   num_players = 2,
                    min_round = 1,
                    max_round = 10,
                   CODETEAM = ["MCO"],
                   selected_players = ["JAMES, MIKE"],
                   min_percent_in = 12)

print(res)