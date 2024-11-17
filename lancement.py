import pandas as pd
import sys
import os
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import webbrowser

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), 'datas')
sys.path.append(os.path.join(os.path.dirname(__file__), 'fonctions'))

import fonctions as f

# Créer l'application Flask
app = Flask(__name__)

# Template HTML avec titre personnalisé pour la saison et la compétition, affichage des logos des équipes et des adversaires
# Dans la variable HTML_TEMPLATE
HTML_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <link rel="icon" href="{{ url_for('static', filename='images/' + competition + '_little.png') }}" type="image/png">
    <title>GP DATA</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <style>
        /* Base Light Theme Styles */
        body {
            background-color: #ffffff;
            color: #000000;
            transition: background-color 0.3s, color 0.3s;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            table-layout: auto;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: center;
            padding: 8px;
            word-wrap: break-word;
        }
        th {
            background-color: #f2f2f2;
            font-size: 12px;
        }
        td {
            font-size: 12px;
        }
        #teamFilter, #opponentFilter {
            margin: 10px 0;
        }
        #teamFilter label, #opponentFilter label {
            display: inline-block;
            margin-right: 15px;
        }
        #logo {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 400px;
            height: auto;
        }

        /* Dark Mode Styles */
        body.dark-mode {
            background-color: #4f4f4f;
            color: #ffffff;
        }
        .dark-mode th {
            background-color: #333333;
        }
        .dark-mode td, .dark-mode th {
            border-color: #444444;
        }
        .dark-mode #teamFilter label, .dark-mode #opponentFilter label {
            color: #ffffff;
        }

        /* Dark Mode Toggle Button */
        .dark-mode-toggle {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 100;
            padding: 10px;
            cursor: pointer;
            font-size: 16px;
            background-color: #f0f0f0;
            color: #000;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .dark-mode .dark-mode-toggle {
            background-color: #444;
            color: #fff;
        }

        #filterContainer {
    display: flex;
    align-items: center;
    gap: 20px;
}

.rectangle-box {
    border: 1px solid #cccccc;    /* Bordure grise légère */
    padding: 10px 15px;           /* Espacement interne pour créer le rectangle */
    border-radius: 8px;           /* Coins arrondis */
    background-color: #f9f9f9;    /* Fond léger */
    box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1); /* Ombre douce pour effet de profondeur */
}

#fieldSelection p {
    margin: 0;
    padding-bottom: 5px;
    font-weight: bold;
}

/* Styles de base pour le conteneur */
#filterContainer {
    display: flex;
    align-items: center;
    gap: 20px;
}

/* Styles pour les rectangles */
.rectangle-box {
    border: 1px solid #cccccc;
    padding: 10px 15px;
    border-radius: 8px;
    background-color: #f9f9f9;
    box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s, border-color 0.3s, box-shadow 0.3s; /* Transition douce */
}

/* Styles pour le mode sombre */
.dark-mode .rectangle-box {
    background-color: #333333;        /* Fond sombre */
    border-color: #555555;            /* Bordure sombre */
    box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.5); /* Ombre plus prononcée */
}

#fieldSelection p {
    margin: 0;
    padding-bottom: 5px;
    font-weight: bold;
}

.dark-mode #fieldSelection p, 
.dark-mode .rectangle-box label, 
.dark-mode .rectangle-box input[type="radio"], 
.dark-mode .rectangle-box input[type="checkbox"] {
    color: #ffffff; /* Texte en blanc en mode sombre */
}

#toggleContainer {
    margin-bottom: 20px; /* Ajoute un espacement sous le bouton pour séparer les lignes */
}

#roundSelectors {
    display: block; /* Force le conteneur de round à occuper toute la ligne */
    margin-top: 10px;
}


    </style>
<script>
    $(document).ready(function() {
        // Initialize DataTable
        var table = $('#dataframe').DataTable({
            responsive: true,
            autoWidth: false,
            pageLength: 50,
            ordering: true,
            order: [[8, 'desc']]
        });

        // Dark Mode Toggle
        const darkModeToggle = document.createElement("div");
        darkModeToggle.className = "dark-mode-toggle";
        darkModeToggle.innerText = "Dark Mode";
        document.body.appendChild(darkModeToggle);
        
        darkModeToggle.addEventListener("click", function() {
            document.body.classList.toggle("dark-mode");
            darkModeToggle.innerText = document.body.classList.contains("dark-mode") ? "Light Mode" : "Dark Mode";
        });

        var selectedPlayer = null; // Initialize selected player
        var selectedTeam = null;
        var selectedOpponent = null;
        var selectedFields = [];

        var playerColumnIndex = table.column(':contains("PLAYER")').index();

        // Function to apply filters based on selected player and team
        function applyFilters() {
            // Apply player filter if a player is selected, otherwise clear
            if (selectedPlayer) {
                table.column(playerColumnIndex).search('^' + selectedPlayer + '$', true, false).draw(); // Exact match for PLAYER column
            } else {
                table.column(playerColumnIndex).search('').draw(); // Clear the filter if no player is selected
            }

            // Apply team filter if teams are selected
            if (selectedTeam) {
                table.column(2).search(selectedTeam.join('|'), true, false);
            } else {
                table.column(2).search('');
            }

            table.draw(); // Refresh the table
        }

        // Update fetch function to use selected fields
        function fetchAndRenderData() {
            var minRound = parseInt($('#minRound').val(), 10);
            var maxRound = parseInt($('#maxRound').val(), 10);
            var order = table.order();
            var mode = $('input[name="mode"]:checked').val();  // Récupère CUMULATED ou AVERAGE
            var percent = $('input[name="percent"]:checked').val();
            $.ajax({
                url: "/get_data",
                type: "POST",
                data: JSON.stringify({ 
                    minRound: minRound, 
                    maxRound: maxRound, 
                    selectedOpponent: selectedOpponent,
                    selectedFields: selectedFields,
                    mode : mode,
                    percent : percent
                }),
                contentType: "application/json",
                success: function(response) {
                    table.destroy();
                    table.search('').columns().search('').draw();
                    var headerRow = $('#dataframe thead tr');
                    headerRow.empty();
                    response.columns.forEach(function(column) {
                        headerRow.append('<th>' + column + '</th>');
                    });

                    table = $('#dataframe').DataTable({
                        responsive: true,
                        autoWidth: false,
                        pageLength: 50,
                        ordering: true,
                        order: [[6, 'desc']],
                        data: response.data,
                        columns: response.columns.map(name => ({ title: name })),
                        language: {
                            search: "Rechercher un joueur" // Custom search label
                        }
                    });

                    playerColumnIndex = table.column(':contains("PLAYER")').index();

                    table.order(order).draw();
                    applyFilters();
                },
                error: function(error) {
                    console.error("Error fetching data:", error);
                }
            });
        }

        // Listen for changes to the selected fields checkboxes
        $('.field-checkbox').on('change', function() {
            // Update selectedFields array with the values of checked checkboxes
            selectedFields = $('.field-checkbox:checked').map(function() {
                return $(this).val();
            }).get();
            
            console.log("Selected fields updated:", selectedFields); // Debugging to confirm selected fields update
            
            // Fetch and render data whenever selected fields change
            fetchAndRenderData();
        });

        // Fetch data initially on page load
        fetchAndRenderData();

        // Listen for round input changes and fetch data
        $('#minRound, #maxRound').on('input', function() {
            fetchAndRenderData();
        });

        $('#minRound').on('input', function() {
            $('#minRoundValue').text($(this).val());
        });
        $('#maxRound').on('input', function() {
            $('#maxRoundValue').text($(this).val());
        });

        // Event listener for PLAYER column clicks to filter by player
        $('#dataframe tbody').on('click', 'td:nth-child(' + (playerColumnIndex + 1) + ')', function() {
            var player = $(this).text();
            selectedPlayer = selectedPlayer === player ? null : player; // Toggle between selected player and null
            applyFilters();
        });

        // Event listener for TEAM column clicks
        $('#dataframe tbody').on('click', 'td:nth-child(2)', function() {
            var team = $(this).text();
            selectedTeam = selectedTeam === team ? null : team;
            applyFilters();
        });

        // Filter by team and opponent selection
        $('#teamFilter input').change(function() {
            selectedTeam = [];
            $('#teamFilter input:checked').each(function() {
                selectedTeam.push($(this).val());
            });
            applyFilters();
        });
        $('input[name="mode"]').on('change', function() {
            fetchAndRenderData();  // Recharge les données au changement de mode
        });

        $('input[name="percent"]').on('change', function() {
            fetchAndRenderData();  // Recharge les données au changement de percent
        });


            $('#toggleOpponentFilter').on('click', function() {
        var opponentFilter = $('#opponentFilter');
        if (opponentFilter.css('display') === 'none') {
            opponentFilter.show();
        } else {
            opponentFilter.hide();
        }
    });



        $('#opponentFilter input').change(function() {
            selectedOpponent = [];
            $('#opponentFilter input:checked').each(function() {
                selectedOpponent.push($(this).val());
            });
            fetchAndRenderData(); // Update data based on selectedOpponent
        });
    });

    // Shut down server when page is closed
    window.addEventListener("beforeunload", function(event) {
        navigator.sendBeacon("/shutdown");
    });

    
</script>
</head>
<body>
    <img id="logo" src="{{ url_for('static', filename='images/' + competition + '.png') }}" alt="Logo de la compétition">

    <h2 style="text-align: center; font-size: 2em; font-weight: bold;">
        Statistiques de la saison {{ season }} - {{ season + 1 }} de l'{{ competition }}
    </h2>

<div id="filterContainer" style="display: flex; align-items: center; gap: 20px;">
    <div class="rectangle-box">
    <div id="aggregationMode" class="rectangle-box">
        <label><input type="radio" name="mode" value="CUMULATED" checked> CUMULATED</label>
        <label><input type="radio" name="mode" value="AVERAGE"> AVERAGE</label>
        </div>

        <div id="percentMode" class="rectangle-box" style="margin-top: 10px;">
            <label><input type="radio" name="percent" value="MADE" checked> MADE</label>
            <label><input type="radio" name="percent" value="PERCENT"> PERCENT</label>
        </div>
    </div>

    <div id="fieldSelection" class="rectangle-box">
        <p>Regroupement :</p>
        <label><input type="checkbox" class="field-checkbox" value="TEAM" checked> TEAM</label>
        <label><input type="checkbox" class="field-checkbox" value="PLAYER" checked> PLAYER</label>
        <label><input type="checkbox" class="field-checkbox" value="ROUND" checked> ROUND</label>
        <label><input type="checkbox" class="field-checkbox" value="OPPONENT"> OPPONENT</label>
    </div>

    <div id="roundSelectors" class="rectangle-box">
        <label for="minRound">ROUND minimum : <span id="minRoundValue">1</span></label>
        <input type="range" id="minRound" min="1" max="{{ max_round }}" value="1" />

        <label for="maxRound">ROUND maximum : <span id="maxRoundValue">{{ max_round }}</span></label>
        <input type="range" id="maxRound" min="1" max="{{ max_round }}" value="{{ max_round }}" />
    </div>
</div>




    <div id="teamFilter">
        <p>Equipes des joueurs :</p>
        {% for team in teams %}
            <label>
                <input type="checkbox" value="{{ team }}">
                <img src="{{ url_for('static', filename='images/' +competition ~ '_' ~ season ~ '_teams/' ~ team ~ '.png') }}" 
                     alt="{{ team }}" width="50" height="50">
            </label>
        {% endfor %}
    </div>
<div id="toggleContainer">
    <button id="toggleOpponentFilter">Afficher/Masquer les Adversaires</button>
    <div id="opponentFilter">
        <p>Adversaires des joueurs:</p>
        {% for opponent in opponents %}
            <label>
                <input type="checkbox" value="{{ opponent }}">
                <img src="{{ url_for('static', filename='images/' + competition ~ '_' ~ season ~ '_teams/' ~ opponent ~ '.png') }}" 
                     alt="{{ opponent }}" width="50" height="50">
            </label>
        {% endfor %}
    </div>
</div>



    <table id="dataframe">
        <thead>
            <tr>
                {% for column in columns %}
                    <th>{{ column }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
                <tr>
                    {% for item in row %}
                        <td>{{ item }}</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""


def get_aggregated_data(df, min_round, max_round, selected_opponents=None,selected_fields = None,mode="CUMULATED",percent = "MADE"):
    selected_fields2 = selected_fields
    if len(selected_fields2) == 0 :
        selected_fields2 = ["PLAYER","ROUND","TEAM","OPPONENT"]

    df = df.drop(columns=["I_PER"])
    df_filtered = df[(df['ROUND'] >= min_round) & (df['ROUND'] <= max_round)].reset_index(drop=True)
    
    if selected_opponents:
        df_filtered = df_filtered[df_filtered['OPPONENT'].isin(selected_opponents)]

    stats = ["TIME_ON"   , "PER" , "PM_ON","PTS",   "DR" ,  "OR" ,  "TR"  , "AS"  , "ST" , "CO" , "1_R" , "1_T" , "2_R" , "2_T" , "3_R" , "3_T" ,  "TO" ,  "FP"  , "CF" , "NCF"]
    aggregation_functions = {
        'TIME_ON': 'sum', 'PER': 'sum', 'PM_ON': 'sum', 'PTS': 'sum', 'DR': 'sum', 'OR': 'sum', 'TR': 'sum',
        'AS': 'sum', 'ST': 'sum', 'CO': 'sum', '1_R': 'sum', '1_T': 'sum', '2_R': 'sum', '2_T': 'sum',
        '3_R': 'sum', '3_T': 'sum', 'TO': 'sum', 'FP': 'sum', 'CF': 'sum', 'NCF': 'sum'
    }

    df_team = df_filtered.groupby(['ROUND','TEAM', 'OPPONENT', 'HOME', 'WIN']).agg(aggregation_functions).reset_index()
    df_team["TIME_ON"] =( (df_team["TIME_ON"]).round(0)).astype(int)
    df_team["PM_ON"] = (df_team["PM_ON"]/5).astype(int)


    if selected_fields2 == ["ROUND"] : 

        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values / 2).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/10).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/10).round(0)).astype(int)


    if selected_fields2 == ["TEAM"] :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if selected_fields2 == ["OPPONENT"] :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if selected_fields2 == ["PLAYER"] :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","TEAM"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["OPPONENT"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"OPPONENT","TEAM"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"ROUND","OPPONENT"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["TEAM"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"ROUND","PLAYER"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["OPPONENT","NUMBER","TEAM"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)


    if set(selected_fields2) == {"TEAM","PLAYER"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)


    if set(selected_fields2) == {"OPPONENT","PLAYER"} and len(selected_fields2) == 2 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","PLAYER","TEAM"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2+["OPPONENT","NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","PLAYER","OPPONENT"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2+["TEAM","NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"TEAM","PLAYER","OPPONENT"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if set(selected_fields2) == {"ROUND","OPPONENT","TEAM"} and len(selected_fields2) == 3 :
        df_grouped = df_filtered.groupby(selected_fields2).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_team.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_team.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_team.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =( (df_grouped["TIME_ON"]/5).round(0)).astype(int)
        df_grouped["PM_ON"] =( (df_grouped["PM_ON"]/5).round(0)).astype(int)

    if set(selected_fields2) == {"ROUND","PLAYER","TEAM","OPPONENT"} and len(selected_fields2) == 4 :
        df_grouped = df_filtered.groupby(selected_fields2+["NUMBER"]).agg(aggregation_functions).reset_index()
        df_grouped["NB_GAME"] = ( (df_filtered.groupby(selected_fields2).size().values).round(0)).astype(int) 
        df_grouped["HOME"] = df_filtered.groupby(selected_fields2)['HOME'].mean().values
        df_grouped["WIN"] = df_filtered.groupby(selected_fields2)['WIN'].mean().values
        df_grouped["TIME_ON"] =(df_grouped["TIME_ON"]).round(2)

    if mode == "AVERAGE" : 
        for s in stats :
            df_grouped[s] = (df_grouped[s]/df_grouped["NB_GAME"]).round(1)

    if (df_grouped['NB_GAME'] == 1).all():
        # If NB_GAME is entirely filled with 1, map HOME and WIN to 'YES'/'NO'
        df_grouped['HOME'] = df_grouped['HOME'].map({1: 'YES', 0: 'NO'})
        df_grouped['WIN'] = df_grouped['WIN'].map({1: 'YES', 0: 'NO'})
    else:
        # Otherwise, convert HOME and WIN to percentages
        df_grouped["HOME"] = (df_grouped["HOME"] * 100).round(0).astype(int).astype(str) + '%'
        df_grouped["WIN"] = (df_grouped["WIN"] * 100).round(0).astype(int).astype(str) + '%'


    df_grouped["I_PER"] = (df_grouped["PER"] / df_grouped["TIME_ON"] ** 0.5).round(1)
    df_grouped["PER"] = (df_grouped["PER"]).round(1)

    # Define the required columns
    required_columns = [
        'ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
        'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF'
    ]

    # Adding missing columns and filling with "---"
    for col in required_columns:
        if col not in df_grouped.columns:
            df_grouped[col] = "---"

    if percent == "PERCENT" : 
        df_grouped["1_R"] = (df_grouped["1_R"] / df_grouped["1_T"] * 100).round(1).fillna(0).astype(float)
        df_grouped["2_R"] = (df_grouped["2_R"] / df_grouped["2_T"] * 100).round(1).fillna(0).astype(float)
        df_grouped["3_R"] = (df_grouped["3_R"] / df_grouped["3_T"] * 100).round(1).fillna(0).astype(float)

        df_grouped.rename(columns={"1_R": '1_P', "2_R": '2_P', "3_R": '3_P'}, inplace=True)

        return df_grouped[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
        'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_P', '1_T', '2_P', '2_T', '3_P', '3_T', 'TO', 'FP', 'CF', 'NCF']]
    
    else :

        return df_grouped[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
        'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
        '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]



@app.route('/get_data', methods=['POST'])
def get_data():
    params = request.get_json()
    min_round = params.get('minRound', 1)
    max_round = params.get('maxRound', df['ROUND'].max())
    selected_opponents = params.get('selectedOpponent', None)
    selected_fields = params.get('selectedFields', ['TEAM', 'PLAYER'])
    mode = params.get('mode', 'CUMULATED')
    percent = params.get('percent', 'MADE')  # Utilisation de percent
    
    # Extraction des données filtrées
    filtered_data = get_aggregated_data(df, min_round, max_round, selected_opponents, selected_fields, mode, percent)
    
    return jsonify({"data": filtered_data.values.tolist(), "columns": filtered_data.columns.tolist()})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    print("Server shutting down...")
    os._exit(0)  # Force termination of the server process

def afficher_dataframe(df, competition, season):
    @app.route('/', methods=['GET'])
    def index():
        teams = sorted(df['TEAM'].unique())
        opponents = sorted(df['OPPONENT'].unique())  # Sort opponents alphabetically
        max_round = df['ROUND'].max()
        return render_template_string(
            HTML_TEMPLATE,
            columns=df.columns,
            data=df.values,
            teams=teams,
            opponents=opponents,  # Pass alphabetically sorted opponents to template
            max_round=max_round,
            competition=competition,
            season=season
        )

    app.run(debug=True)


if __name__ == '__main__':
    df = f.calcul_per2(data_dir, season, competition)
    df.insert(6, "I_PER", df["PER"] / df["TIME_ON"] ** 0.5)
    df["I_PER"] = df["I_PER"].round(2)
    df["NB_GAME"] = 1
    df = df[['ROUND','NB_GAME', 'TEAM', 'OPPONENT', 'HOME', 'WIN', 'NUMBER', 'PLAYER',
    'TIME_ON',"I_PER", 'PER', 'PM_ON', 'PTS', 'DR', 'OR', 'TR', 'AS', 'ST', 'CO',
    '1_R', '1_T', '2_R', '2_T', '3_R', '3_T', 'TO', 'FP', 'CF', 'NCF']]
    
    # Open the default web browser only if not already opened
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000")
    
    # Start the Flask app
    afficher_dataframe(df, competition, season)

