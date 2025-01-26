from PIL import Image, ImageOps,ImageDraw,ImageFont
import pandas as pd
import os

competition = "euroleague"
season = 2024
width = 850 - 60
height = round(850 * 88/63) - 60

font_path_bold = "police_ecriture/GothamBoldItalic.ttf" 
text_color = "black"
border_size = 80
dossier_path = f'images/{competition}_{season}_players'  # Remplacez par le chemin de votre dossier
ct = pd.read_csv("panini_card/color_team.csv")
if competition == "euroleague" :
    background_color = "#FEB673"
    border_color = "#E9540D"  # Color code for the background
else : 
    background_color = "#87C2FA"
    border_color = "#0971CE"

# Liste pour stocker les noms des fichiers PNG sans l'extension
png_files = []

# Parcourir le dossier et ajouter les noms des fichiers PNG à la liste
for filename in os.listdir(dossier_path):
    if filename.endswith('.png'):
        # Retirer l'extension .png et ajouter à la liste
        name_without_extension = filename[:-4]  # Retire les 4 derniers caractères ('.png')
        png_files.append(name_without_extension)

data = pd.read_csv(f"datas/{competition}_{season}_df_images_players.csv")


folder_path = f"panini_card/{competition}_{season}"

# Vérifier si le dossier existe, sinon le créer
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")
    

for index, row in data.iterrows():

    id_players = row["PLAYER_ID"]
    name_players = row["NAME"]
    name_players = f"{name_players.split(', ')[1]}  {name_players.split(', ')[0]}"
    team = row["TEAM"]
    pays = row["NAT"]
    if f"{team}_{id_players}" in png_files :



        panini_image = Image.new("RGB", (width, height), background_color)

        # Définir les coordonnées du bandeau haut
        x0, y0 = 0,  0
        x1, y1 = width, 200

        draw = ImageDraw.Draw(panini_image)

        # Dessiner le bandeau
        draw.rectangle([x0, y0, x1, y1], fill="#E59D63")

        #PLAYER
        player_image_path = f"images/{competition}_{season}_players/{team}_{id_players}.png"
        player_image = Image.open(player_image_path)
        player_image = player_image.resize((770,1027), Image.Resampling.LANCZOS)
        
        player_image = player_image.convert("RGBA")


        panini_image.paste(player_image, (10, 50), mask=player_image)



        # Définir les coordonnées du bandeau bas
        x0, y0 = 0, height - 200 + 50
        x1, y1 = width, height + 50

        # Dessiner le bandeau
        draw.rectangle([x0, y0, x1, y1 + 100], fill="#E59D63")

        #NOM DU JOUEUR
        font_size = 85
        text_width = width 
        text_height = height
        while (text_width > width - 50) | (text_height>80) :
            bbox = draw.textbbox((0, 0), name_players, font=ImageFont.truetype(font_path_bold, font_size))  # Obtenir les coordonnées de la boîte
            text_width = bbox[2] - bbox[0]  # Largeur = x_max - x_min
            text_height = bbox[3] - bbox[1]  # Hauteur = y_max - y_min
            font_size -= 1


        # Calculer la position x pour centrer le texte
        text_position_x = (panini_image.width - text_width) // 2  # Position x centrée
        text_position_y = 977 + round((150-text_height)/2)  # Position y centrée

        draw.text((text_position_x, text_position_y), name_players, fill=text_color, font=ImageFont.truetype(font_path_bold, font_size))
        bbox = draw.textbbox((0, 0), f"{season}-{season+1}", font=ImageFont.truetype(font_path_bold, 35))
        text_width = bbox[2] - bbox[0]  # Largeur = x_max - x_min
        text_height = bbox[3] - bbox[1]  # Hauteur = y_max - y_min
        draw.text(((panini_image.width - text_width) // 2, 50), f"{season}-{season+1}", fill=text_color, font=ImageFont.truetype(font_path_bold, 35))



        #TEAM
        team_image_path = f"images/{competition}_{season}_teams/{team}.png"
        team_image = Image.open(team_image_path)
        # Redimensionner l'image
        new_size = (160,160)
        team_image = team_image.resize(new_size, Image.Resampling.LANCZOS)
        # Convertir en RGBA pour gérer la transparence
        team_image = team_image.convert("RGBA")
        # Coller l'image avec fond tout en préservant la transparence
        panini_image.paste(team_image, (40, 40), mask=team_image)


        #COMPETITON
        competition_image_path = f"images/{competition}_little.png"
        competition_image = Image.open(competition_image_path)
        new_size = (160,160)
        competition_image = competition_image.resize(new_size, Image.Resampling.LANCZOS)
        competition_image = competition_image.convert("RGBA")

        panini_image.paste(competition_image, (width - 200, 40), mask=competition_image)






        # BORDURE
        # Créer l'image avec le contour
        color = ct.loc[ct["TEAM"] == team, "COLOR"].values

        panini_image = ImageOps.expand(panini_image, border=int(border_size*1/5), fill=color[0])

        panini_image = ImageOps.expand(panini_image, border=int(border_size*4/5), fill=border_color)

        panini_image.save(f"{folder_path}/{team}_card_{id_players}.png")