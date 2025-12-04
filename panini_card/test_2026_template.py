from PIL import Image,ImageDraw
import numpy as np
import os
import pandas as pd
# -----------------------------
# PARAMÈTRES GÉNÉRAUX
# -----------------------------
W, H = 850, 1200

#A
# center_color = (255, 229, 3)   
# edge_color = (3, 149, 68)       
#B
# center_color = (28, 165, 176)   
# edge_color = (46, 6, 157) 
#C
# center_color = (90, 8, 152)   
# edge_color = (129, 6, 19) 

#D
center_color = (233, 53, 12)   
edge_color = (250, 184, 1) 

competition = "euroleague"      # exemple (doit exister sous images/euroleague_little.png)
season = 2025
team = "IST"
data = pd.read_csv(f"datas/{competition}_{season}_df_images_players.csv")

row = data.iloc[0]  # Exemple : première ligne du DataFrame


id_players = row["PLAYER_ID"]
name_players = row["NAME"]
name_players = f"{name_players.split(', ')[1]}  {name_players.split(', ')[0]}"
team = row["TEAM"]
team_long = "ANADOLU EFES"
pays = row["NAT"]

# -----------------------------
# FOND DÉGRADÉ RADIAL
# -----------------------------
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
xx, yy = np.meshgrid(x, y)
dist = np.sqrt(xx**2 + yy**2)
dist = np.clip(dist / dist.max(), 0, 1)
dist = dist[..., None]
grad = (np.array(center_color) * (1 - dist) + np.array(edge_color) * dist).astype(np.uint8)
panini_image = Image.fromarray(grad, mode="RGB").convert("RGBA")

# -----------------------------
# AJOUT DU LOGO DE COMPÉTITION
# -----------------------------




# TEAM
team_image_path = f"images/{competition}_{season}_teams/{team}.png"
team_image = Image.open(team_image_path).convert("RGBA")
team_image = team_image.resize((250, 250), Image.Resampling.LANCZOS)
team_position = (60, 120)
panini_image.paste(team_image, team_position, mask=team_image)

# COMPETITION
competition_image_path = f"images/{competition}_little.png"
competition_image = Image.open(competition_image_path).convert("RGBA")
competition_image = competition_image.resize((180, 180), Image.Resampling.LANCZOS)
competition_position = (620, 140)
panini_image.paste(competition_image, competition_position, mask=competition_image)

# ========================
# 📅 SAISON (haut)
# ========================
from PIL import ImageDraw, ImageFont
font_path_bold = "fonts/YourFont-Bold.ttf"
font_path_bold = "police_ecriture/GothamBoldItalic.ttf" 

# Texte de la saison
season_text = f"{season}-{season+1}"

# Police et couleur
font_season = ImageFont.truetype(font_path_bold, 60)
text_color = (255, 255, 255)

# Création de l’objet Draw
draw = ImageDraw.Draw(panini_image)

# Mesurer le texte
bbox = draw.textbbox((0, 0), season_text, font=font_season)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Position centrée
text_position_x = (panini_image.width - text_width) // 2
text_position_y = 50  # marge haute

# Dessiner le texte
draw.text((text_position_x, text_position_y), season_text, fill=text_color, font=font_season)


#PLAYER
player_image_path = f"images/{competition}_{season}_players/{team}_{id_players}.png"
player_image = Image.open(player_image_path)
player_image = player_image.resize((770,1027), Image.Resampling.LANCZOS)
player_image = player_image.convert("RGBA")


panini_image.paste(player_image, ((W-770)//2, 70), mask=player_image)



# Définir les coordonnées du bandeau bas
x0, y0 = 0, H - 250 + 50
x1, y1 = W, H + 50

# Dessiner le bandeau
draw.rectangle([x0, y0, x1, y1 + 100], fill=center_color)




# -----------------------------
# EXPORT
# -----------------------------
out_path = "POSTE_D.png"
panini_image.save(out_path, "PNG")
print("✅ Image enregistrée :", out_path)
