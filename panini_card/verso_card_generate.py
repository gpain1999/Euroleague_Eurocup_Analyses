from PIL import Image, ImageOps,ImageDraw,ImageFont
import pandas as pd
import os

competition = "euroleague"
season = 2024
width = 850 - 80
height = round(850 * 89/63) - 80
border_size = 40
border_color = "#FEB673"
background_color = "black"

panini_image = Image.new("RGB", (width, height), background_color)

folder_path = f"panini_card/{competition}_{season}"


#COMPETITON
competition_image_path = f"images/{competition}_little.png"
competition_image = Image.open(competition_image_path)
new_size = (500,500)
competition_image = competition_image.resize(new_size, Image.Resampling.LANCZOS)
competition_image = competition_image.convert("RGBA")

panini_image.paste(competition_image, ((width - 500)//2, (height - 500)//2), mask=competition_image)

panini_image = ImageOps.expand(panini_image, border=border_size, fill=border_color)


panini_image.save(f"{folder_path}/_verso.png")