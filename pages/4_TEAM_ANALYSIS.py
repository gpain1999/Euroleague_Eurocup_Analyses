import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image

season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

st.set_page_config(
    page_title="PLAYER ANALYSIS",
    layout="wide",  # Active le Wide mode par défaut
    initial_sidebar_state="expanded",  # Si vous avez une barre latérale
)


import fonctions as f

image_path = f"images/{competition}.png"  # Chemin vers l'image

try:
    image = Image.open(image_path)
    # Redimensionner l'image (par exemple, largeur de 300 pixels)
    max_width = 400
    image = image.resize((max_width, int(image.height * (max_width / image.width))))

    # Afficher l'image redimensionnée
    st.image(image, caption=f"Rapport pour {competition}")
except FileNotFoundError:
    st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

    
st.title("Team Analysis - by gpain1999 ")
