import streamlit as st
from PIL import Image

st.set_page_config(page_title="Euroleague Data", layout="wide")
competition = "euroleague"

image_path = f"images/{competition}.png"  # Chemin vers l'image


try:
    image = Image.open(image_path)
    # Redimensionner l'image (par exemple, largeur de 300 pixels)
    max_width = 600
    image = image.resize((max_width, int(image.height * (max_width / image.width))))

    # Afficher l'image redimensionnée
    st.image(image, caption=f"Rapport pour {competition}")
except FileNotFoundError:
    st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

st.title("Application Euroleague - by gpain1999")
st.sidebar.success("Choisissez une page ci-dessus.")
