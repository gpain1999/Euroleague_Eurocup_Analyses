import streamlit as st
from PIL import Image

st.set_page_config(page_title="Euroleague Data", layout="wide")
competition = "euroleague"

image_path = f"images/{competition}.png"  # Chemin vers l'image
image_gp = f"images/logo_gp.png"

try:
    # Charger la première image
    image = Image.open(image_path)
    # Redimensionner la première image (par exemple, largeur de 600 pixels)
    max_width1 = 600
    image = image.resize((max_width1, int(image.height * (max_width1 / image.width))))

    # # Charger la deuxième image
    # image2 = Image.open(image_gp)
    # # Redimensionner la deuxième image (par exemple, largeur de 100 pixels)
    # max_width2 = 100
    # image2 = image2.resize((max_width2, int(image2.height * (max_width2 / image2.width))))

    # Afficher les images côte à côte dans deux colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=f"Rapport pour {competition}")

    # with col2:
        # st.image(image2, caption=f"Rapport pour {competition}")

except FileNotFoundError:
    st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

st.title("Application Euroleague - by gpain1999")
#st.sidebar.success("Choisissez une page ci-dessus.")