import streamlit as st
from PIL import Image
import hmac

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.
    
st.set_page_config(page_title="Euroleague Data", layout="wide")
competition = "euroleague"

image_path = f"images/{competition}.png"  # Chemin vers l'image

try:
    # Charger la première image
    image = Image.open(image_path)
    # Redimensionner la première image (par exemple, largeur de 600 pixels)
    max_width1 = 600
    image = image.resize((max_width1, int(image.height * (max_width1 / image.width))))


    # Afficher les images côte à côte dans deux colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=f"Rapport pour {competition}")


except FileNotFoundError:
    st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 

st.title("Application Euroleague - by gpain1999")
