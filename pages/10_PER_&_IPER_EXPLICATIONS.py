import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
season = 2024
competition = "euroleague"
data_dir = os.path.join(os.path.dirname(__file__), '../datas')
sys.path.append(os.path.join(os.path.dirname(__file__), '../fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory
import fonctions as f

st.set_page_config(
    page_title="PER & I_PER EXPLANATIONS",
    layout="wide",  # Enables wide mode by default
    initial_sidebar_state="expanded",  # If you have a sidebar
)

######################### PARAM
st.sidebar.header("SETTINGS")

zoom = st.sidebar.slider(
    "Choose a zoom value for photos and graphs",
    min_value=0.3,
    max_value=1.0,
    step=0.1,
    value=0.7,  # Valeur initiale
)

image_path = f"images/{competition}.png"  # Chemin vers l'image

col1, col2,t1,t2 = st.columns([1, 2.5,0.5,0.5])

with col1 : 
    try:
        image = Image.open(image_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        max_width = 400
        image = image.resize((max_width, int(image.height * (max_width / image.width))))

        # Afficher l'image redimensionnée
        st.image(image)
    except FileNotFoundError:
        
        st.warning(f"L'image pour {competition} est introuvable à l'emplacement : {image_path}") 
    

with col2 : 

    taille_titre = 70*zoom
    st.markdown(
        f'''
        <p style="font-size:{int(taille_titre)}px; text-align: center; padding: 10pxs;">
            <b>PER&I_PER Explications - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

# Define the PER coefficients data
data = {
    "Statistic": [
        "Defensive Rebound", "Offensive Rebound", "Assist", "Steal", "Block",
        "Turnover", "Drawn Foul", "Missed 2-Point Shot", "Missed 3-Point Shot",
        "Defensive Foul", "Technical/Flagrant Foul", "Points Scored (Scorer)",
        "Points Scored (Teammates)", "Points Conceded"
    ],
    "Coefficient": [
        0.85, 1.35, 0.8, 1.33, 0.5, 
        -1.25, 0.6, -0.75, -0.5, 
        -0.7, -1.25, "80% of value", 
        "5% each", -0.1
    ],
    "Explanation": [
        "Routine action, less impactful than offensive rebounds.",
        "Rare and game-changing, highly impactful.",
        "Slightly devalued, as it doesn't always directly lead to scoring.",
        "Prevents opponent scoring and creates a mental advantage.",
        "Important, but impact depends on subsequent ball recovery.",
        "More penalized, as it cancels a scoring opportunity and may lead to fast breaks.",
        "Useful but less impactful than other actions.",
        "Less penalized to balance impact with 3-point shots.",
        "Less penalized to balance impact with 2-point shots.",
        "Penalized less than offensive fouls, but still impactful.",
        "Heavily penalized due to negative impact on the team.",
        "Scorer receives the majority of the PER value.",
        "Teammates receive a small share for their contribution.",
        "Emphasizes the importance of team defense."
    ]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Streamlit app
st.title("Custom PER Coefficients Table")

# Display the DataFrame in a table format
st.dataframe(df, hide_index=True, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    # Title
    st.title("📊 Understanding I_PER: The Balance Between Quantity and Quality")

    # Introduction
    st.markdown("""
    The **I_PER** (Impact PER) is an advanced metric used to evaluate a player's performance by considering both **total production (PER)** and **playing time in minute (Time)**.  
    It is defined by the formula:
    """)

    # Displaying the main formula in LaTeX
    st.latex(r"I\_PER = \frac{PER}{Time^{0.5}}")

    # Section: The Two Extremes
    st.subheader("🔸 The Two Extremes: Raw PER vs Qualitative PER")

    st.markdown("""
    The idea behind I_PER is to balance between two opposite approaches:
    """)

    # Raw PER (Time power 0)
    st.markdown("#### 📌 1. **Raw PER: Ignoring Playing Time**")
    st.latex(r"QUANTITY = \frac{PER}{Time^0} = \frac{PER}{1} = PER")
    st.markdown("""
    - Here, **playing time is not considered**.  
    - This favors players with high playing volume.  
    - A player accumulating stats over **35 minutes** will have a high PER, even if they lack efficiency.
    """)

    # Qualitative PER (Time power 1)
    st.markdown("#### 📌 2. **Qualitative PER: Evaluating Pure Efficiency**")
    st.latex(r"QUALITY = \frac{PER}{Time^1} = \frac{PER}{Time}")
    st.markdown("""
    - Here, every minute played is accounted for, favoring **efficiency per minute**.  
    - Players performing well in short bursts will have a **high PER/Time**, but they don’t experience the physical wear of a long match.  
    - A player effective in **10 minutes** may have an excellent PER/Time, but it doesn’t guarantee they could maintain the same efficiency over **30 minutes**.
    """)

    # I_PER (The Balance Between Both)
    st.subheader("⚖️ I_PER: The Perfect Balance")

    st.latex(r"BALANCE = I\_PER = \frac{PER}{Time^{0.5}}")

    st.markdown("""
    - **I_PER** considers **playing time** but **without overly penalizing it**.  
    - It prevents **overrating over-utilized players** while avoiding **undervaluing players with limited minutes**.  
    - A player with a **high I_PER** has both **strong production and reasonable efficiency**.
    """)

    # Why is I_PER a good metric?
    st.subheader("🎯 Why is I_PER a Good Metric?")

    st.markdown("""
    ✅ It allows **fair comparisons between players with different playing times**.  
    ✅ It **prevents overrating players who excel in short bursts**.  
    ✅ It **reduces the advantage of high-volume players with low efficiency**.  
    ✅ It answers the key question:  
    > **"Who had a real impact on the game, without playing time distorting the perception?"**  
    """)

    # Conclusion
    st.markdown("""
    🏀 **I_PER is an optimal impact indicator**, as it lies **between raw volume and pure efficiency**, offering a more balanced view of individual performance.
    """)

with col2:
    # Main title
    st.title("🏀 Match Rating Based on I_PER")

    # Explanation of I_PER
    st.markdown("""
    **I_PER** is an advanced metric that evaluates a player's performance by balancing **quantity and quality**.  
    It is defined by:  
    """)

    st.latex(r"I\_PER = \frac{PER}{Time^{0.5}}")

    st.markdown("""
    The higher the **I_PER**, the better the player's performance.  
    This system allows you to obtain a **score out of 10** based on their I_PER.
    """)

    # Correspondence table in DataFrame format

    NOTE = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    Min_I_PER =["-∞", -1.2, -0.7, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5, 2.8, 3.1, 3.9]
    Max_I_PER = [-1.3, -0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3, 3.8, "+∞"]
    
    st.subheader("📊 Rating Table")
    coloration = ['#a50026',
            '#bd1726',
            '#d62f27',
            '#e54e35',
            '#f46d43',
            '#f98e52',
            '#fdad60',
            '#fdc776',
            '#fee08b',
            '#fff0a6',
            '#feffbe',
            '#ecf7a6',
            '#d9ef8b',
            '#bfe47a',
            '#a5d86a',
            '#84ca66',
            '#66bd63',
            '#3faa59',
            '#199750',
            '#0c7f43',
            '#006837']
    n, mini,maxi = st.columns([0.3,0.35,0.35])
    
    with n :
        st.markdown(
                    f'''
                    <p style="font-size:{int(20)}px; text-align: center; background-color: black;color: white; padding: 5px; border-radius: 5px;outline: 3px solid white;">
                        <b>Note</b>
                    </p>
                    ''',
                    unsafe_allow_html=True
                )

        for i in range(21) :
            st.markdown(
                    f'''
                    <p style="font-size:{int(20)}px; text-align: center; background-color: {coloration[i]};color: black; padding: 5px; border-radius: 5px;outline: 3px solid white;">
                        <b>{NOTE[i]}/10</b>
                    </p>
                    ''',
                    unsafe_allow_html=True
                )
    with mini :
        st.markdown(
                    f'''
                    <p style="font-size:{int(20)}px; text-align: center; background-color: black;color: white; padding: 5px; border-radius: 5px;outline: 3px solid white;">
                        <b>I_PER minimum</b>
                    </p>
                    ''',
                    unsafe_allow_html=True
                )
        for i in range(21) :
            st.markdown(
                    f'''
                    <p style="font-size:{int(20)}px; text-align: center; background-color: {coloration[i]};color: black; padding: 5px; border-radius: 5px;outline: 3px solid white;">
                        <b>{Min_I_PER[i]}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True
                )

    
    with maxi :
        st.markdown(
                    f'''
                    <p style="font-size:{int(20)}px; text-align: center; background-color: black;color: white; padding: 5px; border-radius: 5px;outline: 3px solid white;">
                        <b>I_PER maximum</b>
                    </p>
                    ''',
                    unsafe_allow_html=True
                )
        for i in range(21) :
            st.markdown(
                    f'''
                    <p style="font-size:{int(20)}px; text-align: center; background-color: {coloration[i]};color: black; padding: 5px; border-radius: 5px;outline: 3px solid white;">
                        <b>{Max_I_PER[i]}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True
                )
    


