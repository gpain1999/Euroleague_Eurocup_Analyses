from PIL import Image

def png_to_ico(png_path, ico_path, sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]):
    """
    Convertit une image PNG en ICO.
    
    :param png_path: Chemin du fichier PNG source
    :param ico_path: Chemin où enregistrer le fichier ICO
    :param sizes: Tailles d'icône à inclure dans le fichier ICO (par défaut, plusieurs tailles)
    """
    # Ouvre l'image PNG
    img = Image.open(png_path)
    
    # Convertit en ICO et sauvegarde
    img.save(ico_path, format='ICO', sizes=sizes)
    print(f"Icône enregistrée dans {ico_path}")

# Utilisation de la fonction
png_path = "images/euroleague_little.png"
ico_path = "images/euroleague_little.ico"
png_to_ico(png_path, ico_path)
