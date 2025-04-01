from PIL import Image
import os

competition = "euroleague"
season = 2024
# Dossier contenant les images
dossier_images = f"panini_card/{competition}_{season}"

# Liste toutes les images dans le dossier
images = [f for f in os.listdir(dossier_images) if f.lower().endswith(("png", "jpg", "jpeg"))]

# Vérifie la présence du verso
verso_nom = "_verso.png"
if verso_nom not in images:
    raise FileNotFoundError(f"L'image '{verso_nom}' est introuvable dans le dossier.")

# Trie les images sauf le verso
images.remove(verso_nom)
images.sort()

# Charge les images avec PIL
verso = Image.open(os.path.join(dossier_images, verso_nom)).convert("RGB")
image_list = []

for img_name in images:
    img_path = os.path.join(dossier_images, img_name)
    img = Image.open(img_path).convert("RGB")
    image_list.append(img)
    image_list.append(verso)  # Ajoute le verso après chaque image

# Sauvegarde le PDF
if image_list:
    premiere_image = image_list.pop(0)
    pdf_path = os.path.join(dossier_images, "output.pdf")
    premiere_image.save(pdf_path, save_all=True, append_images=image_list)
    print(f"✅ PDF créé avec succès: {pdf_path}")
else:
    print("❌ Aucune image valide trouvée pour la création du PDF.")
