import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Configuration de la page
st.set_page_config(page_title="Digit Predictor", layout="centered")

st.title("🔢 Reconnaissance de Chiffres")
st.write("Dessinez un chiffre ci-dessous pour obtenir une prédiction en temps réel.")

# 1. Chargement du modèle (mis en cache pour la performance)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('DigitRecognizerV2.h5')

model = load_my_model()

# 2. Création de la zone de dessin (Canvas)
# On force une taille carrée pour faciliter le redimensionnement
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 3. Logique de traitement et prédiction
if canvas_result.image_data is not None:
    # Récupérer l'image du canvas (RGBA)
    img = canvas_result.image_data.astype(np.uint8)
    
    # Prétraitement pour correspondre à ton ancien code Flask :
    # a. Convertir en niveaux de gris
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # b. Redimensionner en 28x28 pixels
    img_rescaled = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # c. Transformer en vecteur plat (28*28 = 784) comme dans ton ancien code : x.reshape(1, -1)
    features = img_rescaled.reshape(1, 784).astype('float32')
    
    # d. Normalisation (si ton modèle a été entraîné avec des valeurs entre 0 et 1)
    features = features / 255.0

    # Bouton pour lancer la prédiction
    if st.button('Prédire'):
        probs = model.predict(features)[0]
        pred_class = int(np.argmax(probs))
        maxprob = float(np.max(probs)) * 100
        
        # Affichage des résultats
        st.subheader(f"Résultat : {pred_class}")
        st.write(f"Confiance : {maxprob:.2f}%")
        
        # Petit bonus visuel : graphique des probabilités
        st.bar_chart(probs)
