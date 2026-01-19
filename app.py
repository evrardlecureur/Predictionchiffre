import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Configuration large pour un rendu "Dashboard"
st.set_page_config(page_title="IA Digit Recognizer", layout="wide")

st.title("🔢 Reconnaissance de Chiffres par IA")
st.markdown("---")

# 1. Chargement du modèle
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('DigitRecognizerV2.h5')

model = load_my_model()

# 2. Création de deux colonnes (Gauche: Dessin, Droite: Résultats)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖍️ Zone de dessin")
    st.write("Dessinez un chiffre bien au centre :")
    
    # Le Canvas
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20, # Trait un peu plus épais pour mieux simuler MNIST
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Bouton de réinitialisation (Streamlit recharge la page par défaut, ce qui vide le canvas)
    if st.button("🗑️ Effacer le tableau"):
        st.rerun()

with col2:
    st.subheader("🤖 Analyse de l'IA")
    
    if canvas_result.image_data is not None:
        # Prétraitement
        img = canvas_result.image_data.astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img_rescaled = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Préparation pour le modèle (Vecteur plat 1x784)
        features = img_rescaled.reshape(1, 784).astype('float32') / 255.0

        if st.button('🔍 Prédire maintenant', type="primary"):
            # Prédiction
            probs = model.predict(features)[0]
            pred_class = np.argmax(probs)
            confidence = np.max(probs) * 100
            
            # Affichage stylisé
            st.metric(label="Chiffre prédit", value=pred_class)
            st.write(f"**Indice de confiance :** {confidence:.2f}%")
            
            # Barre de progression pour la confiance
            st.progress(int(confidence))
            
            # Graphique des probabilités pour les autres chiffres
            st.bar_chart(probs)
    else:
        st.info("Dessinez quelque chose à gauche pour lancer l'analyse.")

st.markdown("---")
st.caption("Modèle entraîné sur le dataset MNIST • Déployé via Streamlit Cloud")
