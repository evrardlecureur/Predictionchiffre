import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Configuration
st.set_page_config(page_title="IA Digit Recognizer", layout="wide")

# --- LOGIQUE DE RÉINITIALISATION ---
# On initialise un compteur dans la session s'il n'existe pas
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

def reset_canvas():
    # On incrémente la clé pour forcer Streamlit à recréer un canvas vide
    st.session_state.canvas_key += 1

# -----------------------------------

st.title("🔢 Reconnaissance de Chiffres par IA")
st.markdown("---")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('DigitRecognizerV2.h5')

model = load_my_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖍️ Zone de dessin")
    
    # Le Canvas utilise maintenant la clé dynamique de la session_state
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}", # La clé change au clic sur reset
    )
    
    # Bouton de réinitialisation qui appelle notre fonction
    if st.button("🗑️ Effacer le tableau", on_click=reset_canvas):
        st.info("Tableau réinitialisé !")

with col2:
    st.subheader("🤖 Analyse de l'IA")
    
    # On vérifie si l'utilisateur a dessiné quelque chose (pas juste un tableau vide)
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        
        # On vérifie s'il y a des pixels blancs (pour éviter de prédire sur du vide)
        if np.any(img > 0):
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img_rescaled = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            features = img_rescaled.reshape(1, 784).astype('float32') / 255.0

            if st.button('🔍 Prédire maintenant', type="primary"):
                probs = model.predict(features)[0]
                pred_class = np.argmax(probs)
                confidence = np.max(probs) * 100
                
                st.metric(label="Chiffre prédit", value=pred_class)
                st.write(f"**Confiance :** {confidence:.2f}%")
                st.progress(int(confidence))
                st.bar_chart(probs)
        else:
            st.info("Le tableau est vide. Dessinez un chiffre pour commencer.")
