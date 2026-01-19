import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# configuration sobre
st.set_page_config(page_title="reconnaissance de chiffres", layout="wide")

# gestion de la réinitialisation via la session
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

def reset_canvas():
    st.session_state.canvas_key += 1

# chargement silencieux du moteur
@st.cache_resource
def load_app_engine():
    return tf.keras.models.load_model('DigitRecognizerV2.h5')

engine = load_app_engine()

st.title("dessine-moi un chiffre")
st.write("écrivez un chiffre dans la zone noire, et je vais essayer de le reconnaître.")

# section d'aide simplifiée
with st.expander("comment obtenir un bon résultat ?"):
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", 
             caption="exemples de tracés optimaux", width=350)
    st.info("astuce : dessinez un chiffre assez grand et bien centré pour une reconnaissance précise.")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18, 
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        display_toolbar=False, # cache la barre d'outils technique
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    if st.button("recommencer", on_click=reset_canvas):
        st.rerun()

with col2:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        
        if np.any(img > 0):
            # préparation invisible des données
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img_rescaled = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            features = img_rescaled.reshape(1, 784).astype('float32') / 255.0

            if st.button('identifier', type="primary"):
                # verbose=0 pour cacher les logs techniques
                probs = engine.predict(features, verbose=0) 
                result = np.argmax(probs)
                
                st.subheader("ma réponse :")
                st.header(f"c'est un **{result}**")
                
                confidence = np.max(probs) * 100
                if confidence > 80:
                    st.success(f"j'en suis sûr à {confidence:.0f}%")
                else:
                    st.warning(f"j'hésite un peu ({confidence:.0f}% de certitude)")
        else:
            st.info("en attente d'un tracé...")

st.markdown("---")
st.caption("application de démonstration • 2026")
