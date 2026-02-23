import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import pandas as pd

# configuration
st.set_page_config(page_title="Digit Recognizer", layout="wide")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

def reset_canvas():
    st.session_state.canvas_key += 1

@st.cache_resource
def load_app_engine():
    return tf.keras.models.load_model('DigitRecognizerV2.h5')

engine = load_app_engine()

st.title("Dessiner un chiffre ( 0 - 9 )")
st.write("écrivez un chiffre dans la zone noire, et je vais essayer de le prédire.")

with st.expander("Sur quoi a t-il été entrainé ?"):
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", 
             caption="exemples de la data base", width=350)

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
        display_toolbar=False,
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    if st.button("Delete", on_click=reset_canvas):
        st.rerun()

with col2:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        
        if np.any(img > 0):
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img_rescaled = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            features = img_rescaled.reshape(1,28,28,1).astype('float32') / 255.0

            if st.button('identifier', type="primary"):
                probs = engine.predict(features, verbose=0)[0] 
                result = np.argmax(probs)
                confidence = np.max(probs) * 100
                
                st.subheader("Prédiction :")
                st.header(f"c'est un **{result}**")
                
                # affichage du pourcentage de confiance
                st.write(f"indice de confiance : **{confidence:.2f}%**")
                
                # préparation des données pour le diagramme
                chart_data = pd.DataFrame(
                    probs, 
                    index=[str(i) for i in range(10)], 
                    columns=["certitude"]
                )
                
                st.bar_chart(chart_data)

                if confidence < 80:
                    st.write("L'IA n'est pas sûre !!")
       

st.markdown("---")
st.caption("Page de démonstration @Evrard LECUREUR • 2026")
