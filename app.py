import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline
import json

# ====================== MODERN CONFIG ======================
st.set_page_config(
    page_title="FischID Pro • KI Fisch-Erkennung",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Fancy CSS
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%); color: #e2e8f0; }
    h1 { font-size: 3.5rem; background: linear-gradient(90deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border-radius: 12px; height: 3em; font-weight: bold; }
    .result { background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 16px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🐟 FischID Pro")
st.markdown("**KI-Fisch-Erkennung mit modernem vortrainiertem Modell**")

# ====================== MODELL LADEN ======================
@st.cache_resource(show_spinner="Lade starkes Fisch-Modell...")
def load_model():
    return pipeline(
        "image-classification", 
        model="jeemsterri/fish_classification",
        top_k=5
    )

classifier = load_model()

# fish_data.json (kann dieselbe wie in der ersten App sein)
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# ====================== UPLOAD / KAMERA ======================
st.subheader("📸 Foto aufnehmen oder hochladen")

col1, col2 = st.columns(2)
with col1:
    camera = st.camera_input("Kamera benutzen")
with col2:
    uploaded = st.file_uploader("Oder Bild hochladen", type=["jpg", "jpeg", "png"])

image = None
if camera is not None:
    image = Image.open(camera).convert("RGB")
elif uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

if image is not None:
    st.image(image, caption="Dein Foto", use_column_width=True)

    with st.spinner("🔍 KI analysiert den Fisch..."):
        results = classifier(image)

    # Beste Vorhersage
    top = results[0]
    predicted_label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    # Einfaches Mapping auf deutsche Namen
    mapping = {
        "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch", "Carp": "Karpfen",
        "Trout": "Forelle", "Catfish": "Wels", "Eel": "Aal", "Bream": "Brassen"
    }
    fish_name = mapping.get(predicted_label, predicted_label)

    if confidence >= 80:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {fish_name}")
        st.success(f"**Sicherheit:** {confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Aus welchem Bundesland kommst du?", 
                                  options=list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        
        if info:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with c2:
                st.metric("Schonzeit", info.get('schonzeit', "Keine"))
        else:
            st.info("Keine spezifischen Regelungen in der Datenbank für diese Art im gewählten Bundesland.")
    else:
        st.warning(f"Die KI ist sich nur zu {confidence:.1f}% sicher. Bitte versuche ein klareres Foto.")

# Footer
st.markdown("---")
st.markdown("**FischID Pro** — Moderne KI-App mit vortrainiertem Modell von Hugging Face")
st.caption("Perfekt für Schulprojekte • Kamera-fähig • Schönes Design")
